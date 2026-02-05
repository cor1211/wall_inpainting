# Giải Thích Chi Tiết Pipeline Wall Inpainting

> **Phiên bản:** 1.0.0  
> **Ngày cập nhật:** 2026-02-05  
> **Mục đích:** Tài liệu kỹ thuật đầy đủ bằng tiếng Việt, giải thích mọi luồng hoạt động

---

## Mục Lục

1. [Tổng Quan Kiến Trúc](#1-tổng-quan-kiến-trúc)
2. [Pipeline Training](#2-pipeline-training)
3. [Pipeline Validation](#3-pipeline-validation)
4. [Pipeline Inference](#4-pipeline-inference)
5. [Giải Thích Các Shape](#5-giải-thích-các-shape)
6. [Các Vấn Đề Và Giải Pháp](#6-các-vấn-đề-và-giải-pháp)

---

## 1. Tổng Quan Kiến Trúc

### 1.1 Các Thành Phần Chính

Hệ thống Wall Inpainting bao gồm các thành phần sau:

| Thành phần | Vai trò | Trạng thái khi training |
|------------|---------|-------------------------|
| **VAE (Variational Autoencoder)** | Nén ảnh từ pixel space sang latent space và ngược lại | ❄️ Frozen (không train) |
| **Text Encoder (CLIP)** | Chuyển đổi text prompt thành embedding vector | ❄️ Frozen |
| **UNet2DConditionModel** | Model chính dự đoán noise | ⚡ Trainable (LoRA) |
| **LoRA Adapters** | Low-rank adapters để fine-tune UNet hiệu quả | ⚡ Trainable |
| **IP-Adapter** | Inject image features vào cross-attention | ❌ Không dùng khi training |
| **ControlNet (Depth)** | Điều khiển layout dựa trên depth map | ❌ Không dùng khi training |

### 1.2 Tại Sao Lại Thiết Kế Như Vậy?

**VAE Frozen:** VAE đã được pretrain rất tốt trên hàng tỷ ảnh. Việc fine-tune VAE sẽ:
- Tốn rất nhiều VRAM
- Có thể làm hỏng khả năng encode/decode tổng quát
- Không cần thiết vì LoRA đủ để điều chỉnh hành vi

**Text Encoder Frozen:** Tương tự VAE, CLIP text encoder đã hiểu ngữ nghĩa rất tốt.

**Chỉ Train LoRA:** 
- UNet có ~860 triệu tham số
- LoRA chỉ thêm ~1.6 triệu tham số (0.18%)
- Tiết kiệm VRAM đáng kể (có thể train trên RTX 3060 Ti 12GB)

---

## 2. Pipeline Training

### 2.1 Bước 1: Load Dữ Liệu (WallInpaintingDataset)

#### 2.1.1 Đọc File

```
Đầu vào:
- image_path: đường dẫn tới ảnh RGB gốc
- mask_path: đường dẫn tới mask (vùng trắng = wall cần inpaint)

Đầu ra:
- image: PIL Image RGB, kích thước gốc (ví dụ: 1920×1080)
- mask: PIL Image Grayscale, kích thước gốc
```

**Tại sao?** File ảnh trên disk có thể có bất kỳ kích thước nào. Ta cần load chúng trước khi xử lý.

#### 2.1.2 Mask Erosion (Tùy chọn)

```python
if mask_erosion_size > 0:
    mask = cv2.erode(mask, kernel=(5, 5), iterations=1)
```

**Đầu vào:** mask PIL Image  
**Đầu ra:** mask PIL Image (đã thu nhỏ biên)

**Tại sao Erosion?**
- Dataset có nhiều vùng bị segment nhầm (không phải wall)
- Erosion thu nhỏ mask 5 pixel mỗi phía
- Loại bỏ các vùng nhỏ ở biên (thường là lỗi segmentation)

#### 2.1.3 Hybrid Color Strategy (Chiến lược lấy màu)

```python
if random.random() < random_color_prob:  # 30% ngẫu nhiên
    dominant_color = generate_random_color()  # RGB trong [30, 240]
else:  # 70% trích xuất
    dominant_color = extract_dominant_color(image, mask)  # Median từ wall pixels
```

**Đầu vào:** image, mask  
**Đầu ra:** dominant_color: tuple (R, G, B), giá trị 0-255

**Tại sao Hybrid?**
- **70% Extracted:** Màu thực tế từ tường → realistic, nhưng model chỉ thấy màu "natural"
- **30% Random:** Màu ngẫu nhiên → model học được BẤT KỲ màu nào, tăng generalization

**Tại sao dùng Median?**
- Mean dễ bị ảnh hưởng bởi outliers (bóng, highlight)
- Median robust hơn, cho màu "trung tâm" của vùng wall
- K-means chính xác nhất nhưng chậm

#### 2.1.4 Tạo Solid Color Reference

```python
reference_image = create_solid_color_reference(
    dominant_color,
    size=(512, 512)
)
# Thêm noise Gaussian (std=8) cho texture
# Thêm gradient ánh sáng (0.95 → 1.05)
```

**Đầu vào:** dominant_color (R, G, B)  
**Đầu ra:** reference_image: PIL Image RGB 512×512

**Tại sao thêm noise và gradient?**
- CLIP được train trên ảnh tự nhiên, không phải solid colors
- Solid color thuần túy có rất ít features → CLIP không extract được gì hữu ích
- Noise + gradient giúp CLIP có thêm patterns để encode

#### 2.1.5 Image Transforms

```python
image_transforms = Compose([
    Resize(512),           # Resize cạnh nhỏ nhất về 512
    CenterCrop(512),       # Cắt trung tâm 512×512
    ToTensor(),            # [H,W,3] → [3,H,W], giá trị [0,1]
    Normalize([0.5], [0.5]) # [0,1] → [-1,1]
])
```

**Đầu vào:** PIL Image RGB (bất kỳ size)  
**Đầu ra:** Tensor [3, 512, 512], dtype=float32, range=[-1, 1]

**Tại sao 512×512?**
- Stable Diffusion 1.5 được train trên 512×512
- Latent space sẽ là 64×64 (compression ratio 8x)
- Các kích thước khác có thể hoạt động nhưng không tối ưu

**Tại sao Normalize về [-1, 1]?**
- VAE được train với input range [-1, 1]
- Giúp gradient ổn định hơn khi training

#### 2.1.6 Reference Image Transform

```python
reference_transforms = Compose([
    Resize(224),
    CenterCrop(224),
    ToTensor(),
    Normalize([0.5], [0.5])
])
```

**Đầu ra:** reference_tensor [3, 224, 224], range=[-1, 1]

**Tại sao 224×224?**
- IP-Adapter sử dụng CLIP ViT-H/14 để encode reference image
- CLIP ViT được train trên 224×224 (tiêu chuẩn ImageNet)
- Nếu dùng size khác, CLIP phải interpolate position embeddings → giảm chất lượng

### 2.2 Bước 2: Collate Function (Gom batch)

```python
def collate_fn(examples, tokenizer):
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    masks = torch.stack([e["mask"] for e in examples])
    reference_images = torch.stack([e["reference_image"] for e in examples])
    
    # Tokenize captions
    inputs = tokenizer(captions, padding="max_length", max_length=77, ...)
```

**Đầu vào:** List của dict, mỗi dict là 1 sample  
**Đầu ra:**
- `pixel_values`: [B, 3, 512, 512]
- `masks`: [B, 1, 512, 512]
- `reference_images`: [B, 3, 224, 224]
- `input_ids`: [B, 77]
- `dominant_colors`: [B, 3]

**Tại sao B (batch size)?**
- B=4 trong config hiện tại
- B càng lớn → training càng stable (gradient averaging)
- Nhưng B lớn → tốn nhiều VRAM hơn
- B=4 là cân bằng cho RTX 3060 Ti 12GB

**Tại sao max_length=77?**
- CLIP text encoder có maximum context length = 77 tokens
- Bao gồm [START] token + 75 word tokens + [END] token
- Padding đến 77 để tất cả samples trong batch có cùng length

### 2.3 Bước 3: VAE Encoding

#### 2.3.1 Encode Original Image

```python
latents = vae.encode(pixel_values).latent_dist.sample()
latents = latents * 0.18215  # scaling factor
```

**Đầu vào:** pixel_values [B, 3, 512, 512]  
**Đầu ra:** latents [B, 4, 64, 64]

**Giải thích shape:**
- **B:** batch size (4)
- **4:** số channels trong latent space (VAE architecture quyết định)
- **64×64:** spatial dimensions, bằng 512/8 (compression ratio 8x)

**Tại sao 4 channels?**
- SD 1.5 VAE sử dụng latent dimension = 4
- Đủ để encode thông tin ảnh (thực nghiệm cho thấy 3 channels không đủ, >4 thừa)

**Tại sao × 0.18215?**
- Đây là `vae.config.scaling_factor`
- VAE output có variance khá lớn
- Scaling này normalize latents về range phù hợp cho UNet
- Giá trị 0.18215 được tính từ training data của SD

#### 2.3.2 Tạo Masked Image và Encode

```python
masked_image = pixel_values * (1 - mask)
masked_image_latents = vae.encode(masked_image).latent_dist.sample() * 0.18215
```

**Đầu vào:** 
- pixel_values [B, 3, 512, 512]
- mask [B, 1, 512, 512]

**Đầu ra:** masked_image_latents [B, 4, 64, 64]

**Giải thích:**
- `(1 - mask)`: mask là 1 ở vùng wall, nên (1-mask) = 0 ở vùng wall
- `pixel_values * (1 - mask)`: vùng wall bị blackout, các vùng khác giữ nguyên
- Encode ảnh đã masked để UNet biết context xung quanh

#### 2.3.3 Resize Mask cho Latent Space

```python
mask = F.interpolate(mask, size=(64, 64), mode="nearest")
```

**Đầu vào:** mask [B, 1, 512, 512]  
**Đầu ra:** mask [B, 1, 64, 64]

**Tại sao dùng "nearest"?**
- Mask là binary (0 hoặc 1)
- Bilinear/bicubic sẽ tạo ra các giá trị trung gian (0.5, 0.3, ...)
- Nearest giữ nguyên tính binary của mask

### 2.4 Bước 4: Noise Sampling

```python
noise = torch.randn_like(latents)  # [B, 4, 64, 64]
timesteps = torch.randint(0, 1000, (B,))  # [B]
noisy_latents = scheduler.add_noise(latents, noise, timesteps)
```

**Giải thích từng dòng:**

1. **noise = randn_like(latents):**
   - Sample noise từ N(0, 1)
   - Shape: [B, 4, 64, 64] (giống latents)
   - Đây chính là **GROUND TRUTH** mà model cần dự đoán

2. **timesteps = randint(0, 1000):**
   - Sample ngẫu nhiên timestep cho mỗi sample trong batch
   - Range [0, 1000) tương ứng với noise schedule
   - t=0: ảnh sạch, t=999: pure noise

3. **scheduler.add_noise():**
   - Công thức: `z_t = √(α_t) * z_0 + √(1-α_t) * ε`
   - `α_t` giảm dần theo t (cumulative product của 1-β)
   - t càng lớn → z_t càng gần với pure noise

### 2.5 Bước 5: Xây Dựng UNet Input

```python
latent_model_input = torch.cat([noisy_latents, mask, masked_image_latents], dim=1)
```

**Đầu vào:**
- noisy_latents: [B, 4, 64, 64]
- mask: [B, 1, 64, 64]
- masked_image_latents: [B, 4, 64, 64]

**Đầu ra:** latent_model_input: [B, 9, 64, 64]

**Tại sao 9 channels?**
- **4 channels (noisy_latents):** Latent cần denoise
- **1 channel (mask):** Cho UNet biết vùng nào cần inpaint
- **4 channels (masked_image_latents):** Context xung quanh vùng inpaint

**Đây là đặc trưng của SD Inpainting UNet:**
- UNet thông thường: input 4 channels
- UNet Inpainting: input 9 channels (first conv layer được modify)

### 2.6 Bước 6: Text Encoding

#### 2.6.1 Zero-Prompt Mode

```python
if unconditional_training:
    uncond_tokens = tokenizer([""], padding="max_length", max_length=77, ...)
    encoder_hidden_states = text_encoder(uncond_tokens)[0]  # [1, 77, 768]
    encoder_hidden_states = encoder_hidden_states.expand(B, -1, -1)  # [B, 77, 768]
```

**Tại sao dùng empty string?**
- Zero-Prompt strategy: loại bỏ text guidance
- Buộc model dựa vào image conditioning (IP-Adapter, mask, depth)
- Empty prompt → text embeddings không có semantic information

#### 2.6.2 Normal Mode

```python
else:
    encoder_hidden_states = text_encoder(input_ids)  # [B, 77, 768]
```

**Shape [B, 77, 768]:**
- B: batch size
- 77: số tokens (max length)
- 768: embedding dimension của CLIP text encoder

### 2.7 Bước 7: UNet Forward Pass

```python
noise_pred = unet(
    latent_model_input,      # [B, 9, 64, 64]
    timesteps,               # [B]
    encoder_hidden_states,   # [B, 77, 768]
).sample                     # [B, 4, 64, 64]
```

**UNet Architecture (simplified):**
```
Input: [B, 9, 64, 64]
    ↓
ConvIn: 9 → 320 channels
    ↓
DownBlocks: 64→32→16→8 spatial, increasing channels
    ↓
MidBlock: [B, 1280, 8, 8]
    ↓
UpBlocks: 8→16→32→64 spatial, skip connections from down
    ↓
ConvOut: → 4 channels
    ↓
Output: [B, 4, 64, 64]
```

**Cross-Attention trong UNet:**
- Q (Query): từ image features
- K, V (Key, Value): từ text embeddings [B, 77, 768]
- Output: image features được modulate bởi text

**LoRA Injection:**
- LoRA được thêm vào các attention layers (to_k, to_q, to_v, to_out)
- Formula: `W' = W + BA`, với B [rank, out], A [in, rank]
- rank=8 trong config → mỗi LoRA layer thêm rất ít params

### 2.8 Bước 8: Loss Computation

```python
loss = F.mse_loss(noise_pred, noise, reduction="mean")
```

**Đầu vào:**
- noise_pred: [B, 4, 64, 64] (UNet dự đoán)
- noise: [B, 4, 64, 64] (Ground Truth)

**Đầu ra:** loss: scalar (giá trị duy nhất)

**Tại sao MSE?**
- Diffusion models được derive từ variational bound
- MSE giữa predicted noise và actual noise là simplified objective
- Minimize MSE ≈ Maximize log-likelihood

**Reduction="mean":**
- Average loss qua tất cả elements: B × 4 × 64 × 64 = 65,536 elements
- Cho loss value ổn định, không phụ thuộc vào batch size

### 2.9 Bước 9: Backpropagation

```python
accelerator.backward(loss)
accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm=1.0)
optimizer.step()
lr_scheduler.step()
optimizer.zero_grad()
```

**Chỉ LoRA parameters được update:**
- VAE gradients: không compute (frozen)
- Text encoder gradients: không compute (frozen)
- UNet base weights: không update (frozen)
- LoRA weights (A, B matrices): Được update

---

## 3. Pipeline Validation

### 3.1 Load Validation Data

```python
val_dataset = WallInpaintingDataset(split="validation", max_samples=50)
sample_data = val_dataset[i]
```

**Tương tự training dataset nhưng:**
- Không có random flip
- Không có color jitter
- Không có random color (luôn dùng extracted color)

### 3.2 Tensor to PIL Conversion

```python
source_image = tensor_to_pil(sample_data["pixel_values"])
# [3, 512, 512] tensor → PIL Image 512×512 RGB
```

**Công thức chuyển đổi:**
```python
tensor = (tensor + 1) / 2  # [-1,1] → [0,1]
tensor = tensor.clamp(0, 1)
tensor = (tensor * 255).byte()  # → [0,255]
image = Image.fromarray(tensor.permute(1,2,0).numpy())
```

### 3.3 Inference (Không dùng IP-Adapter)

```python
result = pipeline(
    prompt="",
    image=source_image,
    mask_image=mask_image,
    num_inference_steps=20,
    generator=generator,
).images[0]
```

**Lưu ý quan trọng:**
- Validation dùng StableDiffusionInpaintPipeline chuẩn
- KHÔNG có IP-Adapter (khác với production inference)
- KHÔNG có ControlNet
- Mục đích: đánh giá LoRA thuần, không có image conditioning

### 3.4 Compute Color Metrics

```python
metrics = compute_color_fidelity_metrics(reference_image, result, mask_image)
```

**Các metrics được tính:**

1. **LAB Distance:**
   ```python
   lab_distance = sqrt((L_ref - L_out)² + (a_ref - a_out)² + (b_ref - b_out)²)
   ```
   - Khoảng cách Euclidean trong không gian LAB
   - LAB là perceptually uniform (khác biệt đều cho perception)

2. **Delta-E (CIE76):**
   ```python
   delta_e = lab_distance  # CIE76 = Euclidean in LAB
   ```
   - < 1: không phân biệt được
   - 1-2: chỉ expert thấy
   - 2-10: khác biệt rõ
   - > 10: màu hoàn toàn khác

3. **Hue Error:**
   ```python
   hue_error = abs(H_ref - H_out)  # trong HSV space
   ```
   - Sai lệch về hue (màu sắc)
   - 0 = hoàn hảo, 180 = màu đối

4. **Lightness Difference:**
   ```python
   lightness_diff = abs(L_ref - L_out) / 100
   ```
   - Sai lệch về độ sáng
   - Normalize về [0, 1]

---

## 4. Pipeline Inference (Production)

### 4.1 Khác Biệt So Với Training

| Aspect | Training | Inference |
|--------|----------|-----------|
| IP-Adapter | ❌ Không dùng | ✅ Có dùng |
| ControlNet | ❌ Không dùng | ✅ Có dùng (depth) |
| Reference Image | Derived từ original | User cung cấp |
| Prompt | Empty (Zero-Prompt) | Có thể empty hoặc có text |

### 4.2 Preprocessing

```python
source_resized = source_image.resize((512, 512), LANCZOS)
mask_resized = mask_image.resize((512, 512), NEAREST)
reference_resized = reference_image.resize((224, 224), LANCZOS)
```

**Tại sao reference 224×224?**
- IP-Adapter dùng CLIP ViT để encode
- CLIP ViT expect 224×224 input

### 4.3 Depth Estimation

```python
depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")
depth_map = depth_estimator(source_resized)["depth"]
```

**Đầu ra:** depth_map PIL Image, grayscale, 512×512

**Tại sao dùng depth?**
- Depth giữ structure của phòng (góc tường, đồ vật)
- ControlNet dùng depth để đảm bảo output không thay đổi layout
- Chỉ thay đổi màu sắc/texture, không thay đổi hình dạng

### 4.4 IP-Adapter Processing

```python
# Bên trong pipeline
image_embeds = clip_image_encoder(reference_resized)  # [1, 257, 1024]
projected_embeds = image_projection(image_embeds)      # [1, 4, 768]
```

**Shape [1, 257, 1024]:**
- 1: batch size
- 257: 256 patches (14×14 grid trong 224×224 image) + 1 CLS token
- 1024: CLIP ViT-H embedding dimension

**Shape [1, 4, 768]:**
- IP-Adapter projector reduce từ 257 tokens xuống 4
- 768 match với UNet attention dimension
- 4 tokens đủ để encode color/style information

### 4.5 Denoising Loop

```python
for t in scheduler.timesteps:  # T → 0
    # 1. ControlNet extracts depth features
    down_block_res, mid_block_res = controlnet(latent, t, encoder_hidden_states, depth_map)
    
    # 2. UNet predicts noise
    noise_pred = unet(
        latent,
        t,
        encoder_hidden_states,
        down_block_additional_residuals=down_block_res,
        mid_block_additional_residual=mid_block_res,
        added_cond_kwargs={"image_embeds": ip_adapter_embeds},
    )
    
    # 3. Scheduler step
    latent = scheduler.step(noise_pred, t, latent).prev_sample
```

**IP-Adapter Injection:**
- IP-Adapter modify cross-attention
- Original: `Attention = softmax(Q @ K_text^T) @ V_text`
- With IP: `Attention = softmax(Q @ K_text^T) @ V_text + scale * softmax(Q @ K_image^T) @ V_image`
- scale=1.0 → image features có ảnh hưởng mạnh

### 4.6 VAE Decode

```python
image = vae.decode(latents / 0.18215).sample
# [1, 4, 64, 64] → [1, 3, 512, 512]
```

**Tại sao chia cho 0.18215?**
- Ngược lại với encode (đã nhân 0.18215)
- Đưa latents về range mà VAE decoder expect

---

## 5. Giải Thích Các Shape

### 5.1 Tại Sao Latent 4 Channels?

```
RGB Image: 3 channels (R, G, B)
VAE Latent: 4 channels
```

**Lý do:**
- 4 > 3: có thêm "room" để encode thông tin
- VAE học được cách pack information hiệu quả vào 4 channels
- Thực nghiệm cho thấy 4 channels cho chất lượng reconstruction tốt
- SD 2.x dùng 4 channels, một số model khác dùng nhiều hơn (16, 32)

### 5.2 Tại Sao Compression 8x?

```
Image: 512×512 → Latent: 64×64
Compression ratio: 512/64 = 8
```

**Lý do:**
- 8x là balance giữa quality và efficiency
- Giảm 64 lần số pixels (8×8)
- UNet làm việc trên 64×64 thay vì 512×512 → nhanh hơn nhiều
- Information loss có thể chấp nhận được cho generative task

### 5.3 Tại Sao Text 77 Tokens?

```
CLIP max_position_embeddings = 77
```

**Lý do:**
- CLIP được train với max context 77
- Bao gồm: [BOS] + 75 word tokens + [EOS]
- Đủ cho hầu hết prompts
- Một số model (SDXL) extend lên 150+ tokens

### 5.4 Tại Sao UNet Inpainting 9 Channels?

```
Standard UNet: 4 channels input
Inpainting UNet: 9 channels input

9 = 4 (noisy latent) + 1 (mask) + 4 (masked image latent)
```

**Lý do:**
- UNet cần biết:
  1. Ảnh noisy cần denoise (4 channels)
  2. Vùng nào cần inpaint (1 channel mask)
  3. Context xung quanh (4 channels masked image)
- Concat tất cả làm input cho first conv layer

---

## 6. Các Vấn Đề Và Giải Pháp

### 6.1 Vấn Đề: Train/Inference Mismatch

**Mô tả:**
- Training: UNet học denoise → reconstruct original image
- Inference: Muốn UNet dùng reference color từ IP-Adapter

**Tại sao xảy ra:**
- IP-Adapter KHÔNG được dùng khi training
- LoRA không bao giờ "thấy" IP-Adapter features
- Khi inference, IP-Adapter được thêm vào → distribution shift

**Giải pháp đề xuất:**
1. Train with IP-Adapter active (phức tạp, cần modify training loop)
2. Thêm color consistency loss (tính color distance trong training)
3. Dùng stronger IP-Adapter scale at inference

### 6.2 Vấn Đề: Segmentation Noise

**Mô tả:**
- Mask chứa vùng không phải wall (đồ đạc, sàn nhà, ...)
- Model học paint các vùng đó với wall color

**Giải pháp đã implement:**
1. Mask erosion (mask_erosion_size=5)
2. Quality threshold tăng (0.6 thay vì 0.5)
3. Hybrid color strategy (30% random)

### 6.3 Vấn Đề: Reference Color không được học

**Mô tả:**
- Training không có reference color trong loss
- Model không biết cần match color nào

**Giải pháp đề xuất:**
1. Thêm color reconstruction loss
2. Thêm perceptual loss (LPIPS, SSIM)
3. Train với IP-Adapter active

---

## 7. Tổng Kết

### 7.1 Flow Tóm Tắt

```
TRAINING:
Image → VAE Encode → Latent → Add Noise → UNet → Predict Noise → MSE Loss

VALIDATION:
Image → Pipeline → Output → Compare with Reference → Metrics

INFERENCE:
Image + Mask + Reference → IP-Adapter + ControlNet + UNet → Output
```

### 7.2 Các Shape Quan Trọng

| Tensor | Shape | Range |
|--------|-------|-------|
| Image | [B, 3, 512, 512] | [-1, 1] |
| Latent | [B, 4, 64, 64] | ≈[-4, 4] |
| Mask | [B, 1, 64, 64] | [0, 1] |
| Text Embed | [B, 77, 768] | normalized |
| UNet Input | [B, 9, 64, 64] | varies |
| Noise (GT) | [B, 4, 64, 64] | N(0, 1) |
| Loss | scalar | ≥0 |

### 7.3 Key Takeaways

1. **VAE compress 8x** để UNet làm việc hiệu quả trong latent space
2. **LoRA chỉ 0.18%** tham số nhưng hiệu quả fine-tune
3. **9-channel input** là đặc trưng của SD Inpainting
4. **Ground Truth là NOISE**, không phải image
5. **Train/Inference mismatch** là vấn đề kiến trúc cần giải quyết
