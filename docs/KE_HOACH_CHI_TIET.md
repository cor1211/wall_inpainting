# Kế Hoạch Chi Tiết: Pipeline Đổi Màu Tường với Diffusion

> **Ngày tạo:** 2026-02-07  
> **Mục tiêu:** Xây dựng pipeline training và inference hoàn chỉnh cho bài toán đổi màu tường dựa trên diffusion, tham khảo các paper nổi tiếng như Vi-TryOn, RoomEditor

---

## 📋 Mục Lục

1. [Phân Tích Hiện Trạng](#1-phân-tích-hiện-trạng)
2. [Kiến Trúc Đề Xuất](#2-kiến-trúc-đề-xuất)
3. [Pipeline Training](#3-pipeline-training)
4. [Pipeline Inference](#4-pipeline-inference)
5. [Kế Hoạch Triển Khai](#5-kế-hoạch-triển-khai)
6. [Các Vấn Đề Cần Giải Quyết](#6-các-vấn-đề-cần-giải-quyết)

---

## 1. Phân Tích Hiện Trạng

### 1.1 Cấu Trúc Dataset Hiện Tại (`./dataset_test`)

```
dataset_test/
├── train/
│   ├── images/
│   │   ├── ADE_frame_00000004_original.png    # Ảnh tường cũ
│   │   ├── ADE_frame_00000004_burgundy_0.png # Ảnh tường mới (GT)
│   │   └── ...
│   ├── masks/
│   │   └── ADE_frame_00000004.png            # Mask vùng tường
│   └── metadata.jsonl
└── validation/
    └── ...
```

**Metadata format:**
```json
{
  "source_path": "train/images/ADE_frame_00000004_original.png",
  "target_path": "train/images/ADE_frame_00000004_burgundy_0.png",
  "mask_path": "train/masks/ADE_frame_00000004.png",
  "color_rgb": [128, 0, 32],
  "color_name": "burgundy"
}
```

### 1.2 Kiến Trúc Hiện Tại

#### ✅ Đã có:
- **Base Pipeline** (`pipeline.py`): SD Inpainting + ControlNet Depth + IP-Adapter Plus
- **Segmentation** (`segmentation.py`): SAM2/FastSAM để tạo mask
- **Dataset Preparation** (`prepare_dataset_v2.py`): Tạo training pairs từ source images
- **Training Script** (`train.py`): LoRA training với UNet
- **Inference Script** (`inference.py`): Inference với pipeline

#### ❌ Thiếu/Cần cải thiện:
1. **Module `models/wall_recoloring_pipeline.py`**: Chưa tồn tại, nhưng được import trong `train.py` và `inference.py`
2. **Module `dataset/wall_paint_dataset.py`**: Chưa tồn tại, nhưng được import trong `train.py`
3. **Training Strategy**: Hiện tại train để reconstruct original image, không phù hợp với mục tiêu đổi màu
4. **Condition Integration**: Cách ghép nối các condition (source, mask, ref) chưa tối ưu

### 1.3 Vấn Đề Chính

#### 🔴 Vấn Đề 1: Train/Inference Mismatch
```
TRAINING:
- Input: Ảnh tường cũ (noisy)
- Condition: Source image, mask, color ref
- Target: Ảnh tường cũ (original) ❌ SAI!

INFERENCE:
- Input: Ảnh tường cũ
- Condition: Source image, mask, color ref mới
- Target: Ảnh tường mới ✅ ĐÚNG!

→ Model học reconstruct original, nhưng inference muốn đổi màu!
```

#### 🔴 Vấn Đề 2: Condition Integration Chưa Tối Ưu
- ControlNet: Dùng Canny (trong `inference.py`) nhưng spec nói Depth
- IP-Adapter: Chỉ dùng ở inference, không được train cùng LoRA
- Mask: Được concatenate vào UNet input, nhưng cách sử dụng chưa rõ ràng

#### 🔴 Vấn Đề 3: Thiếu Reference Image trong Training
- Dataset có `color_rgb` nhưng không có reference image
- Training không sử dụng IP-Adapter image embeddings
- Model không học cách sử dụng color reference

---

## 2. Kiến Trúc Đề Xuất

### 2.1 Tham Khảo Các Paper Nổi Tiếng

#### Vi-TryOn (Virtual Try-On)
```
Key Ideas:
1. Dual-branch architecture: Structure + Appearance
2. Warping module để preserve structure
3. Feature fusion ở multiple scales
4. Reference image được encode và inject vào UNet

Áp dụng cho Wall Recoloring:
- Source image → Structure branch (ControlNet)
- Color reference → Appearance branch (IP-Adapter)
- Mask → Guide inpainting region
```

#### RoomEditor
```
Key Ideas:
1. Multi-condition control: Layout + Style + Object
2. Hierarchical conditioning: Global → Local
3. Attention-based feature fusion
4. Progressive refinement

Áp dụng cho Wall Recoloring:
- Global: Depth map (ControlNet)
- Local: Color reference (IP-Adapter)
- Mask: Spatial guidance
```

### 2.2 Kiến Trúc Pipeline Đề Xuất

```
┌─────────────────────────────────────────────────────────────────┐
│              WALL RECOLORING PIPELINE ARCHITECTURE              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUTS:                                                        │
│  ┌──────────────┐  ┌──────────┐  ┌──────────────┐            │
│  │ Source Image │  │   Mask   │  │ Color Ref    │            │
│  │ (Old Wall)   │  │          │  │ (New Color)  │            │
│  └──────┬───────┘  └────┬─────┘  └──────┬───────┘            │
│         │               │               │                      │
│         ▼               ▼               ▼                      │
│  ┌──────────────────────────────────────────────────────┐    │
│  │              PREPROCESSING STAGE                      │    │
│  ├──────────────────────────────────────────────────────┤    │
│  │                                                       │    │
│  │  1. Source → Depth Map (ControlNet Preprocessor)    │    │
│  │  2. Color Ref → CLIP Embedding (IP-Adapter)          │    │
│  │  3. Mask → Latent Mask (Resize to 64x64)            │    │
│  │  4. Source → Masked Source (for inpainting)          │    │
│  │                                                       │    │
│  └──────────────────────────────────────────────────────┘    │
│         │               │               │                      │
│         ▼               ▼               ▼                      │
│  ┌──────────────────────────────────────────────────────┐    │
│  │              CONDITIONING STAGE                       │    │
│  ├──────────────────────────────────────────────────────┤    │
│  │                                                       │    │
│  │  ┌──────────────────────────────────────────────┐    │    │
│  │  │ ControlNet Branch (Structure Preservation) │    │    │
│  │  │ Input: Depth Map                            │    │    │
│  │  │ Output: Down/Mid Block Residuals           │    │    │
│  │  └──────────────────────────────────────────────┘    │    │
│  │                                                       │    │
│  │  ┌──────────────────────────────────────────────┐    │    │
│  │  │ IP-Adapter Branch (Color Transfer)          │    │    │
│  │  │ Input: Color Reference Image                │    │    │
│  │  │ Output: Image Embeddings (added_cond_kwargs)│    │    │
│  │  └──────────────────────────────────────────────┘    │    │
│  │                                                       │    │
│  │  ┌──────────────────────────────────────────────┐    │    │
│  │  │ Inpainting Branch (Masked Generation)        │    │    │
│  │  │ Input: Noisy Latents + Mask + Masked Source │    │    │
│  │  │ Output: Concatenated Input (9 channels)    │    │    │
│  │  └──────────────────────────────────────────────┘    │    │
│  │                                                       │    │
│  └──────────────────────────────────────────────────────┘    │
│         │               │               │                      │
│         └───────────────┴───────────────┘                      │
│                           ▼                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │              UNET GENERATION STAGE                   │    │
│  ├──────────────────────────────────────────────────────┤    │
│  │                                                       │    │
│  │  UNet Input:                                          │    │
│  │  - Latent Model Input: [noisy_latents, mask, masked]│    │
│  │  - Text Embeddings: encoder_hidden_states            │    │
│  │  - ControlNet Residuals: down_block + mid_block     │    │
│  │  - IP-Adapter Embeddings: image_embeds              │    │
│  │                                                       │    │
│  │  UNet Output:                                         │    │
│  │  - Noise Prediction: ε̂                               │    │
│  │                                                       │    │
│  └──────────────────────────────────────────────────────┘    │
│                           ▼                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │              POSTPROCESSING STAGE                    │    │
│  ├──────────────────────────────────────────────────────┤    │
│  │                                                       │    │
│  │  1. Denoising Loop (DDIM/DPM++)                      │    │
│  │  2. VAE Decode: Latents → Image                      │    │
│  │  3. Post-process: Resize, Blend edges               │    │
│  │                                                       │    │
│  └──────────────────────────────────────────────────────┘    │
│                           ▼                                    │
│                    OUTPUT IMAGE                                │
│                  (New Wall Color)                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Chi Tiết Các Component

#### A. ControlNet Branch
```python
# Mục đích: Preserve structure và depth information
ControlNet(
    input: depth_map,  # Từ source image
    condition: depth_map,
    timestep: t,
    encoder_hidden_states: text_embeds
) → (down_block_residuals, mid_block_residual)
```

**Lựa chọn ControlNet:**
- ✅ **Depth** (`control_v11f1p_sd15_depth`): Tốt nhất cho structure preservation
- ❌ **Canny**: Chỉ edge, không có depth info
- ✅ **Canny + Depth**: Có thể combine nếu cần

#### B. IP-Adapter Branch
```python
# Mục đích: Transfer color/style từ reference
CLIPVisionEncoder(
    input: color_reference_image  # [224, 224, 3]
) → image_embeds  # [B, 1024]

# Inject vào UNet qua attention mechanism
UNet.forward(
    ...,
    added_cond_kwargs={"image_embeds": image_embeds}
)
```

**Lựa chọn IP-Adapter:**
- ✅ **IP-Adapter Plus** (`ip-adapter-plus_sd15.bin`): Tốt hơn cho color transfer
- ✅ **IP-Adapter Full**: Nếu cần fine-grained control

#### C. Inpainting Branch
```python
# Mục đích: Guide generation trong masked region
VAE.encode(masked_source) → masked_latents  # [B, 4, 64, 64]
F.interpolate(mask) → mask_latents          # [B, 1, 64, 64]

# Concatenate với noisy latents
unet_input = torch.cat([
    noisy_latents,      # [B, 4, 64, 64]
    mask_latents,        # [B, 1, 64, 64]
    masked_latents       # [B, 4, 64, 64]
], dim=1)  # → [B, 9, 64, 64]
```

---

## 3. Pipeline Training

### 3.1 Training Strategy Đề Xuất

#### ❌ Strategy Cũ (SAI):
```
Input: Ảnh tường cũ (noisy)
Target: Ảnh tường cũ (original)
→ Model học reconstruct original
```

#### ✅ Strategy Mới (ĐÚNG):
```
Input: Ảnh tường mới (noisy) ← Đây là key!
Condition: 
  - Source: Ảnh tường cũ (cho ControlNet)
  - Mask: Vùng tường
  - Color Ref: Màu mới (cho IP-Adapter)
Target: Ảnh tường mới (GT)
→ Model học generate new color từ conditions
```

### 3.2 Training Flow Chi Tiết

```python
def training_step(batch):
    # 1. Load data
    source_image = batch["source"]      # Ảnh tường cũ
    target_image = batch["target"]      # Ảnh tường mới (GT)
    mask = batch["mask"]
    color_ref = batch["color_reference"]  # Color patch mới
    
    # 2. Encode target (GT) to latent
    target_latents = vae.encode(target_image).latent_dist.sample()
    target_latents = target_latents * vae.config.scaling_factor
    
    # 3. Add noise to target (NOT source!)
    noise = torch.randn_like(target_latents)
    timesteps = torch.randint(0, 1000, (bsz,))
    noisy_latents = scheduler.add_noise(target_latents, noise, timesteps)
    
    # 4. Prepare conditions
    # A. ControlNet: Use SOURCE image (old wall)
    depth_map = depth_estimator(source_image)
    controlnet_output = controlnet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=text_embeds,
        controlnet_cond=depth_map
    )
    
    # B. IP-Adapter: Use COLOR REFERENCE (new color)
    color_ref_224 = F.interpolate(color_ref, (224, 224))
    color_ref_normalized = normalize_clip(color_ref_224)
    image_embeds = image_encoder(color_ref_normalized).image_embeds
    added_cond_kwargs = {"image_embeds": image_embeds}
    
    # C. Inpainting: Prepare masked source
    masked_source = source_image * (1 - mask)  # Use SOURCE, not target!
    masked_source_latents = vae.encode(masked_source).latent_dist.sample()
    masked_source_latents = masked_source_latents * vae.config.scaling_factor
    mask_latents = F.interpolate(mask, size=(64, 64), mode="nearest")
    
    # 5. UNet forward
    unet_input = torch.cat([
        noisy_latents,      # Noisy TARGET latents
        mask_latents,
        masked_source_latents  # Masked SOURCE latents
    ], dim=1)
    
    noise_pred = unet(
        unet_input,
        timesteps,
        encoder_hidden_states=text_embeds,
        down_block_additional_residuals=controlnet_output[0],
        mid_block_additional_residual=controlnet_output[1],
        added_cond_kwargs=added_cond_kwargs
    ).sample
    
    # 6. Loss: Predict noise added to TARGET
    loss = F.mse_loss(noise_pred, noise)
    return loss
```

### 3.3 Dataset Format Cho Training

```python
class WallPaintDataset(Dataset):
    """
    Dataset cho training wall recoloring.
    
    Expected structure:
    - source_path: Ảnh tường cũ
    - target_path: Ảnh tường mới (GT)
    - mask_path: Mask vùng tường
    - color_reference: Color patch mới (tạo từ color_rgb)
    """
    
    def __getitem__(self, idx):
        # Load images
        source = load_image(self.samples[idx]["source_path"])
        target = load_image(self.samples[idx]["target_path"])
        mask = load_mask(self.samples[idx]["mask_path"])
        
        # Create color reference patch
        color_rgb = self.samples[idx]["color_rgb"]
        color_ref = create_color_patch(color_rgb, size=512)
        
        # Augmentations
        source, target, mask, color_ref = self.transform(
            source, target, mask, color_ref
        )
        
        return {
            "source": source,           # [3, 512, 512]
            "target": target,           # [3, 512, 512] - GT
            "mask": mask,               # [1, 512, 512]
            "color_reference": color_ref,  # [3, 512, 512]
            "prompt": "interior wall, high quality"
        }
```

### 3.4 LoRA Training Configuration

```python
# LoRA config cho UNet
lora_config = LoraConfig(
    r=16,                    # Rank (tăng từ 8 → 16 cho color task)
    lora_alpha=32,           # Alpha = 2 * r
    target_modules=[
        "to_k", "to_q", "to_v", "to_out.0",  # Attention layers
        # Có thể thêm:
        # "ff.net.0.proj", "ff.net.2"  # FFN layers (nếu cần)
    ],
    lora_dropout=0.0,
    bias="none",
)

# Freeze các components khác
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
controlnet.requires_grad_(False)
image_encoder.requires_grad_(False)  # IP-Adapter encoder

# Chỉ train UNet LoRA
unet.requires_grad_(False)
unet = get_peft_model(unet, lora_config)
```

---

## 4. Pipeline Inference

### 4.1 Inference Flow

```python
def inference(source_image, mask, color_reference):
    # 1. Preprocess
    source_512 = resize(source_image, (512, 512))
    mask_512 = resize(mask, (512, 512))
    color_ref_224 = resize(color_reference, (224, 224))
    
    # 2. Prepare conditions
    depth_map = depth_estimator(source_512)
    color_embeds = image_encoder(normalize_clip(color_ref_224)).image_embeds
    
    # 3. Initialize latents (random noise)
    latents = torch.randn((1, 4, 64, 64))
    
    # 4. Prepare inpainting inputs
    masked_source = source_512 * (1 - mask_512)
    masked_latents = vae.encode(masked_source).latent_dist.sample()
    mask_latents = F.interpolate(mask_512, (64, 64), mode="nearest")
    
    # 5. Denoising loop
    scheduler.set_timesteps(50)
    for t in scheduler.timesteps:
        # ControlNet
        controlnet_output = controlnet(
            latents, t, text_embeds, depth_map
        )
        
        # UNet input
        unet_input = torch.cat([
            latents,
            mask_latents,
            masked_latents
        ], dim=1)
        
        # UNet prediction
        noise_pred = unet(
            unet_input,
            t,
            encoder_hidden_states=text_embeds,
            down_block_additional_residuals=controlnet_output[0],
            mid_block_additional_residual=controlnet_output[1],
            added_cond_kwargs={"image_embeds": color_embeds}
        ).sample
        
        # Step
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    # 6. Decode
    image = vae.decode(latents / vae.config.scaling_factor).sample
    image = postprocess(image)
    
    return image
```

### 4.2 Hyperparameters Inference

```python
INFERENCE_CONFIG = {
    "num_inference_steps": 50,           # DDIM steps
    "guidance_scale": 5.0,                # CFG scale
    "controlnet_conditioning_scale": 0.8, # ControlNet strength
    "ip_adapter_scale": 0.7,              # IP-Adapter strength
    "strength": 1.0,                      # Denoising strength
}
```

---

## 5. Kế Hoạch Triển Khai

### Phase 1: Tạo Missing Modules (Ưu tiên cao)

#### 5.1 Tạo `models/wall_recoloring_pipeline.py`
```python
# File này sẽ wrap pipeline hiện tại và thêm các utilities
def get_wall_recoloring_pipeline(
    base_model_path,
    controlnet_path,
    ip_adapter_scale=0.7,
    device="cuda"
):
    # Load và combine các components
    # Return pipeline ready to use
```

#### 5.2 Tạo `dataset/wall_paint_dataset.py`
```python
# File này sẽ implement WallPaintDataset
class WallPaintDataset:
    # Load từ metadata.jsonl
    # Return source, target, mask, color_reference
```

### Phase 2: Fix Training Strategy

#### 5.3 Sửa `train.py`
- [ ] Thay đổi target từ source → target image
- [ ] Thêm color reference vào training loop
- [ ] Fix ControlNet để dùng Depth thay vì Canny
- [ ] Đảm bảo IP-Adapter được sử dụng trong training

### Phase 3: Cải Thiện Dataset

#### 5.4 Cải thiện `prepare_dataset_v2.py`
- [ ] Đảm bảo tạo color reference patches
- [ ] Validate dataset format
- [ ] Add data augmentation

### Phase 4: Testing & Validation

#### 5.5 Tạo validation script
- [ ] Visual validation trong training
- [ ] Quantitative metrics (color accuracy, structure preservation)
- [ ] A/B testing với different strategies

---

## 6. Các Vấn Đề Cần Giải Quyết

### 6.1 Train/Inference Alignment ✅
**Giải pháp:** Train với target image làm noisy input, source image làm condition

### 6.2 Color Reference Generation
**Vấn đề:** Dataset chỉ có `color_rgb`, cần tạo color patch
**Giải pháp:** 
```python
def create_color_patch(rgb, size=512):
    # Tạo solid color patch với texture nhẹ
    # Có thể thêm gradient, noise để CLIP encode tốt hơn
```

### 6.3 ControlNet Type
**Vấn đề:** `inference.py` dùng Canny, nhưng spec nói Depth
**Giải pháp:** Standardize về Depth cho structure preservation tốt hơn

### 6.4 IP-Adapter Training
**Vấn đề:** IP-Adapter weights frozen, không được train
**Giải pháp:** 
- Option 1: Giữ frozen, chỉ train LoRA (đơn giản hơn)
- Option 2: Fine-tune IP-Adapter (phức tạp hơn, cần nhiều data)

### 6.5 Mask Quality
**Vấn đề:** Mask có thể có noise, không chính xác
**Giải pháp:** 
- Erosion/dilation để clean mask
- Validation mask quality trong dataset prep

---

## 7. Next Steps

### Immediate (Tuần 1):
1. ✅ Tạo `models/wall_recoloring_pipeline.py`
2. ✅ Tạo `dataset/wall_paint_dataset.py`
3. ✅ Fix `train.py` với training strategy mới
4. ✅ Test training với dataset_test

### Short-term (Tuần 2-3):
1. Cải thiện dataset preparation
2. Add validation metrics
3. Hyperparameter tuning
4. Documentation

### Long-term (Tháng 2+):
1. Scale up dataset
2. Advanced techniques (multi-scale, progressive refinement)
3. Production deployment

---

## 8. References

- **Vi-TryOn**: Virtual Try-On with Diffusion Models
- **RoomEditor**: Room Editing with Diffusion Models
- **Paint-by-Example**: Reference-based Inpainting
- **IP-Adapter**: Effective Image Adapter for Diffusion Models
- **ControlNet**: Adding Conditional Control to Diffusion Models

---

**Tác giả:** AI Assistant  
**Ngày cập nhật:** 2026-02-07
