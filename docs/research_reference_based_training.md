# Nghiên Cứu: Phương Pháp Training Reference-Based Inpainting

> **Phiên bản:** 1.0.0  
> **Ngày tạo:** 2026-02-05  
> **Mục đích:** Tổng hợp các phương pháp training từ các repo nổi tiếng để áp dụng cho Wall Inpainting

---

## Mục Lục

1. [Tổng Quan Các Phương Pháp](#1-tổng-quan-các-phương-pháp)
2. [Paint-by-Example (CVPR 2023)](#2-paint-by-example)
3. [AnyDoor (CVPR 2024)](#3-anydoor)
4. [IP-Adapter Training](#4-ip-adapter-training)
5. [Kết Hợp LoRA + IP-Adapter](#5-kết-hợp-lora--ip-adapter)
6. [Đề Xuất Cho Wall Inpainting](#6-đề-xuất-cho-wall-inpainting)
7. [Implementation Plan](#7-implementation-plan)

---

## 1. Tổng Quan Các Phương Pháp

### 1.1 So Sánh Các Approach

| Phương pháp | Training | Reference Encoder | Điểm mạnh | Điểm yếu |
|-------------|----------|-------------------|-----------|----------|
| **Paint-by-Example** | Self-supervised | CLIP Image | Không cần paired data | Dễ copy-paste |
| **AnyDoor** | Self-supervised + Video | DINOv2 + Detail Extractor | Object identity mạnh | Phức tạp |
| **IP-Adapter** | Only cross-attention | CLIP ViT-H | Nhẹ, dễ tích hợp | Thiếu fine-grained control |
| **LoRA chỉ riêng** | Noise prediction | N/A | Đơn giản | Không biết về reference |
| **LoRA + IP-Adapter** | Combined | CLIP ViT-H | Tốt nhất cho color | Cần custom training |

### 1.2 Vấn Đề Chính Của Approach Hiện Tại

```
HIỆN TẠI:
- Training: LoRA học denoise về original image
- Inference: IP-Adapter inject reference color
- KẾT QUẢ: LoRA bị "surprised" bởi IP-Adapter signals

CẦN FIX:
- Training: LoRA phải thấy reference image và học cách sử dụng nó
```

---

## 2. Paint-by-Example

### 2.1 Ý Tưởng Chính

Paint-by-Example là phương pháp **reference-based inpainting** đầu tiên sử dụng diffusion models.

**Key Insight:** Không cần paired data (source + reference + ground truth). Thay vào đó, dùng **self-supervised learning**:

```
Training Data:
- Input: Ảnh bị mask ngẫu nhiên
- Reference: Crop từ chính ảnh đó (vùng bị mask)
- Ground Truth: Ảnh gốc

Model học: "Điền vào vùng mask sao cho giống reference"
```

### 2.2 Kiến Trúc

```
┌─────────────────────────────────────────────────────────────┐
│                    PAINT-BY-EXAMPLE                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Reference Image ──► CLIP Image Encoder ──► Image Embedding│
│                          │                                  │
│                          ▼                                  │
│  ┌────────────────────────────────────────────────────┐    │
│  │              Stable Diffusion UNet                  │    │
│  │  ┌──────────────────────────────────────────────┐  │    │
│  │  │ Cross-Attention: thay Text bằng Image Embed │  │    │
│  │  └──────────────────────────────────────────────┘  │    │
│  └────────────────────────────────────────────────────┘    │
│                          │                                  │
│  Source + Mask ─────────►│                                  │
│                          ▼                                  │
│                    Output Image                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Kỹ Thuật Quan Trọng

**1. Information Bottleneck:**
- Không cho reference image quá rõ ràng
- Augment mạnh reference (blur, color jitter, crop)
- Ngăn model "copy-paste" nguyên reference

```python
# Augmentation cho reference
reference_augment = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
])
```

**2. Classifier-Free Guidance (cho Image):**
- Không chỉ CFG cho text, mà còn cho image
- 10% thời gian, drop reference → model học inpaint không cần reference

```python
# Training với image dropping
if random.random() < 0.1:
    image_embeds = torch.zeros_like(image_embeds)  # Drop reference
```

**3. Self-Supervised Triplet:**
```python
# Tạo training triplet từ 1 ảnh
def create_training_sample(image):
    # Random mask
    mask = generate_random_mask(image.size)
    
    # Reference = crop từ vùng mask
    masked_region = image.crop(mask.bbox)
    reference = augment(masked_region)
    
    # Input
    masked_image = image * (1 - mask)
    
    return masked_image, mask, reference, image  # ground truth = original
```

### 2.4 Áp Dụng Cho Wall Inpainting

**Điểm phù hợp:**
- Self-supervised → không cần paired data
- Image conditioning → color transfer

**Điểm cần điều chỉnh:**
- Reference là solid color, không phải crop từ ảnh
- Không cần information bottleneck mạnh (solid color đã là bottleneck)

---

## 3. AnyDoor

### 3.1 Ý Tưởng Chính

AnyDoor được thiết kế cho **object insertion** với identity preservation mạnh.

**Key Innovation:**
- Dùng **DINOv2** thay vì CLIP (tốt hơn cho object identity)
- **Frequency-aware detail extractor** để giữ texture
- **Video dataset** để học object dynamics

### 3.2 Kiến Trúc

```
┌─────────────────────────────────────────────────────────────┐
│                       ANYDOOR                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Reference Object ──┬──► DINOv2 ──────────► ID Tokens      │
│                     │                           │           │
│                     └──► High-Pass Filter ──► Detail Maps  │
│                                    │            │           │
│                                    ▼            ▼           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                 Stable Diffusion UNet                 │  │
│  │  ┌─────────────────┐  ┌──────────────────────────┐   │  │
│  │  │ Cross-Attention │  │ Gated Self-Attention     │   │  │
│  │  │ (ID Tokens)     │  │ (Detail Maps injection)  │   │  │
│  │  └─────────────────┘  └──────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  Scene + Target Box ─────────────────────────────────────► │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Kỹ Thuật Quan Trọng

**1. DINOv2 cho Identity:**
```python
# DINOv2 tốt hơn CLIP cho object-level features
dino_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

# Lấy CLS token + patch tokens
features = dino_encoder(reference_image)  # [1, 257, 1024]
id_tokens = features[:, 0:4, :]  # Lấy 4 tokens đầu cho identity
```

**2. Frequency-Aware Detail Extractor:**
```python
# High-pass filter để lấy texture/detail
def extract_detail_features(image):
    # Low frequency (structure)
    low_freq = gaussian_blur(image, sigma=5)
    
    # High frequency (detail)
    high_freq = image - low_freq
    
    # Encode high-freq features
    detail_features = detail_encoder(high_freq)
    return detail_features
```

**3. Video Dataset cho Dynamics:**
- Train trên video: cùng object xuất hiện ở nhiều frame
- Model học object có thể thay đổi pose, lighting, etc.
- Giúp generalize tốt hơn

### 3.4 Áp Dụng Cho Wall Inpainting

**Điểm phù hợp:**
- DINOv2 có thể tốt hơn CLIP cho color matching
- Detail extractor có thể giữ texture wall

**Điểm cần điều chỉnh:**
- Wall không cần identity preservation mạnh như object
- Video dataset không áp dụng được (wall không di chuyển)

---

## 4. IP-Adapter Training

### 4.1 Cách IP-Adapter Được Train

IP-Adapter chỉ train **Image Projection layers** và **Cross-Attention layers cho image**, UNet frozen.

```python
# Trainable parameters trong IP-Adapter
trainable_modules = [
    "image_proj",           # Project CLIP image embeds
    "to_k_ip",              # Key projection cho image attention
    "to_v_ip",              # Value projection cho image attention
]

# Frozen parameters
frozen_modules = [
    "unet",                 # Base UNet
    "text_encoder",         # CLIP text
    "image_encoder",        # CLIP image (pretrained)
]
```

### 4.2 Training Objective

```python
# IP-Adapter training giống diffusion training thông thường
# Chỉ khác là có thêm image conditioning

def training_step(batch):
    images = batch["images"]
    reference_images = batch["reference_images"]  # Key difference!
    
    # Encode reference
    image_embeds = clip_image_encoder(reference_images)
    projected_embeds = image_proj(image_embeds)
    
    # Encode image to latent
    latents = vae.encode(images).latent_dist.sample()
    
    # Add noise
    noise = torch.randn_like(latents)
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
    
    # UNet forward with image conditioning
    noise_pred = unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=text_embeds,
        added_cond_kwargs={"image_embeds": projected_embeds},
    ).sample
    
    # Loss
    loss = F.mse_loss(noise_pred, noise)
    return loss
```

### 4.3 Áp Dụng Cho Wall Inpainting

**QUAN TRỌNG:** IP-Adapter được train để model BIẾT cách sử dụng image reference.

Nếu chúng ta train LoRA MÀ KHÔNG CÓ IP-Adapter active:
- LoRA không bao giờ thấy image reference
- LoRA không học cách phối hợp với IP-Adapter

---

## 5. Kết Hợp LoRA + IP-Adapter

### 5.1 Phương Pháp 1: Sequential Training (Đơn giản)

```
Bước 1: Train IP-Adapter (hoặc dùng pretrained)
Bước 2: Freeze IP-Adapter, train LoRA với IP-Adapter active

# Khi train LoRA:
- IP-Adapter ACTIVE nhưng FROZEN
- LoRA weights là trainable
- LoRA học cách collaborate với IP-Adapter
```

```python
# Sequential training pseudo-code
def train_lora_with_ipadapter():
    # Load pretrained IP-Adapter
    pipe.load_ip_adapter("h94/IP-Adapter", weight_name="ip-adapter-plus_sd15.bin")
    pipe.set_ip_adapter_scale(1.0)
    
    # Freeze IP-Adapter
    for name, param in pipe.unet.named_parameters():
        if "ip" in name.lower():
            param.requires_grad = False
    
    # Add LoRA (trainable)
    lora_config = LoraConfig(r=8, lora_alpha=16, ...)
    pipe.unet = get_peft_model(pipe.unet, lora_config)
    
    # Training loop
    for batch in dataloader:
        # IP-Adapter processes reference
        image_embeds = encode_reference(batch["reference_image"])
        
        # Forward với cả IP-Adapter và LoRA
        noise_pred = pipe.unet(
            noisy_latents,
            timesteps,
            text_embeds,
            added_cond_kwargs={"image_embeds": image_embeds},
        ).sample
        
        loss = F.mse_loss(noise_pred, noise)
        loss.backward()  # Chỉ LoRA weights được update
```

### 5.2 Phương Pháp 2: Joint Training (Phức tạp hơn)

```
Train cả IP-Adapter layers và LoRA cùng lúc:
- IP-Adapter cross-attention: trainable
- LoRA adapters: trainable
- Base UNet: frozen
```

```python
# Joint training pseudo-code
def train_joint():
    # Trainable parameters
    trainable_params = []
    
    # LoRA params
    trainable_params.extend([p for n, p in unet.named_parameters() if "lora" in n])
    
    # IP-Adapter params
    trainable_params.extend([p for n, p in unet.named_parameters() if "ip" in n])
    
    optimizer = AdamW(trainable_params, lr=1e-5)
    
    # Training với cả 2
    for batch in dataloader:
        # Reference processing
        image_embeds = clip_image_encoder(batch["reference_image"])
        projected_embeds = image_proj(image_embeds)  # Trainable!
        
        # UNet forward
        noise_pred = unet(
            noisy_latents,
            timesteps,
            text_embeds,
            added_cond_kwargs={"image_embeds": projected_embeds},
        ).sample
        
        loss = F.mse_loss(noise_pred, noise)
        loss.backward()  # Update cả IP-Adapter và LoRA
```

### 5.3 Phương Pháp 3: Color Consistency Loss (Auxiliary)

Thêm loss phụ để ép model match màu:

```python
def compute_color_loss(output_latents, reference_color, mask):
    # Decode latents to image
    output_image = vae.decode(output_latents / 0.18215).sample
    
    # Convert to LAB
    output_lab = rgb_to_lab(output_image)
    reference_lab = rgb_to_lab(reference_color)
    
    # Compute color distance trong masked region
    masked_output = output_lab * mask
    color_loss = F.mse_loss(masked_output, reference_lab.expand_as(masked_output))
    
    return color_loss

# Total loss
total_loss = noise_loss + lambda_color * color_loss
```

---

## 6. Đề Xuất Cho Wall Inpainting

### 6.1 Phương Pháp Được Đề Xuất

**Ưu tiên 1: Sequential Training (LoRA + IP-Adapter Frozen)**

Lý do:
- Đơn giản nhất để implement
- Không cần modify IP-Adapter
- LoRA học cách "nghe" IP-Adapter

**Ưu tiên 2: Thêm Color Consistency Loss**

Lý do:
- Ép explicit color matching trong loss
- Không làm tăng độ phức tạp kiến trúc nhiều

### 6.2 Dataset Preparation

```python
# Mỗi sample cần có:
{
    "image": original_image,            # Ảnh gốc
    "mask": wall_mask,                  # Mask vùng wall
    "reference_image": solid_color,     # Solid color reference (đã có)
    "reference_color_rgb": (R, G, B),   # Màu target
}
```

### 6.3 Training Configuration Đề Xuất

```yaml
# Config mới cho IP-Adapter active training
training:
  # Enable IP-Adapter during training
  use_ip_adapter_training: true
  ip_adapter_model: "h94/IP-Adapter"
  ip_adapter_weight_name: "ip-adapter-plus_sd15.bin"
  ip_adapter_scale: 1.0
  freeze_ip_adapter: true  # Sequential approach
  
  # Color consistency loss
  use_color_loss: true
  color_loss_weight: 0.1
  color_loss_start_step: 1000  # Bắt đầu sau khi model converge cơ bản
  
  # LoRA config
  lora_rank: 16  # Tăng từ 8 lên 16 cho capacity
  lora_alpha: 32
```

---

## 7. Implementation Plan

### 7.1 Các Bước Thực Hiện

```
PHASE 1: Chuẩn bị (1-2 ngày)
├── [ ] Tải IP-Adapter weights
├── [ ] Modify training script để load IP-Adapter
├── [ ] Test inference với IP-Adapter

PHASE 2: Sequential Training (3-5 ngày)
├── [ ] Implement IP-Adapter integration trong training loop
├── [ ] Freeze IP-Adapter, train LoRA
├── [ ] Validate: reference column phải là solid color, output phải match

PHASE 3: Color Loss (2-3 ngày)
├── [ ] Implement color_consistency_loss
├── [ ] Integrate vào training loop
├── [ ] Tune lambda_color

PHASE 4: Evaluation (1-2 ngày)
├── [ ] So sánh metrics: old LoRA vs new LoRA
├── [ ] Visual comparison trên validation set
├── [ ] Document kết quả
```

### 7.2 Code Changes Cần Thiết

**File: `train_lora_inpainting.py`**
- Thêm load IP-Adapter
- Thêm image encoding trong training loop
- Thêm `added_cond_kwargs` vào UNet call

**File: `losses.py`**
- Implement `ColorConsistencyLoss`

**File: `configs/lora_training.yaml`**
- Thêm IP-Adapter config section
- Thêm color loss config

---

## 8. Kết Luận

### 8.1 Key Takeaways

1. **Paint-by-Example** dạy chúng ta: Self-supervised + Information Bottleneck
2. **AnyDoor** dạy chúng ta: DINOv2 tốt hơn CLIP cho object features
3. **IP-Adapter** dạy chúng ta: Chỉ train cross-attention layers
4. **Combined approach** là tốt nhất: LoRA + IP-Adapter cùng training

### 8.2 Recommendation Cuối Cùng

Cho Wall Inpainting task:

```
BEST APPROACH: Sequential Training
- Load pretrained IP-Adapter Plus
- Freeze IP-Adapter weights
- Train LoRA với IP-Adapter active
- Optional: Add color consistency loss

WHY?
- LoRA học cách phối hợp với IP-Adapter
- Không cần modify IP-Adapter (đã tốt cho color)
- Training stable hơn joint training
```
