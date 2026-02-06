# Training Strategy Fixes - Chi Tiết Các Thay Đổi

> **Ngày:** 2026-02-07  
> **File:** `train.py`  
> **Mục đích:** Tài liệu chi tiết các thay đổi để tuân thủ training strategy mới

---

## 📋 Tóm Tắt Thay Đổi

### ✅ Đã Sửa

1. **Training Strategy** - Sửa logic training để học đổi màu thay vì reconstruct
2. **ControlNet Type** - Đổi từ Canny sang Depth
3. **Dataset Configuration** - Cấu hình đúng cho training strategy mới
4. **Validation Function** - Sửa để match với training logic
5. **Learning Rate Scheduler** - Thêm LR scheduler
6. **Checkpointing** - Cải thiện checkpoint saving
7. **Comments** - Thêm comments chi tiết cho từng phase

---

## 🔧 Chi Tiết Các Thay Đổi

### 1. Training Strategy (CRITICAL FIX)

#### ❌ Trước (SAI):
```python
# Dùng target làm GT nhưng không rõ ràng về strategy
pixel_values = batch["targets"]  # Không rõ là target hay source
```

#### ✅ Sau (ĐÚNG):
```python
# Rõ ràng: Dùng TARGET image (new wall color) làm GT
target_pixel_values = batch["target"]  # TARGET = new color
target_latents = vae.encode(target_pixel_values_normalized)  # Encode TARGET
noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)  # Add noise to TARGET
```

**Logic:**
- **Input (noisy)**: Target image (ảnh tường mới) - Đây là key!
- **Conditions**:
  - ControlNet: Source image (ảnh tường cũ) → preserve structure
  - IP-Adapter: Color reference (màu mới) → transfer color
  - Masked source: Source image (ảnh tường cũ) → inpainting context
- **Target (GT)**: Target image (ảnh tường mới)

### 2. ControlNet Type

#### ❌ Trước:
```python
parser.add_argument("--controlnet_model", default="lllyasviel/control_v11p_sd15_canny")
```

#### ✅ Sau:
```python
parser.add_argument("--controlnet_model", default="lllyasviel/control_v11f1p_sd15_depth")
```

**Lý do:** Depth tốt hơn Canny cho structure preservation trong wall recoloring.

### 3. Dataset Configuration

#### ❌ Trước:
```python
train_dataset = WallPaintDataset(
    data_json=args.data_json,
    image_size=args.resolution,
    reconstruction_ratio=0.5  # 50% reconstruct source, 50% use target
)
```

#### ✅ Sau:
```python
train_dataset = WallPaintDataset(
    data_json=args.data_json,
    image_size=args.resolution,
    reconstruction_ratio=0.0,  # Always use target (new color) as GT
    use_depth=True,  # Use depth map for ControlNet
    use_canny=False,
    random_flip=True
)
```

**Lý do:** 
- `reconstruction_ratio=0.0`: Luôn dùng target (new color) làm GT, không reconstruct source
- `use_depth=True`: Dùng depth map thay vì canny

### 4. Training Loop - Phân Chia Rõ Ràng Các Phase

#### ✅ Phase 1: Prepare Target Latents
```python
# Encode TARGET image (new wall color) to latent space
target_pixel_values = batch["target"].to(dtype=weight_dtype)
target_pixel_values_normalized = target_pixel_values * 2.0 - 1.0
target_latents = vae.encode(target_pixel_values_normalized).latent_dist.sample()
target_latents = target_latents * vae.config.scaling_factor
```

#### ✅ Phase 2: Add Noise to Target
```python
# Sample noise and add to TARGET latents
noise = torch.randn_like(target_latents)  # ε ~ N(0, I) - GT for loss
timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,))
noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)
```

#### ✅ Phase 3: Prepare Conditions
```python
# A. Text Embeddings
encoder_hidden_states = text_encoder(inputs.input_ids.to(device))[0]

# B. ControlNet: Use SOURCE image (old wall) for structure
control_images_normalized = batch["conditional_images"] * 2.0 - 1.0
down_block_res_samples, mid_block_res_sample = controlnet(
    noisy_latents,  # Noisy TARGET latents
    timesteps,
    encoder_hidden_states=encoder_hidden_states,
    controlnet_cond=control_images_normalized,  # Depth from SOURCE
)

# C. IP-Adapter: Use COLOR REFERENCE (new color) for color transfer
pixel_values_ip = batch["color_patches"]  # Color reference
image_embeds = image_encoder(pixel_values_ip_normalized).image_embeds
added_cond_kwargs = {"image_embeds": image_embeds}
```

#### ✅ Phase 4: Prepare Inpainting Inputs
```python
# Use SOURCE image (old wall) for masked_source
masked_source_pixel = batch["masked_sources"]  # SOURCE with mask applied
masked_latents = vae.encode(masked_source_pixel_normalized).latent_dist.sample()

# Concatenate: [noisy_latents, mask, masked_source_latents]
unet_input = torch.cat([noisy_latents, mask_latents, masked_latents], dim=1)
```

#### ✅ Phase 5: UNet Prediction
```python
# UNet predicts noise added to TARGET latents
noise_pred = unet(
    unet_input,
    timesteps,
    encoder_hidden_states=encoder_hidden_states,  # Text
    down_block_additional_residuals=down_block_res_samples,  # ControlNet (structure)
    mid_block_additional_residual=mid_block_res_sample,  # ControlNet (structure)
    added_cond_kwargs=added_cond_kwargs  # IP-Adapter (color)
).sample
```

#### ✅ Phase 6: Loss Computation
```python
# Loss: MSE between predicted noise and actual noise
# Model learns to predict noise added to TARGET image
loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
```

### 5. Validation Function

#### ✅ Sửa để match với training logic:
- Dùng source image cho ControlNet và masked source
- Dùng color reference cho IP-Adapter
- Start từ random noise (như inference)
- Generate với các conditions đúng

### 6. Learning Rate Scheduler

#### ✅ Thêm:
```python
lr_scheduler = get_scheduler(
    args.lr_scheduler,
    optimizer=optimizer,
    num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
    num_training_steps=len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps,
)

# Trong training loop:
lr_scheduler.step()
```

### 7. Checkpointing

#### ✅ Cải thiện:
```python
# Save checkpoint mỗi 50 steps
if global_step > 0 and global_step % 50 == 0:
    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    accelerator.save_state(checkpoint_dir)

# Save final checkpoint
final_checkpoint_dir = os.path.join(args.output_dir, "checkpoint-final")
accelerator.save_state(final_checkpoint_dir)
```

### 8. Gradient Clipping

#### ✅ Thêm:
```python
if accelerator.sync_gradients:
    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
```

---

## 📊 So Sánh Training Strategy

| Aspect | ❌ Trước (SAI) | ✅ Sau (ĐÚNG) |
|--------|----------------|----------------|
| **Noisy Input** | Source image (old wall) | **Target image (new wall)** |
| **ControlNet** | Source image | Source image ✅ |
| **IP-Adapter** | Không rõ | Color reference ✅ |
| **Masked Source** | Không rõ | Source image ✅ |
| **Target (GT)** | Source image (old) | **Target image (new)** |
| **Model học** | Reconstruct old color | **Generate new color** |

---

## 🎯 Kết Quả Mong Đợi

Sau khi sửa, model sẽ:
1. ✅ Học generate màu mới từ color reference
2. ✅ Preserve structure từ source image (ControlNet)
3. ✅ Transfer color từ reference (IP-Adapter)
4. ✅ Maintain context từ masked source (Inpainting)

---

## ⚠️ Lưu Ý

1. **Dataset Keys**: Đảm bảo dataset trả về đúng keys:
   - `target` (không phải `targets`)
   - `mask` (không phải `masks`)
   - `masked_sources` (không phải `masked_source`)

2. **ControlNet Input**: ControlNet nhận depth map từ SOURCE image, không phải target

3. **IP-Adapter Input**: IP-Adapter nhận color reference (new color), không phải source image

4. **Masked Source**: Dùng SOURCE image với mask, không phải target

---

## 🧪 Testing

Để test training script:

```bash
python train.py \
    --data_json dataset_test/train/metadata.jsonl \
    --validation_json dataset_test/validation/metadata.jsonl \
    --output_dir output/test_training \
    --train_batch_size 2 \
    --num_train_epochs 1 \
    --resolution 512
```

Kiểm tra:
- ✅ Loss giảm dần
- ✅ Validation images được generate
- ✅ Checkpoints được save
- ✅ Không có errors

---

**Tác giả:** AI Assistant  
**Ngày:** 2026-02-07
