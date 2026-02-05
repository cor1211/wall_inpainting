# Training Mechanics Deep Dive: Wall Inpainting

> **CRITICAL DOCUMENT**: Explains Ground Truth, Loss, and Potential Conflicts  
> **Version**: 1.0.0  
> **Last Updated**: 2026-02-05

---

## Table of Contents

1. [Ground Truth Definition](#1-ground-truth-definition)
2. [Loss Computation](#2-loss-computation)
3. [The Critical Conflict](#3-the-critical-conflict)
4. [Segmentation Noise Problem](#4-segmentation-noise-problem)
5. [Reference Color Strategy Analysis](#5-reference-color-strategy-analysis)
6. [Recommendations](#6-recommendations)

---

## 1. Ground Truth Definition

### What is GT in Diffusion Model Training?

In Stable Diffusion Inpainting training, **Ground Truth is NOT the final image**. Instead:

```
Ground Truth = ε (epsilon) = The NOISE that was added to the latent
```

### Training Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SD INPAINTING TRAINING FLOW                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  STEP 1: Encode original image to latent space                         │
│  ┌──────────────┐     ┌───────┐     ┌──────────────┐                   │
│  │ Original     │ ──► │  VAE  │ ──► │ z₀ (latent)  │                   │
│  │ Image        │     │Encoder│     │ [4, 64, 64]  │                   │
│  └──────────────┘     └───────┘     └──────────────┘                   │
│                                              │                          │
│  STEP 2: Sample random noise ε and timestep t                          │
│                                              ▼                          │
│  ┌──────────────┐     ε ~ N(0, I)    ┌──────────────┐                  │
│  │ Random Noise │ ──────────────────►│ z_t = αₜz₀   │                  │
│  │ ε            │     + add noise    │    + σₜε     │                  │
│  └──────────────┘                    └──────────────┘                  │
│         │                                    │                          │
│         │                                    ▼                          │
│         │             ┌────────────────────────────────────┐           │
│         │             │ UNet Inpainting Input:             │           │
│         │             │ [z_t | mask | masked_image_latent] │           │
│         │             │ [4 + 1 + 4 = 9 channels]           │           │
│         │             └────────────────────────────────────┘           │
│         │                            │                                  │
│         │                            ▼                                  │
│         │             ┌────────────────────────────────────┐           │
│         │             │       UNet Prediction              │           │
│         │             │       ε̂ (predicted noise)          │           │
│         │             └────────────────────────────────────┘           │
│         │                            │                                  │
│         ▼                            ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │                   LOSS = MSE(ε̂, ε)                       │ ◄── GT   │
│  │                   (predicted noise vs actual noise)      │          │
│  └──────────────────────────────────────────────────────────┘          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Insight

> **The model learns to predict what noise was added to the latent, NOT what the final image should look like.**

During inference, the predicted noise is iteratively subtracted to reconstruct the clean image.

---

## 2. Loss Computation

### Current Implementation (from `train_lora_inpainting.py`)

```python
# Line 394-434 in train_lora_inpainting.py

# Sample noise (THIS IS THE GROUND TRUTH)
noise = torch.randn_like(latents)   # ε ~ N(0, I)

# Add noise to latents at timestep t
noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
# z_t = √(α_t) * z_0 + √(1 - α_t) * ε

# UNet predicts the noise
noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states).sample

# LOSS: MSE between predicted noise and actual noise
loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
```

### What This Means

| Component | Value | Meaning |
|-----------|-------|---------|
| `noise` | Random tensor | Ground Truth (ε) |
| `noise_pred` | UNet output | Model's prediction (ε̂) |
| `loss` | MSE value | How wrong the prediction is |

### Important: What is NOT in the Loss

The following are **NOT directly in the loss function**:
- ❌ Reference color matching
- ❌ IP-Adapter features
- ❌ Final image quality
- ❌ Color fidelity metrics

> **The model only learns to denoise. Color matching is an EMERGENT property, not directly optimized.**

---

## 3. The Critical Conflict

### ⚠️ WARNING: Major Architectural Issue

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 THE TRAIN/INFERENCE MISMATCH                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  TRAINING TIME:                                                         │
│  ┌─────────────────────────────────────────────┐                       │
│  │ Inputs:                                     │                       │
│  │   • Original Image (with wall content)     │                       │
│  │   • Mask (areas to inpaint)                │                       │
│  │   • Text Embedding (or empty for Zero-Prompt)│                     │
│  │                                             │                       │
│  │ GT Loss Target:                             │                       │
│  │   • Reconstruct EXACT ORIGINAL PIXELS      │ ◄── PROBLEM!          │
│  │   • (the noise that recreates original)    │                       │
│  │                                             │                       │
│  │ NOT USED:                                   │                       │
│  │   • IP-Adapter ❌                           │                       │
│  │   • Reference Color ❌                      │                       │
│  └─────────────────────────────────────────────┘                       │
│                                                                         │
│  INFERENCE TIME:                                                        │
│  ┌─────────────────────────────────────────────┐                       │
│  │ Inputs:                                     │                       │
│  │   • Original Image (with wall to repaint)  │                       │
│  │   • Mask                                   │                       │
│  │   • Reference Color Image (solid color)   │ ◄── NEW!              │
│  │   • IP-Adapter (injects reference features)│ ◄── NEW!              │
│  │                                             │                       │
│  │ Desired Output:                             │                       │
│  │   • Wall repainted with REFERENCE COLOR    │                       │
│  │   • NOT the original wall color            │                       │
│  └─────────────────────────────────────────────┘                       │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│  CONFLICT: Model is trained to reconstruct original, but we want it    │
│            to use reference color at inference time.                   │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why This Matters

1. **Training teaches**: "Given noisy wall, predict noise to get back original wall"
2. **Inference expects**: "Given noisy wall + reference color, paint wall with reference color"

These are **fundamentally different tasks**.

### The IP-Adapter Gap

```python
# TRAINING: No IP-Adapter
noise_pred = unet(latent_input, timesteps, text_embedding).sample

# INFERENCE: With IP-Adapter
noise_pred = unet(latent_input, timesteps, text_embedding, 
                  added_cond_kwargs={"image_embeds": ip_adapter_embeds}).sample
```

> The LoRA is trained without ever seeing IP-Adapter signals. At inference, IP-Adapter is suddenly added, creating a distribution shift.

---

## 4. Segmentation Noise Problem

### The Mask Quality Issue

You mentioned: *"dataset có nhiều khu vực không phải wall/ceil bị segment nhầm"*

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SEGMENTATION NOISE IMPACT                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  IDEAL MASK:                          NOISY MASK:                      │
│  ┌────────────────────┐              ┌────────────────────┐            │
│  │ ░░░░░░░░░░░░░░░░░░ │              │ ░░░░░░░░░░░░░░░░░░ │            │
│  │ ░░░░████████░░░░░░ │              │ ░░░░████████░░░░░░ │            │
│  │ ░░░░████████░░░░░░ │ Wall only   │ ░░░░████████░░░░░░ │ Wall       │
│  │ ░░░░████████░░░░░░ │              │ ░░░░████████░████░ │ + Furniture│
│  │ ░░░░░░░░░░░░░░░░░░ │              │ ░░██░░░░░░░░░████░ │ + Floor    │
│  └────────────────────┘              └────────────────────┘            │
│                                                                         │
│  TRAINING EFFECT:                                                       │
│  - Model learns: "Masked region should look like [wall+furniture+floor]"│
│  - Model generalizes: "Paint everything in mask with wall color"       │
│  - FAILURE: Furniture gets painted with wall color at inference        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Quantifying the Problem

| Mask Quality | Expected Behavior | Actual Behavior |
|--------------|-------------------|-----------------|
| Clean (only wall) | Paint wall correctly | ✅ Works |
| Noisy (includes furniture) | Ignore furniture | ❌ Paints furniture too |
| Very noisy (>30% non-wall) | Catastrophic | ❌ Completely wrong output |

### Mitigation Strategies

1. **Data Filtering**: Remove samples with low IoU between mask and actual wall
2. **Mask Erosion**: Shrink masks to reduce edge artifacts
3. **Quality Score Threshold**: Increase `quality_threshold` in config
4. **Adversarial Training**: Add discriminator to detect non-wall regions

---

## 5. Reference Color Strategy Analysis

### Current: Median Color Extraction

```python
# Current implementation in dataset_fix.py
dominant_color = np.median(wall_pixels, axis=0)
reference_image = solid_color(dominant_color)
```

**Pros:**
- Realistic wall colors
- Robust to shadows/highlights

**Cons:**
- Model only sees "natural" colors
- May not generalize to unusual colors (bright green, etc.)

### Alternative: Random Color Sampling

```python
# Proposed alternative
random_color = (
    random.randint(0, 255),
    random.randint(0, 255), 
    random.randint(0, 255)
)
reference_image = solid_color(random_color)
```

**Pros:**
- Full color space coverage
- Better generalization to any color

**Cons:**
- Train/inference mismatch (training sees random, inference sees realistic)
- May confuse the model initially

### Recommended: Hybrid Strategy

```python
# Best of both worlds
if random.random() < 0.3:  # 30% random colors
    color = random_color()
else:  # 70% extracted colors
    color = extract_median(wall_pixels)
```

**Analysis Table:**

| Strategy | Generalization | Realism | Recommended |
|----------|---------------|---------|-------------|
| Median only | ⭐⭐ | ⭐⭐⭐⭐⭐ | For production |
| Random only | ⭐⭐⭐⭐⭐ | ⭐⭐ | For research |
| Hybrid (30/70) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | **Best choice** |
| K-means cluster | ⭐⭐⭐ | ⭐⭐⭐⭐ | For textured walls |

---

## 6. Recommendations

### Immediate Fixes

1. **Add Color Augmentation**
   ```yaml
   # In lora_training.yaml
   reference:
     strategy: "hybrid"
     random_color_prob: 0.3
   ```

2. **Increase Mask Quality Threshold**
   ```yaml
   dataset:
     quality_threshold: 0.7  # was 0.5
   ```

3. **Add Mask Erosion**
   ```python
   mask = cv2.erode(mask, kernel=np.ones((5,5)), iterations=2)
   ```

### Long-term Solutions

1. **IP-Adapter Integrated Training** (Most Important)
   - Train LoRA WITH IP-Adapter active
   - Loss becomes aware of reference color

2. **Color Consistency Loss**
   - Add auxiliary loss: `LAB_distance(output_color, reference_color)`
   - Periodically decode latents and compute color loss

3. **Mask Quality Estimation**
   - Train a small classifier to score mask quality
   - Filter or weight samples by quality

### Priority Order

| Priority | Task | Impact | Effort |
|----------|------|--------|--------|
| P0 | Hybrid color strategy | High | Low |
| P0 | Fix depth map N/A | Medium | Low |
| P1 | Increase quality threshold | High | None |
| P2 | Mask erosion | Medium | Low |
| P3 | IP-Adapter training | Very High | High |
| P3 | Color consistency loss | High | Medium |

---

## Summary

> **The fundamental issue is that LoRA training optimizes for noise prediction without any awareness of the reference color. IP-Adapter provides reference color at inference time, but the LoRA was never trained to use it.**

To truly fix color fidelity, you need either:
1. Train with IP-Adapter active (complex but correct)
2. Add color consistency loss (moderate complexity)
3. Use larger/better IP-Adapter weights (simple but limited)
