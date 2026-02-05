# Scientific Workflow: Zero-Prompt Wall Inpainting

> **Version:** 2.1.0 (Production Ready)  
> **Last Updated:** 2026-02-05  
> **Author:** CV/AI Research Team  
> **Status:** ✅ Implemented and Tested

---

## Quick Start

```bash
# Fresh training with all fixes
accelerate launch --mixed_precision=fp16 train_lora_inpainting.py \
    --config configs/lora_training.yaml

# Key config changes applied:
# - random_color_prob: 0.3 (30% random colors for generalization)
# - mask_erosion_size: 5 (reduce segmentation noise)
# - quality_threshold: 0.6 (filter low-quality masks)
```

---

1. [Problem Statement](#problem-statement)
2. [Root Cause Analysis](#root-cause-analysis)
3. [Solution Architecture](#solution-architecture)
4. [Data Pipeline](#data-pipeline)
5. [Training Configuration](#training-configuration)
6. [Validation & Metrics](#validation--metrics)
7. [Troubleshooting Guide](#troubleshooting-guide)
8. [Experimental Workflow](#experimental-workflow)

---

## Problem Statement

### Goal
Train an SD 1.5 Inpainting model where users provide:
- **Source Image** with a segmented wall region
- **Binary Mask** indicating the wall area
- **Reference Color** (solid color or texture sample)

The model must repaint the wall with **exact color fidelity** to the reference.

### Observed Issues

| Issue | Symptom | Root Cause |
|-------|---------|------------|
| Color Mismatch | Output color differs from reference | Data leakage (original image used as reference) |
| Flat Gray Depth Maps | Validation grids show gray squares | No normalization applied to depth tensors |
| Zero Metrics | LAB=0.00, Hue=0.00 in logs | Metrics computed incorrectly (wrong mask application) |

---

## Root Cause Analysis

### Issue 1: Data Leakage / Distribution Mismatch

```
TRAINING TIME:                    INFERENCE TIME:
┌─────────────────┐               ┌─────────────────┐
│ Original Image  │ ──► Reference │ Solid Color     │ ──► Reference
│ (with shadows,  │               │ (clean, uniform)│
│  texture, noise)│               │                 │
└─────────────────┘               └─────────────────┘
        ↓                                 ↓
   MISMATCH! The model learns to expect complex images but 
   receives solid colors at inference.
```

**Solution:** Extract dominant color from mask region → Create solid color reference during training.

### Issue 2: Depth Map Visualization

The raw depth tensor values (e.g., 0.1 to 2.5) were directly converted to uint8 without normalization, resulting in values clustering around 0-6 (appearing as near-black or flat gray).

**Solution:** Apply min-max normalization → scale to [0, 255] → apply colormap.

### Issue 3: Metric Calculation

Previous implementation:
```python
ref_lab_mean = ref_lab.mean(axis=(0, 1))  # ❌ Mean of ENTIRE reference
out_lab_masked = out_lab[mask_bool].mean(axis=0)  # Partial fix
```

The reference mean was computed over the entire reference image, but since reference was the original image (not solid color), this was meaningless.

**Solution:** Use solid color reference (uniform color) → compare with output's masked region only.

---

## Solution Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FIXED TRAINING PIPELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ Original     │    │ Segmentation │    │ DOMINANT COLOR       │  │
│  │ Image        │ +  │ Mask         │ ─► │ EXTRACTION           │  │
│  └──────────────┘    └──────────────┘    │ (from masked pixels) │  │
│                                          └──────────────────────┘  │
│                                                     │               │
│                                                     ▼               │
│                                          ┌──────────────────────┐  │
│                                          │ SOLID COLOR          │  │
│                                          │ REFERENCE (512x512)  │  │
│                                          │ + subtle texture     │  │
│                                          └──────────────────────┘  │
│                                                     │               │
│                                                     ▼               │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    SD 1.5 INPAINTING                          │  │
│  │  • UNet with LoRA adapters                                    │  │
│  │  • IP-Adapter Plus (receives solid color reference)           │  │
│  │  • ControlNet Depth (optional)                                │  │
│  │  • Unconditional text embedding (Zero-Prompt strategy)        │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Pipeline

### Dataset Class: `WallInpaintingDataset`

**File:** [dataset_fix.py](file:///mnt/data1tb/vinh/wall_inpainting/dataset_fix.py)

#### Key Methods

##### 1. Dominant Color Extraction

```python
def extract_dominant_color(image, mask) -> Tuple[int, int, int]:
    """
    Extracts the dominant RGB color from pixels within the mask region.
    
    Methods available:
    - "median": Robust to outliers (recommended)
    - "mean": Simple average
    - "kmeans": Clustering for textured walls
    """
```

**Algorithm (Median Method):**
1. Load original image and mask
2. Create boolean mask: `wall_mask = mask > 127`
3. Extract pixels: `wall_pixels = image[wall_mask]`
4. Compute median: `color = np.median(wall_pixels, axis=0)`

##### 2. Solid Color Reference Generation

```python
def create_solid_color_reference(color, size=(512, 512)) -> Image:
    """
    Creates a reference image that simulates inference-time input.
    
    Features:
    - Solid base color
    - Optional Gaussian noise (for CLIP feature extraction)
    - Optional lighting gradient (realistic variation)
    """
```

#### Data Flow Diagram

```
┌────────────────┐     ┌────────────────┐
│ images/001.png │     │ masks/001.png  │
└───────┬────────┘     └───────┬────────┘
        │                      │
        ▼                      ▼
   ┌─────────────────────────────────────┐
   │   extract_dominant_color()          │
   │   - Mask pixels: img[mask > 0.5]    │
   │   - Compute median RGB              │
   └───────────────┬─────────────────────┘
                   │
                   ▼ RGB(185, 160, 140)
   ┌─────────────────────────────────────┐
   │   create_solid_color_reference()    │
   │   - 512×512 solid color image       │
   │   - Add noise (σ=8) for CLIP        │
   │   - Add gradient (0.95-1.05)        │
   └───────────────┬─────────────────────┘
                   │
                   ▼
   ┌─────────────────────────────────────┐
   │   Output Dictionary:                │
   │   - pixel_values: [-1, 1] tensor    │
   │   - mask: [0, 1] tensor             │
   │   - reference_image: [-1, 1] tensor │<── KEY FIX
   │   - dominant_color: (R, G, B)       │
   └─────────────────────────────────────┘
```

---

## Training Configuration

### Config File: `lora_training.yaml`

```yaml
training:
  # Zero-Prompt Strategy
  unconditional_training: true    # Use empty prompt embeddings
  
  # Reference Image Settings (NEW)
  reference:
    extraction_method: "median"   # "median", "mean", or "kmeans"
    add_texture: true             # Add noise for CLIP
    texture_noise_std: 8.0
    add_lighting_gradient: true
    
  # Loss (Optional Enhancement)
  use_color_loss: false           # Enable LAB color consistency loss
  color_loss_weight: 0.1
```

### Training Command

```bash
# Standard training
accelerate launch --mixed_precision=fp16 train_lora_inpainting.py \
    --config configs/lora_training.yaml

# With custom dataset
accelerate launch --mixed_precision=fp16 train_lora_inpainting.py \
    --config configs/lora_training.yaml \
    --dataset-class dataset_fix.WallInpaintingDataset
```

---

## Validation & Metrics

### Validation Utilities

**File:** [validation_utils.py](file:///mnt/data1tb/vinh/wall_inpainting/validation_utils.py)

### Color Fidelity Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **LAB Distance** | `√(ΔL² + Δa² + Δb²)` | Perceptual color distance (lower = better) |
| **Delta-E** | CIE76 formula | Industry standard color difference |
| **Hue Error** | `min(|H₁-H₂|, 180-|H₁-H₂|)` | Circular hue difference in degrees |
| **Chroma Diff** | `√(Δa² + Δb²)` | Color saturation difference |

### Fixed Metric Computation

```python
def compute_color_fidelity_metrics(reference, output, mask):
    """
    CRITICAL: Metrics are computed ONLY on masked region.
    
    1. reference: Solid color image (uniform)
       → Use entire image mean (it's all the same color)
    
    2. output: Model prediction
       → Use ONLY pixels where mask > 0.5
    
    3. Handle edge cases:
       → Empty mask: return zeros
       → NaN values: replace with fallback
    """
```

### Depth Map Visualization

```python
def normalize_depth_map(depth, colormap="inferno"):
    """
    FIXED: Proper depth visualization.
    
    1. Handle NaN/Inf: np.nan_to_num()
    2. Min-max normalize: (depth - min) / (max - min)
    3. Scale to uint8: * 255
    4. Apply colormap: cv2.applyColorMap(INFERNO)
    """
```

**Before Fix:**
![Gray squares - no normalization](./docs/depth_before.png)

**After Fix:**
![Colorized depth with INFERNO colormap](./docs/depth_after.png)

---

## Troubleshooting Guide

### Problem: LAB Distance = 0.0

**Symptoms:**
```
02/05/2026 10:15:23 - INFO - Color metrics: LAB=0.00, Hue=0.00
```

**Diagnosis:**
1. Check if mask is all zeros (empty)
2. Check if reference == output (copying bug)
3. Check if metrics function has try/except returning 0

**Solution:**
```python
# Verify mask has pixels
mask_bool = mask_arr > 0.5
if mask_bool.sum() < 10:
    print("WARNING: Nearly empty mask detected!")
```

### Problem: Reference Color Doesn't Match

**Symptoms:** Model outputs correct structure but wrong color

**Diagnosis:**
1. Is `extract_dominant_color()` being called?
2. Is the reference image actually solid color?
3. Check IP-Adapter scale

**Solution:**
```python
# Debug: Save reference images during training
ref_img = create_solid_color_reference(dominant_color)
ref_img.save(f"debug/reference_{idx}.png")
```

### Problem: Training Loss NaN

**Symptoms:** `loss: nan` after N steps

**Diagnosis:**
1. Check for division by zero in color extraction
2. Verify mask has valid pixels
3. Check reference tensor normalization

**Solution:**
```python
# Add safety checks
if wall_mask.sum() < 100:
    # Fallback to image mean
    color = image.mean(axis=(0, 1))
```

---

## Experimental Workflow

### Phase 1: Baseline Establishment

1. **Run validation on untrained model**
   ```bash
   python validate.py --checkpoint none --samples 50
   ```
   
2. **Record baseline metrics**
   - Expected: High LAB distance (40-80)
   - High hue error (20-60)

### Phase 2: Training with Fixed Dataset

1. **Enable fixed dataset**
   ```python
   from dataset_fix import WallInpaintingDataset
   ```

2. **Train for 1000 steps**
   ```bash
   accelerate launch train_lora_inpainting.py --max-steps 1000
   ```

3. **Check validation grids**
   - Reference column should show solid colors
   - Depth column should show colorized depth maps

### Phase 3: Metric Analysis

1. **Compare metrics across checkpoints**
   ```python
   import json
   from pathlib import Path
   
   metrics_files = sorted(Path("validation_grids").glob("*_metrics.json"))
   for f in metrics_files:
       data = json.load(open(f))
       step = data["step"]
       lab = data["aggregated_metrics"]["val/lab_distance_mean"]
       print(f"Step {step}: LAB={lab:.2f}")
   ```

2. **Target Metrics**
   
   | Metric | Target (Good) | Target (Excellent) |
   |--------|---------------|-------------------|
   | LAB Distance | < 15 | < 5 |
   | Hue Error | < 10° | < 3° |
   | Delta-E | < 10 | < 3 |

### Phase 4: Ablation Studies

| Experiment | Change | Expected Outcome |
|------------|--------|------------------|
| A1 | `color_extraction_method="kmeans"` | Better for textured walls |
| A2 | `add_texture=False` | May hurt CLIP features |
| A3 | `unconditional_training=False` | Baseline comparison |
| A4 | Add `ColorConsistencyLoss` | Lower LAB distance |

---

## File Reference

| File | Purpose |
|------|---------|
| [dataset_fix.py](file:///mnt/data1tb/vinh/wall_inpainting/dataset_fix.py) | Fixed dataset with dominant color extraction |
| [validation_utils.py](file:///mnt/data1tb/vinh/wall_inpainting/validation_utils.py) | Fixed metrics and depth visualization |
| [losses.py](file:///mnt/data1tb/vinh/wall_inpainting/losses.py) | Color consistency loss functions |
| [train_lora_inpainting.py](file:///mnt/data1tb/vinh/wall_inpainting/train_lora_inpainting.py) | Training script |
| [lora_training.yaml](file:///mnt/data1tb/vinh/wall_inpainting/configs/lora_training.yaml) | Training configuration |

---

## Quick Start Checklist

- [ ] Replace `InpaintingDataset` with `WallInpaintingDataset` from `dataset_fix.py`
- [ ] Import validation utilities from `validation_utils.py`
- [ ] Enable `unconditional_training: true` in config
- [ ] Set `num_validation_samples: 50` for rigorous testing
- [ ] Run training and verify:
  - [ ] Reference column shows solid colors
  - [ ] Depth column shows colorized maps
  - [ ] Metrics are non-zero and decreasing
