# Dataset Reconstruction Plan for Wall Inpainting

## Mục tiêu

Xây dựng dataset **color-neutral** với đa dạng màu sắc để train LoRA không bị color bias.

---

## Phase 1: Thu thập ảnh nguồn

### 1.1 Nguồn ảnh
- **Interior images:** Unsplash, Pexels, ADE20K, SUN RGB-D
- **Tiêu chí:** Góc nhìn rõ tường, ánh sáng tự nhiên, đa dạng phong cách

### 1.2 Yêu cầu kỹ thuật
| Thuộc tính | Yêu cầu |
|------------|---------|
| Resolution | ≥ 512x512 |
| Tường chiếm | 10-60% diện tích |
| Đa dạng | Màu tường, ánh sáng, góc chụp |

---

## Phase 2: Segmentation với SAM2

### 2.1 Tại sao SAM2?
- **6x faster** và chính xác hơn SAM1
- Zero-shot generalization tốt
- Hỗ trợ interactive refinement

### 2.2 Pipeline Segmentation

```
Input Image → SAM2 Auto-segment → Filter by size/position → Manual Review (optional)
```

### 2.3 Cải tiến `segmentation.py`

```python
class WallSegmenterV2:
    def __init__(self):
        # Sử dụng SAM2 thay vì FastSAM
        self.sam2 = sam2_model_registry["sam2_hiera_large"](checkpoint)
        self.predictor = SAM2ImagePredictor(self.sam2)
    
    def segment_with_prompts(self, image, points=None, boxes=None):
        """Interactive segmentation với point/box prompts."""
        self.predictor.set_image(image)
        masks, scores, _ = self.predictor.predict(
            point_coords=points,
            point_labels=[1] * len(points) if points else None,
            box=boxes,
        )
        return masks[scores.argmax()]
    
    def auto_segment_walls(self, image):
        """Auto-detect walls với heuristics."""
        # SAM2 auto mode + filter by area, position
        ...
```

### 2.4 Interactive Refinement Tool

Tạo simple GUI để:
1. Hiển thị auto-generated mask
2. Click để add/remove regions
3. Save refined mask

---

## Phase 3: Multi-Color Reference Augmentation ⭐

### 3.1 Chiến lược tạo màu

Mỗi source image → **5-10 color variants:**

```python
COLOR_PALETTE = {
    # Cool colors (để balance dataset)
    "blue": [(100, 130, 180), (70, 100, 150), (150, 180, 210)],
    "green": [(140, 170, 140), (100, 140, 100), (180, 200, 180)],
    "teal": [(100, 150, 160), (80, 130, 140)],
    
    # Warm colors
    "beige": [(220, 200, 180), (240, 220, 195)],
    "gray": [(180, 180, 180), (150, 150, 155), (200, 200, 200)],
    
    # Bold colors
    "terracotta": [(180, 100, 80), (200, 120, 100)],
    "navy": [(50, 60, 90), (40, 50, 80)],
}
```

### 3.2 Realistic Color Transfer với Lighting

```python
def apply_wall_color_with_lighting(
    image: np.ndarray,
    mask: np.ndarray,
    target_color: Tuple[int, int, int],
    preserve_lighting: bool = True,
) -> np.ndarray:
    """
    Apply color while preserving shadows and lighting.
    """
    # 1. Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]  # Luminance (lighting info)
    
    # 2. Extract lighting from original wall
    wall_l = l_channel[mask > 127]
    
    # 3. Create target color in LAB
    target_lab = cv2.cvtColor(
        np.array([[target_color]], dtype=np.uint8),
        cv2.COLOR_RGB2LAB
    )[0, 0]
    
    # 4. Apply target color while preserving luminance
    new_wall = np.zeros_like(image)
    new_wall[:, :, 0] = l_channel if preserve_lighting else target_lab[0]
    new_wall[:, :, 1] = target_lab[1]  # A channel (color)
    new_wall[:, :, 2] = target_lab[2]  # B channel (color)
    
    # 5. Convert back and blend
    new_rgb = cv2.cvtColor(new_wall, cv2.COLOR_LAB2RGB)
    
    # 6. Blend with original at mask edges for smooth transition
    mask_float = mask.astype(float) / 255.0
    mask_float = cv2.GaussianBlur(mask_float, (5, 5), 0)
    
    result = image * (1 - mask_float[..., None]) + new_rgb * mask_float[..., None]
    return result.astype(np.uint8)
```

### 3.3 Output Structure

```
dataset_v2/
├── train/
│   ├── images/
│   │   ├── room001_blue.png      # Recolored
│   │   ├── room001_green.png
│   │   ├── room001_original.png  # Original
│   │   └── ...
│   ├── masks/
│   │   └── room001.png           # Shared mask
│   └── metadata.jsonl
└── validation/
    └── ...
```

---

## Phase 4: Color-Neutral Captions

### 4.1 Chiến lược Caption

**KHÔNG** đề cập màu sắc trong caption:

```python
NEUTRAL_CAPTIONS = [
    "interior room with painted walls, professional photography",
    "modern interior design, high quality photo",
    "living room interior, architectural photography",
    "residential space with wall finish",
]
```

### 4.2 Tại sao?

- Model học **structure/texture** từ text
- Model học **color** từ reference image (IP-Adapter hoặc inpainting target)
- Tránh color bias như dataset hiện tại

---

## Phase 5: Quality Validation

### 5.1 Automated Checks

```python
def validate_sample(image, mask, recolored):
    checks = {
        "mask_coverage": 0.05 < mask.mean() < 0.6,
        "color_applied": color_difference(image, recolored, mask) > threshold,
        "lighting_preserved": luminance_correlation(image, recolored, mask) > 0.8,
        "no_artifacts": edge_quality(recolored, mask) > 0.7,
    }
    return all(checks.values())
```

### 5.2 Manual Review

- Sample 10% để kiểm tra chất lượng
- Focus: Edge quality, color accuracy, lighting preservation

---

## Implementation Timeline

| Phase | Thời gian | Output |
|-------|-----------|--------|
| 1. Thu thập ảnh | 2-3 ngày | 500-1000 source images |
| 2. SAM2 Segmentation | 2-3 ngày | Masks + refinement tool |
| 3. Color Augmentation | 1-2 ngày | 5-10x samples |
| 4. Validation | 1 ngày | Clean dataset |
| 5. Export | 0.5 ngày | Training-ready format |

---

## Files cần tạo/modify

| File | Mục đích |
|------|----------|
| `[NEW] sam2_segmenter.py` | SAM2 wrapper + interactive refinement |
| `[NEW] color_augmentor.py` | Multi-color generation với lighting |
| `[MODIFY] prepare_training_data.py` | Neutral captions, new workflow |
| `[NEW] tools/mask_editor.py` | Simple GUI for mask editing |
