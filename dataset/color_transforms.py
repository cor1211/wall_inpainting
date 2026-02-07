import cv2
import numpy as np
from typing import Tuple

def apply_color_shift(image: np.ndarray, mask: np.ndarray, target_color_rgb: Tuple[int, int, int]) -> np.ndarray:
    """
    Apply target color to the masked region of the image while preserving texture (Luminance).
    Uses CIELAB color space:
    - Retains L channel from original image
    - Uses a, b channels from target color
    
    Args:
        image: RGB numpy array (H, W, 3)
        mask: Binary mask (H, W), 1 where color should be applied
        target_color_rgb: Tuple (R, G, B)
        
    Returns:
        Augmented RGB image
    """
    if not np.any(mask > 0.5):
        return image.copy()
        
    # Standardize inputs
    mask_bool = mask > 0.5
    
    # 1. Convert Image to Lab
    # Convert RGB to Lab
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    
    # 2. Convert Target Color to Lab
    # Make a 1x1 pixel to convert
    target_pixel = np.array([[target_color_rgb]], dtype=np.uint8)
    target_lab = cv2.cvtColor(target_pixel, cv2.COLOR_RGB2LAB)[0, 0]
    target_l, target_a, target_b = target_lab
    
    # 3. Blend logic
    # We want to keep L from image, but use A and B from target.
    # However, simply swapping A/B can look unnatural if the original wall was very dark or very bright.
    # A purely flat color A/B might also look "too perfect".
    # For now, let's try strict "Color" mode: New L = Old L, New AB = Target AB.
    
    # Create output channels
    out_a = a_channel.copy()
    out_b = b_channel.copy()
    
    # IMPROVED LOGIC:
    # Instead of keeping L exact (which fails for Dark <-> Light swaps),
    # we use the Target's L as base, and add the original's L variation (texture).
    
    # 1. Calculate original texture (variation from mean)
    # Masked region only
    masked_l = l_channel[mask_bool]
    if masked_l.size > 0:
        mean_orig_l = np.mean(masked_l)
    else:
        mean_orig_l = 127.5
        
    l_variation = l_channel - mean_orig_l
    
    # 2. New L = Target L + Scaled Texture
    # Scale factor 0.7 reduces the harshness of the texture on the new color
    new_l = target_l + l_variation * 0.7
    new_l = np.clip(new_l, 0, 255).astype(np.uint8)
    
    # 3. Apply changes (only in masked region)
    # Note: We compute new_l for whole image but only apply at mask
    out_l = l_channel.copy()
    out_l[mask_bool] = new_l[mask_bool]
    
    out_a[mask_bool] = target_a
    out_b[mask_bool] = target_b
    
    # Merge back
    out_lab = cv2.merge([out_l, out_a, out_b])
    
    # Convert back to RGB
    out_image = cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB)
    
    # Composite: output where mask is True, original elsewhere
    result = image.copy()
    result[mask_bool] = out_image[mask_bool]
    
    return result

def blend_color_multiply(image: np.ndarray, mask: np.ndarray, target_color_rgb: Tuple[int, int, int]) -> np.ndarray:
    """
    Alternative method using Multiply blend mode.
    Good for white/light walls being painted darker.
    Bad for dark walls being painted lighter.
    """
    mask_bool = mask > 0.5
    if not np.any(mask_bool):
        return image.copy()

    # Normalize image to 0-1
    img_float = image.astype(np.float32) / 255.0
    
    # Target color 0-1
    color_float = np.array(target_color_rgb, dtype=np.float32) / 255.0
    
    # Grayscale version of image (intensity)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray_3d = np.stack([gray, gray, gray], axis=-1)
    
    # Multiply: Result = Gray * Color
    # Usually we want to preserve some original tonality, but this assumes we are painting "over" the wall
    # A better approx of "Multiply" layer in PS:
    colored_region = gray_3d * color_float
    
    # Rescale to match original mean luminance? optional
    
    out_image = (colored_region * 255).clip(0, 255).astype(np.uint8)
    
    result = image.copy()
    result[mask_bool] = out_image[mask_bool]
    
    return result
