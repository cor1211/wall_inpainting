"""
Color Augmentor for Wall Inpainting Dataset.

Generates multiple color variants of wall regions while preserving
realistic lighting, shadows, and texture.

Usage:
    from color_augmentor import ColorAugmentor
    
    augmentor = ColorAugmentor()
    variants = augmentor.generate_color_variants(image, mask, num_colors=5)
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, List, Tuple, Dict, Optional
import cv2
import colorsys
import random


# ============== Color Palettes ==============

# Balanced color palette with cool, warm, and neutral colors
COLOR_PALETTE = {
    # Cool colors (under-represented in typical datasets)
    "sky_blue": [(135, 180, 220), (150, 190, 230), (120, 165, 210)],
    "navy": [(45, 55, 85), (55, 65, 100), (35, 45, 75)],
    "sage_green": [(150, 175, 150), (140, 165, 140), (160, 185, 160)],
    "forest_green": [(70, 100, 75), (80, 110, 85), (60, 90, 65)],
    "teal": [(100, 150, 155), (90, 140, 145), (110, 160, 165)],
    "mint": [(175, 210, 195), (185, 220, 205), (165, 200, 185)],
    
    # Warm neutrals  
    "warm_white": [(250, 245, 235), (245, 240, 230), (255, 250, 240)],
    "cream": [(250, 240, 220), (245, 235, 215), (255, 245, 225)],
    "beige": [(225, 210, 190), (220, 205, 185), (230, 215, 195)],
    "taupe": [(180, 165, 150), (175, 160, 145), (185, 170, 155)],
    
    # Cool neutrals
    "cool_white": [(245, 248, 252), (240, 243, 248), (250, 252, 255)],
    "light_gray": [(200, 200, 205), (195, 195, 200), (205, 205, 210)],
    "gray": [(160, 160, 165), (155, 155, 160), (165, 165, 170)],
    "charcoal": [(80, 80, 85), (75, 75, 80), (85, 85, 90)],
    
    # Accent colors
    "terracotta": [(185, 110, 90), (180, 105, 85), (190, 115, 95)],
    "dusty_rose": [(200, 160, 160), (195, 155, 155), (205, 165, 165)],
    "mustard": [(210, 180, 100), (205, 175, 95), (215, 185, 105)],
    "coral": [(230, 150, 130), (225, 145, 125), (235, 155, 135)],
    "lavender": [(190, 180, 210), (185, 175, 205), (195, 185, 215)],
    "blush": [(235, 200, 195), (230, 195, 190), (240, 205, 200)],
}


class ColorAugmentor:
    """
    Generate color variants of wall regions with realistic lighting.
    
    Key features:
    - Preserves shadows and lighting from original image
    - Generates diverse color palette
    - Supports texture preservation
    - LAB color space for natural color transfer
    """
    
    def __init__(
        self,
        color_palette: Optional[Dict[str, List[Tuple[int, int, int]]]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize color augmentor.
        
        Args:
            color_palette: Custom color palette. Uses default if None.
            seed: Random seed for reproducibility.
        """
        self.color_palette = color_palette or COLOR_PALETTE
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def apply_wall_color(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        target_color: Tuple[int, int, int],
        preserve_lighting: bool = True,
        blend_edges: bool = True,
        edge_blend_size: int = 5,
    ) -> np.ndarray:
        """
        Apply color to wall region while preserving lighting.
        
        Uses LAB color space to separate luminance (lighting) from
        chrominance (color), allowing realistic color transfer.
        
        Args:
            image: RGB image as numpy array (H, W, 3).
            mask: Binary mask (H, W) where 1 = wall.
            target_color: Target RGB color (0-255).
            preserve_lighting: If True, preserve original lighting/shadows.
            blend_edges: If True, smooth blend at mask edges.
            edge_blend_size: Kernel size for edge blending.
            
        Returns:
            Recolored RGB image.
        """
        image = image.astype(np.float32)
        mask_binary = (mask > 127).astype(np.float32)
        
        # Convert to LAB color space
        # LAB separates: L (lightness), A (green-red), B (blue-yellow)
        image_lab = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Get original luminance from wall region
        original_l = image_lab[:, :, 0].copy()
        
        # Convert target color to LAB
        target_rgb = np.array([[target_color]], dtype=np.uint8)
        target_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
        
        # Create new LAB image for wall
        new_lab = image_lab.copy()
        
        if preserve_lighting:
            # Keep original luminance (preserves shadows/highlights)
            # Only change A and B channels (color information)
            
            # Calculate luminance adjustment factor
            # This helps maintain relative brightness of target color
            target_l = target_lab[0]
            original_mean_l = original_l[mask_binary > 0.5].mean() if mask_binary.sum() > 0 else 128
            
            # Scale luminance to match target color's brightness level
            # but preserve the lighting variation (shadows, highlights)
            l_variation = original_l - original_mean_l
            new_l = target_l + l_variation * 0.7  # Keep 70% of lighting variation
            new_l = np.clip(new_l, 0, 255)
            
            new_lab[:, :, 0] = new_l
        else:
            # Replace luminance entirely
            new_lab[:, :, 0] = target_lab[0]
        
        # Apply target color (A and B channels)
        new_lab[:, :, 1] = target_lab[1]
        new_lab[:, :, 2] = target_lab[2]
        
        # Convert back to RGB
        new_rgb = cv2.cvtColor(new_lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
        
        # Blend with original at edges for smooth transition
        if blend_edges:
            # Create soft mask with blurred edges
            mask_soft = cv2.GaussianBlur(
                mask_binary,
                (edge_blend_size * 2 + 1, edge_blend_size * 2 + 1),
                0
            )
        else:
            mask_soft = mask_binary
        
        # Expand mask for broadcasting
        mask_3d = mask_soft[:, :, np.newaxis]
        
        # Blend: result = original * (1 - mask) + new * mask
        result = image * (1 - mask_3d) + new_rgb * mask_3d
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def generate_random_color(self) -> Tuple[str, Tuple[int, int, int]]:
        """
        Generate a random realistic wall color.
        
        Returns:
            Tuple of (color_name, (R, G, B))
        """
        # Strategy: Sample in HSV space for better control over saturation/brightness
        
        # 1. Choose color category probabilistically to ensure variety
        # - 30% Light Neutrals (White, Off-white, Cream, Light Gray)
        # - 20% Dark Neutrals (Charcoal, Dark Gray, Navy)
        # - 30% Desaturated Colors (Sage, Dusty Blue, Terracotta, Mauve)
        # - 20% Saturated Colors (Accent walls: Teal, Yellow, Dark Green)
        category = random.choices(
            ["light_neutral", "dark_neutral", "desaturated", "saturated"],
            weights=[0.3, 0.2, 0.3, 0.2]
        )[0]
        
        h = random.random()  # 0.0 - 1.0
        
        if category == "light_neutral":
            s = random.uniform(0.0, 0.15)
            v = random.uniform(0.85, 0.98)
            name_prefix = "light"
        elif category == "dark_neutral":
            s = random.uniform(0.0, 0.2)
            v = random.uniform(0.15, 0.4)
            name_prefix = "dark"
        elif category == "desaturated":
            s = random.uniform(0.15, 0.4)
            v = random.uniform(0.4, 0.8)
            name_prefix = "dusty"
        else: # saturated
            s = random.uniform(0.4, 0.85)  # Cap saturation at 0.85 to avoid neon
            v = random.uniform(0.3, 0.8)
            name_prefix = "vibrant"
            
        # Convert HSV to RGB
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        rgb = (int(r * 255), int(g * 255), int(b * 255))
        
        # Generate descriptive name
        base_name = get_color_name(rgb)
        
        # Add random hex to ensure uniqueness in filenames
        hex_code = f"{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        full_name = f"{base_name}_{hex_code}"
        
        return full_name, rgb

    def generate_color_variants(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        mask: Union[str, Path, np.ndarray, Image.Image],
        num_colors: int = 5,
        include_original: bool = True,
        preserve_lighting: bool = True,
    ) -> List[Dict]:
        """
        Generate multiple color variants of an image using unlimited random colors.
        """
        # Load image and mask
        if isinstance(image, (str, Path)):
            image = np.array(Image.open(image).convert("RGB"))
        elif isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        
        if isinstance(mask, (str, Path)):
            mask = np.array(Image.open(mask).convert("L"))
        elif isinstance(mask, Image.Image):
            mask = np.array(mask.convert("L"))
        
        results = []
        
        # Include original
        if include_original:
            results.append({
                "image": image.copy(),
                "color_name": "original",
                "color_rgb": self._extract_dominant_color(image, mask),
            })
        
        # Generate random unique colors
        for _ in range(num_colors):
            color_name, color_rgb = self.generate_random_color()
            
            recolored = self.apply_wall_color(
                image, mask, color_rgb,
                preserve_lighting=preserve_lighting,
            )
            
            results.append({
                "image": recolored,
                "color_name": color_name,
                "color_rgb": color_rgb,
            })
        
        return results
    
    def _extract_dominant_color(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[int, int, int]:
        """Extract dominant color from masked region."""
        mask_binary = mask > 127
        
        if mask_binary.sum() == 0:
            return (200, 200, 200)
        
        # Get pixels in mask
        pixels = image[mask_binary].reshape(-1, 3).astype(np.float32)
        
        # Sample if too many
        if len(pixels) > 5000:
            indices = np.random.choice(len(pixels), 5000, replace=False)
            pixels = pixels[indices]
        
        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv2.kmeans(pixels, 1, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
        
        return tuple(centers[0].astype(int).tolist())
    
    def process_dataset(
        self,
        image_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        output_dir: Union[str, Path],
        num_colors_per_image: int = 5,
        include_original: bool = True,
    ) -> Dict:
        """
        Process entire dataset with color augmentation.
        
        Args:
            image_dir: Directory containing source images.
            mask_dir: Directory containing masks.
            output_dir: Output directory.
            num_colors_per_image: Colors per image.
            include_original: Include original in output.
            
        Returns:
            Statistics about processed dataset.
        """
        from tqdm import tqdm
        
        image_dir = Path(image_dir)
        mask_dir = Path(mask_dir)
        output_dir = Path(output_dir)
        
        output_images = output_dir / "images"
        output_masks = output_dir / "masks"
        output_images.mkdir(parents=True, exist_ok=True)
        output_masks.mkdir(parents=True, exist_ok=True)
        
        # Find all images with masks
        image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
        
        stats = {
            "total_source": 0,
            "total_generated": 0,
            "colors_used": {},
        }
        
        metadata = []
        
        for img_path in tqdm(image_files, desc="Augmenting"):
            mask_path = mask_dir / f"{img_path.stem}.png"
            
            if not mask_path.exists():
                continue
            
            stats["total_source"] += 1
            
            try:
                # Generate variants
                variants = self.generate_color_variants(
                    img_path, mask_path,
                    num_colors=num_colors_per_image,
                    include_original=include_original,
                )
                
                # Copy mask once
                mask_output = output_masks / f"{img_path.stem}.png"
                if not mask_output.exists():
                    Image.open(mask_path).save(mask_output)
                
                # Save each variant
                for variant in variants:
                    color_name = variant["color_name"]
                    suffix = "" if color_name == "original" else f"_{color_name}"
                    
                    output_name = f"{img_path.stem}{suffix}.png"
                    output_path = output_images / output_name
                    
                    Image.fromarray(variant["image"]).save(output_path)
                    
                    # Track statistics
                    stats["total_generated"] += 1
                    stats["colors_used"][color_name] = stats["colors_used"].get(color_name, 0) + 1
                    
                    # Add to metadata
                    metadata.append({
                        "image": output_name,
                        "mask": f"{img_path.stem}.png",
                        "color_name": color_name,
                        "color_rgb": variant["color_rgb"],
                    })
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        # Save metadata
        import json
        with open(output_dir / "augmentation_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        return stats


def get_color_name(rgb: Tuple[int, int, int]) -> str:
    """
    Get human-readable color name from RGB.
    
    Args:
        rgb: RGB color tuple.
        
    Returns:
        Color name string.
    """
    r, g, b = rgb
    h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
    
    # Neutral colors (low saturation)
    if s < 0.15:
        if l > 0.9: return "white"
        elif l > 0.75: return "light gray"
        elif l > 0.5: return "gray"
        elif l > 0.25: return "dark gray"
        else: return "black"
    
    # By hue
    hue_deg = h * 360
    
    if hue_deg < 15 or hue_deg >= 345:
        return "red" if s > 0.5 else "dusty rose"
    elif hue_deg < 45:
        return "orange" if s > 0.5 else "terracotta"
    elif hue_deg < 70:
        return "yellow" if s > 0.5 else "mustard"
    elif hue_deg < 150:
        return "green" if s > 0.5 else "sage"
    elif hue_deg < 200:
        return "teal" if s > 0.5 else "dusty teal"
    elif hue_deg < 260:
        return "blue" if s > 0.5 else "dusty blue"
    elif hue_deg < 290:
        return "purple" if s > 0.5 else "lavender"
    else:
        return "pink" if l > 0.6 else "magenta"


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python color_augmentor.py <image_path> <mask_path> [num_colors]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    mask_path = sys.argv[2]
    num_colors = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
    augmentor = ColorAugmentor(seed=42)
    variants = augmentor.generate_color_variants(
        image_path, mask_path,
        num_colors=num_colors,
        include_original=True,
    )
    
    print(f"Generated {len(variants)} variants:")
    for v in variants:
        print(f"  - {v['color_name']}: RGB{v['color_rgb']}")
        
        # Save
        output_path = Path(image_path).stem + f"_{v['color_name']}.png"
        Image.fromarray(v["image"]).save(output_path)
        print(f"    Saved: {output_path}")
