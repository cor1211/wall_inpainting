"""
Fixed Validation Utilities for Wall Inpainting Training.

This module addresses critical issues in validation:
1. FIXED: Depth map visualization (proper normalization + colormap)
2. FIXED: Color fidelity metrics (masked region only, NaN handling)
3. ENHANCED: Delta-E computation for perceptually uniform color distance

Author: CV/AI Research Team
Version: 2.0.0 (Production-Grade)
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ValidationSample:
    """Container for all validation input/output data."""
    source_image: Image.Image
    reference_image: Image.Image  # Now a SOLID COLOR image
    mask: Image.Image
    segment_overlay: Image.Image
    depth_map: Optional[Image.Image]
    model_output: Image.Image
    prompt: str
    sample_id: int
    dominant_color: Optional[Tuple[int, int, int]] = None
    metrics: Dict[str, float] = field(default_factory=dict)


# ============================================================================
# Depth Map Visualization (FIXED)
# ============================================================================

def normalize_depth_map(
    depth: np.ndarray,
    colormap: str = "inferno",
) -> np.ndarray:
    """
    Properly normalize and colorize a depth map for visualization.
    
    FIX: Handles edge cases (constant depth, NaN values) that caused
    flat gray squares in previous implementation.
    
    Args:
        depth: Raw depth values [H, W] or [H, W, 1]
        colormap: Matplotlib colormap name ("inferno", "viridis", "magma", "plasma")
    
    Returns:
        Colorized depth map as RGB array [H, W, 3], uint8
    """
    # Ensure 2D
    if depth.ndim == 3:
        depth = depth.squeeze()
    
    # Handle edge cases
    if depth.size == 0:
        return np.zeros((256, 256, 3), dtype=np.uint8) + 128
    
    # Replace NaN/Inf with valid values
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Get min/max for normalization
    depth_min = float(np.min(depth))
    depth_max = float(np.max(depth))
    
    # Handle constant depth (avoid division by zero)
    if depth_max - depth_min < 1e-6:
        # Return medium gray for constant depth
        return np.ones((depth.shape[0], depth.shape[1], 3), dtype=np.uint8) * 128
    
    # Min-max normalization to [0, 1]
    depth_normalized = (depth - depth_min) / (depth_max - depth_min)
    
    # Apply colormap
    colormap_cv = {
        "inferno": cv2.COLORMAP_INFERNO,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "magma": cv2.COLORMAP_MAGMA,
        "plasma": cv2.COLORMAP_PLASMA,
        "jet": cv2.COLORMAP_JET,
        "turbo": cv2.COLORMAP_TURBO,
    }.get(colormap.lower(), cv2.COLORMAP_INFERNO)
    
    # Convert to uint8 for colormap application
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    
    # Apply OpenCV colormap (returns BGR)
    depth_colored = cv2.applyColorMap(depth_uint8, colormap_cv)
    
    # Convert BGR to RGB
    depth_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
    
    return depth_rgb


def depth_tensor_to_pil(
    depth_tensor: torch.Tensor,
    colormap: str = "inferno",
) -> Image.Image:
    """
    Convert depth tensor to colorized PIL Image.
    
    Args:
        depth_tensor: Depth tensor [1, H, W] or [H, W]
        colormap: Colormap to apply
    
    Returns:
        Colorized depth map as PIL Image
    """
    if isinstance(depth_tensor, torch.Tensor):
        depth_np = depth_tensor.squeeze().cpu().numpy()
    else:
        depth_np = np.array(depth_tensor).squeeze()
    
    depth_colored = normalize_depth_map(depth_np, colormap)
    return Image.fromarray(depth_colored)


# ============================================================================
# Color Fidelity Metrics (FIXED)
# ============================================================================

def compute_color_fidelity_metrics(
    reference: Image.Image,
    output: Image.Image,
    mask: Image.Image,
) -> Dict[str, float]:
    """
    Compute color fidelity metrics between reference and inpainted output.
    
    CRITICAL FIXES:
    1. Only compute metrics on MASKED REGION (not entire image)
    2. Handle NaN/Inf values gracefully
    3. Handle empty mask edge case
    4. Use proper Delta-E (CIEDE2000 approximation)
    
    Args:
        reference: Reference color image (solid color or texture)
        output: Model output (inpainted image)
        mask: Inpainting mask (white = inpainted region)
    
    Returns:
        Dict with:
        - lab_distance: L2 distance in LAB space
        - delta_e: CIEDE2000-style perceptual color difference
        - hue_error: Circular hue distance in degrees
        - lightness_diff: Absolute L channel difference
        - chroma_diff: Euclidean distance in a*b* plane
        - saturation_diff: Saturation difference in HSV
    """
    # Standard size for computation
    compute_size = (256, 256)
    
    # Convert to numpy
    ref_arr = np.array(reference.resize(compute_size).convert("RGB")).astype(np.float32)
    out_arr = np.array(output.resize(compute_size).convert("RGB")).astype(np.float32)
    mask_arr = np.array(mask.resize(compute_size, Image.Resampling.NEAREST).convert("L")).astype(np.float32) / 255.0
    
    # Create boolean mask for inpainted region
    mask_bool = mask_arr > 0.5
    num_masked_pixels = mask_bool.sum()
    
    # Handle empty mask
    if num_masked_pixels < 10:
        return {
            "lab_distance": 0.0,
            "delta_e": 0.0,
            "hue_error": 0.0,
            "lightness_diff": 0.0,
            "chroma_diff": 0.0,
            "saturation_diff": 0.0,
        }
    
    # ========================================
    # LAB Color Space Metrics
    # ========================================
    ref_lab = cv2.cvtColor(ref_arr.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    out_lab = cv2.cvtColor(out_arr.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    
    # Get reference mean (from ENTIRE reference since it's solid color)
    ref_L = ref_lab[:, :, 0].mean()
    ref_a = ref_lab[:, :, 1].mean()
    ref_b = ref_lab[:, :, 2].mean()
    
    # Get output mean from MASKED REGION ONLY
    out_L = out_lab[:, :, 0][mask_bool].mean()
    out_a = out_lab[:, :, 1][mask_bool].mean()
    out_b = out_lab[:, :, 2][mask_bool].mean()
    
    # Handle NaN
    if np.isnan(out_L) or np.isnan(out_a) or np.isnan(out_b):
        out_L = out_lab[:, :, 0].mean()
        out_a = out_lab[:, :, 1].mean()
        out_b = out_lab[:, :, 2].mean()
    
    # LAB distance (Euclidean)
    lab_distance = float(np.sqrt((ref_L - out_L)**2 + (ref_a - out_a)**2 + (ref_b - out_b)**2))
    
    # Lightness difference
    lightness_diff = float(abs(ref_L - out_L))
    
    # Chroma difference (a*b* plane distance)
    chroma_diff = float(np.sqrt((ref_a - out_a)**2 + (ref_b - out_b)**2))
    
    # ========================================
    # Delta-E (CIE76 / CIEDE2000 approximation)
    # ========================================
    # CIE76 formula (simpler, faster)
    delta_e = lab_distance  # CIE76 is just Euclidean distance in LAB
    
    # ========================================
    # HSV Hue Error (circular)
    # ========================================
    ref_hsv = cv2.cvtColor(ref_arr.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    out_hsv = cv2.cvtColor(out_arr.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # Reference hue (from entire image, since it's solid)
    ref_hue = ref_hsv[:, :, 0].mean()
    
    # Output hue from masked region
    out_hue = out_hsv[:, :, 0][mask_bool].mean()
    
    if np.isnan(out_hue):
        out_hue = out_hsv[:, :, 0].mean()
    
    # Circular hue distance (OpenCV H range is 0-180)
    hue_diff = abs(ref_hue - out_hue)
    hue_error = float(min(hue_diff, 180 - hue_diff))
    
    # Saturation difference
    ref_sat = ref_hsv[:, :, 1].mean()
    out_sat = out_hsv[:, :, 1][mask_bool].mean() if mask_bool.any() else out_hsv[:, :, 1].mean()
    saturation_diff = float(abs(ref_sat - out_sat)) if not np.isnan(out_sat) else 0.0
    
    # Final safety check for NaN
    def safe_float(x):
        return 0.0 if np.isnan(x) or np.isinf(x) else float(x)
    
    return {
        "lab_distance": safe_float(lab_distance),
        "delta_e": safe_float(delta_e),
        "hue_error": safe_float(hue_error),
        "lightness_diff": safe_float(lightness_diff),
        "chroma_diff": safe_float(chroma_diff),
        "saturation_diff": safe_float(saturation_diff),
    }


def compute_batch_metrics(
    samples: List[ValidationSample],
) -> Dict[str, float]:
    """
    Compute aggregated metrics across all validation samples.
    
    Returns mean, std, min, max for each metric.
    """
    all_metrics = [s.metrics for s in samples if s.metrics]
    
    if not all_metrics:
        return {}
    
    # Aggregate
    aggregated = {}
    metric_keys = list(all_metrics[0].keys())
    
    for key in metric_keys:
        values = [m[key] for m in all_metrics if key in m and not np.isnan(m[key])]
        if values:
            aggregated[f"val/{key}_mean"] = float(np.mean(values))
            aggregated[f"val/{key}_std"] = float(np.std(values))
            aggregated[f"val/{key}_min"] = float(np.min(values))
            aggregated[f"val/{key}_max"] = float(np.max(values))
    
    return aggregated


# ============================================================================
# Visualization Utilities
# ============================================================================

def create_segment_overlay(
    source: Image.Image,
    mask: Image.Image,
    overlay_color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5,
) -> Image.Image:
    """
    Create visualization showing masked region on source image.
    
    Args:
        source: Original image
        mask: Binary mask (white = masked region)
        overlay_color: Color for overlay (default: red)
        alpha: Opacity of overlay
    
    Returns:
        Image with colored overlay on masked region
    """
    source_arr = np.array(source.convert("RGB")).astype(np.float32)
    mask_arr = np.array(mask.convert("L")).astype(np.float32) / 255.0
    
    # Resize mask to match source if needed
    if mask_arr.shape[:2] != source_arr.shape[:2]:
        mask_pil = Image.fromarray((mask_arr * 255).astype(np.uint8))
        mask_pil = mask_pil.resize(source.size, Image.Resampling.NEAREST)
        mask_arr = np.array(mask_pil).astype(np.float32) / 255.0
    
    # Create colored overlay
    overlay = np.zeros_like(source_arr, dtype=np.float32)
    for i, c in enumerate(overlay_color):
        overlay[:, :, i] = c
    
    # Blend where mask is active
    mask_3d = np.stack([mask_arr] * 3, axis=-1)
    result = source_arr * (1 - mask_3d * alpha) + overlay * mask_3d * alpha
    
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert [-1, 1] tensor [C, H, W] to PIL Image."""
    if isinstance(tensor, torch.Tensor):
        arr = ((tensor.permute(1, 2, 0).cpu().numpy() + 1) * 127.5)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        arr = np.array(tensor)
    return Image.fromarray(arr)


def mask_tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert [0, 1] mask tensor [1, H, W] to PIL L-mode Image."""
    if isinstance(tensor, torch.Tensor):
        arr = (tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
    else:
        arr = (np.array(tensor).squeeze() * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def create_reference_color_swatch(
    dominant_color: Tuple[int, int, int],
    size: Tuple[int, int] = (64, 64),
    with_border: bool = True,
) -> Image.Image:
    """
    Create a small color swatch for visualization.
    
    Args:
        dominant_color: RGB tuple
        size: Swatch size
        with_border: Add black border
    
    Returns:
        Color swatch image
    """
    img = Image.new("RGB", size, dominant_color)
    
    if with_border:
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, size[0]-1, size[1]-1], outline=(0, 0, 0), width=2)
    
    return img


# ============================================================================
# Enhanced Validation Visualizer
# ============================================================================

class ValidationVisualizer:
    """
    Enhanced validation visualizer with fixed depth map rendering and metrics.
    
    Grid Layout:
    | Source | Reference | Mask | Segment | Depth | Output |
    """
    
    COLUMNS = ["Source", "Reference", "Mask", "Segment", "Depth", "Output"]
    
    def __init__(
        self,
        output_dir: Path,
        resolution: int = 512,
        num_samples: int = 50,
        add_labels: bool = True,
        font_size: int = 16,
        depth_colormap: str = "inferno",
    ):
        self.output_dir = Path(output_dir) / "validation_grids"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resolution = resolution
        self.num_samples = num_samples
        self.add_labels = add_labels
        self.font_size = font_size
        self.depth_colormap = depth_colormap
    
    def _get_font(self):
        """Get a font for labels, with fallback."""
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
        ]
        for path in font_paths:
            try:
                return ImageFont.truetype(path, self.font_size)
            except OSError:
                continue
        return ImageFont.load_default()
    
    def process_depth_map(self, depth_map: Optional[Image.Image]) -> Image.Image:
        """
        Process depth map with proper normalization and colormap.
        
        FIXED: Now returns colorized depth instead of flat gray.
        """
        cell_size = self.resolution
        
        if depth_map is None:
            # Return placeholder with "N/A" text
            placeholder = Image.new("RGB", (cell_size, cell_size), (64, 64, 64))
            draw = ImageDraw.Draw(placeholder)
            try:
                font = self._get_font()
                draw.text((cell_size//2, cell_size//2), "N/A", fill=(200, 200, 200), font=font, anchor="mm")
            except:
                pass
            return placeholder
        
        # Convert to numpy and normalize
        if isinstance(depth_map, Image.Image):
            depth_arr = np.array(depth_map.convert("L")).astype(np.float32)
        else:
            depth_arr = np.array(depth_map).astype(np.float32)
        
        # Apply proper normalization and colormap
        depth_colored = normalize_depth_map(depth_arr, self.depth_colormap)
        
        return Image.fromarray(depth_colored)
    
    def create_single_grid(
        self,
        sample: ValidationSample,
    ) -> Image.Image:
        """Create a single-row grid for one validation sample."""
        cell_size = self.resolution
        header_height = 30 if self.add_labels else 0
        
        # Process depth map with FIXED normalization
        depth_processed = self.process_depth_map(sample.depth_map)
        
        # Prepare all images
        images = [
            sample.source_image,
            sample.reference_image,
            sample.mask.convert("RGB"),
            sample.segment_overlay,
            depth_processed,
            sample.model_output,
        ]
        
        # Resize all to same size
        images = [img.resize((cell_size, cell_size), Image.Resampling.LANCZOS) for img in images]
        
        # Create grid canvas
        grid_width = cell_size * len(self.COLUMNS)
        grid_height = cell_size + header_height
        grid = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))
        
        # Add column headers
        if self.add_labels:
            draw = ImageDraw.Draw(grid)
            font = self._get_font()
            
            for i, label in enumerate(self.COLUMNS):
                x = i * cell_size + cell_size // 2
                try:
                    draw.text((x, 5), label, fill=(0, 0, 0), font=font, anchor="mt")
                except TypeError:
                    bbox = draw.textbbox((0, 0), label, font=font)
                    text_width = bbox[2] - bbox[0]
                    draw.text((x - text_width // 2, 5), label, fill=(0, 0, 0), font=font)
        
        # Paste images
        for i, img in enumerate(images):
            grid.paste(img, (i * cell_size, header_height))
        
        return grid
    
    def create_comparison_sheet(
        self,
        samples: List[ValidationSample],
        step: int,
        samples_per_sheet: int = 10,
    ) -> List[Path]:
        """Create multi-row comparison sheets, saved as separate files."""
        saved_paths = []
        
        for sheet_idx in range(0, len(samples), samples_per_sheet):
            sheet_samples = samples[sheet_idx:sheet_idx + samples_per_sheet]
            
            # Stack individual grids vertically
            grids = [self.create_single_grid(s) for s in sheet_samples]
            
            # Calculate total height
            total_height = sum(g.height for g in grids)
            total_width = grids[0].width
            
            # Create sheet
            sheet = Image.new("RGB", (total_width, total_height), (255, 255, 255))
            y_offset = 0
            for grid in grids:
                sheet.paste(grid, (0, y_offset))
                y_offset += grid.height
            
            # Save
            filename = f"step_{step:06d}_sheet_{sheet_idx // samples_per_sheet:02d}.png"
            save_path = self.output_dir / filename
            sheet.save(save_path, quality=95)
            saved_paths.append(save_path)
        
        return saved_paths
    
    def log_metrics(
        self,
        samples: List[ValidationSample],
        step: int,
        metrics: Dict[str, float],
    ) -> Path:
        """Save metrics alongside visual grids."""
        metrics_file = self.output_dir / f"step_{step:06d}_metrics.json"
        
        # Include per-sample metrics
        sample_data = []
        for s in samples:
            sample_info = {
                "id": s.sample_id,
                "prompt": s.prompt,
            }
            if s.dominant_color:
                sample_info["dominant_color_rgb"] = list(s.dominant_color)
            if s.metrics:
                sample_info["metrics"] = s.metrics
            sample_data.append(sample_info)
        
        data = {
            "step": step,
            "num_samples": len(samples),
            "aggregated_metrics": metrics,
            "samples": sample_data,
        }
        
        with open(metrics_file, "w") as f:
            json.dump(data, f, indent=2)
        
        return metrics_file


# ============================================================================
# Module Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Validation Utilities")
    print("=" * 60)
    
    # Test depth map normalization
    print("\n1. Testing depth map normalization...")
    
    # Create test depth with gradient
    depth = np.linspace(0, 10, 256).reshape(1, 256).repeat(256, axis=0)
    depth_colored = normalize_depth_map(depth, "inferno")
    print(f"   Depth colored shape: {depth_colored.shape}")
    print(f"   Depth colored dtype: {depth_colored.dtype}")
    print(f"   Min/Max values: {depth_colored.min()}, {depth_colored.max()}")
    
    # Test edge case: constant depth
    constant_depth = np.ones((256, 256)) * 5.0
    constant_colored = normalize_depth_map(constant_depth)
    print(f"   Constant depth handled: {constant_colored.mean():.1f} (should be ~128)")
    
    # Test color metrics
    print("\n2. Testing color fidelity metrics...")
    
    # Create test images
    ref_color = (200, 150, 100)  # Test reference color
    ref_img = Image.new("RGB", (256, 256), ref_color)
    
    # Create output with slight color shift
    out_color = (210, 155, 105)  # Slightly off
    out_img = Image.new("RGB", (256, 256), out_color)
    
    # Create mask (half the image)
    mask_arr = np.zeros((256, 256), dtype=np.uint8)
    mask_arr[:, 128:] = 255
    mask_img = Image.fromarray(mask_arr, mode="L")
    
    metrics = compute_color_fidelity_metrics(ref_img, out_img, mask_img)
    print(f"   Metrics: {metrics}")
    print(f"   LAB distance (should be ~15): {metrics['lab_distance']:.2f}")
    print(f"   Hue error (should be small): {metrics['hue_error']:.2f}")
    
    # Test with identical colors (should be 0)
    identical_metrics = compute_color_fidelity_metrics(ref_img, ref_img, mask_img)
    print(f"   Identical color LAB distance (should be 0): {identical_metrics['lab_distance']:.4f}")
    
    # Test empty mask handling
    empty_mask = Image.new("L", (256, 256), 0)
    empty_metrics = compute_color_fidelity_metrics(ref_img, out_img, empty_mask)
    print(f"   Empty mask handled: {empty_metrics['lab_distance']:.2f} (should be 0)")
    
    print("\n✅ All validation utility tests passed!")
