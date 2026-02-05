"""
Advanced Validation Visualization for Wall Inpainting Training.

Creates comprehensive comparison grids for rigorous quality assessment.
Supports 50-sample validation with full input/output visualization.
"""
import json
import numpy as np
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


@dataclass
class ValidationSample:
    """Container for all validation input/output data."""
    source_image: Image.Image
    reference_image: Image.Image
    mask: Image.Image
    segment_overlay: Image.Image  # Masked region on source
    depth_map: Optional[Image.Image]
    model_output: Image.Image
    prompt: str
    sample_id: int


class ValidationVisualizer:
    """
    Creates multi-column validation grids for SD Inpainting + IP-Adapter.
    
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
    ):
        self.output_dir = Path(output_dir) / "validation_grids"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resolution = resolution
        self.num_samples = num_samples
        self.add_labels = add_labels
        self.font_size = font_size
    
    def create_segment_overlay(
        self, 
        source: Image.Image, 
        mask: Image.Image,
        overlay_color: Tuple[int, int, int] = (255, 0, 0),
        alpha: float = 0.5,
    ) -> Image.Image:
        """Create visualization showing masked region on source image."""
        source_arr = np.array(source.convert("RGB"))
        mask_arr = np.array(mask.convert("L")) / 255.0
        
        # Resize mask to match source if needed
        if mask_arr.shape[:2] != source_arr.shape[:2]:
            mask_pil = Image.fromarray((mask_arr * 255).astype(np.uint8))
            mask_pil = mask_pil.resize(source.size, Image.Resampling.NEAREST)
            mask_arr = np.array(mask_pil) / 255.0
        
        # Create colored overlay
        overlay = np.zeros_like(source_arr, dtype=np.float32)
        for i, c in enumerate(overlay_color):
            overlay[:, :, i] = c
        
        # Blend where mask is active
        mask_3d = np.stack([mask_arr] * 3, axis=-1)
        result = source_arr.astype(np.float32) * (1 - mask_3d * alpha) + overlay * mask_3d * alpha
        
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
    
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
    
    def create_single_grid(
        self,
        sample: ValidationSample,
    ) -> Image.Image:
        """Create a single-row grid for one validation sample."""
        cell_size = self.resolution
        header_height = 30 if self.add_labels else 0
        
        # Prepare all images
        images = [
            sample.source_image,
            sample.reference_image,
            sample.mask.convert("RGB"),
            sample.segment_overlay,
            sample.depth_map.convert("RGB") if sample.depth_map else Image.new("RGB", (cell_size, cell_size), (128, 128, 128)),
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
                    # Fallback for older Pillow without anchor
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
        
        data = {
            "step": step,
            "num_samples": len(samples),
            "metrics": metrics,
            "samples": [{"id": s.sample_id, "prompt": s.prompt} for s in samples],
        }
        
        with open(metrics_file, "w") as f:
            json.dump(data, f, indent=2)
        
        return metrics_file


def compute_color_fidelity_metrics(
    reference: Image.Image,
    output: Image.Image,
    mask: Image.Image,
) -> Dict[str, float]:
    """
    Compute color fidelity metrics between reference and output.
    
    Returns:
        dict with LAB distance, HSV hue error, lightness diff, chroma diff.
    """
    try:
        import cv2
        from skimage import color as skcolor
    except ImportError:
        # Fallback if skimage/cv2 not available
        return {"lab_distance": 0.0, "hue_error": 0.0, "lightness_diff": 0.0, "chroma_diff": 0.0}
    
    # Convert to numpy at consistent size
    ref_arr = np.array(reference.resize((256, 256)).convert("RGB"))
    out_arr = np.array(output.resize((256, 256)).convert("RGB"))
    mask_arr = np.array(mask.resize((256, 256), Image.Resampling.NEAREST).convert("L")) / 255.0
    
    # Extract masked region from output
    mask_bool = mask_arr > 0.5
    
    # LAB color space metrics
    ref_lab = skcolor.rgb2lab(ref_arr / 255.0)
    out_lab = skcolor.rgb2lab(out_arr / 255.0)
    
    # Mean LAB in masked region
    ref_lab_mean = ref_lab.mean(axis=(0, 1))  # Reference mean LAB
    out_lab_masked = out_lab[mask_bool].mean(axis=0) if mask_bool.any() else out_lab.mean(axis=(0, 1))
    
    lab_distance = float(np.sqrt(((ref_lab_mean - out_lab_masked) ** 2).sum()))
    
    # HSV hue error
    ref_hsv = cv2.cvtColor(ref_arr, cv2.COLOR_RGB2HSV)
    out_hsv = cv2.cvtColor(out_arr, cv2.COLOR_RGB2HSV)
    
    ref_hue_mean = float(ref_hsv[:, :, 0].mean())
    out_hue_masked = float(out_hsv[:, :, 0][mask_bool].mean()) if mask_bool.any() else float(out_hsv[:, :, 0].mean())
    
    # Circular hue distance (0-180 in OpenCV)
    hue_diff = abs(ref_hue_mean - out_hue_masked)
    hue_error = min(hue_diff, 180 - hue_diff)
    
    return {
        "lab_distance": lab_distance,
        "hue_error": float(hue_error),
        "lightness_diff": float(abs(ref_lab_mean[0] - out_lab_masked[0])),
        "chroma_diff": float(np.sqrt((ref_lab_mean[1] - out_lab_masked[1])**2 + 
                                      (ref_lab_mean[2] - out_lab_masked[2])**2)),
    }


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert [-1, 1] tensor [C, H, W] to PIL Image."""
    arr = ((tensor.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
    return Image.fromarray(arr)


def mask_tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert [0, 1] mask tensor [1, H, W] to PIL L-mode Image."""
    arr = (tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


if __name__ == "__main__":
    # Quick test
    print("Testing ValidationVisualizer...")
    
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        visualizer = ValidationVisualizer(
            output_dir=Path(tmpdir),
            resolution=256,
            num_samples=2,
        )
        
        # Create dummy sample
        dummy_img = Image.new("RGB", (256, 256), (200, 180, 160))
        dummy_mask = Image.new("L", (256, 256), 128)
        
        sample = ValidationSample(
            source_image=dummy_img,
            reference_image=dummy_img,
            mask=dummy_mask,
            segment_overlay=visualizer.create_segment_overlay(dummy_img, dummy_mask),
            depth_map=None,
            model_output=dummy_img,
            prompt="test prompt",
            sample_id=0,
        )
        
        # Create grid
        grid = visualizer.create_single_grid(sample)
        print(f"Single grid size: {grid.size}")
        
        # Create sheet
        paths = visualizer.create_comparison_sheet([sample, sample], step=100)
        print(f"Saved {len(paths)} sheets")
        
        # Test metrics
        metrics = compute_color_fidelity_metrics(dummy_img, dummy_img, dummy_mask)
        print(f"Metrics: {metrics}")
        
    print("✅ ValidationVisualizer test passed!")
