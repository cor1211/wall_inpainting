"""
SAM2 Wall Segmenter for Dataset Creation.

Uses Segment Anything Model 2 (SAM2) for high-quality wall segmentation
with interactive refinement support.

Usage:
    from sam2_segmenter import SAM2WallSegmenter
    
    segmenter = SAM2WallSegmenter()
    mask = segmenter.auto_segment_walls(image_path)
    
    # Interactive refinement
    mask = segmenter.refine_with_points(image, mask, add_points, remove_points)
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Optional, List, Tuple, Dict
import cv2
import torch

# SAM2 imports (will be installed separately)
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("SAM2 not installed. Install with: pip install segment-anything-2")


class SAM2WallSegmenter:
    """
    Wall segmentation using SAM2 with heuristic filtering.
    
    Features:
    - Auto-segment with wall detection heuristics
    - Interactive point-based refinement
    - Box prompt support
    - Batch processing
    """
    
    # SAM2 model configs
    MODEL_CONFIGS = {
        "tiny": ("sam2_hiera_t.yaml", "sam2_hiera_tiny.pt"),
        "small": ("sam2_hiera_s.yaml", "sam2_hiera_small.pt"),
        "base": ("sam2_hiera_b+.yaml", "sam2_hiera_base_plus.pt"),
        "large": ("sam2_hiera_l.yaml", "sam2_hiera_large.pt"),
    }
    
    
    
    def __init__(
        self,
        model_size: str = "large",
        checkpoint_dir: str = "models/sam2",
        device: Optional[str] = None,
        auto_download: bool = True,
    ):
        """
        Initialize SAM2 segmenter.
        """
        if not SAM2_AVAILABLE:
            raise ImportError(
                "SAM2 package not installed.\n"
                "Please run: pip install git+https://github.com/facebookresearch/segment-anything-2.git"
            )
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_size = model_size
        
        # Load SAM2 model
        if model_size not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model size: {model_size}. Choose from {list(self.MODEL_CONFIGS.keys())}")
            
        config_name, checkpoint_name = self.MODEL_CONFIGS[model_size]
        
        # Setup paths
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / checkpoint_name
        config_path = checkpoint_dir / config_name
        
        # Download checkpoint if needed
        if not checkpoint_path.exists():
            if auto_download:
                print(f"Downloading SAM2 {model_size} checkpoint...")
                self._download_file(
                    f"https://dl.fbaipublicfiles.com/segment_anything_2/072824/{checkpoint_name}",
                    checkpoint_path
                )
            else:
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading SAM2 {model_size} model...")
        
        # Resolve to absolute paths and use forward slashes for Windows compatibility with Hydra
        config_path_str = str(config_path.resolve()).replace("\\", "/")
        checkpoint_path_str = str(checkpoint_path.resolve()).replace("\\", "/")
        
        try:
            self.sam2 = build_sam2(config_path_str, checkpoint_path_str, device=self.device)
            self.predictor = SAM2ImagePredictor(self.sam2)
            print(f"SAM2 loaded on {self.device}")
        except Exception as e:
            print(f"Error loading SAM2: {e}")
            print("Troubleshooting: Ensure config file is valid and paths are correct.")
            # If standard build failed, try a more direct approach by checking sys.path or mocking hydra context if we were advanced,
            # but for now let's just fail loudly so user sees the real error (path issue)
            raise e

        
        # Current image state
        self._current_image = None
        self._image_set = False

    def _download_file(self, url: str, output_path: Path):
        """Download file from URL."""
        import urllib.request
        import ssl
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create unverified context to avoid SSL errors often found in some environments
        context = ssl._create_unverified_context()
        
        try:
            print(f"Downloading {url} to {output_path}...")
            with urllib.request.urlopen(url, context=context) as response, open(output_path, 'wb') as out_file:
                out_file.write(response.read())
            print("Download complete.")
        except Exception as e:
            raise RuntimeError(f"Failed to download {url}: {e}")
    
    def set_image(self, image: Union[str, Path, np.ndarray, Image.Image]) -> None:
        """
        Set the current image for segmentation.
        
        Args:
            image: Image path, numpy array, or PIL Image.
        """
        if isinstance(image, (str, Path)):
            image = np.array(Image.open(image).convert("RGB"))
        elif isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        
        self._current_image = image
        self.predictor.set_image(image)
        self._image_set = True
    
    def segment_with_points(
        self,
        positive_points: List[Tuple[int, int]],
        negative_points: Optional[List[Tuple[int, int]]] = None,
        multimask_output: bool = False,
    ) -> np.ndarray:
        """
        Segment using point prompts.
        
        Args:
            positive_points: List of (x, y) points to include.
            negative_points: List of (x, y) points to exclude.
            multimask_output: If True, return multiple mask options.
            
        Returns:
            Binary mask (H, W) or list of masks if multimask_output.
        """
        if not self._image_set:
            raise ValueError("Call set_image() first")
        
        # Prepare points
        points = list(positive_points)
        labels = [1] * len(positive_points)
        
        if negative_points:
            points.extend(negative_points)
            labels.extend([0] * len(negative_points))
        
        point_coords = np.array(points)
        point_labels = np.array(labels)
        
        # Predict
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask_output,
        )
        
        if multimask_output:
            return masks, scores
        else:
            # Return best mask
            return masks[scores.argmax()]
    
    def segment_with_box(
        self,
        box: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """
        Segment using bounding box prompt.
        
        Args:
            box: (x1, y1, x2, y2) bounding box.
            
        Returns:
            Binary mask (H, W).
        """
        if not self._image_set:
            raise ValueError("Call set_image() first")
        
        box_array = np.array([box])
        
        masks, scores, logits = self.predictor.predict(
            box=box_array,
            multimask_output=False,
        )
        
        return masks[0]
    
    def auto_segment_walls(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        min_area_ratio: float = 0.05,
        max_area_ratio: float = 0.6,
        exclude_bottom_ratio: float = 0.15,
        include_ceiling: bool = False,
    ) -> np.ndarray:
        """
        Automatically segment walls using SAM2 + heuristics.
        
        Strategy:
        1. Use SAM2's automatic mask generation
        2. Filter by size (walls are typically 5-60% of image)
        3. Filter by position (exclude floor at bottom)
        4. Merge remaining masks
        
        Args:
            image: Input image.
            min_area_ratio: Minimum mask area as ratio of image area.
            max_area_ratio: Maximum mask area as ratio of image area.
            exclude_bottom_ratio: Exclude masks that are mostly in bottom portion.
            include_ceiling: If True, include ceiling-like regions at top.
            
        Returns:
            Binary mask (H, W) where 1 = wall.
        """
        # Load image
        if isinstance(image, (str, Path)):
            image = np.array(Image.open(image).convert("RGB"))
        elif isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        
        self.set_image(image)
        h, w = image.shape[:2]
        total_area = h * w
        
        # Use automatic mask generation
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        
        mask_generator = SAM2AutomaticMaskGenerator(
            model=self.sam2,
            points_per_side=32,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=1000,
        )
        
        masks = mask_generator.generate(image)
        
        # Filter masks
        wall_masks = []
        
        for mask_data in masks:
            mask = mask_data["segmentation"]
            area = mask.sum()
            area_ratio = area / total_area
            
            # Size filter
            if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                continue
            
            # Position filter - exclude floor-like regions
            mask_y = np.where(mask)[0]
            if len(mask_y) == 0:
                continue
            
            centroid_y = mask_y.mean() / h
            
            # Skip if mostly in bottom portion (likely floor)
            if centroid_y > (1 - exclude_bottom_ratio):
                continue
            
            # Skip if all in bottom 20% (definitely floor)
            if mask_y.min() / h > 0.8:
                continue
            
            # Include ceiling check
            if not include_ceiling and centroid_y < 0.15 and mask_y.max() / h < 0.3:
                continue  # Skip pure ceiling regions
            
            wall_masks.append(mask)
        
        # Merge masks
        if not wall_masks:
            return np.zeros((h, w), dtype=np.uint8)
        
        merged = np.zeros((h, w), dtype=np.uint8)
        for mask in wall_masks:
            merged = np.maximum(merged, mask.astype(np.uint8))
        
        # Post-processing: dilate slightly to fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        merged = cv2.dilate(merged, kernel, iterations=1)
        
        return merged
    
    def refine_mask(
        self,
        current_mask: np.ndarray,
        add_points: Optional[List[Tuple[int, int]]] = None,
        remove_points: Optional[List[Tuple[int, int]]] = None,
    ) -> np.ndarray:
        """
        Refine an existing mask with additional point prompts.
        
        Args:
            current_mask: Current binary mask.
            add_points: Points to add to mask.
            remove_points: Points to remove from mask.
            
        Returns:
            Refined binary mask.
        """
        if not self._image_set:
            raise ValueError("Call set_image() first")
        
        # Use current mask as initial prompt
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            mask_input=current_mask[None, :, :],
            multimask_output=False,
        )
        
        refined = masks[0]
        
        # Add/remove regions based on points
        if add_points:
            for point in add_points:
                add_mask = self.segment_with_points([point])
                refined = np.maximum(refined, add_mask)
        
        if remove_points:
            for point in remove_points:
                remove_mask = self.segment_with_points([point])
                refined = refined * (1 - remove_mask)
        
        return (refined > 0.5).astype(np.uint8)
    
    def process_batch(
        self,
        image_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        include_ceiling: bool = False,
    ) -> Dict[str, str]:
        """
        Process multiple images and save masks.
        
        Args:
            image_paths: List of image paths.
            output_dir: Directory to save masks.
            include_ceiling: Include ceiling in masks.
            
        Returns:
            Dict mapping image paths to mask paths.
        """
        from tqdm import tqdm
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for img_path in tqdm(image_paths, desc="Segmenting"):
            img_path = Path(img_path)
            
            try:
                mask = self.auto_segment_walls(
                    img_path,
                    include_ceiling=include_ceiling,
                )
                
                # Skip if empty mask
                if mask.sum() < 1000:
                    print(f"Skipping {img_path.name}: no wall detected")
                    continue
                
                # Save mask
                mask_path = output_dir / f"{img_path.stem}.png"
                Image.fromarray(mask * 255).save(mask_path)
                
                results[str(img_path)] = str(mask_path)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        return results


# Fallback to existing WallSegmenter if SAM2 not available
class FastSAMFallback:
    """Fallback to existing WallSegmenter if SAM2 is not available."""
    
    def __init__(self):
        # Try to use existing segmentation.py
        try:
            from segmentation import WallSegmenter
            self._segmenter = WallSegmenter()
            self._use_existing = True
            print("Using existing WallSegmenter as fallback")
        except ImportError:
            # Try FastSAM directly
            try:
                from ultralytics import FastSAM
                self.model = FastSAM("FastSAM-x.pt")
                self._use_existing = False
            except ImportError:
                raise ImportError(
                    "No segmentation model available. Install one of:\n"
                    "  - SAM2: pip install git+https://github.com/facebookresearch/segment-anything-2.git\n"
                    "  - Or ensure segmentation.py is available"
                )
    
    def auto_segment_walls(self, image, include_ceiling=False, **kwargs):
        if self._use_existing:
            # Use semantic strategy if ceiling requested, otherwise auto
            strategy = "semantic" if include_ceiling else "auto"
            
            mask = self._segmenter.get_wall_mask(
                image, 
                return_pil=False,
                strategy=strategy,
                include_ceiling=include_ceiling
            )
            
            if isinstance(mask, np.ndarray):
                return (mask > 127).astype(np.uint8)
            return np.array(mask.convert("L")) > 127
        else:
            # FastSAM fallback doesn't support ceiling specifically
            results = self.model(image, retina_masks=True)
            # Return largest mask as wall
            if results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                areas = [m.sum() for m in masks]
                return masks[np.argmax(areas)].astype(np.uint8)
            return np.zeros((512, 512), dtype=np.uint8)


def get_segmenter(prefer_sam2: bool = True, **kwargs):
    """
    Factory function to get best available segmenter.
    
    Args:
        prefer_sam2: If True, prefer SAM2 over FastSAM.
        **kwargs: Arguments to pass to segmenter.
        
    Returns:
        SAM2WallSegmenter or FastSAMFallback.
    """
    if prefer_sam2 and SAM2_AVAILABLE:
        try:
            return SAM2WallSegmenter(**kwargs)
        except FileNotFoundError as e:
            print(f"SAM2 checkpoints not found: {e}")
            print("Falling back to FastSAM...")
    
    return FastSAMFallback()


if __name__ == "__main__":
    # Quick test
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python sam2_segmenter.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    segmenter = get_segmenter(prefer_sam2=True)
    mask = segmenter.auto_segment_walls(image_path)
    
    # Save result
    output_path = Path(image_path).stem + "_mask.png"
    Image.fromarray(mask * 255).save(output_path)
    print(f"Mask saved to: {output_path}")
