"""
Segmentation Module for AI Interior Wall Re-skinning

Implements multiple strategies for robust wall detection:
1. FastSAM + CLIP filtering (semantic understanding)
2. Heuristic filtering (size/position based)
3. Semantic segmentation with OneFormer (fallback)

Uses Ultralytics FastSAM and OpenAI CLIP for optimal results.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Optional, List, Tuple
import torch
import warnings

# Suppress some warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class WallSegmenter:
    """
    Multi-strategy wall segmentation class.
    
    Strategies:
    - 'clip': FastSAM + CLIP filtering (best accuracy, slower)
    - 'heuristic': FastSAM + size/position filtering (faster, good for typical rooms)
    - 'semantic': OneFormer semantic segmentation (most accurate, requires more memory)
    - 'auto': Try strategies in order until good result found
    """
    
    def __init__(
        self,
        fastsam_model: str = "FastSAM-x.pt",
        device: Optional[str] = None,
        clip_model: str = "ViT-B/32",
    ):
        """
        Initialize the wall segmenter.
        
        Args:
            fastsam_model: Path to FastSAM model weights.
            device: Device to run on (auto-detect if None).
            clip_model: CLIP model variant to use.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fastsam_model_path = fastsam_model
        self.clip_model_name = clip_model
        
        # Lazy loading
        self._fastsam = None
        self._clip_model = None
        self._clip_preprocess = None
        self._oneformer = None
        
    @property
    def fastsam(self):
        """Lazy load FastSAM model."""
        if self._fastsam is None:
            from ultralytics import FastSAM
            self._fastsam = FastSAM(self.fastsam_model_path)
        return self._fastsam
    
    @property
    def clip(self):
        """Lazy load CLIP model."""
        if self._clip_model is None:
            try:
                import clip
                self._clip_model, self._clip_preprocess = clip.load(
                    self.clip_model_name, device=self.device
                )
            except ImportError:
                raise ImportError(
                    "CLIP not installed. Install with: "
                    "pip install git+https://github.com/openai/CLIP.git"
                )
        return self._clip_model, self._clip_preprocess
    
    def segment_everything(
        self,
        image_path: Union[str, Path],
        conf: float = 0.4,
        iou: float = 0.9,
        imgsz: int = 1024,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Segment all objects in the image using FastSAM.
        
        Returns:
            Tuple of (list of binary masks, original image as numpy array)
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load original image
        original_image = np.array(Image.open(image_path).convert("RGB"))
        
        # Run FastSAM
        results = self.fastsam(
            source=str(image_path),
            device=self.device,
            retina_masks=True,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
        )
        
        masks = []
        if results and len(results) > 0 and results[0].masks is not None:
            mask_data = results[0].masks.data.cpu().numpy()
            h, w = original_image.shape[:2]
            
            for mask in mask_data:
                # Resize mask to match original image
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(
                        mask.astype(np.float32), (w, h),
                        interpolation=cv2.INTER_LINEAR
                    )
                masks.append((mask > 0.5).astype(np.uint8))
        
        return masks, original_image
    
    def get_clip_score(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        text_prompts: List[str],
    ) -> float:
        """
        Calculate CLIP similarity score between masked region and text prompts.
        
        Args:
            image: Original image as numpy array (H, W, 3).
            mask: Binary mask (H, W).
            text_prompts: List of text prompts to match against.
            
        Returns:
            Maximum similarity score across prompts.
        """
        import clip
        
        clip_model, clip_preprocess = self.clip
        
        # Get bounding box of mask
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return 0.0
        
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        
        # Add some padding
        h, w = image.shape[:2]
        pad = 10
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        
        # Crop and mask the region
        cropped = image[y1:y2, x1:x2].copy()
        cropped_mask = mask[y1:y2, x1:x2]
        
        # Apply mask (set background to white for better CLIP matching)
        cropped[cropped_mask == 0] = [255, 255, 255]
        
        # Preprocess for CLIP
        pil_image = Image.fromarray(cropped)
        image_input = clip_preprocess(pil_image).unsqueeze(0).to(self.device)
        
        # Encode text prompts
        text_tokens = clip.tokenize(text_prompts).to(self.device)
        
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_tokens)
            
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity
            similarity = (image_features @ text_features.T).squeeze(0)
            
        return similarity.max().item()
    
    def filter_by_clip(
        self,
        masks: List[np.ndarray],
        image: np.ndarray,
        text_prompts: List[str] = None,
        threshold: float = 0.15,
        negative_prompts: List[str] = None,
        min_area_ratio: float = 0.03,
        max_area_ratio: float = 0.5,
        top_k: int = 5,
        negative_weight: float = 0.3,
    ) -> List[np.ndarray]:
        """
        Filter masks by CLIP similarity score combined with size heuristics.
        
        Strategy:
        1. Filter by size (walls are typically 3-50% of image)
        2. Filter by position (exclude floor-like regions at bottom)
        3. Rank by CLIP score (positive - negative * weight)
        4. Select top K candidates above threshold
        
        Args:
            masks: List of binary masks.
            image: Original image.
            text_prompts: Positive prompts (what we want).
            threshold: Minimum CLIP score to consider.
            negative_prompts: Negative prompts to exclude.
            min_area_ratio: Minimum area ratio for pre-filtering.
            max_area_ratio: Maximum area ratio for pre-filtering.
            top_k: Maximum number of top masks to select.
            negative_weight: Weight for negative score subtraction.
            
        Returns:
            Filtered list of masks.
        """
        if text_prompts is None:
            text_prompts = [
                "a wall",
                "interior wall", 
                "room wall",
                "white wall",
                "beige wall",
                "painted wall surface",
            ]
        
        if negative_prompts is None:
            negative_prompts = [
                "a person",
                "human body",
                "furniture",
                "wooden cabinet",
                "shelf",
                "table",
                "ceiling",
                "floor",
                "window frame",
            ]
        
        h, w = image.shape[:2]
        image_area = h * w
        
        # Step 1: Pre-filter by size and position
        candidates = []
        for i, mask in enumerate(masks):
            mask_area = mask.sum()
            area_ratio = mask_area / image_area
            
            # Size filter
            if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                continue
            
            # Position filter - exclude floor (mostly in bottom 20%)
            ys, xs = np.where(mask > 0)
            if len(ys) == 0:
                continue
            
            mean_y = ys.mean() / h
            if mean_y > 0.85:  # Mostly at bottom of image
                continue
            
            candidates.append((i, mask, area_ratio))
        
        if not candidates:
            return []
        
        # Step 2: Calculate CLIP scores for candidates
        scored_masks = []
        for idx, mask, area_ratio in candidates:
            try:
                # Positive score
                pos_score = self.get_clip_score(image, mask, text_prompts)
                
                # Negative score
                neg_score = self.get_clip_score(image, mask, negative_prompts)
                
                # Combined score: positive minus weighted negative
                final_score = pos_score - neg_score * negative_weight
                
                if final_score >= threshold:
                    scored_masks.append((mask, final_score, area_ratio, pos_score, neg_score))
            except Exception as e:
                continue
        
        if not scored_masks:
            # Fallback: if no masks pass threshold, take largest candidates
            candidates.sort(key=lambda x: x[2], reverse=True)  # Sort by area
            return [c[1] for c in candidates[:min(3, len(candidates))]]
        
        # Step 3: Sort by score and select top K
        scored_masks.sort(key=lambda x: x[1], reverse=True)
        selected = [m[0] for m in scored_masks[:top_k]]
        
        return selected
    
    def filter_by_heuristic(
        self,
        masks: List[np.ndarray],
        image: np.ndarray,
        min_area_ratio: float = 0.02,
        max_area_ratio: float = 0.7,
        exclude_bottom_ratio: float = 0.15,
        require_edge_touch: bool = False,
    ) -> List[np.ndarray]:
        """
        Filter masks by heuristic rules (size, position, shape).
        
        Args:
            masks: List of binary masks.
            image: Original image.
            min_area_ratio: Minimum area as ratio of image area.
            max_area_ratio: Maximum area as ratio of image area.
            exclude_bottom_ratio: Exclude masks mostly in bottom X% (floor).
            require_edge_touch: If True, mask must touch image edge.
            
        Returns:
            Filtered list of masks.
        """
        h, w = image.shape[:2]
        image_area = h * w
        
        selected_masks = []
        
        for mask in masks:
            mask_area = mask.sum()
            area_ratio = mask_area / image_area
            
            # Check area bounds
            if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                continue
            
            # Check if mostly in bottom region (likely floor)
            ys, xs = np.where(mask > 0)
            if len(ys) == 0:
                continue
            
            mean_y = ys.mean()
            if mean_y > h * (1 - exclude_bottom_ratio):
                continue  # Skip floor-like regions
            
            # Check edge touch
            if require_edge_touch:
                touches_edge = (
                    mask[0, :].any() or  # Top
                    mask[-1, :].any() or  # Bottom
                    mask[:, 0].any() or  # Left
                    mask[:, -1].any()  # Right
                )
                if not touches_edge:
                    continue
            
            # Calculate convexity (walls tend to have lower convexity due to occlusions)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                contour = max(contours, key=cv2.contourArea)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    convexity = mask_area / hull_area
                    # Walls typically have convexity between 0.3-0.9
                    if convexity < 0.1:  # Too fragmented
                        continue
            
            selected_masks.append(mask)
        
        return selected_masks
    
    def merge_masks(
        self,
        masks: List[np.ndarray],
        dilate_kernel_size: int = 7,
    ) -> np.ndarray:
        """
        Merge multiple masks into a single binary mask.
        
        Args:
            masks: List of binary masks.
            dilate_kernel_size: Kernel size for dilation.
            
        Returns:
            Merged binary mask.
        """
        if not masks:
            return None
        
        # Combine all masks
        h, w = masks[0].shape
        combined = np.zeros((h, w), dtype=np.uint8)
        
        for mask in masks:
            combined = np.maximum(combined, mask)
        
        # Apply morphological operations
        if dilate_kernel_size > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (dilate_kernel_size, dilate_kernel_size)
            )
            # Close small holes first
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
            # Then dilate to cover edges
            combined = cv2.dilate(combined, kernel, iterations=1)
        
        return combined
    
    def get_wall_mask_semantic(
        self,
        image_path: Union[str, Path],
        include_ceiling: bool = False,
        dilate_kernel_size: int = 7,
    ) -> np.ndarray:
        """
        Extract wall mask using Mask2Former semantic segmentation.
        
        Uses Mask2Former pre-trained on ADE20K dataset which includes:
        - Class 0: "wall" 
        - Class 5: "ceiling" (optional)
        
        Args:
            image_path: Path to input image.
            include_ceiling: If True, also include ceiling in mask.
            dilate_kernel_size: Kernel size for mask dilation.
            
        Returns:
            Binary mask where 1 represents wall regions.
        """
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        
        # Load model (lazy load and cache)
        if not hasattr(self, '_mask2former_model') or self._mask2former_model is None:
            print("Loading Mask2Former model...")
            self._mask2former_processor = AutoImageProcessor.from_pretrained(
                "facebook/mask2former-swin-base-ade-semantic"
            )
            self._mask2former_model = Mask2FormerForUniversalSegmentation.from_pretrained(
                "facebook/mask2former-swin-base-ade-semantic"
            )
            self._mask2former_model.to(self.device)
            self._mask2former_model.eval()
            print("Mask2Former model loaded!")
        
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        original_size = image.size  # (width, height)
        
        inputs = self._mask2former_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self._mask2former_model(**inputs)
        
        # Post-process to get semantic segmentation map
        segmentation = self._mask2former_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]  # (height, width)
        )[0]
        
        segmentation = segmentation.cpu().numpy()
        
        # ADE20K class IDs:
        # 0: wall
        # 5: ceiling  
        # Full list: https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/data/object150_info.csv
        
        wall_class_id = 0
        ceiling_class_id = 5
        
        # Create wall mask
        wall_mask = (segmentation == wall_class_id).astype(np.uint8)
        
        if include_ceiling:
            ceiling_mask = (segmentation == ceiling_class_id).astype(np.uint8)
            wall_mask = np.maximum(wall_mask, ceiling_mask)
        
        # Resize to original size if needed
        if wall_mask.shape[:2] != (original_size[1], original_size[0]):
            wall_mask = cv2.resize(
                wall_mask, 
                (original_size[0], original_size[1]),
                interpolation=cv2.INTER_NEAREST
            )
        
        # Apply morphological operations
        if dilate_kernel_size > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (dilate_kernel_size, dilate_kernel_size)
            )
            wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel)
            wall_mask = cv2.dilate(wall_mask, kernel, iterations=1)
        
        return wall_mask
    
    def get_wall_mask(
        self,
        image_path: Union[str, Path],
        strategy: str = "auto",
        dilate_kernel_size: int = 7,
        clip_threshold: float = 0.2,
        return_pil: bool = True,
        **kwargs,
    ) -> Union[Image.Image, np.ndarray]:
        """
        Extract wall mask using the specified strategy.
        
        Args:
            image_path: Path to input image.
            strategy: 'clip', 'heuristic', 'semantic', or 'auto'.
            dilate_kernel_size: Kernel size for mask dilation.
            clip_threshold: CLIP similarity threshold (for 'clip' strategy).
            return_pil: If True, return PIL Image.
            **kwargs: Additional arguments for specific strategies.
            
        Returns:
            Binary mask where white (255) represents wall regions.
        """
        image_path = Path(image_path)
        
        if strategy == "auto":
            # Try strategies in order: semantic is most accurate
            for strat in ["semantic", "clip", "heuristic"]:
                try:
                    mask = self.get_wall_mask(
                        image_path, 
                        strategy=strat,
                        dilate_kernel_size=dilate_kernel_size,
                        clip_threshold=clip_threshold,
                        return_pil=False,
                        **kwargs,
                    )
                    if mask is not None and mask.sum() > 0:
                        # Check if reasonable coverage (2-80% of image)
                        ratio = mask.sum() / mask.size
                        if 0.02 < ratio < 0.8:
                            binary_mask = (mask > 0).astype(np.uint8) * 255
                            if return_pil:
                                return Image.fromarray(binary_mask, mode="L")
                            return binary_mask
                except Exception as e:
                    print(f"Strategy '{strat}' failed: {e}")
                    continue
            
            # Return empty mask if all strategies fail
            img = Image.open(image_path)
            empty_mask = np.zeros((img.size[1], img.size[0]), dtype=np.uint8)
            if return_pil:
                return Image.fromarray(empty_mask, mode="L")
            return empty_mask
        
        # Handle semantic strategy separately (no FastSAM needed)
        if strategy == "semantic":
            include_ceiling = kwargs.get("include_ceiling", True)
            wall_mask = self.get_wall_mask_semantic(
                image_path, 
                include_ceiling=include_ceiling,
                dilate_kernel_size=dilate_kernel_size
            )
            binary_mask = (wall_mask > 0).astype(np.uint8) * 255
            if return_pil:
                return Image.fromarray(binary_mask, mode="L")
            return binary_mask
        
        # Segment everything (for clip and heuristic strategies)
        masks, image = self.segment_everything(image_path, **kwargs)
        
        if not masks:
            h, w = image.shape[:2]
            empty = np.zeros((h, w), dtype=np.uint8)
            if return_pil:
                return Image.fromarray(empty, mode="L")
            return empty
        
        # Apply strategy
        if strategy == "clip":
            selected = self.filter_by_clip(
                masks, image, threshold=clip_threshold
            )
        elif strategy == "heuristic":
            selected = self.filter_by_heuristic(masks, image)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Merge masks
        if not selected:
            h, w = image.shape[:2]
            empty = np.zeros((h, w), dtype=np.uint8)
            if return_pil:
                return Image.fromarray(empty, mode="L")
            return empty
        
        combined = self.merge_masks(selected, dilate_kernel_size)
        binary_mask = (combined > 0).astype(np.uint8) * 255
        
        if return_pil:
            return Image.fromarray(binary_mask, mode="L")
        
        return binary_mask


# Convenience function for backward compatibility
def get_wall_mask(
    image_path: Union[str, Path],
    text_prompt: str = "wall",
    model_path: str = "FastSAM-x.pt",
    confidence: float = 0.4,
    iou_threshold: float = 0.9,
    dilate_kernel_size: int = 7,
    device: Optional[str] = None,
    strategy: str = "auto",
    return_pil: bool = True,
) -> Union[Image.Image, np.ndarray]:
    """
    Extract wall mask from an interior image.
    
    This is a convenience function that creates a WallSegmenter instance
    and calls get_wall_mask().
    
    Args:
        image_path: Path to the input image.
        text_prompt: Text prompt (used for CLIP filtering).
        model_path: Path to FastSAM model weights.
        confidence: Confidence threshold for FastSAM.
        iou_threshold: IoU threshold for NMS.
        dilate_kernel_size: Kernel size for mask dilation.
        device: Device to run on.
        strategy: 'clip', 'heuristic', 'semantic', or 'auto'.
        return_pil: If True, return PIL Image.
        
    Returns:
        Binary mask where white (255) represents wall regions.
    """
    segmenter = WallSegmenter(fastsam_model=model_path, device=device)
    
    return segmenter.get_wall_mask(
        image_path=image_path,
        strategy=strategy,
        dilate_kernel_size=dilate_kernel_size,
        return_pil=return_pil,
        conf=confidence,
        iou=iou_threshold,
    )


def visualize_mask(
    image_path: Union[str, Path],
    mask: Union[Image.Image, np.ndarray],
    alpha: float = 0.5,
    color: tuple = (255, 0, 0),
    save_path: Optional[Union[str, Path]] = None,
) -> Image.Image:
    """
    Visualize the mask overlaid on the original image.
    
    Args:
        image_path: Path to the original image.
        mask: Binary mask (PIL Image or numpy array).
        alpha: Transparency of the overlay (0-1).
        color: RGB color for the mask overlay.
        save_path: Optional path to save the visualization.
        
    Returns:
        PIL Image with mask overlay.
    """
    original = Image.open(image_path).convert("RGB")
    
    if isinstance(mask, Image.Image):
        mask_np = np.array(mask)
    else:
        mask_np = mask
    
    if mask_np.shape[:2] != (original.size[1], original.size[0]):
        mask_np = cv2.resize(
            mask_np, (original.size[0], original.size[1]),
            interpolation=cv2.INTER_NEAREST
        )
    
    original_np = np.array(original)
    mask_bool = mask_np > 127
    
    result = original_np.copy()
    result[mask_bool] = (
        original_np[mask_bool] * (1 - alpha) + np.array(color) * alpha
    ).astype(np.uint8)
    
    result_image = Image.fromarray(result)
    
    if save_path:
        result_image.save(save_path)
        print(f"Visualization saved to: {save_path}")
    
    return result_image


def visualize_all_masks(
    image_path: Union[str, Path],
    masks: List[np.ndarray],
    save_path: Optional[Union[str, Path]] = None,
) -> Image.Image:
    """
    Visualize all masks with different colors for debugging.
    
    Args:
        image_path: Path to original image.
        masks: List of binary masks.
        save_path: Optional path to save visualization.
        
    Returns:
        PIL Image with colored masks overlay.
    """
    original = np.array(Image.open(image_path).convert("RGB"))
    result = original.copy()
    
    # Generate random colors
    np.random.seed(42)
    colors = np.random.randint(0, 255, (len(masks), 3))
    
    for mask, color in zip(masks, colors):
        mask_bool = mask > 0
        result[mask_bool] = (
            original[mask_bool] * 0.5 + color * 0.5
        ).astype(np.uint8)
    
    result_image = Image.fromarray(result)
    
    if save_path:
        result_image.save(save_path)
        print(f"Visualization saved to: {save_path}")
    
    return result_image


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract wall mask from interior image"
    )
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument(
        "--strategy", type=str, default="auto",
        choices=["auto", "clip", "heuristic", "semantic"],
        help="Segmentation strategy"
    )
    parser.add_argument(
        "--model", type=str, default="FastSAM-x.pt",
        help="Path to FastSAM model"
    )
    parser.add_argument(
        "--dilate", type=int, default=7,
        help="Dilation kernel size"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.2,
        help="CLIP similarity threshold"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for mask"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Save visualization overlay"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (cuda/cpu)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Visualize all masks for debugging"
    )
    
    args = parser.parse_args()
    
    print(f"Processing: {args.image_path}")
    print(f"Strategy: {args.strategy}")
    
    # Create segmenter
    segmenter = WallSegmenter(
        fastsam_model=args.model,
        device=args.device
    )
    
    # Debug mode: show all masks
    if args.debug:
        masks, image = segmenter.segment_everything(args.image_path)
        print(f"Found {len(masks)} masks")
        
        debug_path = Path(args.image_path).parent / f"{Path(args.image_path).stem}_all_masks.png"
        visualize_all_masks(args.image_path, masks, save_path=debug_path)
    
    # Get wall mask
    mask = segmenter.get_wall_mask(
        image_path=args.image_path,
        strategy=args.strategy,
        dilate_kernel_size=args.dilate,
        clip_threshold=args.threshold,
    )
    
    # Determine output path
    image_path = Path(args.image_path)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = image_path.parent / f"{image_path.stem}_mask.png"
    
    # Save mask
    mask.save(output_path)
    print(f"Mask saved to: {output_path}")
    
    # Save visualization if requested
    if args.visualize:
        vis_path = image_path.parent / f"{image_path.stem}_visualization.png"
        visualize_mask(args.image_path, mask, save_path=vis_path)
