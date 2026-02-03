"""
Mask Refinement Script for Wall Inpainting Dataset

Refines wall/ceiling masks by:
1. Class-based filtering using ADE20K segmentation maps
2. Morphological post-processing (hole filling, edge smoothing)
3. Connected component analysis

Based on:
- ADE20K class definitions (wall=0, ceiling=5)
- SegRefiner methodology (discrete diffusion for edge refinement)

Usage:
    python refine_masks.py --input dataset/wall_dataset --output dataset/refined_wall_dataset
    python refine_masks.py --input dataset/wall_dataset --output dataset/refined_wall_dataset --seg-path path/to/ade20k_seg
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, asdict
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2


# ============== Configuration ==============

# ADE20K class IDs (0-indexed, after subtracting 1 from stored values)
WALL_CLASS_ID = 0
CEILING_CLASS_ID = 5
TARGET_CLASSES = {WALL_CLASS_ID, CEILING_CLASS_ID}

# Objects to EXCLUDE from surface masks
EXCLUDE_CLASS_IDS = {
    8, 14, 18, 19, 25, 35, 36, 39, 45, 48, 
    62, 66, 83, 89, 93, 100, 124
}

# Morphological parameters
MORPH_KERNEL_SIZE = 5
CLOSE_ITERATIONS = 2
OPEN_ITERATIONS = 1
EDGE_SMOOTH_KERNEL = 3

# Quality thresholds
MIN_SURFACE_RATIO = 0.03
MAX_SURFACE_RATIO = 0.85
MIN_COMPONENT_SIZE = 500  # pixels


# ============== Refinement Functions ==============

def load_image(path: Path) -> np.ndarray:
    """Load image as numpy array."""
    return np.array(Image.open(path))


def save_mask(mask: np.ndarray, path: Path):
    """Save mask as PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8)).save(path)


def refine_by_class_filter(
    mask: np.ndarray,
    seg_map: np.ndarray,
    include_wall: bool = True,
    include_ceiling: bool = True,
) -> np.ndarray:
    """
    Filter mask to include only wall/ceiling pixels.
    
    Args:
        mask: Original binary mask (H, W)
        seg_map: ADE20K segmentation map with class indices
        include_wall: Include wall class
        include_ceiling: Include ceiling class
        
    Returns:
        Refined mask with only valid surface pixels
    """
    binary_mask = mask > 127
    
    # Create valid surface mask
    valid_surface = np.zeros_like(binary_mask)
    
    if include_wall:
        valid_surface |= (seg_map == WALL_CLASS_ID)
    if include_ceiling:
        valid_surface |= (seg_map == CEILING_CLASS_ID)
    
    # Apply filter
    refined = binary_mask & valid_surface
    
    return (refined * 255).astype(np.uint8)


def remove_excluded_objects(
    mask: np.ndarray,
    seg_map: np.ndarray,
) -> np.ndarray:
    """
    Remove pixels belonging to excluded objects from mask.
    """
    binary_mask = mask > 127
    
    excluded = np.zeros_like(binary_mask)
    for cls_id in EXCLUDE_CLASS_IDS:
        excluded |= (seg_map == cls_id)
    
    refined = binary_mask & ~excluded
    
    return (refined * 255).astype(np.uint8)


def morphological_cleanup(
    mask: np.ndarray,
    kernel_size: int = MORPH_KERNEL_SIZE,
    close_iter: int = CLOSE_ITERATIONS,
    open_iter: int = OPEN_ITERATIONS,
) -> np.ndarray:
    """
    Apply morphological operations to clean up mask.
    
    Operations:
    1. Close: Fill small holes
    2. Open: Remove small noise
    3. Gaussian blur + threshold: Smooth edges
    """
    binary = (mask > 127).astype(np.uint8)
    
    # Structuring element
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, 
        (kernel_size, kernel_size)
    )
    
    # Close operation (fill holes)
    closed = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, kernel, iterations=close_iter
    )
    
    # Open operation (remove noise)
    opened = cv2.morphologyEx(
        closed, cv2.MORPH_OPEN, kernel, iterations=open_iter
    )
    
    # Edge smoothing
    if EDGE_SMOOTH_KERNEL > 0:
        smoothed = cv2.GaussianBlur(
            opened.astype(np.float32), 
            (EDGE_SMOOTH_KERNEL, EDGE_SMOOTH_KERNEL), 
            0
        )
        opened = (smoothed > 0.5).astype(np.uint8)
    
    return opened * 255


def filter_small_components(
    mask: np.ndarray,
    min_size: int = MIN_COMPONENT_SIZE,
) -> np.ndarray:
    """
    Remove connected components smaller than min_size.
    """
    binary = (mask > 127).astype(np.uint8)
    
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    
    # Filter by size
    filtered = np.zeros_like(binary)
    for label in range(1, num_labels):  # Skip background (0)
        size = stats[label, cv2.CC_STAT_AREA]
        if size >= min_size:
            filtered[labels == label] = 1
    
    return filtered * 255


def refine_edges_bilateral(
    mask: np.ndarray,
    original_image: np.ndarray,
    d: int = 5,
    sigma_color: float = 50,
    sigma_space: float = 50,
) -> np.ndarray:
    """
    Refine mask edges using bilateral filtering guided by original image.
    This helps align mask edges with actual image edges.
    """
    # Convert mask to float
    mask_float = mask.astype(np.float32) / 255.0
    
    # Create soft mask using bilateral filter
    if len(original_image.shape) == 3:
        guide = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    else:
        guide = original_image
    
    # Edge detection on original
    edges = cv2.Canny(guide, 50, 150)
    edge_dilated = cv2.dilate(edges, None, iterations=2)
    
    # Apply bilateral filter to mask
    filtered = cv2.bilateralFilter(
        mask_float, d, sigma_color / 255.0, sigma_space
    )
    
    # Threshold back to binary
    refined = (filtered > 0.5).astype(np.uint8) * 255
    
    return refined


def calculate_quality_score(
    original_mask: np.ndarray,
    refined_mask: np.ndarray,
    seg_map: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Calculate quality metrics for refined mask.
    """
    orig_binary = original_mask > 127
    ref_binary = refined_mask > 127
    
    total_pixels = orig_binary.size
    orig_ratio = orig_binary.sum() / total_pixels
    ref_ratio = ref_binary.sum() / total_pixels
    
    # IoU between original and refined
    intersection = (orig_binary & ref_binary).sum()
    union = (orig_binary | ref_binary).sum()
    iou = intersection / union if union > 0 else 0.0
    
    # Reduction ratio
    reduction = 1 - (ref_binary.sum() / orig_binary.sum()) if orig_binary.sum() > 0 else 0
    
    # Contamination check
    contamination = 0.0
    if seg_map is not None:
        for cls_id in EXCLUDE_CLASS_IDS:
            contamination += ((seg_map == cls_id) & ref_binary).sum()
        contamination = contamination / ref_binary.sum() if ref_binary.sum() > 0 else 0
    
    return {
        "original_ratio": float(orig_ratio),
        "refined_ratio": float(ref_ratio),
        "iou": float(iou),
        "reduction": float(reduction),
        "contamination": float(contamination),
    }


def refine_single_mask(
    mask_path: Path,
    image_path: Path,
    seg_path: Optional[Path] = None,
    use_class_filter: bool = True,
    use_morphological: bool = True,
    use_edge_refinement: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Refine a single mask using all available methods.
    
    Args:
        mask_path: Path to original mask
        image_path: Path to original image
        seg_path: Path to ADE20K segmentation map (optional)
        use_class_filter: Apply class-based filtering
        use_morphological: Apply morphological cleanup
        use_edge_refinement: Apply edge refinement
        
    Returns:
        Refined mask and quality metrics
    """
    mask = load_image(mask_path)
    original_mask = mask.copy()
    
    # Load segmentation map if available
    seg_map = None
    if seg_path and seg_path.exists():
        seg_map = load_image(seg_path)
        # ADE20K uses R channel for class index (for RGB encoded)
        if len(seg_map.shape) == 3:
            seg_map = seg_map[:, :, 0]
        # Subtract 1 to get 0-indexed classes (0 in stored = unlabeled)
        seg_map = seg_map.astype(np.int32) - 1
    
    # Load original image
    original_image = load_image(image_path)
    
    # Apply refinement pipeline
    refined = mask
    
    # Step 1: Class-based filtering (if segmentation available)
    if use_class_filter and seg_map is not None:
        refined = refine_by_class_filter(refined, seg_map)
        refined = remove_excluded_objects(refined, seg_map)
    
    # Step 2: Morphological cleanup
    if use_morphological:
        refined = morphological_cleanup(refined)
    
    # Step 3: Remove small components
    refined = filter_small_components(refined)
    
    # Step 4: Edge refinement (optional, slower)
    if use_edge_refinement and refined.sum() > 0:
        refined = refine_edges_bilateral(refined, original_image)
    
    # Calculate quality metrics
    quality = calculate_quality_score(original_mask, refined, seg_map)
    
    # Check validity
    surface_ratio = refined.sum() / (255 * refined.size)
    quality["is_valid"] = MIN_SURFACE_RATIO <= surface_ratio <= MAX_SURFACE_RATIO
    quality["surface_ratio"] = float(surface_ratio)
    
    return refined, quality


def process_dataset(
    input_path: Path,
    output_path: Path,
    seg_path: Optional[Path] = None,
    split: str = "train",
    use_class_filter: bool = True,
    use_edge_refinement: bool = False,  # Slower, optional
) -> Dict[str, Any]:
    """
    Process entire dataset split.
    """
    input_masks = input_path / split / "masks"
    input_images = input_path / split / "images"
    output_masks = output_path / split / "masks"
    output_images = output_path / split / "images"
    
    if not input_masks.exists():
        raise FileNotFoundError(f"Input masks not found: {input_masks}")
    
    # Create output directories
    output_masks.mkdir(parents=True, exist_ok=True)
    output_images.mkdir(parents=True, exist_ok=True)
    
    mask_files = list(input_masks.glob("*.png"))
    
    print(f"Processing {len(mask_files)} masks in {split}...")
    
    results = {
        "total": len(mask_files),
        "valid": 0,
        "invalid": 0,
        "samples": [],
    }
    
    for mask_file in tqdm(mask_files, desc=f"Refining {split}"):
        image_name = mask_file.stem
        
        # Find corresponding files
        image_path = input_images / f"{image_name}.png"
        if not image_path.exists():
            image_path = input_images / f"{image_name}.jpg"
        
        seg_file = None
        if seg_path:
            seg_file = seg_path / f"{image_name}_seg.png"
            if not seg_file.exists():
                seg_file = seg_path / f"{image_name}.png"
            if not seg_file.exists():
                seg_file = None
        
        # Refine mask
        try:
            refined_mask, quality = refine_single_mask(
                mask_path=mask_file,
                image_path=image_path,
                seg_path=seg_file,
                use_class_filter=use_class_filter and seg_file is not None,
                use_edge_refinement=use_edge_refinement,
            )
            
            # Save refined mask
            save_mask(refined_mask, output_masks / f"{image_name}.png")
            
            # Copy original image
            if image_path.exists():
                shutil.copy2(image_path, output_images / image_path.name)
            
            if quality["is_valid"]:
                results["valid"] += 1
            else:
                results["invalid"] += 1
            
            results["samples"].append({
                "name": image_name,
                **quality,
            })
            
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            results["invalid"] += 1
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Refine wall/ceiling masks for LoRA training"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input dataset path"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output refined dataset path"
    )
    parser.add_argument(
        "--seg-path",
        type=str,
        default=None,
        help="Path to ADE20K segmentation maps"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="both",
        choices=["train", "validation", "both"],
        help="Dataset split to process"
    )
    parser.add_argument(
        "--edge-refine",
        action="store_true",
        help="Enable edge refinement (slower)"
    )
    parser.add_argument(
        "--no-class-filter",
        action="store_true",
        help="Disable class-based filtering"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    seg_path = Path(args.seg_path) if args.seg_path else None
    
    if args.split == "both":
        splits = ["train", "validation"]
    else:
        splits = [args.split]
    
    all_results = {}
    
    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing {split} split")
        print("=" * 60)
        
        results = process_dataset(
            input_path=input_path,
            output_path=output_path,
            seg_path=seg_path,
            split=split,
            use_class_filter=not args.no_class_filter,
            use_edge_refinement=args.edge_refine,
        )
        
        all_results[split] = {
            "total": results["total"],
            "valid": results["valid"],
            "invalid": results["invalid"],
        }
        
        print(f"\n{split} Results:")
        print(f"  Total: {results['total']}")
        print(f"  Valid: {results['valid']} ({100*results['valid']/results['total']:.1f}%)")
        print(f"  Invalid: {results['invalid']}")
    
    # Save processing report
    report_path = output_path / "refinement_report.json"
    with open(report_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Copy/create dataset info
    dataset_info = {
        "source": str(input_path),
        "refinement_applied": True,
        "class_filter": not args.no_class_filter,
        "edge_refinement": args.edge_refine,
        "splits": all_results,
    }
    
    info_path = output_path / "dataset_info.json"
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Refinement complete!")
    print(f"Output saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
