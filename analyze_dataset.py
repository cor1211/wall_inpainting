"""
Dataset Analysis Script for Wall Inpainting

Analyzes the wall/ceiling segmentation dataset to:
1. Calculate quality metrics (surface ratio, contamination rate)
2. Identify problematic masks
3. Generate statistics report

Based on ADE20K class definitions:
- wall: class 0
- ceiling: class 5

Usage:
    python analyze_dataset.py --input dataset/wall_dataset --output analysis_report.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2


# ============== ADE20K Class Definitions ==============

# Target classes for wall/ceiling
WALL_CLASS_ID = 0
CEILING_CLASS_ID = 5
TARGET_CLASSES = {WALL_CLASS_ID, CEILING_CLASS_ID}

# Objects that should NOT be in wall/ceiling masks
EXCLUDE_CLASSES = {
    8: "window",
    14: "door",
    18: "painting/picture",
    19: "poster",
    25: "lamp",
    35: "mirror",
    36: "rug/carpet",
    39: "shelf",
    45: "cabinet",
    48: "curtain",
    62: "television",
    66: "plant",
    83: "clock",
    89: "chandelier",
    93: "sconce",
    100: "fan",
    124: "air conditioner",
    # Add more as needed
}


# ============== Data Classes ==============

@dataclass
class MaskQualityMetrics:
    """Quality metrics for a single mask."""
    image_name: str
    surface_ratio: float
    wall_ratio: float
    ceiling_ratio: float
    contamination_ratio: float
    num_connected_components: int
    edge_sharpness: float
    has_valid_surface: bool
    contaminating_classes: List[int]


@dataclass
class DatasetStatistics:
    """Overall dataset statistics."""
    total_samples: int
    valid_samples: int
    invalid_samples: int
    mean_surface_ratio: float
    std_surface_ratio: float
    mean_contamination_ratio: float
    contamination_distribution: Dict[int, int]
    outlier_count: int
    samples_with_issues: List[str]


# ============== Analysis Functions ==============

def load_mask(mask_path: Path) -> np.ndarray:
    """Load mask as numpy array."""
    mask = np.array(Image.open(mask_path))
    return mask


def load_segmentation_map(seg_path: Path) -> Optional[np.ndarray]:
    """
    Load ADE20K segmentation map if available.
    Returns class indices for each pixel.
    """
    if not seg_path.exists():
        return None
    
    seg = np.array(Image.open(seg_path))
    
    # ADE20K stores class_id + 1 (0 means unlabeled)
    # So we subtract 1 to get actual class IDs
    if len(seg.shape) == 3:
        # RGB encoded, convert to class index
        seg = seg[:, :, 0]  # Use R channel
    
    return seg


def calculate_edge_sharpness(mask: np.ndarray) -> float:
    """
    Calculate edge sharpness using gradient magnitude.
    Higher values = sharper edges.
    """
    # Convert to float
    mask_float = mask.astype(np.float32) / 255.0
    
    # Calculate gradients
    grad_x = cv2.Sobel(mask_float, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(mask_float, cv2.CV_32F, 0, 1, ksize=3)
    
    # Gradient magnitude
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    
    # Mean gradient at edges
    edge_mask = gradient > 0.1
    if edge_mask.sum() == 0:
        return 0.0
    
    return float(gradient[edge_mask].mean())


def count_connected_components(mask: np.ndarray) -> int:
    """Count number of connected components in mask."""
    binary = (mask > 127).astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(binary)
    return num_labels - 1  # Subtract background


def analyze_contamination(
    mask: np.ndarray, 
    seg_map: Optional[np.ndarray]
) -> Tuple[float, List[int]]:
    """
    Analyze mask contamination from non-wall/ceiling objects.
    
    Returns:
        contamination_ratio: Fraction of mask pixels belonging to excluded classes
        contaminating_classes: List of class IDs found in mask
    """
    if seg_map is None:
        return 0.0, []
    
    # Get mask region
    binary_mask = mask > 127
    
    if binary_mask.sum() == 0:
        return 0.0, []
    
    # Check which classes are in the mask
    classes_in_mask = np.unique(seg_map[binary_mask])
    
    contaminating = []
    contamination_pixels = 0
    
    for cls_id in classes_in_mask:
        if cls_id in EXCLUDE_CLASSES:
            contaminating.append(int(cls_id))
            contamination_pixels += ((seg_map == cls_id) & binary_mask).sum()
    
    contamination_ratio = contamination_pixels / binary_mask.sum()
    
    return float(contamination_ratio), contaminating


def analyze_single_mask(
    image_name: str,
    mask_path: Path,
    seg_path: Optional[Path] = None,
    min_surface_ratio: float = 0.05,
    max_surface_ratio: float = 0.80,
) -> MaskQualityMetrics:
    """
    Analyze quality metrics for a single mask.
    """
    mask = load_mask(mask_path)
    seg_map = load_segmentation_map(seg_path) if seg_path else None
    
    h, w = mask.shape[:2] if len(mask.shape) >= 2 else (mask.shape[0], 1)
    total_pixels = h * w
    
    # Binary mask
    binary = mask > 127
    mask_pixels = binary.sum()
    
    # Surface ratio
    surface_ratio = mask_pixels / total_pixels
    
    # Wall/ceiling ratio (if segmentation map available)
    wall_ratio = 0.0
    ceiling_ratio = 0.0
    
    if seg_map is not None:
        wall_pixels = ((seg_map == WALL_CLASS_ID) & binary).sum()
        ceiling_pixels = ((seg_map == CEILING_CLASS_ID) & binary).sum()
        
        if mask_pixels > 0:
            wall_ratio = wall_pixels / mask_pixels
            ceiling_ratio = ceiling_pixels / mask_pixels
    
    # Contamination
    contamination_ratio, contaminating_classes = analyze_contamination(mask, seg_map)
    
    # Connected components
    num_components = count_connected_components(mask)
    
    # Edge sharpness
    edge_sharpness = calculate_edge_sharpness(mask)
    
    # Valid surface check
    has_valid_surface = (
        min_surface_ratio <= surface_ratio <= max_surface_ratio and
        contamination_ratio < 0.05  # Less than 5% contamination
    )
    
    return MaskQualityMetrics(
        image_name=image_name,
        surface_ratio=float(surface_ratio),
        wall_ratio=float(wall_ratio),
        ceiling_ratio=float(ceiling_ratio),
        contamination_ratio=float(contamination_ratio),
        num_connected_components=int(num_components),
        edge_sharpness=float(edge_sharpness),
        has_valid_surface=bool(has_valid_surface),  # Convert numpy.bool_ to Python bool
        contaminating_classes=contaminating_classes,
    )


def analyze_dataset(
    dataset_path: Path,
    seg_path: Optional[Path] = None,
    split: str = "train",
) -> Tuple[List[MaskQualityMetrics], DatasetStatistics]:
    """
    Analyze entire dataset split.
    
    Args:
        dataset_path: Path to dataset root
        seg_path: Path to ADE20K segmentation maps (optional)
        split: "train" or "validation"
        
    Returns:
        List of per-sample metrics and overall statistics
    """
    masks_dir = dataset_path / split / "masks"
    
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")
    
    mask_files = list(masks_dir.glob("*.png"))
    
    print(f"Analyzing {len(mask_files)} masks in {split} split...")
    
    metrics_list = []
    contamination_distribution = defaultdict(int)
    samples_with_issues = []
    
    for mask_file in tqdm(mask_files, desc=f"Analyzing {split}"):
        image_name = mask_file.stem
        
        # Try to find corresponding segmentation map
        seg_file = None
        if seg_path:
            seg_file = seg_path / f"{image_name}_seg.png"
            if not seg_file.exists():
                seg_file = seg_path / f"{image_name}.png"
            if not seg_file.exists():
                seg_file = None
        
        metrics = analyze_single_mask(
            image_name=image_name,
            mask_path=mask_file,
            seg_path=seg_file,
        )
        
        metrics_list.append(metrics)
        
        # Track contamination
        for cls_id in metrics.contaminating_classes:
            contamination_distribution[cls_id] += 1
        
        # Track issues
        if not metrics.has_valid_surface:
            samples_with_issues.append(image_name)
    
    # Calculate overall statistics
    surface_ratios = [m.surface_ratio for m in metrics_list]
    contamination_ratios = [m.contamination_ratio for m in metrics_list]
    
    mean_surface = np.mean(surface_ratios)
    std_surface = np.std(surface_ratios)
    
    # Identify outliers (> 3 std from mean)
    outliers = [
        m.image_name for m in metrics_list
        if abs(m.surface_ratio - mean_surface) > 3 * std_surface
    ]
    
    stats = DatasetStatistics(
        total_samples=len(metrics_list),
        valid_samples=sum(1 for m in metrics_list if m.has_valid_surface),
        invalid_samples=sum(1 for m in metrics_list if not m.has_valid_surface),
        mean_surface_ratio=float(mean_surface),
        std_surface_ratio=float(std_surface),
        mean_contamination_ratio=float(np.mean(contamination_ratios)),
        contamination_distribution=dict(contamination_distribution),
        outlier_count=len(outliers),
        samples_with_issues=samples_with_issues[:100],  # Limit to 100
    )
    
    return metrics_list, stats


def save_report(
    metrics_list: List[MaskQualityMetrics],
    stats: DatasetStatistics,
    output_path: Path,
):
    """Save analysis report to JSON."""
    report = {
        "statistics": asdict(stats),
        "sample_metrics": [asdict(m) for m in metrics_list[:1000]],  # Sample
        "class_names": {str(k): v for k, v in EXCLUDE_CLASSES.items()},
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nReport saved to: {output_path}")


def print_summary(stats: DatasetStatistics):
    """Print summary to console."""
    print("\n" + "=" * 60)
    print("DATASET ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total samples:     {stats.total_samples}")
    print(f"Valid samples:     {stats.valid_samples} ({100*stats.valid_samples/stats.total_samples:.1f}%)")
    print(f"Invalid samples:   {stats.invalid_samples} ({100*stats.invalid_samples/stats.total_samples:.1f}%)")
    print(f"Mean surface ratio: {stats.mean_surface_ratio:.3f} ± {stats.std_surface_ratio:.3f}")
    print(f"Mean contamination: {stats.mean_contamination_ratio:.4f}")
    print(f"Outliers:          {stats.outlier_count}")
    
    if stats.contamination_distribution:
        print("\nContamination by class:")
        for cls_id, count in sorted(
            stats.contamination_distribution.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]:
            cls_name = EXCLUDE_CLASSES.get(int(cls_id), f"class_{cls_id}")
            print(f"  {cls_name}: {count} samples")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze wall/ceiling segmentation dataset"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to dataset root"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="analysis_report.json",
        help="Output report path"
    )
    parser.add_argument(
        "--seg-path",
        type=str,
        default=None,
        help="Path to ADE20K segmentation maps (for contamination analysis)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "both"],
        help="Dataset split to analyze"
    )
    
    args = parser.parse_args()
    
    dataset_path = Path(args.input)
    seg_path = Path(args.seg_path) if args.seg_path else None
    
    if args.split == "both":
        splits = ["train", "validation"]
    else:
        splits = [args.split]
    
    all_metrics = []
    
    for split in splits:
        metrics_list, stats = analyze_dataset(
            dataset_path=dataset_path,
            seg_path=seg_path,
            split=split,
        )
        all_metrics.extend(metrics_list)
        print_summary(stats)
    
    # Save combined report
    output_path = Path(args.output)
    combined_stats = DatasetStatistics(
        total_samples=len(all_metrics),
        valid_samples=sum(1 for m in all_metrics if m.has_valid_surface),
        invalid_samples=sum(1 for m in all_metrics if not m.has_valid_surface),
        mean_surface_ratio=float(np.mean([m.surface_ratio for m in all_metrics])),
        std_surface_ratio=float(np.std([m.surface_ratio for m in all_metrics])),
        mean_contamination_ratio=float(np.mean([m.contamination_ratio for m in all_metrics])),
        contamination_distribution={},
        outlier_count=0,
        samples_with_issues=[],
    )
    
    save_report(all_metrics, combined_stats, output_path)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
