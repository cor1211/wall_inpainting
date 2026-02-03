"""
Dataset Validation Script

Validates refined dataset before LoRA training:
1. Automated quality checks (surface ratio, contamination)
2. Statistical consistency verification
3. Train/val data integrity checks

Usage:
    python validate_dataset.py --input dataset/refined_wall_dataset --report validation_report.json
"""

import os
import json
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2


# ============== Validation Thresholds ==============

MIN_SURFACE_RATIO = 0.03
MAX_SURFACE_RATIO = 0.85
MAX_CONTAMINATION = 0.05
MIN_SAMPLES = 100
MAX_CONNECTED_COMPONENTS = 5


@dataclass
class ValidationResult:
    """Result of validation for a single sample."""
    name: str
    is_valid: bool
    issues: List[str]
    surface_ratio: float
    num_components: int


@dataclass
class DatasetValidation:
    """Overall validation result."""
    total_samples: int
    valid_samples: int
    failed_samples: int
    pass_rate: float
    issues_summary: Dict[str, int]
    failed_sample_names: List[str]
    duplicates_found: int
    train_val_overlap: int


# ============== Validation Functions ==============

def calculate_file_hash(path: Path) -> str:
    """Calculate MD5 hash of file for duplicate detection."""
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def validate_single_sample(
    image_path: Path,
    mask_path: Path,
) -> ValidationResult:
    """Validate a single image-mask pair."""
    issues = []
    
    # Check files exist
    if not image_path.exists():
        issues.append("image_missing")
        return ValidationResult(
            name=mask_path.stem,
            is_valid=False,
            issues=issues,
            surface_ratio=0.0,
            num_components=0,
        )
    
    if not mask_path.exists():
        issues.append("mask_missing")
        return ValidationResult(
            name=image_path.stem,
            is_valid=False,
            issues=issues,
            surface_ratio=0.0,
            num_components=0,
        )
    
    # Load files
    try:
        image = np.array(Image.open(image_path))
        mask = np.array(Image.open(mask_path))
    except Exception as e:
        issues.append(f"load_error: {str(e)}")
        return ValidationResult(
            name=mask_path.stem,
            is_valid=False,
            issues=issues,
            surface_ratio=0.0,
            num_components=0,
        )
    
    # Check dimensions match
    img_h, img_w = image.shape[:2]
    mask_h, mask_w = mask.shape[:2] if len(mask.shape) >= 2 else (mask.shape[0], 1)
    
    if (img_h, img_w) != (mask_h, mask_w):
        issues.append(f"size_mismatch: img={img_w}x{img_h}, mask={mask_w}x{mask_h}")
    
    # Calculate surface ratio
    binary_mask = mask > 127
    surface_ratio = binary_mask.sum() / binary_mask.size
    
    if surface_ratio < MIN_SURFACE_RATIO:
        issues.append(f"surface_too_small: {surface_ratio:.3f}")
    elif surface_ratio > MAX_SURFACE_RATIO:
        issues.append(f"surface_too_large: {surface_ratio:.3f}")
    
    # Count connected components
    binary_uint8 = binary_mask.astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(binary_uint8)
    num_components = num_labels - 1  # Exclude background
    
    if num_components > MAX_CONNECTED_COMPONENTS:
        issues.append(f"too_many_components: {num_components}")
    
    if num_components == 0 and surface_ratio > 0:
        issues.append("empty_mask")
    
    return ValidationResult(
        name=mask_path.stem,
        is_valid=len(issues) == 0,
        issues=issues,
        surface_ratio=float(surface_ratio),
        num_components=num_components,
    )


def check_duplicates(images_dir: Path) -> List[Tuple[str, str]]:
    """Find duplicate images based on file hash."""
    hashes = {}
    duplicates = []
    
    for img_path in images_dir.glob("*.*"):
        if img_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            file_hash = calculate_file_hash(img_path)
            
            if file_hash in hashes:
                duplicates.append((hashes[file_hash], img_path.name))
            else:
                hashes[file_hash] = img_path.name
    
    return duplicates


def check_train_val_overlap(
    train_images: Path,
    val_images: Path,
) -> Set[str]:
    """Check for overlap between train and validation sets."""
    train_hashes = {}
    
    for img_path in train_images.glob("*.*"):
        if img_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            file_hash = calculate_file_hash(img_path)
            train_hashes[file_hash] = img_path.name
    
    overlap = set()
    for img_path in val_images.glob("*.*"):
        if img_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            file_hash = calculate_file_hash(img_path)
            if file_hash in train_hashes:
                overlap.add(img_path.name)
    
    return overlap


def validate_split(
    dataset_path: Path,
    split: str,
) -> Tuple[List[ValidationResult], Dict[str, int]]:
    """Validate a dataset split."""
    images_dir = dataset_path / split / "images"
    masks_dir = dataset_path / split / "masks"
    
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")
    
    mask_files = list(masks_dir.glob("*.png"))
    
    results = []
    issues_summary = {}
    
    for mask_path in tqdm(mask_files, desc=f"Validating {split}"):
        # Find corresponding image
        image_name = mask_path.stem
        image_path = images_dir / f"{image_name}.png"
        if not image_path.exists():
            image_path = images_dir / f"{image_name}.jpg"
        
        result = validate_single_sample(image_path, mask_path)
        results.append(result)
        
        for issue in result.issues:
            issue_type = issue.split(":")[0]
            issues_summary[issue_type] = issues_summary.get(issue_type, 0) + 1
    
    return results, issues_summary


def validate_dataset(
    dataset_path: Path,
) -> DatasetValidation:
    """Validate entire dataset."""
    all_results = []
    all_issues = {}
    
    # Validate each split
    for split in ["train", "validation"]:
        split_path = dataset_path / split
        if split_path.exists():
            results, issues = validate_split(dataset_path, split)
            all_results.extend(results)
            for k, v in issues.items():
                all_issues[k] = all_issues.get(k, 0) + v
    
    # Check duplicates
    duplicates = []
    for split in ["train", "validation"]:
        images_dir = dataset_path / split / "images"
        if images_dir.exists():
            duplicates.extend(check_duplicates(images_dir))
    
    # Check train/val overlap
    train_images = dataset_path / "train" / "images"
    val_images = dataset_path / "validation" / "images"
    overlap = set()
    if train_images.exists() and val_images.exists():
        overlap = check_train_val_overlap(train_images, val_images)
    
    valid_count = sum(1 for r in all_results if r.is_valid)
    failed_names = [r.name for r in all_results if not r.is_valid]
    
    return DatasetValidation(
        total_samples=len(all_results),
        valid_samples=valid_count,
        failed_samples=len(all_results) - valid_count,
        pass_rate=valid_count / len(all_results) if all_results else 0,
        issues_summary=all_issues,
        failed_sample_names=failed_names[:100],
        duplicates_found=len(duplicates),
        train_val_overlap=len(overlap),
    )


def print_validation_report(validation: DatasetValidation):
    """Print validation summary."""
    print("\n" + "=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)
    print(f"Total samples:      {validation.total_samples}")
    print(f"Valid samples:      {validation.valid_samples}")
    print(f"Failed samples:     {validation.failed_samples}")
    print(f"Pass rate:          {100 * validation.pass_rate:.1f}%")
    print(f"Duplicates found:   {validation.duplicates_found}")
    print(f"Train/val overlap:  {validation.train_val_overlap}")
    
    if validation.issues_summary:
        print("\nIssues breakdown:")
        for issue, count in sorted(validation.issues_summary.items(), 
                                   key=lambda x: x[1], reverse=True):
            print(f"  {issue}: {count}")
    
    # Status
    print("\n" + "-" * 60)
    if validation.pass_rate >= 0.95 and validation.train_val_overlap == 0:
        print("✅ VALIDATION PASSED - Dataset ready for training")
    elif validation.pass_rate >= 0.90:
        print("⚠️ VALIDATION WARNING - Some issues detected, review recommended")
    else:
        print("❌ VALIDATION FAILED - Dataset needs refinement")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Validate refined dataset for LoRA training"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to refined dataset"
    )
    parser.add_argument(
        "--report", "-r",
        type=str,
        default="validation_report.json",
        help="Output report path"
    )
    
    args = parser.parse_args()
    
    dataset_path = Path(args.input)
    
    print(f"Validating dataset: {dataset_path}")
    
    validation = validate_dataset(dataset_path)
    
    print_validation_report(validation)
    
    # Save report
    report_path = Path(args.report)
    with open(report_path, 'w') as f:
        json.dump(asdict(validation), f, indent=2)
    
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
