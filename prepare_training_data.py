"""
Training Data Preparation Script

Prepares refined dataset for LoRA fine-tuning:
1. Generates metadata with quality scores
2. Creates train/val splits with proper format
3. Optionally generates color-augmented pairs (self-supervised)

Output format compatible with diffusers training scripts.

Usage:
    python prepare_training_data.py --input dataset/refined_wall_dataset --output dataset/training_ready
"""

import os
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
import colorsys


# ============== Configuration ==============

RANDOM_SEED = 42
DEFAULT_CAPTION = "a room with painted walls, interior design, photorealistic"
WALL_CAPTIONS = [
    "interior room with {color} walls",
    "a room with {color} painted walls, interior photography",
    "interior design with {color} wall color, professional photo",
    "{color} walls in modern interior",
]

COLOR_NAMES = [
    "white", "beige", "cream", "gray", "light gray",
    "warm white", "off-white", "ivory", "taupe", "greige",
    "sage green", "dusty blue", "blush pink", "soft yellow",
    "terracotta", "navy blue", "forest green", "burgundy",
]


# ============== Data Classes ==============

@dataclass
class TrainingSample:
    """A single training sample."""
    image_path: str
    mask_path: str
    caption: str
    quality_score: float
    surface_ratio: float


@dataclass
class TrainingDataset:
    """Complete training dataset metadata."""
    train_samples: List[TrainingSample]
    val_samples: List[TrainingSample]
    total_train: int
    total_val: int
    mean_quality: float


# ============== Helper Functions ==============

def calculate_quality_score(mask: np.ndarray) -> float:
    """
    Calculate quality score based on mask properties.
    Score 0-1, higher is better.
    """
    binary = mask > 127
    
    # Surface ratio score (prefer 10-50%)
    ratio = binary.sum() / binary.size
    ratio_score = 1.0 - abs(ratio - 0.3) * 2  # Peak at 30%
    ratio_score = max(0, min(1, ratio_score))
    
    # Connected components score (prefer 1-2 components)
    num_labels, _ = cv2.connectedComponents(binary.astype(np.uint8))
    num_components = num_labels - 1
    comp_score = 1.0 if num_components <= 2 else 1.0 / num_components
    
    # Edge smoothness (prefer smooth edges)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(binary.astype(np.uint8), kernel)
    eroded = cv2.erode(binary.astype(np.uint8), kernel)
    edge = dilated - eroded
    edge_ratio = edge.sum() / binary.sum() if binary.sum() > 0 else 1
    edge_score = 1.0 - min(edge_ratio * 5, 1.0)  # Penalize rough edges
    
    # Combined score
    quality = 0.4 * ratio_score + 0.3 * comp_score + 0.3 * edge_score
    
    return float(quality)


def generate_color_caption(base_color: Optional[str] = None) -> str:
    """Generate caption with color description."""
    color = base_color or random.choice(COLOR_NAMES)
    template = random.choice(WALL_CAPTIONS)
    return template.format(color=color)


def extract_dominant_color(
    image: np.ndarray,
    mask: np.ndarray,
) -> Tuple[int, int, int]:
    """Extract dominant color from masked region."""
    binary_mask = mask > 127
    
    if binary_mask.sum() == 0:
        return (200, 200, 200)  # Default gray
    
    # Get pixels in mask
    masked_pixels = image[binary_mask]
    
    # Use k-means to find dominant color
    pixels = masked_pixels.reshape(-1, 3).astype(np.float32)
    
    # Sample if too many pixels
    if len(pixels) > 10000:
        indices = np.random.choice(len(pixels), 10000, replace=False)
        pixels = pixels[indices]
    
    # K-means with k=1 for dominant color
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(pixels, 1, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
    
    dominant = centers[0].astype(int)
    return tuple(dominant.tolist())


def color_to_name(rgb: Tuple[int, int, int]) -> str:
    """Convert RGB to approximate color name."""
    r, g, b = rgb
    h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
    
    # Neutral colors
    if s < 0.15:
        if l > 0.9:
            return "white"
        elif l > 0.7:
            return "light gray"
        elif l > 0.4:
            return "gray"
        else:
            return "dark gray"
    
    # Warm neutrals
    if s < 0.3 and 0.05 < h < 0.15:
        if l > 0.8:
            return "cream"
        elif l > 0.6:
            return "beige"
        else:
            return "taupe"
    
    # Named colors by hue
    hue_deg = h * 360
    
    if hue_deg < 15 or hue_deg >= 345:
        return "red" if s > 0.5 else "dusty rose"
    elif hue_deg < 45:
        return "orange" if s > 0.5 else "terracotta"
    elif hue_deg < 65:
        return "yellow" if s > 0.5 else "soft yellow"
    elif hue_deg < 150:
        return "green" if s > 0.4 else "sage green"
    elif hue_deg < 200:
        return "teal" if s > 0.4 else "dusty teal"
    elif hue_deg < 260:
        return "blue" if s > 0.4 else "dusty blue"
    elif hue_deg < 290:
        return "purple" if s > 0.4 else "lavender"
    else:
        return "pink" if l > 0.6 else "magenta"


def process_sample(
    image_path: Path,
    mask_path: Path,
    output_images: Path,
    output_masks: Path,
    generate_captions: bool = True,
) -> Optional[TrainingSample]:
    """Process a single sample for training."""
    try:
        # Load image and mask
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        
        # Calculate quality
        quality_score = calculate_quality_score(mask)
        
        # Surface ratio
        binary = mask > 127
        surface_ratio = binary.sum() / binary.size
        
        # Skip if invalid
        if surface_ratio < 0.03 or quality_score < 0.3:
            return None
        
        # Generate caption
        if generate_captions:
            dominant_color = extract_dominant_color(image, mask)
            color_name = color_to_name(dominant_color)
            caption = generate_color_caption(color_name)
        else:
            caption = DEFAULT_CAPTION
        
        # Copy files
        sample_name = image_path.stem
        
        out_image = output_images / f"{sample_name}.png"
        out_mask = output_masks / f"{sample_name}.png"
        
        Image.fromarray(image).save(out_image)
        Image.fromarray(mask).save(out_mask)
        
        return TrainingSample(
            image_path=str(out_image.relative_to(output_images.parent.parent)),
            mask_path=str(out_mask.relative_to(output_masks.parent.parent)),
            caption=caption,
            quality_score=quality_score,
            surface_ratio=surface_ratio,
        )
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def prepare_split(
    input_path: Path,
    output_path: Path,
    split: str,
    generate_captions: bool = True,
) -> List[TrainingSample]:
    """Prepare a single dataset split."""
    input_images = input_path / split / "images"
    input_masks = input_path / split / "masks"
    
    output_images = output_path / split / "images"
    output_masks = output_path / split / "masks"
    
    output_images.mkdir(parents=True, exist_ok=True)
    output_masks.mkdir(parents=True, exist_ok=True)
    
    mask_files = list(input_masks.glob("*.png"))
    
    samples = []
    
    for mask_path in tqdm(mask_files, desc=f"Preparing {split}"):
        image_name = mask_path.stem
        image_path = input_images / f"{image_name}.png"
        if not image_path.exists():
            image_path = input_images / f"{image_name}.jpg"
        
        sample = process_sample(
            image_path=image_path,
            mask_path=mask_path,
            output_images=output_images,
            output_masks=output_masks,
            generate_captions=generate_captions,
        )
        
        if sample:
            samples.append(sample)
    
    return samples


def save_metadata(
    dataset: TrainingDataset,
    output_path: Path,
):
    """Save training metadata files."""
    # Save JSON metadata
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump({
            "total_train": dataset.total_train,
            "total_val": dataset.total_val,
            "mean_quality": dataset.mean_quality,
        }, f, indent=2)
    
    # Save train JSONL (for diffusers)
    train_jsonl = output_path / "train" / "metadata.jsonl"
    with open(train_jsonl, 'w') as f:
        for sample in dataset.train_samples:
            f.write(json.dumps(asdict(sample)) + "\n")
    
    # Save val JSONL
    val_jsonl = output_path / "validation" / "metadata.jsonl"
    with open(val_jsonl, 'w') as f:
        for sample in dataset.val_samples:
            f.write(json.dumps(asdict(sample)) + "\n")
    
    # Save simple captions file (alternative format)
    train_captions = output_path / "train" / "captions.txt"
    with open(train_captions, 'w', encoding='utf-8') as f:
        for sample in dataset.train_samples:
            name = Path(sample.image_path).stem
            f.write(f"{name}|{sample.caption}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data for LoRA fine-tuning"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to refined dataset"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output training data path"
    )
    parser.add_argument(
        "--no-captions",
        action="store_true",
        help="Skip caption generation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Preparing training data...")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    train_samples = []
    val_samples = []
    
    for split in ["train", "validation"]:
        if (input_path / split).exists():
            samples = prepare_split(
                input_path=input_path,
                output_path=output_path,
                split=split,
                generate_captions=not args.no_captions,
            )
            
            if split == "train":
                train_samples = samples
            else:
                val_samples = samples
    
    # Calculate statistics
    all_samples = train_samples + val_samples
    mean_quality = np.mean([s.quality_score for s in all_samples]) if all_samples else 0
    
    dataset = TrainingDataset(
        train_samples=train_samples,
        val_samples=val_samples,
        total_train=len(train_samples),
        total_val=len(val_samples),
        mean_quality=float(mean_quality),
    )
    
    # Save metadata
    save_metadata(dataset, output_path)
    
    print("\n" + "=" * 60)
    print("TRAINING DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"Train samples: {dataset.total_train}")
    print(f"Val samples:   {dataset.total_val}")
    print(f"Mean quality:  {dataset.mean_quality:.3f}")
    print(f"\nOutput saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
