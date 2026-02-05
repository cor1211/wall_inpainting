"""
Dataset Preparation Pipeline V2.

Complete pipeline for creating wall inpainting training dataset:
1. Segment walls using SAM2
2. Generate multiple color variants
3. Create color-neutral captions
4. Export in training-ready format

Usage:
    python prepare_dataset_v2.py \
        --input raw_images/ \
        --output dataset/training_v2/ \
        --colors-per-image 5
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


# ============== Configuration ==============

RANDOM_SEED = 42

# Color-neutral captions (NO color mentions!)
NEUTRAL_CAPTIONS = [
    "interior room with painted walls, professional photography",
    "modern interior design, high quality photo",
    "residential space with wall finish, architectural photography",
    "room interior with smooth wall surface",
    "indoor space with painted walls, natural lighting",
    "interior photograph of a room with wall surfaces",
    "home interior with wall treatment, professional shot",
    "architectural interior with wall finish",
]


# ============== Data Classes ==============

@dataclass
class TrainingSample:
    """A single training sample."""
    image_path: str
    mask_path: str
    caption: str
    color_name: str
    color_rgb: Tuple[int, int, int]
    quality_score: float
    surface_ratio: float
    source_image: str  # Original source image name


@dataclass  
class DatasetStats:
    """Dataset statistics."""
    total_source_images: int
    total_train_samples: int
    total_val_samples: int
    colors_distribution: Dict[str, int]
    mean_quality_score: float
    mean_surface_ratio: float


# ============== Helper Functions ==============

def calculate_quality_score(mask: np.ndarray, image: np.ndarray) -> float:
    """
    Calculate quality score based on mask and image properties.
    
    Factors:
    - Surface ratio (prefer 10-50%)
    - Connected components (prefer 1-2)
    - Edge smoothness
    - Lighting variation (good = more variation = more realistic)
    """
    binary = mask > 127
    
    # Surface ratio score
    ratio = binary.sum() / binary.size
    ratio_score = 1.0 - abs(ratio - 0.3) * 2
    ratio_score = max(0, min(1, ratio_score))
    
    # Connected components score
    num_labels, _ = cv2.connectedComponents(binary.astype(np.uint8))
    num_components = num_labels - 1
    comp_score = 1.0 if num_components <= 2 else 0.5 / num_components
    
    # Edge smoothness
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(binary.astype(np.uint8), kernel)
    eroded = cv2.erode(binary.astype(np.uint8), kernel)
    edge = dilated - eroded
    edge_ratio = edge.sum() / binary.sum() if binary.sum() > 0 else 1
    edge_score = 1.0 - min(edge_ratio * 5, 1.0)
    
    # Lighting variation in wall region (good = realistic shadows)
    if binary.sum() > 0:
        wall_pixels = image[binary]
        if len(wall_pixels.shape) == 2:
            wall_pixels = wall_pixels.reshape(-1, 1)
        elif len(wall_pixels.shape) == 1:
            wall_pixels = wall_pixels.reshape(-1, 1)
        
        # Convert to grayscale for luminance
        if wall_pixels.shape[-1] == 3:
            gray = 0.299 * wall_pixels[:, 0] + 0.587 * wall_pixels[:, 1] + 0.114 * wall_pixels[:, 2]
        else:
            gray = wall_pixels.mean(axis=-1) if len(wall_pixels.shape) > 1 else wall_pixels
        
        std = np.std(gray)
        # Some variation is good (realistic), too much is bad (noisy)
        lighting_score = min(std / 30, 1.0) if std < 60 else max(0, 1.0 - (std - 60) / 60)
    else:
        lighting_score = 0.5
    
    # Combined score
    quality = 0.3 * ratio_score + 0.2 * comp_score + 0.25 * edge_score + 0.25 * lighting_score
    
    return float(quality)


def get_neutral_caption() -> str:
    """Get a random color-neutral caption."""
    return random.choice(NEUTRAL_CAPTIONS)


# ============== Main Pipeline ==============

class DatasetPipeline:
    """
    Complete dataset preparation pipeline.
    
    Steps:
    1. Load source images and masks
    2. Apply color augmentation
    3. Calculate quality scores
    4. Generate neutral captions
    5. Split into train/val
    6. Export in training format
    """
    
    def __init__(
        self,
        colors_per_image: int = 5,
        val_ratio: float = 0.1,
        min_quality: float = 0.3,
        min_surface_ratio: float = 0.05,
        seed: int = RANDOM_SEED,
    ):
        self.colors_per_image = colors_per_image
        self.val_ratio = val_ratio
        self.min_quality = min_quality
        self.min_surface_ratio = min_surface_ratio
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize augmentor
        from color_augmentor import ColorAugmentor
        self.augmentor = ColorAugmentor(seed=seed)
    
    def process_source_with_mask(
        self,
        image_path: Path,
        mask_path: Path,
    ) -> List[TrainingSample]:
        """
        Process a source image with existing mask.
        
        Args:
            image_path: Path to source image.
            mask_path: Path to mask.
            
        Returns:
            List of training samples (one per color variant).
        """
        # Load image and mask
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        
        # Check mask validity
        binary = mask > 127
        surface_ratio = binary.sum() / binary.size
        
        if surface_ratio < self.min_surface_ratio:
            return []
        
        # Calculate quality
        quality_score = calculate_quality_score(mask, image)
        
        if quality_score < self.min_quality:
            return []
        
        # Generate color variants
        variants = self.augmentor.generate_color_variants(
            image, mask,
            num_colors=self.colors_per_image,
            include_original=True,
            preserve_lighting=True,
        )
        
        # Create samples
        samples = []
        
        for variant in variants:
            sample = TrainingSample(
                image_path="",  # Will be set when saving
                mask_path="",
                caption=get_neutral_caption(),
                color_name=variant["color_name"],
                color_rgb=variant["color_rgb"],
                quality_score=quality_score,
                surface_ratio=surface_ratio,
                source_image=image_path.name,
            )
            sample._image_data = variant["image"]  # Temporary storage
            sample._mask_data = mask
            samples.append(sample)
        
        return samples
    
    def process_source_without_mask(
        self,
        image_path: Path,
        segmenter,
        include_ceiling: bool = False,
    ) -> List[TrainingSample]:
        """
        Process a source image without mask (will segment first).
        
        Args:
            image_path: Path to source image.
            segmenter: SAM2 or fallback segmenter.
            include_ceiling: Include ceiling in mask.
            
        Returns:
            List of training samples.
        """
        # Segment walls
        mask = segmenter.auto_segment_walls(
            image_path,
            include_ceiling=include_ceiling,
        )
        
        # Check if valid mask was generated
        if mask.sum() < 1000:
            return []
        
        # Convert to proper format
        mask_pil = Image.fromarray(mask * 255).convert("L")
        
        # Create temporary mask path (in memory)
        temp_mask_path = Path("/tmp") / f"{image_path.stem}_mask.png"
        mask_pil.save(temp_mask_path)
        
        # Process with mask
        samples = self.process_source_with_mask(image_path, temp_mask_path)
        
        # Store actual mask data
        for sample in samples:
            sample._mask_data = mask * 255
        
        return samples
    
    def run(
        self,
        input_dir: Path,
        output_dir: Path,
        use_existing_masks: bool = True,
        segment_new: bool = False,
        include_ceiling: bool = False,
        proposal_only: bool = False,
    ) -> DatasetStats:
        """
        Run the complete pipeline with incremental saving.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # Setup directories
        if proposal_only:
            # Proposal mode: just save masks and preview
            proposal_dir = output_dir / "proposals"
            proposal_masks = proposal_dir / "masks"
            proposal_images = proposal_dir / "images" # Copy images for reviewer convenience
            proposal_masks.mkdir(parents=True, exist_ok=True)
            proposal_images.mkdir(parents=True, exist_ok=True)
        else:
            # Full dataset mode
            train_images = output_dir / "train" / "images"
            train_masks = output_dir / "train" / "masks"
            val_images = output_dir / "validation" / "images"
            val_masks = output_dir / "validation" / "masks"
            
            for d in [train_images, train_masks, val_images, val_masks]:
                d.mkdir(parents=True, exist_ok=True)
                
            # Metadata files
            train_jsonl = open(output_dir / "train" / "metadata.jsonl", "w")
            val_jsonl = open(output_dir / "validation" / "metadata.jsonl", "w")
        
        # Find source images
        image_dir = input_dir / "images" if (input_dir / "images").exists() else input_dir
        mask_dir = input_dir / "masks" if (input_dir / "masks").exists() else None
        
        image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
        print(f"Found {len(image_files)} source images")
        
        # Initialize segmenter
        # Use FastSAM/WallSegmenter as default to avoid heavy SAM2 download
        from sam2_segmenter import get_segmenter
        segmenter = get_segmenter(prefer_sam2=False)
        
        # Statistics tracking
        stats = DatasetStats(
            total_source_images=len(image_files),
            total_train_samples=0,
            total_val_samples=0,
            colors_distribution={},
            mean_quality_score=0.0,
            mean_surface_ratio=0.0,
        )
        
        total_quality = 0.0
        total_surface = 0.0
        
        # Process and save incrementally
        for img_path in tqdm(image_files, desc="Processing"):
            
            # PROPOSAL MODE: Just generate masks
            if proposal_only:
                if not segmenter:
                    continue
                    
                mask = segmenter.auto_segment_walls(
                    img_path, 
                    include_ceiling=include_ceiling
                )
                
                # Filter bad masks
                if mask.sum() < 1000:
                    continue
                
                # Save to proposals
                base_name = img_path.stem
                Image.fromarray(mask * 255).save(proposal_masks / f"{base_name}.png")
                
                # Copy source image for convenience
                # Use shutil copy for speed
                import shutil
                shutil.copy2(img_path, proposal_images / img_path.name)
                
                stats.total_source_images += 1
                continue

            # FULL DATASET MODE
            samples = []
            
            # Check for existing mask
            if mask_dir and use_existing_masks:
                mask_path = mask_dir / f"{img_path.stem}.png"
                if mask_path.exists():
                    samples = self.process_source_with_mask(img_path, mask_path)
            
            # Segment if no existing mask and enabled
            if not samples and segment_new and segmenter:
                samples = self.process_source_without_mask(
                    img_path, 
                    segmenter,
                    include_ceiling=include_ceiling
                )
            
            if not samples:
                continue
                
            # Assign to train or val (randomly per source image)
            is_val = random.random() < self.val_ratio
            split = "validation" if is_val else "train"
            
            img_out_dir = val_images if is_val else train_images
            mask_out_dir = val_masks if is_val else train_masks
            jsonl_file = val_jsonl if is_val else train_jsonl
            
            if is_val:
                stats.total_val_samples += len(samples)
            else:
                stats.total_train_samples += len(samples)
            
            # Save samples
            for sample in samples:
                # Stats update
                stats.colors_distribution[sample.color_name] = stats.colors_distribution.get(sample.color_name, 0) + 1
                total_quality += sample.quality_score
                total_surface += sample.surface_ratio
                
                # Generate filename
                suffix = "" if sample.color_name == "original" else f"_{sample.color_name}"
                img_name = f"{Path(sample.source_image).stem}{suffix}.png"
                mask_name = f"{Path(sample.source_image).stem}.png"
                
                # Save image
                img_path = img_out_dir / img_name
                Image.fromarray(sample._image_data).save(img_path)
                
                # Save mask (check exist to avoid rewrite)
                mask_path = mask_out_dir / mask_name
                if not mask_path.exists():
                    Image.fromarray(sample._mask_data).save(mask_path)
                
                # Set paths for metadata
                sample.image_path = f"{split}/images/{img_name}"
                sample.mask_path = f"{split}/masks/{mask_name}"
                
                # Write metadata
                data = asdict(sample)
                data["color_rgb"] = list(data["color_rgb"])
                # Note: Dynamic attributes start with _ are not in asdict result
                jsonl_file.write(json.dumps(data) + "\n")
                
                # Free memory
                delattr(sample, "_image_data")
                delattr(sample, "_mask_data")
            
            # Flush periodically
            jsonl_file.flush()
        
        if proposal_only:
            print(f"Generated proposals in {output_dir}/proposals")
            return stats

        # Finalize stats
        total_samples = stats.total_train_samples + stats.total_val_samples
        if total_samples > 0:
            stats.mean_quality_score = total_quality / total_samples
            stats.mean_surface_ratio = total_surface / total_samples
        
        # Close files
        train_jsonl.close()
        val_jsonl.close()
        
        # Save stats
        with open(output_dir / "dataset_stats.json", "w") as f:
            json.dump(asdict(stats), f, indent=2)
            
        self._print_summary(stats)
        return stats
    

    
    def _print_summary(self, stats: DatasetStats) -> None:
        """Print dataset summary."""
        print("\n" + "=" * 60)
        print("DATASET PREPARATION COMPLETE")
        print("=" * 60)
        print(f"Source images:    {stats.total_source_images}")
        print(f"Train samples:    {stats.total_train_samples}")
        print(f"Val samples:      {stats.total_val_samples}")
        print(f"Mean quality:     {stats.mean_quality_score:.3f}")
        print(f"Mean surface:     {stats.mean_surface_ratio:.3f}")
        print("\nColor distribution:")
        for color, count in sorted(stats.colors_distribution.items(), key=lambda x: -x[1]):
            print(f"  {color:20s}: {count:4d}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare wall inpainting training dataset with color augmentation"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input directory with images and optional masks/"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for training data"
    )
    parser.add_argument(
        "--colors-per-image",
        type=int,
        default=5,
        help="Number of color variants per image"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio"
    )
    parser.add_argument(
        "--segment",
        action="store_true",
        help="Segment images without masks using SAM2"
    )
    parser.add_argument(
        "--include-ceiling",
        action="store_true",
        help="Include ceiling in segmentation mask (requires semantic model)"
    )
    parser.add_argument(
        "--proposal-only",
        action="store_true",
        help="Only generate masks for review, do not augment yet"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Proposal only implies segmentation
    if args.proposal_only and not args.segment:
        print("Notice: --proposal-only implies --segment. Enabling segmentation.")
        args.segment = True
    
    pipeline = DatasetPipeline(
        colors_per_image=args.colors_per_image,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    
    pipeline.run(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        use_existing_masks=True,
        segment_new=args.segment,
        include_ceiling=args.include_ceiling,
        proposal_only=args.proposal_only,
    )


if __name__ == "__main__":
    main()
