"""
Dataset Preparation Pipeline V2 (Optimized).

High-performance script for creating Wall Recoloring dataset pairs.
Focuses strictly on I/O and applying texture-preserving color transformations.

Workflow:
1. Read Source Image & Mask
2. Apply Texture-Preserving Color Shift -> Generate Target Image
3. Save Pair (Source, Target, Mask) -> Write Metadata (JSONL)

Usage:
    python prepare_dataset_v2.py --input raw_data/ --output dataset_v2/ --max-samples 100
"""

import os
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from PIL import Image
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable
            print(f"Processing: {kwargs.get('desc', '')}")
            
        def __iter__(self):
            return iter(self.iterable)
            
        def set_postfix(self, **kwargs):
            pass

# Import optimized color transform logic
from dataset.color_transforms import apply_color_shift
from dataset.wall_colors import sample_random_color

# Configuration
RANDOM_SEED = 42

@dataclass
class TrainingPair:
    """Efficient data structure for training pairs."""
    source_path: str   # Relative path to Original (Condition)
    target_path: str   # Relative path to Recolored (Noisy Input)
    mask_path: str     # Relative path to Mask
    color_rgb: List[int] # Target Color [R, G, B]
    color_name: str    # Color name (for prompting/debugging)

class DatasetPipeline:
    def __init__(
        self,
        output_dir: Path,
        colors_per_image: int = 5,
        val_ratio: float = 0.1,
        seed: int = RANDOM_SEED,
    ):
        self.output_dir = output_dir
        self.colors_per_image = colors_per_image
        self.val_ratio = val_ratio
        
        # Setup Output Directories
        self.dirs = {
            "train": {
                "images": output_dir / "train" / "images",
                "masks": output_dir / "train" / "masks",
                "meta": output_dir / "train" / "metadata.jsonl"
            },
            "validation": {
                "images": output_dir / "validation" / "images",
                "masks": output_dir / "validation" / "masks",
                "meta": output_dir / "validation" / "metadata.jsonl"
            }
        }
        
        # Create directories and open file handles
        self.handles = {}
        for split, paths in self.dirs.items():
            paths["images"].mkdir(parents=True, exist_ok=True)
            paths["masks"].mkdir(parents=True, exist_ok=True)
            self.handles[split] = open(paths["meta"], "w", encoding='utf-8')

        random.seed(seed)
        np.random.seed(seed)

    def close(self):
        """Close file handles."""
        for f in self.handles.values():
            f.close()

    def process_single_image(self, img_path: Path, mask_path: Path) -> int:
        """
        Process a single source image: generate N pairs and save immediately.
        Returns number of pairs generated.
        """
        try:
            # 1. Fast Load (Lazy if possible, but we need numpy for transform)
            # Use distinct variable names to avoid closure issues
            src_img_pil = Image.open(img_path).convert("RGB")
            mask_pil = Image.open(mask_path).convert("L")
            
            # Simple validation - fast check size matches
            if src_img_pil.size != mask_pil.size:
                # Resize mask to match image
                mask_pil = mask_pil.resize(src_img_pil.size, Image.NEAREST)

            src_arr = np.array(src_img_pil)
            mask_arr = np.array(mask_pil)
            mask_float = (mask_arr > 127).astype(np.float32)

            # Skip if mask is empty
            if mask_float.sum() < 100:
                return 0

            # 2. Decide Split (per source image to avoid leakage)
            split = "validation" if random.random() < self.val_ratio else "train"
            split_dirs = self.dirs[split]
            
            base_name = img_path.stem

            # 3. Save Common Files (Source & Mask) ONCE per image?
            # Actually, typically we save them once.
            # Filenames
            src_filename = f"{base_name}_original.png"
            mask_filename = f"{base_name}.png"
            
            rel_src_path = f"{split}/images/{src_filename}"
            rel_mask_path = f"{split}/masks/{mask_filename}"
            
            abs_src_path = split_dirs["images"] / src_filename
            abs_mask_path = split_dirs["masks"] / mask_filename

            # Write Source & Mask if not exists
            if not abs_src_path.exists():
                src_img_pil.save(abs_src_path, optimize=False) # optimize=False for speed
            
            if not abs_mask_path.exists():
                mask_pil.save(abs_mask_path, optimize=False)

            # 4. Generate Variants
            metadata_entries = []
            
            for i in range(self.colors_per_image):
                # Generate random color
                color_name, color_rgb = sample_random_color()
                
                # Apply Transform
                # Apply texture-preserving color shift
                # Input: Source Image. Target: New Color.
                target_arr = apply_color_shift(src_arr, mask_float, color_rgb)
                
                # Save Target
                target_filename = f"{base_name}_{color_name}_{i}.png" # Add index to avoid collisions
                rel_target_path = f"{split}/images/{target_filename}"
                abs_target_path = split_dirs["images"] / target_filename
                
                Image.fromarray(target_arr).save(abs_target_path, optimize=False)

                # Create Metadata
                pair = TrainingPair(
                    source_path=rel_src_path,
                    target_path=rel_target_path,
                    mask_path=rel_mask_path,
                    color_rgb=list(color_rgb),
                    color_name=color_name
                )
                metadata_entries.append(asdict(pair))

            # 5. Flush Metadata
            for entry in metadata_entries:
                self.handles[split].write(json.dumps(entry) + "\n")
            
            return len(metadata_entries)

        except Exception as e:
            print(f"[Error] Failed processing {img_path.name}: {e}")
            return 0

    def run(self, input_dir: Path, max_samples: int = None):
        """Main execution loop."""
        input_dir = Path(input_dir)
        
        # Discovery
        img_dir = input_dir / "images"
        mask_dir = input_dir / "masks"
        
        if not img_dir.exists() or not mask_dir.exists():
            image_files = sorted(list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")))
            # Assume masks are in same dir with suffix
            pairs = []
            for img in image_files:
                # Try finding mask with _mask suffix or same name in masks/ folder
                # Logic: user usually has a structure. Let's assume input has 'images' and 'masks' subdirs
                # based on previous script usage. If not, fail fast.
                pass 
        else:
            all_images = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))

        print(f"Found {len(all_images)} source images.")
        
        # Iteration Control
        if max_samples:
            all_images = all_images[:max_samples]
            print(f"Limiting to first {max_samples} images.")

        count = 0
        pbar = tqdm(all_images, desc="Generating Pairs")
        
        for img_path in pbar:
            # Find mask
            mask_path = mask_dir / f"{img_path.stem}.png"
            if not mask_path.exists():
                # Try .jpg mask? unlikely for masks but possible
                continue
            
            num_pairs = self.process_single_image(img_path, mask_path)
            count += num_pairs
            
            if num_pairs > 0:
                pbar.set_postfix(pairs=count)

        self.close()
        print(f"\ndataset generation complete. Generated {count} pairs in {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Optimized Wall Recoloring Dataset Generator")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input directory (must contain 'images' and 'masks' subdirs)")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--max-samples", "-n", type=int, default=None, help="Limit number of source images to process")
    parser.add_argument("--colors", type=int, default=5, help="Color variants per image")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    pipeline = DatasetPipeline(
        output_dir=Path(args.output),
        colors_per_image=args.colors,
        seed=args.seed
    )
    
    pipeline.run(Path(args.input), max_samples=args.max_samples)

if __name__ == "__main__":
    main()
