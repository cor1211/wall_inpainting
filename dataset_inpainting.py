"""
Custom Dataset for Inpainting Training with JSONL metadata.

Loads image, mask, and caption triplets from training_ready format.
"""
import json
import random
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms


class InpaintingDataset(Dataset):
    """
    Dataset for training SD Inpainting models.
    
    Expected directory structure:
    training_ready/
    ├── train/
    │   ├── images/
    │   ├── masks/
    │   ├── metadata.jsonl
    │   └── captions.txt
    └── validation/
        ├── images/
        ├── masks/
        └── metadata.jsonl
    
    Each line in metadata.jsonl:
    {
        "image_path": "train/images/xxx.png",
        "mask_path": "train/masks/xxx.png", 
        "caption": "interior room with gray walls",
        "quality_score": 0.8,
        "surface_ratio": 0.4
    }
    """
    
    def __init__(
        self,
        data_root: Union[str, Path],
        split: str = "train",
        resolution: int = 512,
        center_crop: bool = True,
        random_flip: bool = True,
        quality_threshold: float = 0.5,
        tokenizer = None,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            data_root: Root directory containing training_ready folder
            split: "train" or "validation"
            resolution: Target resolution for images (512 for SD 1.5)
            center_crop: Whether to center crop images
            random_flip: Whether to randomly flip images horizontally
            quality_threshold: Minimum quality score to include sample
            tokenizer: Tokenizer for encoding captions
            max_samples: Maximum number of samples (for debugging)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.tokenizer = tokenizer
        
        # Load metadata
        split_dir = self.data_root / split
        if split == "validation":
            split_dir = self.data_root / "validation"
        
        metadata_path = split_dir / "metadata.jsonl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        self.samples = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line.strip())
                # Filter by quality
                if sample.get("quality_score", 1.0) >= quality_threshold:
                    self.samples.append(sample)
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        print(f"Loaded {len(self.samples)} samples from {split} split")
        
        # Define transforms
        self.image_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution) if center_crop else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
        ])
        
        self.mask_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(resolution) if center_crop else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
        ])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Normalize paths (handle Windows-style paths in metadata)
        image_rel_path = sample["image_path"].replace("\\", "/")
        mask_rel_path = sample["mask_path"].replace("\\", "/")
        
        # Build full paths
        image_path = self.data_root / image_rel_path
        mask_path = self.data_root / mask_rel_path
        
        # Load images
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        # Random horizontal flip (same transform for both)
        if self.random_flip and self.split == "train" and random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Apply transforms
        image_tensor = self.image_transforms(image)
        mask_tensor = self.mask_transforms(mask)
        
        # Prepare caption
        caption = sample.get("caption", "interior wall")
        
        # Tokenize if tokenizer provided
        input_ids = None
        if self.tokenizer is not None:
            inputs = self.tokenizer(
                caption,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = inputs.input_ids[0]
        
        result = {
            "pixel_values": image_tensor,
            "mask": mask_tensor,
            "caption": caption,
            "quality_score": sample.get("quality_score", 1.0),
        }
        
        if input_ids is not None:
            result["input_ids"] = input_ids
        
        return result


class InpaintingCollator:
    """
    Collator for batching inpainting samples.
    Prepares inputs in the format expected by SD Inpainting pipeline.
    """
    
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
    
    def __call__(self, samples: list) -> Dict[str, torch.Tensor]:
        pixel_values = torch.stack([s["pixel_values"] for s in samples])
        masks = torch.stack([s["mask"] for s in samples])
        captions = [s["caption"] for s in samples]
        
        batch = {
            "pixel_values": pixel_values,
            "masks": masks,
            "captions": captions,
        }
        
        # Add tokenized inputs if available
        if "input_ids" in samples[0]:
            input_ids = torch.stack([s["input_ids"] for s in samples])
            batch["input_ids"] = input_ids
        
        return batch


def prepare_mask_and_masked_image(image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare mask and masked image for inpainting.
    
    Args:
        image: Image tensor [B, 3, H, W] in [-1, 1]
        mask: Mask tensor [B, 1, H, W] in [0, 1]
        
    Returns:
        mask: Processed mask [B, 1, H, W]
        masked_image: Image with masked regions zeroed [B, 3, H, W]
    """
    # Binarize mask
    mask = (mask > 0.5).float()
    
    # Create masked image (set masked regions to 0)
    masked_image = image * (1 - mask)
    
    return mask, masked_image


if __name__ == "__main__":
    # Test dataset loading
    import sys
    
    data_root = Path("/mnt/data1tb/vinh/wall_inpainting/training_ready")
    
    print("Testing InpaintingDataset...")
    dataset = InpaintingDataset(
        data_root=data_root,
        split="train",
        resolution=512,
        max_samples=10,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test single sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Image shape: {sample['pixel_values'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    print(f"Caption: {sample['caption']}")
    print(f"Quality score: {sample['quality_score']:.3f}")
    
    # Test collator
    collator = InpaintingCollator()
    batch = collator([dataset[i] for i in range(3)])
    print(f"\nBatch pixel_values shape: {batch['pixel_values'].shape}")
    print(f"Batch masks shape: {batch['masks'].shape}")
    print(f"Batch captions: {batch['captions']}")
    
    print("\n✅ Dataset test passed!")
