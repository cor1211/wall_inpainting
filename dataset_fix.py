"""
Fixed Dataset for Wall Inpainting Training with Proper Reference Image Generation.

This module addresses the DATA LEAKAGE problem by generating synthetic reference images
that simulate inference-time conditions during training.

Key Fix:
- Instead of using the original image as reference (which contains noise/shadows/details),
  we extract the DOMINANT COLOR from the segmentation mask region and create a SOLID COLOR
  image to use as the IP-Adapter reference input.

This ensures train-time and inference-time distributions match.

Author: CV/AI Research Team
Version: 2.0.0 (Fixed Data Distribution)
"""
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class WallInpaintingDataset(Dataset):
    """
    Production-grade Dataset for Wall Inpainting with Zero-Prompt Strategy.
    
    Key Features:
    1. Extracts DOMINANT COLOR from wall region (masked area) in original image
    2. Creates SOLID COLOR REFERENCE IMAGE to simulate inference conditions
    3. Adds subtle texture/noise to solid color for CLIP feature extraction
    4. Returns all required inputs for SD 1.5 Inpainting + IP-Adapter pipeline
    
    This fixes the train/inference distribution mismatch problem.
    
    Expected directory structure:
    training_ready/
    ├── train/
    │   ├── images/     # Original interior images
    │   ├── masks/      # Segmentation masks (white = wall area)
    │   └── metadata.jsonl
    └── validation/
        ├── images/
        ├── masks/
        └── metadata.jsonl
    """
    
    def __init__(
        self,
        data_root: Union[str, Path],
        split: str = "train",
        resolution: int = 512,
        center_crop: bool = True,
        random_flip: bool = True,
        quality_threshold: float = 0.5,
        tokenizer=None,
        max_samples: Optional[int] = None,
        # Reference image generation config
        add_reference_texture: bool = True,
        texture_noise_std: float = 8.0,
        add_lighting_gradient: bool = True,
        # Color extraction config
        color_extraction_method: str = "median",  # "median", "mean", or "kmeans"
        color_jitter_prob: float = 0.0,  # Probability of adding color jitter during training
        color_jitter_range: float = 10.0,  # Max jitter in LAB space
        # HYBRID STRATEGY: Mix extracted colors with random colors
        random_color_prob: float = 0.0,  # Probability of using completely random color (0-1)
        # Mask erosion to reduce segmentation noise
        mask_erosion_size: int = 0,  # Kernel size for erosion (0 = disabled)
    ):
        """
        Initialize dataset with dominant color extraction.
        
        Args:
            data_root: Root directory containing training_ready folder
            split: "train" or "validation"
            resolution: Target resolution for images (512 for SD 1.5)
            center_crop: Whether to center crop images
            random_flip: Whether to randomly flip images horizontally
            quality_threshold: Minimum quality score to include sample
            tokenizer: Tokenizer for encoding captions (optional for zero-prompt)
            max_samples: Maximum number of samples (for debugging)
            add_reference_texture: Add subtle noise to solid color for CLIP
            texture_noise_std: Standard deviation of texture noise
            add_lighting_gradient: Add subtle lighting gradient variation
            color_extraction_method: How to compute dominant color
            color_jitter_prob: Data augmentation - jitter target color
            color_jitter_range: Range of color jitter in LAB space
        """
        self.data_root = Path(data_root)
        self.split = split
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.tokenizer = tokenizer
        
        # Reference generation config
        self.add_reference_texture = add_reference_texture
        self.texture_noise_std = texture_noise_std
        self.add_lighting_gradient = add_lighting_gradient
        self.color_extraction_method = color_extraction_method
        self.color_jitter_prob = color_jitter_prob
        self.color_jitter_range = color_jitter_range
        self.random_color_prob = random_color_prob
        self.mask_erosion_size = mask_erosion_size
        
        # Load metadata
        split_dir = self.data_root / split
        metadata_path = split_dir / "metadata.jsonl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        self.samples = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line.strip())
                if sample.get("quality_score", 1.0) >= quality_threshold:
                    self.samples.append(sample)
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        print(f"[WallInpaintingDataset] Loaded {len(self.samples)} samples from {split}")
        print(f"[WallInpaintingDataset] Color extraction method: {color_extraction_method}")
        print(f"[WallInpaintingDataset] Reference texture: {add_reference_texture}")
        print(f"[WallInpaintingDataset] Random color probability: {random_color_prob:.0%}")
        print(f"[WallInpaintingDataset] Mask erosion: {mask_erosion_size}px")
        
        # Define image transforms
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
        
        # Reference image transform (for IP-Adapter, typically 224x224)
        self.reference_transforms = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def extract_dominant_color(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> Tuple[int, int, int]:
        """
        Extract the dominant color from the wall region (masked area).
        
        This is the CORE FIX for the data leakage problem.
        
        Args:
            image: Original RGB image
            mask: Binary mask (white = wall region to extract color from)
        
        Returns:
            Tuple[int, int, int]: Dominant RGB color (0-255)
        """
        img_arr = np.array(image)  # [H, W, 3]
        mask_arr = np.array(mask.convert("L"))  # [H, W]
        
        # Create boolean mask for wall region
        wall_mask = mask_arr > 127  # Binary threshold
        
        # Check if mask has enough pixels
        if wall_mask.sum() < 100:
            # Fallback: use image mean if mask is too small
            return tuple(img_arr.mean(axis=(0, 1)).astype(np.uint8).tolist())
        
        # Extract pixels from wall region
        wall_pixels = img_arr[wall_mask]  # [N, 3]
        
        if self.color_extraction_method == "mean":
            # Simple mean - fast but affected by outliers
            color = wall_pixels.mean(axis=0).astype(np.uint8)
        
        elif self.color_extraction_method == "median":
            # Median - robust to outliers (shadows, highlights)
            color = np.median(wall_pixels, axis=0).astype(np.uint8)
        
        elif self.color_extraction_method == "kmeans":
            # K-means clustering to find dominant color cluster
            # More accurate for textured walls, but slower
            from sklearn.cluster import KMeans
            
            # Subsample for speed
            if len(wall_pixels) > 5000:
                indices = np.random.choice(len(wall_pixels), 5000, replace=False)
                wall_pixels_sample = wall_pixels[indices]
            else:
                wall_pixels_sample = wall_pixels
            
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=3)
            kmeans.fit(wall_pixels_sample)
            
            # Find largest cluster
            labels, counts = np.unique(kmeans.labels_, return_counts=True)
            dominant_cluster = labels[np.argmax(counts)]
            color = kmeans.cluster_centers_[dominant_cluster].astype(np.uint8)
        
        else:
            raise ValueError(f"Unknown color extraction method: {self.color_extraction_method}")
        
        return tuple(color.tolist())
    
    def create_solid_color_reference(
        self,
        color: Tuple[int, int, int],
        size: Tuple[int, int] = (512, 512),
    ) -> Image.Image:
        """
        Create a solid color reference image with optional texture.
        
        Pure solid colors can break CLIP feature extraction (not enough variance),
        so we add subtle noise and gradient to give the encoder something to work with.
        
        Args:
            color: RGB color tuple (0-255)
            size: Output image size
        
        Returns:
            PIL Image with solid color + optional texture
        """
        img = Image.new("RGB", size, color)
        
        if self.add_reference_texture or self.add_lighting_gradient:
            arr = np.array(img, dtype=np.float32)
            
            if self.add_reference_texture:
                # Add subtle Gaussian noise for texture
                noise = np.random.normal(0, self.texture_noise_std, arr.shape)
                arr = arr + noise
            
            if self.add_lighting_gradient:
                # Add subtle vertical gradient to simulate lighting variation
                h, w = size[1], size[0]
                gradient = np.linspace(0.95, 1.05, h).reshape(-1, 1, 1)
                gradient = np.tile(gradient, (1, w, 3))
                arr = arr * gradient
            
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
        
        return img
    
    def apply_color_jitter(
        self,
        color: Tuple[int, int, int],
    ) -> Tuple[int, int, int]:
        """
        Apply random color jitter for data augmentation.
        
        Operates in LAB space for perceptually uniform jitter.
        """
        if random.random() > self.color_jitter_prob:
            return color
        
        # Convert to LAB
        rgb = np.array([[color]], dtype=np.uint8)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Add jitter to a and b channels (color, not lightness)
        jitter = np.random.uniform(-self.color_jitter_range, self.color_jitter_range, 2)
        lab[0, 0, 1:] += jitter
        
        # Clip and convert back
        lab = np.clip(lab, [0, 0, 0], [255, 255, 255]).astype(np.uint8)
        rgb_jittered = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return tuple(rgb_jittered[0, 0].tolist())
    
    def generate_random_color(self) -> Tuple[int, int, int]:
        """
        Generate a completely random RGB color.
        
        This enables the model to learn ANY color mapping, not just
        colors that appear naturally in interior images.
        
        Returns:
            Tuple[int, int, int]: Random RGB color (0-255)
        """
        return (
            random.randint(30, 240),   # Avoid extreme blacks/whites
            random.randint(30, 240),
            random.randint(30, 240),
        )
    
    def apply_mask_erosion(self, mask: Image.Image) -> Image.Image:
        """
        Apply morphological erosion to mask to reduce segmentation noise.
        
        Erosion shrinks the mask boundary, removing small incorrectly
        segmented regions at the edges.
        
        Args:
            mask: Binary mask image
        
        Returns:
            Eroded mask image
        """
        if self.mask_erosion_size <= 0:
            return mask
        
        mask_arr = np.array(mask)
        kernel = np.ones((self.mask_erosion_size, self.mask_erosion_size), np.uint8)
        eroded = cv2.erode(mask_arr, kernel, iterations=1)
        
        return Image.fromarray(eroded)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single training sample with properly generated reference image.
        
        This method implements the FIXED data pipeline that:
        1. Loads original image and mask
        2. Extracts dominant wall color from masked region
        3. Creates synthetic solid color reference image
        4. Returns all inputs needed for SD Inpainting + IP-Adapter
        """
        sample = self.samples[idx]
        
        # Normalize paths
        image_rel_path = sample["image_path"].replace("\\", "/")
        mask_rel_path = sample["mask_path"].replace("\\", "/")
        
        # Load images
        image_path = self.data_root / image_rel_path
        mask_path = self.data_root / mask_rel_path
        
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        # ========================================
        # Apply mask erosion to reduce segmentation noise
        # ========================================
        mask = self.apply_mask_erosion(mask)
        
        # ========================================
        # HYBRID COLOR STRATEGY
        # ========================================
        # Choose between extracted color (realistic) and random color (generalization)
        use_random_color = self.split == "train" and random.random() < self.random_color_prob
        
        if use_random_color:
            # Generate completely random color for better generalization
            dominant_color = self.generate_random_color()
        else:
            # Extract dominant color from wall region (realistic)
            dominant_color = self.extract_dominant_color(image, mask)
            
            # Apply color jitter augmentation (training only)
            if self.split == "train":
                dominant_color = self.apply_color_jitter(dominant_color)
        
        # ========================================
        # Create solid color reference image
        # ========================================
        # This simulates what users will provide at inference time
        reference_image = self.create_solid_color_reference(
            dominant_color,
            size=(self.resolution, self.resolution)
        )
        
        # Random horizontal flip (same transform for image, mask, and reference)
        if self.random_flip and self.split == "train" and random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            # Reference is solid color, flip doesn't change it, but for consistency:
            reference_image = reference_image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Apply transforms
        image_tensor = self.image_transforms(image)
        mask_tensor = self.mask_transforms(mask)
        reference_tensor = self.reference_transforms(reference_image)
        
        # Caption (empty for zero-prompt strategy)
        caption = sample.get("caption", "")
        
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
            # Core inputs
            "pixel_values": image_tensor,           # Original image [-1, 1]
            "mask": mask_tensor,                     # Binary mask [0, 1]
            "reference_image": reference_tensor,     # Solid color reference [-1, 1]
            
            # Metadata
            "dominant_color": torch.tensor(dominant_color, dtype=torch.uint8),
            "caption": caption,
            "quality_score": sample.get("quality_score", 1.0),
            "image_path": str(image_path),
        }
        
        if input_ids is not None:
            result["input_ids"] = input_ids
        
        return result


class WallInpaintingCollator:
    """
    Collator for batching wall inpainting samples.
    
    Properly handles reference images and dominant colors.
    """
    
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
    
    def __call__(self, samples: list) -> Dict[str, torch.Tensor]:
        batch = {
            "pixel_values": torch.stack([s["pixel_values"] for s in samples]),
            "masks": torch.stack([s["mask"] for s in samples]),
            "reference_images": torch.stack([s["reference_image"] for s in samples]),
            "dominant_colors": torch.stack([s["dominant_color"] for s in samples]),
            "captions": [s["caption"] for s in samples],
        }
        
        if "input_ids" in samples[0]:
            batch["input_ids"] = torch.stack([s["input_ids"] for s in samples])
        
        return batch


def dominant_color_to_pil(color: torch.Tensor, size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """Convert dominant color tensor to PIL Image for visualization."""
    if isinstance(color, torch.Tensor):
        color = tuple(color.cpu().numpy().tolist())
    return Image.new("RGB", size, color)


# ============================================================================
# Backward Compatibility Aliases
# ============================================================================
InpaintingDataset = WallInpaintingDataset
InpaintingCollator = WallInpaintingCollator


if __name__ == "__main__":
    """Test the fixed dataset implementation."""
    import tempfile
    from pathlib import Path
    
    print("=" * 60)
    print("Testing WallInpaintingDataset with Dominant Color Extraction")
    print("=" * 60)
    
    data_root = Path("/mnt/data1tb/vinh/wall_inpainting/training_ready")
    
    if data_root.exists():
        # Test with real data
        print("\nLoading dataset...")
        dataset = WallInpaintingDataset(
            data_root=data_root,
            split="train",
            resolution=512,
            max_samples=5,
            color_extraction_method="median",
            add_reference_texture=True,
        )
        
        print(f"\nDataset size: {len(dataset)}")
        
        # Test single sample
        sample = dataset[0]
        print(f"\nSample keys: {list(sample.keys())}")
        print(f"Image shape: {sample['pixel_values'].shape}")
        print(f"Mask shape: {sample['mask'].shape}")
        print(f"Reference shape: {sample['reference_image'].shape}")
        print(f"Dominant color (RGB): {tuple(sample['dominant_color'].tolist())}")
        print(f"Caption: {sample['caption']}")
        
        # Visualize
        from validation_visualizer import tensor_to_pil, mask_tensor_to_pil
        
        source_img = tensor_to_pil(sample['pixel_values'])
        ref_img = tensor_to_pil(sample['reference_image'])
        mask_img = mask_tensor_to_pil(sample['mask'])
        
        # Save test output
        test_dir = Path("./test_output")
        test_dir.mkdir(exist_ok=True)
        source_img.save(test_dir / "source.png")
        ref_img.save(test_dir / "reference_solid_color.png")
        mask_img.save(test_dir / "mask.png")
        
        print(f"\nTest images saved to {test_dir}")
        
        # Test collator
        collator = WallInpaintingCollator()
        batch = collator([dataset[i] for i in range(3)])
        print(f"\nBatch pixel_values shape: {batch['pixel_values'].shape}")
        print(f"Batch reference_images shape: {batch['reference_images'].shape}")
        print(f"Batch dominant_colors: {batch['dominant_colors']}")
        
        print("\n✅ Dataset test passed!")
    else:
        print(f"Data root not found: {data_root}")
        print("Skipping real data test")
