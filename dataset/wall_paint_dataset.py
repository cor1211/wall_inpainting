"""
Wall Paint Dataset for Training

Dataset loader for wall recoloring task.
Loads source images, target images (GT), masks, and color references.
"""

import json
import random
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F

from dataset.wall_colors import create_color_patch


class WallPaintDataset(Dataset):
    """
    Dataset for training wall recoloring model.
    
    Expected metadata.jsonl format:
    {
        "source_path": "train/images/xxx_original.png",
        "target_path": "train/images/xxx_color.png",
        "mask_path": "train/masks/xxx.png",
        "color_rgb": [128, 0, 32],
        "color_name": "burgundy"
    }
    
    The dataset returns:
    - source: Original wall image (for ControlNet condition)
    - target: Recolored wall image (GT for training)
    - mask: Wall region mask
    - color_patches: Color reference patch (for IP-Adapter)
    - conditional_images: Preprocessed image for ControlNet (depth/canny)
    - masked_sources: Source image with masked region
    - prompts: Text prompt
    """
    
    def __init__(
        self,
        data_json: Union[str, Path],
        image_size: int = 512,
        reconstruction_ratio: float = 0.5,
        use_depth: bool = True,
        use_canny: bool = False,
        random_flip: bool = True,
    ):
        """
        Initialize Wall Paint Dataset.
        
        Args:
            data_json: Path to metadata.jsonl file
            image_size: Target image size (square)
            reconstruction_ratio: Ratio of samples to use reconstruction loss
                                 (0.0 = always use target, 1.0 = always reconstruct source)
            use_depth: If True, use depth map for ControlNet (recommended)
            use_canny: If True, use canny edges for ControlNet (alternative to depth)
            random_flip: If True, apply random horizontal flip augmentation
        """
        self.data_json = Path(data_json)
        self.image_size = image_size
        self.reconstruction_ratio = reconstruction_ratio
        self.use_depth = use_depth
        self.use_canny = use_canny
        self.random_flip = random_flip
        
        # Load metadata
        self.samples = []
        with open(self.data_json, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line.strip()))
        
        print(f"Loaded {len(self.samples)} samples from {self.data_json}")
        
        # Setup transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), Image.LANCZOS),
            transforms.ToTensor(),
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), Image.NEAREST),
            transforms.ToTensor(),
        ])
        
        # Lazy load depth estimator if needed
        self._depth_estimator = None
        self._canny_processor = None
        
    def _get_depth_estimator(self):
        """Lazy load depth estimator."""
        if self._depth_estimator is None:
            try:
                from transformers import pipeline
                self._depth_estimator = pipeline(
                    "depth-estimation",
                    model="Intel/dpt-large",
                    device=0 if torch.cuda.is_available() else -1,
                )
            except Exception as e:
                print(f"Warning: Could not load depth estimator: {e}")
                self._depth_estimator = False  # Mark as unavailable
        return self._depth_estimator
    
    def _get_canny_processor(self):
        """Lazy load canny processor."""
        if self._canny_processor is None:
            try:
                import cv2
                self._canny_processor = cv2
            except ImportError:
                print("Warning: OpenCV not available for Canny processing")
                self._canny_processor = False
        return self._canny_processor
    
    def _create_conditional_image(self, image: Image.Image) -> Image.Image:
        """
        Create conditional image for ControlNet.
        
        Returns depth map if use_depth=True, else canny edges.
        """
        if self.use_depth:
            depth_estimator = self._get_depth_estimator()
            if depth_estimator and depth_estimator is not False:
                try:
                    result = depth_estimator(image)
                    depth_map = result["depth"] if isinstance(result, dict) else result
                    # Resize to target size
                    depth_map = depth_map.resize((self.image_size, self.image_size), Image.LANCZOS)
                    return depth_map
                except Exception as e:
                    print(f"Warning: Depth estimation failed: {e}, using grayscale")
                    # Fallback to grayscale
                    return image.convert("L").resize((self.image_size, self.image_size))
            else:
                # Fallback to grayscale
                return image.convert("L").resize((self.image_size, self.image_size))
        
        elif self.use_canny:
            canny_processor = self._get_canny_processor()
            if canny_processor and canny_processor is not False:
                try:
                    import cv2
                    img_array = np.array(image)
                    canny = cv2.Canny(img_array, 100, 200)
                    canny_rgb = np.stack([canny] * 3, axis=-1)
                    canny_img = Image.fromarray(canny_rgb)
                    return canny_img.resize((self.image_size, self.image_size), Image.LANCZOS)
                except Exception as e:
                    print(f"Warning: Canny processing failed: {e}, using grayscale")
                    return image.convert("L").resize((self.image_size, self.image_size))
        
        # Default: grayscale
        return image.convert("L").resize((self.image_size, self.image_size))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a training sample.
        
        Returns dictionary with keys:
        - source: Source image tensor [3, H, W] (0-1)
        - target: Target image tensor [3, H, W] (0-1) - GT
        - mask: Mask tensor [1, H, W] (0-1)
        - color_patches: Color reference tensor [3, 224, 224] (0-1) for IP-Adapter
        - conditional_images: ControlNet condition [3, H, W] (0-1)
        - masked_sources: Masked source [3, H, W] (0-1)
        - prompts: Text prompt string
        """
        sample = self.samples[idx]
        
        # Get data root (parent of train/validation directory)
        data_root = self.data_json.parent.parent
        
        # Load images
        source_path = data_root / sample["source_path"]
        target_path = data_root / sample["target_path"]
        mask_path = data_root / sample["mask_path"]
        
        source_img = Image.open(source_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L")
        
        # Random flip augmentation
        if self.random_flip and random.random() > 0.5:
            source_img = source_img.transpose(Image.FLIP_LEFT_RIGHT)
            target_img = target_img.transpose(Image.FLIP_LEFT_RIGHT)
            mask_img = mask_img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Apply transforms
        source = self.image_transform(source_img)  # [3, H, W] (0-1)
        target = self.image_transform(target_img)  # [3, H, W] (0-1)
        mask = self.mask_transform(mask_img)  # [1, H, W] (0-1)
        
        # Create color reference patch
        color_rgb = sample.get("color_rgb", [128, 128, 128])
        color_patch_array = create_color_patch(
            color_rgb,
            size=(224, 224),  # IP-Adapter expects 224x224
            add_texture=True,
            add_gradient=True
        )
        color_patch_pil = Image.fromarray(color_patch_array)
        color_patches = transforms.ToTensor()(color_patch_pil)  # [3, 224, 224] (0-1)
        
        # Create conditional image (depth/canny)
        conditional_image = self._create_conditional_image(source_img)
        conditional_images = self.image_transform(conditional_image.convert("RGB"))  # [3, H, W] (0-1)
        
        # Create masked source (for inpainting)
        masked_sources = source * (1 - mask)  # [3, H, W] (0-1)
        
        # Prompt
        prompt = sample.get("prompt", "interior wall, high quality, realistic")
        
        return {
            "source": source,
            "target": target,
            "mask": mask,
            "color_patches": color_patches,
            "conditional_images": conditional_images,
            "masked_sources": masked_sources,
            "prompts": prompt,
        }
    
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate function for DataLoader.
        
        Stacks tensors and converts lists to tensors where needed.
        """
        # Stack all tensor fields
        result = {}
        for key in ["source", "target", "mask", "color_patches", "conditional_images", "masked_sources"]:
            if key in batch[0]:
                result[key] = torch.stack([item[key] for item in batch])
        
        # Keep prompts as list (will be tokenized in training loop)
        result["prompts"] = [item["prompts"] for item in batch]
        
        return result
