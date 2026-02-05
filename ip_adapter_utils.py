"""
IP-Adapter Utilities for Training

This module provides utilities for integrating IP-Adapter into the LoRA training pipeline.
Following the approach from Paint-by-Example and AnyDoor papers:
- IP-Adapter is loaded and FROZEN during LoRA training
- LoRA learns to collaborate with IP-Adapter for reference-based generation
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, List
from pathlib import Path
import logging
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

logger = logging.getLogger(__name__)


class IPAdapterImageEncoder:
    """
    Encodes reference images for IP-Adapter conditioning.
    
    Uses CLIP ViT-H/14 image encoder (same as IP-Adapter Plus).
    """
    
    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.device = device
        self.dtype = dtype
        self._image_encoder = None
        self._image_processor = None
        
    def load(self):
        """Load CLIP image encoder."""
        if self._image_encoder is not None:
            return
            
        logger.info("Loading CLIP Image Encoder for IP-Adapter...")
        
        # IP-Adapter Plus (and apparently Standard sd15 too) uses ViT-H/14
        self._image_processor = CLIPImageProcessor.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        )
        self._image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        ).to(self.device, dtype=self.dtype)
        self._image_encoder.eval()
        
        # Freeze encoder
        for param in self._image_encoder.parameters():
            param.requires_grad = False
            
        logger.info("CLIP Image Encoder loaded and frozen")
    
    @torch.no_grad()
    def encode(
        self,
        images: Union[Image.Image, List[Image.Image], torch.Tensor],
    ) -> torch.Tensor:
        """
        Encode images to CLIP embeddings for IP-Adapter.
        
        Args:
            images: PIL Image(s) or tensor of shape [B, 3, 224, 224]
            
        Returns:
            image_embeds: [B, 1, 768] for IP-Adapter
        """
        self.load()
        
        # Handle PIL images
        if isinstance(images, Image.Image):
            images = [images]
        
        if isinstance(images, list) and isinstance(images[0], Image.Image):
            # Process with CLIP processor
            inputs = self._image_processor(
                images=images,
                return_tensors="pt",
            ).to(self.device, dtype=self.dtype)
            pixel_values = inputs.pixel_values
        else:
            # Already tensor
            pixel_values = images.to(self.device, dtype=self.dtype)
        
        # Encode
        outputs = self._image_encoder(pixel_values)
        image_embeds = outputs.image_embeds  # [B, 768]
        
        # Add sequence dimension for IP-Adapter
        image_embeds = image_embeds.unsqueeze(1)  # [B, 1, 768]
        
        return image_embeds
    
    def encode_batch(
        self,
        batch_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode a batch of reference images from training dataloader.
        
        Args:
            batch_tensor: [B, 3, 224, 224] tensor from dataloader
            
        Returns:
            image_embeds: [B, 1, 768]
        """
        self.load()
        
        # Denormalize from [-1, 1] to [0, 1] for CLIP
        # CLIP expects specific normalization, handled by processor
        # But our tensors are in [-1, 1], need to convert
        images = (batch_tensor + 1) / 2  # [-1, 1] -> [0, 1]
        images = images.clamp(0, 1)
        
        # CLIP normalization (processor does this but we have tensor)
        # mean = [0.48145466, 0.4578275, 0.40821073]
        # std = [0.26862954, 0.26130258, 0.27577711]
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device, dtype=self.dtype)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device, dtype=self.dtype)
        
        images = (images - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
        
        # Encode
        outputs = self._image_encoder(images.to(self.dtype))
        image_embeds = outputs.image_embeds  # [B, 768]
        image_embeds = image_embeds.unsqueeze(1)  # [B, 1, 768]
        
        return image_embeds


def load_ip_adapter_for_training(
    pipe,
    model_id: str = "h94/IP-Adapter",
    weight_name: str = "ip-adapter-plus_sd15.bin",
    scale: float = 1.0,
    freeze: bool = True,
):
    """
    Load IP-Adapter into a diffusers pipeline for training.
    
    Args:
        pipe: StableDiffusionInpaintPipeline or similar
        model_id: HuggingFace model ID
        weight_name: Weight file name
        scale: IP-Adapter scale (1.0 = full strength)
        freeze: Whether to freeze IP-Adapter weights
        
    Returns:
        pipe: Pipeline with IP-Adapter loaded
    """
    logger.info(f"Loading IP-Adapter from {model_id}/{weight_name}")
    
    pipe.load_ip_adapter(
        model_id,
        subfolder="models",
        weight_name=weight_name,
    )
    pipe.set_ip_adapter_scale(scale)
    
    if freeze:
        freeze_ip_adapter_weights(pipe.unet)
        logger.info("IP-Adapter weights frozen")
    
    logger.info(f"IP-Adapter loaded with scale={scale}")
    return pipe


def freeze_ip_adapter_weights(unet: nn.Module):
    """
    Freeze IP-Adapter specific weights in UNet.
    
    IP-Adapter adds these modules:
    - image_proj (image projection)
    - to_k_ip, to_v_ip (cross-attention for image)
    """
    frozen_count = 0
    for name, param in unet.named_parameters():
        # IP-Adapter related parameters
        if any(keyword in name.lower() for keyword in ["ip_adapter", "image_proj", "to_k_ip", "to_v_ip"]):
            param.requires_grad = False
            frozen_count += 1
    
    logger.info(f"Frozen {frozen_count} IP-Adapter parameters")


def get_ip_adapter_image_embeds(
    pipe,
    reference_images: Union[Image.Image, List[Image.Image], torch.Tensor],
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Get IP-Adapter image embeddings using the pipeline's built-in encoder.
    
    This uses the IP-Adapter's image encoder that was loaded with the adapter.
    
    Args:
        pipe: Pipeline with IP-Adapter loaded
        reference_images: Reference images
        device: Device
        dtype: Data type
        
    Returns:
        image_embeds: Tensor ready for added_cond_kwargs
    """
    # The diffusers pipeline handles this internally
    # We need to access the image encoder from ip_adapter
    
    if hasattr(pipe, "image_encoder") and pipe.image_encoder is not None:
        # Use pipeline's image encoder
        if isinstance(reference_images, torch.Tensor):
            # Convert tensor to PIL for pipeline
            images_pil = []
            for i in range(reference_images.shape[0]):
                img = reference_images[i]
                img = ((img + 1) / 2 * 255).clamp(0, 255).byte()
                img = img.permute(1, 2, 0).cpu().numpy()
                images_pil.append(Image.fromarray(img))
            reference_images = images_pil
        
        # Encode using pipeline's prepare method
        image_embeds = pipe.prepare_ip_adapter_image_embeds(
            ip_adapter_image=reference_images,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,  # No CFG during training
        )
        return image_embeds
    else:
        raise RuntimeError("Pipeline does not have image_encoder. Is IP-Adapter loaded?")


class IPAdapterTrainingHelper:
    """
    Helper class to manage IP-Adapter during LoRA training.
    
    Usage:
        helper = IPAdapterTrainingHelper(pipe, config)
        helper.setup()
        
        for batch in dataloader:
            image_embeds = helper.encode_references(batch["reference_images"])
            noise_pred = unet(
                ...,
                added_cond_kwargs={"image_embeds": image_embeds},
            )
    """
    
    def __init__(
        self,
        pipe,
        ip_adapter_config: dict,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.pipe = pipe
        self.config = ip_adapter_config
        self.device = device
        self.dtype = dtype
        self.is_setup = False
        
        # Separate encoder for training (more control)
        self.image_encoder = IPAdapterImageEncoder(device, dtype)
        
    def setup(self):
        """Setup IP-Adapter for training."""
        if self.is_setup:
            return
            
        # Load IP-Adapter
        load_ip_adapter_for_training(
            self.pipe,
            model_id=self.config.get("model_id", "h94/IP-Adapter"),
            weight_name=self.config.get("weight_name", "ip-adapter-plus_sd15.bin"),
            scale=self.config.get("scale", 1.0),
            freeze=self.config.get("freeze_weights", True),
        )
        
        # Load image encoder
        self.image_encoder.load()
        
        self.is_setup = True
        logger.info("IPAdapterTrainingHelper setup complete")
    
    def encode_references(self, reference_images: torch.Tensor) -> torch.Tensor:
        """
        Encode reference images for training.
        
        Args:
            reference_images: [B, 3, 224, 224] from dataloader
            
        Returns:
            image_embeds: [B, 1, 768] for added_cond_kwargs
        """
        return self.image_encoder.encode_batch(reference_images)
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get list of trainable parameters (excludes frozen IP-Adapter)."""
        trainable = []
        for name, param in self.pipe.unet.named_parameters():
            if param.requires_grad:
                trainable.append(param)
        return trainable
