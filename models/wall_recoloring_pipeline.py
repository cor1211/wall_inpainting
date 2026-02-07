"""
Wall Recoloring Pipeline Module

Wrapper for Stable Diffusion Inpainting + ControlNet + IP-Adapter pipeline.
This module provides a unified interface for loading and using the wall recoloring pipeline.
"""

import torch
from pathlib import Path
from typing import Optional, Union
from PIL import Image

from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection


def get_wall_recoloring_pipeline(
    base_model_path: str = "runwayml/stable-diffusion-inpainting",
    controlnet_path: str = "lllyasviel/control_v11f1p_sd15_depth",
    ip_adapter_scale: float = 0.7,
    device: Union[str, torch.device] = "cuda",
    torch_dtype: Optional[torch.dtype] = None,
) -> StableDiffusionControlNetInpaintPipeline:
    """
    Load and configure the wall recoloring pipeline.
    
    This function loads:
    1. Stable Diffusion Inpainting base model
    2. ControlNet (Depth) for structure preservation
    3. IP-Adapter Plus for color/style transfer
    
    Args:
        base_model_path: HuggingFace model ID or local path for SD Inpainting
        controlnet_path: HuggingFace model ID or local path for ControlNet Depth
        ip_adapter_scale: IP-Adapter strength (0.0-1.0). Higher = stronger color transfer
        device: Device to load models on ("cuda", "cpu", or torch.device)
        torch_dtype: Data type for models (None = auto-detect based on device)
    
    Returns:
        Configured StableDiffusionControlNetInpaintPipeline ready for inference
    
    Example:
        >>> pipe = get_wall_recoloring_pipeline(
        ...     base_model_path="runwayml/stable-diffusion-inpainting",
        ...     controlnet_path="lllyasviel/control_v11f1p_sd15_depth",
        ...     ip_adapter_scale=0.7,
        ...     device="cuda"
        ... )
        >>> result = pipe(
        ...     prompt="interior wall",
        ...     image=source_image,
        ...     mask_image=mask,
        ...     control_image=depth_map,
        ...     ip_adapter_image=color_reference
        ... ).images[0]
    """
    # Auto-detect dtype if not specified
    if torch_dtype is None:
        if device == "cuda" and torch.cuda.is_available():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
    
    print(f"Loading Wall Recoloring Pipeline...")
    print(f"  Device: {device}")
    print(f"  Dtype: {torch_dtype}")
    print(f"  Base Model: {base_model_path}")
    print(f"  ControlNet: {controlnet_path}")
    
    # 1. Load ControlNet (Depth)
    print("  [1/3] Loading ControlNet Depth...")
    controlnet = ControlNetModel.from_pretrained(
        controlnet_path,
        torch_dtype=torch_dtype,
    )
    
    # 2. Load Main Pipeline (SD Inpainting + ControlNet)
    print("  [2/3] Loading Stable Diffusion Inpainting...")
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=torch_dtype,
        safety_checker=None,  # Disable safety checker for faster inference
    )
    
    # 3. Load IP-Adapter Plus
    print("  [3/3] Loading IP-Adapter Plus...")
    try:
        pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="models",
            weight_name="ip-adapter-plus_sd15.bin"
        )
        pipe.set_ip_adapter_scale(ip_adapter_scale)
        print(f"  IP-Adapter scale set to {ip_adapter_scale}")
    except Exception as e:
        print(f"  Warning: Failed to load IP-Adapter Plus: {e}")
        print("  Trying standard IP-Adapter...")
        try:
            pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="models",
                weight_name="ip-adapter_sd15.bin"
            )
            pipe.set_ip_adapter_scale(ip_adapter_scale)
        except Exception as e2:
            print(f"  Error: Could not load IP-Adapter: {e2}")
            print("  Pipeline will work without IP-Adapter (color transfer may be limited)")
    
    # 4. Move to device
    pipe = pipe.to(device)
    
    # 5. Enable optimizations
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("  xformers memory efficient attention enabled")
    except Exception:
        print("  xformers not available, using default attention")
    
    print("Pipeline loaded successfully!")
    
    return pipe


def extract_components(pipeline: StableDiffusionControlNetInpaintPipeline):
    """
    Extract individual components from pipeline for training.
    
    Args:
        pipeline: The loaded pipeline
    
    Returns:
        Dictionary with components:
        - tokenizer: CLIPTokenizer
        - text_encoder: CLIPTextModel
        - vae: AutoencoderKL
        - unet: UNet2DConditionModel
        - controlnet: ControlNetModel
        - image_encoder: CLIPVisionModelWithProjection (for IP-Adapter)
        - scheduler: DDPMScheduler
    """
    return {
        "tokenizer": pipeline.tokenizer,
        "text_encoder": pipeline.text_encoder,
        "vae": pipeline.vae,
        "unet": pipeline.unet,
        "controlnet": pipeline.controlnet,
        "image_encoder": getattr(pipeline, "image_encoder", None),  # May not exist if IP-Adapter not loaded
        "scheduler": pipeline.scheduler,
    }
