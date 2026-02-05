"""
Configuration Module for AI Interior Wall Re-skinning

Centralized configuration for all pipeline parameters.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple


# Base directories
PROJECT_ROOT = Path(__file__).parent.absolute()
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SAMPLES_DIR = PROJECT_ROOT / "tests" / "samples"


@dataclass
class SegmentationConfig:
    """Configuration for wall segmentation."""
    
    # FastSAM model
    fastsam_model: str = "FastSAM-x.pt"
    fastsam_model_url: str = "https://github.com/ultralytics/assets/releases/download/v8.1.0/FastSAM-x.pt"
    
    # CLIP model for filtering
    clip_model: str = "ViT-B/32"
    
    # Default strategy
    default_strategy: str = "semantic"
    
    # Mask processing
    dilate_kernel_size: int = 7
    clip_threshold: float = 0.2
    
    # FastSAM parameters
    confidence: float = 0.4
    iou_threshold: float = 0.9


@dataclass
class PipelineConfig:
    """Configuration for generative pipeline."""
    
    # Model IDs (HuggingFace)
    base_model: str = "runwayml/stable-diffusion-inpainting"
    controlnet_model: str = "lllyasviel/control_v11f1p_sd15_depth"
    ip_adapter_model: str = "h94/IP-Adapter"
    ip_adapter_weights: str = "ip-adapter_sd15.bin"
    depth_model: str = "Intel/dpt-large"
    
    # Generation parameters
    num_inference_steps: int = 30
    controlnet_conditioning_scale: float = 0.8
    ip_adapter_scale: float = 1.0  # Maximum for better color transfer
    strength: float = 0.99
    guidance_scale: float = 5.0  # Balanced: not too high (color bias), not too low (weak transfer)
    
    # Image sizes
    output_size: Tuple[int, int] = (512, 512)
    reference_size: Tuple[int, int] = (224, 224)
    
    # Prompts (empty to let IP-Adapter control color)
    default_prompt: str = ""
    default_negative_prompt: str = "blurry, low quality, artifacts, distortion"
    
    # Optimization
    enable_cpu_offload: bool = True
    enable_xformers: bool = True


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters."""
    
    # Enable LoRA loading (DISABLED - LoRA has color bias from training dataset)
    enabled: bool = False
    
    # Default LoRA checkpoint path
    default_path: str = "lora_checkpoints"
    
    # Default LoRA strength (0.0 - 1.0)
    default_scale: float = 1.0
    
    # Allow dynamic LoRA selection via API
    allow_dynamic_selection: bool = False


@dataclass
class APIConfig:
    """Configuration for FastAPI server."""
    
    host: str = "0.0.0.0"
    port: int = 8000
    max_file_size_mb: int = 20
    allowed_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")
    
    # Rate limiting
    max_requests_per_minute: int = 10
    
    # Temporary files
    temp_dir: str = field(default_factory=lambda: str(OUTPUT_DIR / "temp"))
    cleanup_temp_files: bool = True


@dataclass 
class Config:
    """Main configuration class combining all configs."""
    
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    api: APIConfig = field(default_factory=APIConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    
    # Global settings
    device: Optional[str] = None  # None = auto-detect
    seed: Optional[int] = None
    debug: bool = False
    
    def __post_init__(self):
        """Initialize directories."""
        MODELS_DIR.mkdir(exist_ok=True)
        OUTPUT_DIR.mkdir(exist_ok=True)


# Default configuration instance
config = Config()


def get_device() -> str:
    """Get the device to use (CUDA if available, else CPU)."""
    import torch
    if config.device:
        return config.device
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_fastsam_model_path() -> Path:
    """Get path to FastSAM model, downloading if necessary."""
    model_path = MODELS_DIR / config.segmentation.fastsam_model
    if not model_path.exists():
        print(f"FastSAM model not found at {model_path}")
        print("Run 'python download_models.py' to download required models.")
    return model_path
