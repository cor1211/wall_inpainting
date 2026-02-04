"""
Generative Pipeline for AI Interior Wall Re-skinning

Uses Stable Diffusion 1.5 Inpainting with:
- ControlNet Depth for structure preservation
- IP-Adapter for color/texture transfer from reference image
"""

import torch
from pathlib import Path
from typing import Union, Optional, Tuple
from PIL import Image
import numpy as np

# Check for CUDA availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


class WallReskinPipeline:
    """
    Pipeline for wall re-skinning using Stable Diffusion Inpainting.
    
    Components:
    - SD 1.5 Inpainting: Base model for masked region generation
    - ControlNet Depth: Preserves 3D structure and perspective
    - IP-Adapter: Transfers color/style from reference image
    - Midas Depth Estimator: Generates depth maps for ControlNet
    """
    
    def __init__(
        self,
        base_model: str = "runwayml/stable-diffusion-inpainting",
        controlnet_model: str = "lllyasviel/control_v11f1p_sd15_depth",
        ip_adapter_model: str = "h94/IP-Adapter",
        device: Optional[str] = None,
        enable_cpu_offload: bool = True,
        lora_path: Optional[str] = None,
        lora_scale: float = 1.0,
    ):
        """
        Initialize the wall re-skinning pipeline.
        
        Args:
            base_model: HuggingFace ID for SD Inpainting model.
            controlnet_model: HuggingFace ID for ControlNet Depth.
            ip_adapter_model: HuggingFace ID for IP-Adapter.
            device: Device to run on (auto-detect if None).
            enable_cpu_offload: Enable CPU offloading for memory optimization.
            lora_path: Optional path to LoRA checkpoint directory.
            lora_scale: LoRA strength multiplier (0.0-1.0).
        """
        self.device = device or DEVICE
        self.dtype = DTYPE if self.device == "cuda" else torch.float32
        self.enable_cpu_offload = enable_cpu_offload
        
        self.base_model_id = base_model
        self.controlnet_model_id = controlnet_model
        self.ip_adapter_model_id = ip_adapter_model
        
        # LoRA configuration
        self.lora_path = lora_path
        self.lora_scale = lora_scale
        
        # Lazy loading
        self._pipe = None
        self._depth_estimator = None
        
        lora_info = f", LoRA: {lora_path}" if lora_path else ""
        print(f"WallReskinPipeline initialized (device: {self.device}{lora_info})")
    
    def _load_pipeline(self):
        """Load the diffusion pipeline with ControlNet and IP-Adapter."""
        if self._pipe is not None:
            return
        
        print("Loading models... (this may take a few minutes on first run)")
        
        from diffusers import (
            StableDiffusionControlNetInpaintPipeline, 
            ControlNetModel,
        )
        
        # 1. Load ControlNet (Depth)
        print("  Loading ControlNet Depth...")
        controlnet = ControlNetModel.from_pretrained(
            self.controlnet_model_id,
            torch_dtype=self.dtype,
        )
        
        # 2. Load Main Pipeline (SD 1.5 Inpainting)
        print("  Loading Stable Diffusion Inpainting...")
        self._pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            self.base_model_id,
            controlnet=controlnet,
            torch_dtype=self.dtype,
            safety_checker=None,
        )
        
        # 3. Load IP-Adapter (For Style/Color Transfer)
        print("  Loading IP-Adapter...")
        self._pipe.load_ip_adapter(
            self.ip_adapter_model_id, 
            subfolder="models", 
            weight_name="ip-adapter_sd15.bin"
        )
        self._pipe.set_ip_adapter_scale(0.7)  # Default scale
        
        # Load LoRA BEFORE enabling CPU offload (PEFT is incompatible with offload hooks)
        self._load_lora()
        
        # 4. Optimization
        # Note: CPU offload is disabled when LoRA is active due to PEFT hook conflicts
        if self.enable_cpu_offload and self.lora_path is None:
            self._pipe.enable_model_cpu_offload()
        else:
            self._pipe.to(self.device)
            if self.lora_path is not None:
                print("  Note: CPU offload disabled (incompatible with LoRA/PEFT)")
        
        # Try to enable xformers for memory efficiency
        try:
            self._pipe.enable_xformers_memory_efficient_attention()
            print("  xformers enabled for memory efficiency")
        except Exception:
            print("  xformers not available, using default attention")
        
        print("Models loaded successfully!")
    
    def _load_depth_estimator(self):
        """Load depth estimator using transformers pipeline."""
        if self._depth_estimator is not None:
            return
        
        from transformers import pipeline as tf_pipeline
        
        print("Loading Depth Estimator (Intel DPT-Large)...")
        self._depth_estimator = tf_pipeline(
            "depth-estimation", 
            model="Intel/dpt-large",
            device=0 if self.device == "cuda" else -1,
        )
        print("Depth estimator loaded!")
    
    def _load_lora(self):
        """Load LoRA weights if specified."""
        if self.lora_path is None:
            return
        
        from lora_utils import load_lora_weights
        
        load_lora_weights(
            self._pipe,
            self.lora_path,
            lora_scale=self.lora_scale,
            adapter_name="wall_inpainting",
        )
    
    def set_lora(
        self,
        lora_path: Optional[str] = None,
        scale: float = 1.0,
    ) -> None:
        """
        Load or change LoRA weights dynamically.
        
        Args:
            lora_path: Path to new LoRA checkpoint. If None, unload current LoRA.
            scale: LoRA strength multiplier.
        """
        # Ensure pipeline is loaded
        self._load_pipeline()
        
        from lora_utils import load_lora_weights, unload_lora
        
        if lora_path is None:
            # Unload current LoRA
            unload_lora(self._pipe, adapter_name="wall_inpainting")
            self.lora_path = None
            self.lora_scale = 1.0
        else:
            # Load new LoRA
            load_lora_weights(
                self._pipe,
                lora_path,
                lora_scale=scale,
                adapter_name="wall_inpainting",
            )
            self.lora_path = lora_path
            self.lora_scale = scale
    
    @property
    def pipe(self):
        """Lazy-load and return the diffusion pipeline."""
        self._load_pipeline()
        return self._pipe
    
    @property
    def depth_estimator(self):
        """Lazy-load and return the depth estimator."""
        self._load_depth_estimator()
        return self._depth_estimator
    
    def generate_depth_map(
        self, 
        image: Union[str, Path, Image.Image],
    ) -> Image.Image:
        """
        Generate depth map from image using Intel DPT model.
        
        Args:
            image: Input image (path or PIL Image).
            
        Returns:
            Depth map as PIL Image.
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        
        # Get depth prediction
        result = self.depth_estimator(image)
        depth_map = result["depth"]
        
        # Ensure it's a PIL Image
        if not isinstance(depth_map, Image.Image):
            depth_map = Image.fromarray(depth_map)
        
        # Resize to match input
        if depth_map.size != image.size:
            depth_map = depth_map.resize(image.size, Image.Resampling.LANCZOS)
        
        return depth_map
    
    def process(
        self,
        source_image: Union[str, Path, Image.Image],
        mask_image: Union[str, Path, Image.Image],
        reference_image: Union[str, Path, Image.Image],
        prompt: str = "high quality wall, interior design, photorealistic",
        negative_prompt: str = "blurry, low quality, artifacts, distortion, furniture changes, people",
        num_inference_steps: int = 30,
        controlnet_conditioning_scale: float = 0.8,
        ip_adapter_scale: float = 0.7,
        strength: float = 0.99,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        output_size: Tuple[int, int] = (512, 512),
    ) -> Image.Image:
        """
        Process an image to change wall color/texture.
        
        Args:
            source_image: Source interior image.
            mask_image: Binary mask (white = wall to change).
            reference_image: Reference image for color/texture.
            prompt: Text prompt for generation.
            negative_prompt: Negative prompt.
            num_inference_steps: Number of denoising steps.
            controlnet_conditioning_scale: ControlNet strength (0-1).
            ip_adapter_scale: IP-Adapter strength (0-1).
            strength: Denoising strength (0-1, higher = more change).
            guidance_scale: CFG scale.
            seed: Random seed for reproducibility.
            output_size: Output image size (default 512x512 for SD 1.5).
            
        Returns:
            Generated image with new wall color/texture.
        """
        # Load images
        if isinstance(source_image, (str, Path)):
            source_image = Image.open(source_image).convert("RGB")
        if isinstance(mask_image, (str, Path)):
            mask_image = Image.open(mask_image).convert("L")
        if isinstance(reference_image, (str, Path)):
            reference_image = Image.open(reference_image).convert("RGB")
        
        # Store original size
        original_size = source_image.size
        
        # Resize for SD 1.5
        source_resized = source_image.resize(output_size, Image.Resampling.LANCZOS)
        mask_resized = mask_image.resize(output_size, Image.Resampling.NEAREST)
        
        # Reference image for IP-Adapter needs 224x224
        reference_resized = reference_image.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Generate depth map
        depth_map = self.generate_depth_map(source_resized)
        
        # Convert mask to RGB for pipeline
        mask_rgb = mask_resized.convert("RGB")
        
        # Set IP-Adapter scale
        self.pipe.set_ip_adapter_scale(ip_adapter_scale)
        
        # Set up generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Generate
        print(f"Generating with {num_inference_steps} steps...")
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=source_resized,
            mask_image=mask_rgb,
            control_image=depth_map,
            ip_adapter_image=reference_resized,
            num_inference_steps=num_inference_steps,
            generator=generator,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            strength=strength,
            guidance_scale=guidance_scale,
        ).images[0]
        
        # Resize back to original size
        if result.size != original_size:
            result = result.resize(original_size, Image.Resampling.LANCZOS)
        
        return result
    
    def process_batch(
        self,
        source_images: list,
        mask_images: list,
        reference_image: Union[str, Path, Image.Image],
        **kwargs,
    ) -> list:
        """
        Process multiple images with the same reference.
        
        Args:
            source_images: List of source images.
            mask_images: List of corresponding masks.
            reference_image: Single reference image for all.
            **kwargs: Additional arguments for process().
            
        Returns:
            List of generated images.
        """
        results = []
        for src, mask in zip(source_images, mask_images):
            result = self.process(src, mask, reference_image, **kwargs)
            results.append(result)
        return results


def create_solid_color_reference(
    color: Tuple[int, int, int],
    size: Tuple[int, int] = (224, 224),
) -> Image.Image:
    """
    Create a solid color reference image for IP-Adapter.
    
    Args:
        color: RGB color tuple (0-255).
        size: Image size.
        
    Returns:
        Solid color PIL Image.
    """
    return Image.new("RGB", size, color)


def create_gradient_reference(
    color1: Tuple[int, int, int],
    color2: Tuple[int, int, int],
    size: Tuple[int, int] = (224, 224),
    direction: str = "vertical",
) -> Image.Image:
    """
    Create a gradient reference image for IP-Adapter.
    
    Args:
        color1: Start color RGB.
        color2: End color RGB.
        size: Image size.
        direction: 'vertical' or 'horizontal'.
        
    Returns:
        Gradient PIL Image.
    """
    import numpy as np
    
    w, h = size
    if direction == "vertical":
        gradient = np.linspace(0, 1, h).reshape(-1, 1)
        gradient = np.tile(gradient, (1, w))
    else:
        gradient = np.linspace(0, 1, w).reshape(1, -1)
        gradient = np.tile(gradient, (h, 1))
    
    # Interpolate colors
    c1 = np.array(color1)
    c2 = np.array(color2)
    
    result = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(3):
        result[:, :, i] = (c1[i] * (1 - gradient) + c2[i] * gradient).astype(np.uint8)
    
    return Image.fromarray(result)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Wall Re-skinning Pipeline"
    )
    parser.add_argument("source", type=str, help="Path to source image")
    parser.add_argument("--mask", type=str, required=True, help="Path to mask image")
    parser.add_argument("--reference", type=str, help="Path to reference image")
    parser.add_argument("--color", type=str, help="Solid color (e.g., '255,200,150')")
    parser.add_argument("--output", type=str, default="output.png", help="Output path")
    parser.add_argument("--steps", type=int, default=30, help="Inference steps")
    parser.add_argument("--controlnet-scale", type=float, default=0.8, help="ControlNet scale")
    parser.add_argument("--ip-scale", type=float, default=0.7, help="IP-Adapter scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    args = parser.parse_args()
    
    # Create reference image
    if args.reference:
        reference = args.reference
    elif args.color:
        color = tuple(map(int, args.color.split(",")))
        reference = create_solid_color_reference(color)
        print(f"Using solid color: {color}")
    else:
        print("Error: Either --reference or --color must be specified")
        exit(1)
    
    # Run pipeline
    pipeline = WallReskinPipeline()
    
    result = pipeline.process(
        source_image=args.source,
        mask_image=args.mask,
        reference_image=reference,
        num_inference_steps=args.steps,
        controlnet_conditioning_scale=args.controlnet_scale,
        ip_adapter_scale=args.ip_scale,
        seed=args.seed,
    )
    
    result.save(args.output)
    print(f"Result saved to: {args.output}")
