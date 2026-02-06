
import argparse
import os
import torch
from pathlib import Path
from PIL import Image
import numpy as np

from models.wall_recoloring_pipeline import get_wall_recoloring_pipeline
from dataset.wall_colors import create_color_patch

def main():
    parser = argparse.ArgumentParser(description="Inference for Wall Recoloring")
    parser.add_argument("--image", type=str, required=True, help="Input Source Image path")
    parser.add_argument("--mask", type=str, required=True, help="Input Mask path")
    parser.add_argument("--color_hex", type=str, default="#FF0000", help="Target Color Hex")
    parser.add_argument("--output", type=str, default="output/result.png")
    parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-inpainting")
    # Path to your trained LoRA or just base model for testing
    parser.add_argument("--lora_weights", type=str, default=None)
    parser.add_argument("--controlnet_model", type=str, default="lllyasviel/control_v11p_sd15_canny")
    parser.add_argument("--ip_adapter_scale", type=float, default=0.6)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # 1. Load Pipeline
    print(f"Loading pipeline on {args.device}...")
    pipe = get_wall_recoloring_pipeline(
        base_model_path=args.base_model,
        controlnet_path=args.controlnet_model,
        ip_adapter_scale=args.ip_adapter_scale,
        device=args.device
    )
    
    if args.lora_weights:
        print(f"Loading LoRA from {args.lora_weights}")
        pipe.load_lora_weights(args.lora_weights)
        
    # 2. Prepare Inputs
    source_img = Image.open(args.image).convert("RGB").resize((512, 512))
    mask_img = Image.open(args.mask).convert("L").resize((512, 512))
    
    # Control Image (Canny) - ControlNet pipeline expects the condition image
    # If using Canny ControlNet, we might need to preprocess.
    # But usually ControlNetModel expects the already processed Canny map OR the pipeline handles it?
    # StableDiffusionControlNetPipeline expects the condition image (e.g. Canny map).
    # "lllyasviel/control_v11p_sd15_canny" expects a Canny edge map.
    import cv2
    image_np = np.array(source_img)
    image_canny = cv2.Canny(image_np, 100, 200)
    image_canny = np.stack([image_canny] * 3, axis=-1)
    control_image = Image.fromarray(image_canny)
    
    # IP-Adapter Image (Color Patch)
    # Convert Hex to RGB
    hex_color = args.color_hex.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    ip_image = create_color_patch(rgb, size=512)
    # ip_image is numpy array (512, 512, 3)
    ip_image_pil = Image.fromarray(ip_image)
    
    # 3. Generate
    prompt = "high quality, realistic wall"
    
    # Inpainting Pipeline signature:
    # prompt, image (source), mask_image, control_image (for ControlNet), ip_adapter_image (for IP-Adapter)
    # Note: `StableDiffusionControlNetInpaintPipeline` typically takes:
    # prompt, image (this is the starting image to inpaint), mask_image, control_image
    # IP Adapter adds `ip_adapter_image` arg via mixin.
    
    print("Generating...")
    result = pipe(
        prompt=prompt,
        image=source_img,
        mask_image=mask_img,
        control_image=control_image,
        ip_adapter_image=ip_image_pil,
        num_inference_steps=30,
        strength=1.0 # Denoising strength (1.0 = full generation in masked area)
    ).images[0]
    
    # 4. Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result.save(args.output)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
