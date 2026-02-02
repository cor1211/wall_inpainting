"""
Main Execution Script for AI Interior Wall Re-skinning

Orchestrates the flow:
1. Load source image
2. Generate wall mask using segmentation.py
3. Process with pipeline.py to change wall color
4. Save result
"""

import argparse
from pathlib import Path
from PIL import Image

from segmentation import WallSegmenter
from pipeline import WallReskinPipeline, create_solid_color_reference


def main():
    parser = argparse.ArgumentParser(
        description="AI Interior Wall Re-skinning - Change wall color/texture in interior images"
    )
    
    # Required arguments
    parser.add_argument(
        "--source", "-s", 
        type=str, 
        required=True,
        help="Path to source interior image"
    )
    
    # Reference (one of these required)
    ref_group = parser.add_mutually_exclusive_group(required=True)
    ref_group.add_argument(
        "--reference", "-r",
        type=str,
        help="Path to reference image for color/texture"
    )
    ref_group.add_argument(
        "--color", "-c",
        type=str,
        help="Solid color in format 'R,G,B' (e.g., '255,200,150')"
    )
    
    # Output
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path (default: source_reskin.png)"
    )
    
    # Mask options
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Custom mask path (if not provided, auto-generate using segmentation)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["auto", "semantic", "clip", "heuristic"],
        default="semantic",
        help="Segmentation strategy (default: semantic)"
    )
    parser.add_argument(
        "--include-ceiling",
        action="store_true",
        help="Include ceiling in wall mask"
    )
    
    # Pipeline options
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of inference steps (default: 30)"
    )
    parser.add_argument(
        "--controlnet-scale",
        type=float,
        default=0.8,
        help="ControlNet conditioning scale (default: 0.8)"
    )
    parser.add_argument(
        "--ip-scale",
        type=float,
        default=0.7,
        help="IP-Adapter scale (default: 0.7)"
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.99,
        help="Denoising strength (default: 0.99)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    # Debug options
    parser.add_argument(
        "--save-mask",
        action="store_true",
        help="Save generated mask"
    )
    parser.add_argument(
        "--save-depth",
        action="store_true",
        help="Save depth map"
    )
    
    args = parser.parse_args()
    
    # Validate source
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: Source image not found: {source_path}")
        return 1
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = source_path.parent / f"{source_path.stem}_reskin.png"
    
    print("=" * 60)
    print("AI Interior Wall Re-skinning")
    print("=" * 60)
    print(f"Source: {source_path}")
    
    # Step 1: Get or generate mask
    if args.mask:
        print(f"Using custom mask: {args.mask}")
        mask_image = Image.open(args.mask).convert("L")
    else:
        print(f"Generating wall mask (strategy: {args.strategy})...")
        segmenter = WallSegmenter()
        mask_image = segmenter.get_wall_mask(
            source_path,
            strategy=args.strategy,
            include_ceiling=args.include_ceiling,
            return_pil=True,
        )
        
        if args.save_mask:
            mask_path = source_path.parent / f"{source_path.stem}_mask.png"
            mask_image.save(mask_path)
            print(f"Mask saved to: {mask_path}")
    
    # Check if mask has content
    mask_array = list(mask_image.getdata())
    if sum(mask_array) == 0:
        print("Warning: Empty mask - no wall detected!")
        print("Try using --strategy semantic or providing a custom --mask")
        return 1
    
    # Step 2: Create reference
    if args.reference:
        print(f"Reference: {args.reference}")
        reference_image = Image.open(args.reference).convert("RGB")
    else:
        color = tuple(map(int, args.color.split(",")))
        print(f"Color: RGB{color}")
        reference_image = create_solid_color_reference(color)
    
    # Step 3: Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = WallReskinPipeline()
    
    # Step 4: Process
    print("\nProcessing...")
    result = pipeline.process(
        source_image=source_path,
        mask_image=mask_image,
        reference_image=reference_image,
        num_inference_steps=args.steps,
        controlnet_conditioning_scale=args.controlnet_scale,
        ip_adapter_scale=args.ip_scale,
        strength=args.strength,
        seed=args.seed,
    )
    
    # Save depth map if requested
    if args.save_depth:
        source_img = Image.open(source_path).convert("RGB")
        depth = pipeline.generate_depth_map(source_img)
        depth_path = source_path.parent / f"{source_path.stem}_depth.png"
        depth.save(depth_path)
        print(f"Depth map saved to: {depth_path}")
    
    # Step 5: Save result
    result.save(output_path)
    print(f"\n{'=' * 60}")
    print(f"Result saved to: {output_path}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
