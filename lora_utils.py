"""
LoRA Utilities for Wall Inpainting Pipeline.

Provides functions for loading, managing, and switching LoRA adapters
in the Stable Diffusion Inpainting pipeline.

Usage:
    from lora_utils import load_lora_weights, get_lora_metadata

    # Load LoRA into existing pipeline
    load_lora_weights(pipeline.pipe, "lora_checkpoints", scale=0.8)
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

import torch


def load_lora_weights(
    pipeline,
    lora_path: Union[str, Path],
    lora_scale: float = 1.0,
    adapter_name: str = "default",
) -> None:
    """
    Load LoRA weights into a Stable Diffusion pipeline's UNet.
    
    Args:
        pipeline: Diffusers pipeline with a UNet component.
        lora_path: Path to LoRA checkpoint directory (containing adapter_config.json
                   and adapter_model.safetensors).
        lora_scale: LoRA strength multiplier (0.0 = disabled, 1.0 = full strength).
        adapter_name: Name for the adapter (for multi-LoRA support).
    
    Raises:
        FileNotFoundError: If LoRA path doesn't exist or is invalid.
        ValueError: If lora_scale is out of valid range.
    
    Example:
        >>> from diffusers import StableDiffusionInpaintPipeline
        >>> pipe = StableDiffusionInpaintPipeline.from_pretrained(...)
        >>> load_lora_weights(pipe, "lora_checkpoints", scale=0.8)
    """
    lora_path = Path(lora_path)
    
    # Validate path
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA path not found: {lora_path}")
    
    # Check for required files
    config_file = lora_path / "adapter_config.json"
    weights_file = lora_path / "adapter_model.safetensors"
    
    if not config_file.exists():
        raise FileNotFoundError(
            f"adapter_config.json not found in {lora_path}. "
            "Ensure this is a valid PEFT LoRA checkpoint."
        )
    
    if not weights_file.exists():
        # Try .bin format
        weights_file = lora_path / "adapter_model.bin"
        if not weights_file.exists():
            raise FileNotFoundError(
                f"adapter_model.safetensors or .bin not found in {lora_path}"
            )
    
    # Validate scale
    if not 0.0 <= lora_scale <= 2.0:
        raise ValueError(f"lora_scale must be between 0.0 and 2.0, got {lora_scale}")
    
    # Load using PEFT
    from peft import PeftModel
    
    print(f"Loading LoRA from {lora_path} (scale={lora_scale})...")
    
    # Check if UNet already has PEFT adapters
    if hasattr(pipeline.unet, 'peft_config'):
        # Already has adapters, load additional one
        pipeline.unet.load_adapter(str(lora_path), adapter_name=adapter_name)
    else:
        # First adapter, wrap with PeftModel
        pipeline.unet = PeftModel.from_pretrained(
            pipeline.unet,
            str(lora_path),
            adapter_name=adapter_name,
        )
    
    # Set active adapter and scale
    pipeline.unet.set_adapter(adapter_name)
    
    # Store scale for reference (used during inference)
    if not hasattr(pipeline, '_lora_scales'):
        pipeline._lora_scales = {}
    pipeline._lora_scales[adapter_name] = lora_scale
    
    print(f"LoRA '{adapter_name}' loaded successfully!")


def unload_lora(pipeline, adapter_name: str = "default") -> None:
    """
    Unload a LoRA adapter from the pipeline.
    
    Args:
        pipeline: Diffusers pipeline with LoRA loaded.
        adapter_name: Name of adapter to unload.
    """
    if not hasattr(pipeline.unet, 'peft_config'):
        print("No LoRA adapters loaded.")
        return
    
    try:
        pipeline.unet.delete_adapter(adapter_name)
        if hasattr(pipeline, '_lora_scales') and adapter_name in pipeline._lora_scales:
            del pipeline._lora_scales[adapter_name]
        print(f"LoRA '{adapter_name}' unloaded.")
    except Exception as e:
        print(f"Failed to unload LoRA: {e}")


def set_lora_scale(pipeline, scale: float, adapter_name: str = "default") -> None:
    """
    Adjust the strength of a loaded LoRA adapter.
    
    Args:
        pipeline: Diffusers pipeline with LoRA loaded.
        scale: New scale value (0.0 to 2.0).
        adapter_name: Name of adapter to adjust.
    """
    if not hasattr(pipeline, '_lora_scales'):
        pipeline._lora_scales = {}
    pipeline._lora_scales[adapter_name] = scale
    
    # PEFT doesn't have built-in per-adapter scaling at inference,
    # so we store it and apply during forward pass if needed
    print(f"LoRA '{adapter_name}' scale set to {scale}")


def get_lora_metadata(lora_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extract metadata from a LoRA checkpoint.
    
    Args:
        lora_path: Path to LoRA checkpoint directory.
    
    Returns:
        Dictionary with LoRA configuration:
        {
            "rank": int,
            "alpha": int,
            "target_modules": list,
            "dropout": float,
            "base_model": str,
        }
    """
    lora_path = Path(lora_path)
    config_file = lora_path / "adapter_config.json"
    
    if not config_file.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {lora_path}")
    
    with open(config_file, "r") as f:
        config = json.load(f)
    
    return {
        "rank": config.get("r", 0),
        "alpha": config.get("lora_alpha", 0),
        "target_modules": config.get("target_modules", []),
        "dropout": config.get("lora_dropout", 0.0),
        "base_model": config.get("base_model_name_or_path", "unknown"),
        "peft_type": config.get("peft_type", "LORA"),
    }


def list_available_loras(lora_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    List all available LoRA checkpoints in a directory.
    
    Args:
        lora_dir: Directory containing LoRA checkpoints.
    
    Returns:
        List of dictionaries with LoRA info:
        [
            {"name": "default", "path": "/path/to/lora", "rank": 8, ...},
            ...
        ]
    """
    lora_dir = Path(lora_dir)
    loras = []
    
    # Check if the directory itself is a LoRA checkpoint
    if (lora_dir / "adapter_config.json").exists():
        try:
            meta = get_lora_metadata(lora_dir)
            loras.append({
                "name": lora_dir.name,
                "path": str(lora_dir),
                **meta
            })
        except Exception:
            pass
    
    # Check subdirectories
    for subdir in lora_dir.iterdir():
        if subdir.is_dir() and (subdir / "adapter_config.json").exists():
            try:
                meta = get_lora_metadata(subdir)
                loras.append({
                    "name": subdir.name,
                    "path": str(subdir),
                    **meta
                })
            except Exception:
                pass
    
    return loras


def merge_lora_weights(
    pipeline,
    adapter_name: str = "default",
    safe_merge: bool = True,
) -> None:
    """
    Merge LoRA weights into the base model permanently.
    
    WARNING: This is irreversible. The LoRA cannot be unloaded after merging.
    Use this for production deployment where you want maximum inference speed.
    
    Args:
        pipeline: Diffusers pipeline with LoRA loaded.
        adapter_name: Name of adapter to merge.
        safe_merge: If True, check for NaN/Inf before merging.
    """
    if not hasattr(pipeline.unet, 'peft_config'):
        print("No LoRA adapters to merge.")
        return
    
    print(f"Merging LoRA '{adapter_name}' into base model...")
    print("WARNING: This operation is irreversible!")
    
    # Merge and unload
    pipeline.unet = pipeline.unet.merge_and_unload(
        safe_merge=safe_merge,
        adapter_names=[adapter_name] if adapter_name else None,
    )
    
    print("LoRA merged successfully. Model is now a standard UNet.")


def get_lora_state(pipeline) -> Dict[str, Any]:
    """
    Get current LoRA state of a pipeline.
    
    Args:
        pipeline: Diffusers pipeline.
    
    Returns:
        Dictionary with current LoRA state:
        {
            "has_lora": bool,
            "active_adapters": list,
            "scales": dict,
        }
    """
    has_lora = hasattr(pipeline.unet, 'peft_config')
    
    if not has_lora:
        return {
            "has_lora": False,
            "active_adapters": [],
            "scales": {},
        }
    
    active = list(pipeline.unet.peft_config.keys()) if has_lora else []
    scales = getattr(pipeline, '_lora_scales', {})
    
    return {
        "has_lora": True,
        "active_adapters": active,
        "scales": scales,
    }


if __name__ == "__main__":
    # Quick test
    print("Testing LoRA utilities...")
    
    test_path = Path("lora_checkpoints")
    if test_path.exists():
        print("\n1. Testing metadata extraction...")
        meta = get_lora_metadata(test_path)
        print(f"   Rank: {meta['rank']}")
        print(f"   Alpha: {meta['alpha']}")
        print(f"   Target modules: {meta['target_modules']}")
        
        print("\n2. Listing available LoRAs...")
        loras = list_available_loras(test_path)
        for lora in loras:
            print(f"   - {lora['name']}: rank={lora['rank']}")
    else:
        print(f"Test directory {test_path} not found.")
    
    print("\nLoRA utilities test complete!")
