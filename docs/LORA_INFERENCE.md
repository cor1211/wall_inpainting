# LoRA Inference Guide

## Overview

This guide explains how to use LoRA (Low-Rank Adaptation) checkpoints with the wall inpainting pipeline for improved inference quality.

## Quick Start

### CLI Usage

```bash
# Basic inference with LoRA
python main.py --source room.jpg --color "200,180,160" \
    --lora lora_checkpoints --output result.png

# Adjust LoRA strength
python main.py --source room.jpg --color "200,180,160" \
    --lora lora_checkpoints --lora-scale 0.8
```

### Python API

```python
from pipeline import WallReskinPipeline

# Initialize with LoRA
pipeline = WallReskinPipeline(
    lora_path="lora_checkpoints",
    lora_scale=1.0,  # 0.0-1.0, higher = stronger effect
)

# Process image
result = pipeline.process(
    source_image="room.jpg",
    mask_image="mask.png",
    reference_image="reference.jpg",
)
result.save("output.png")
```

## Configuration

### Load Order

```
Base Model (SD Inpainting)
    ↓
ControlNet Depth
    ↓
IP-Adapter
    ↓
LoRA Weights  ← Applied last
```

### Scale Parameter

| Scale | Effect |
|-------|--------|
| 0.0 | LoRA disabled (base model only) |
| 0.5 | Moderate enhancement |
| 0.8 | Recommended default |
| 1.0 | Full LoRA strength |

## Dynamic LoRA Switching

```python
# Change LoRA at runtime
pipeline.set_lora("path/to/other_lora", scale=0.7)

# Unload LoRA entirely
pipeline.set_lora(None)
```

## Available Functions

```python
from lora_utils import (
    load_lora_weights,    # Load LoRA into pipeline
    unload_lora,          # Remove LoRA
    get_lora_metadata,    # Get LoRA config info
    list_available_loras, # List all LoRAs in directory
    merge_lora_weights,   # Permanently merge into base (irreversible)
)
```

## Checkpoint Structure

```
lora_checkpoints/
├── adapter_config.json        # LoRA architecture (rank, alpha, targets)
├── adapter_model.safetensors  # Trained weights (~6MB for r=8)
└── training_state.json        # Training metadata
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError` | Ensure `adapter_config.json` exists in path |
| Poor quality output | Try adjusting `lora_scale` (0.5-1.0) |
| VRAM issues | LoRA adds ~10-50MB; consider `enable_cpu_offload=True` |
