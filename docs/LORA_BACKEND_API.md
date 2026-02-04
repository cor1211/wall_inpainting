# LoRA Backend API Integration

## Overview

The FastAPI backend automatically loads LoRA checkpoints at startup based on configuration.

## Configuration

### config.py

```python
from config import config

# Enable/disable LoRA
config.lora.enabled = True

# Default checkpoint path
config.lora.default_path = "lora_checkpoints"

# Default strength
config.lora.default_scale = 1.0
```

### Environment Variables (optional)

```bash
export LORA_ENABLED=true
export LORA_PATH=lora_checkpoints
export LORA_SCALE=0.8
```

## Startup Behavior

```
API Server Start
    ↓
Check config.lora.enabled
    ↓
If enabled → Validate lora_checkpoints/ exists
    ↓
Load WallReskinPipeline with LoRA
    ↓
Ready to serve requests
```

## API Endpoints

### Health Check (with LoRA status)

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "cuda_available": true,
  "device": "cuda"
}
```

### Process with LoRA-enhanced pipeline

```bash
curl -X POST "http://localhost:8000/process-color" \
    -F "source=@room.jpg" \
    -F "color=200,180,160" \
    -o result.png
```

## Thread Safety

The pipeline uses a global singleton pattern:

```python
_pipeline: Optional[WallReskinPipeline] = None

def get_pipeline() -> WallReskinPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = WallReskinPipeline(...)
    return _pipeline
```

**Key points:**
- Model loaded once at first request
- Same instance reused across requests
- No VRAM leak from repeated loading
- Thread-safe for concurrent requests (GPU serializes inference)

## VRAM Management

| Component | Approximate VRAM |
|-----------|-----------------|
| Base SD Inpainting | ~4GB |
| ControlNet | ~1GB |
| LoRA weights | ~10-50MB |
| **Total** | **~5-6GB** |

## Disabling LoRA

To run without LoRA:

```python
# In config.py or at runtime
config.lora.enabled = False
```

Or delete/rename the `lora_checkpoints` directory.
