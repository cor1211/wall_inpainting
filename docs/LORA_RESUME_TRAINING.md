# LoRA Resume Training Guide

## Overview

This guide covers how to resume training from checkpoints and manage training state.

## Resume Training

### From Intermediate Checkpoint

```bash
# Resume from specific checkpoint
python train_lora_inpainting.py \
    --resume lora_checkpoints/checkpoint-2000 \
    --max-steps 5000
```

### What Gets Restored

| Component | File | Restored |
|-----------|------|----------|
| LoRA weights | `model.safetensors` | ✅ |
| Optimizer state | `optimizer.bin` | ✅ |
| LR scheduler | `scheduler.bin` | ✅ |
| AMP scaler | `scaler.pt` | ✅ |
| RNG states | `random_states_0.pkl` | ✅ |
| Global step | `training_state.json` | ✅ |

## Resume vs Fine-tune

| Scenario | Command | Effect |
|----------|---------|--------|
| **Resume** | `--resume checkpoint-2000` | Continue from step 2000 with optimizer state |
| **Fine-tune** | No `--resume` flag | Start fresh with pretrained LoRA as init |

### Fine-tune from Existing LoRA

```python
# In config, set:
checkpointing:
  resume_from_checkpoint: "lora_checkpoints"
```

This loads LoRA weights but resets optimizer (useful for domain adaptation).

## Checkpoint Structure

```
lora_checkpoints/
├── adapter_config.json        # LoRA config (always present)
├── adapter_model.safetensors  # Final/best weights
├── training_state.json        # Training progress
│   └── {global_step, epoch, loss, status}
├── checkpoint-1000/
│   ├── model.safetensors      # Full accelerator state
│   ├── optimizer.bin
│   ├── scheduler.bin
│   ├── scaler.pt
│   └── random_states_0.pkl
└── validation_samples/
    └── step_1000_sample_0.png
```

## Training State Metadata

`training_state.json`:
```json
{
  "global_step": 2000,
  "epoch": 1,
  "loss": 0.0234,
  "learning_rate": 9.8e-5,
  "checkpoint_path": "lora_checkpoints/checkpoint-2000",
  "status": "completed"
}
```

## Best Practices

1. **Regular checkpoints**: Set `checkpointing_steps: 1000` or lower
2. **Keep metadata**: Don't delete `training_state.json`
3. **Verify resume**: Check logs for "Resumed at global_step=N"
4. **Cleanup old checkpoints**: Keep only last 2-3 for disk space

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Could not determine global_step" | Ensure `training_state.json` exists |
| Training restarts from 0 | Check `--resume` path is correct |
| Loss spikes after resume | Normal; optimizer momentum rebuilds |
