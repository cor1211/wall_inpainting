#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for Stable Diffusion Inpainting.

Trains LoRA adapters on UNet for wall texture/color inpainting task.

Usage:
    # Dry run
    python train_lora_inpainting.py --dry-run
    
    # Quick test
    python train_lora_inpainting.py --max-steps 100 --subset 100
    
    # Full training with accelerate
    accelerate launch --mixed_precision=fp16 train_lora_inpainting.py \
        --config configs/lora_training.yaml
"""
import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import gc
import json
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from dataset_inpainting import InpaintingDataset, InpaintingCollator, prepare_mask_and_masked_image

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for SD Inpainting")
    
    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default="configs/lora_training.yaml",
        help="Path to config file",
    )
    
    # Override options
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--subset", type=int, default=None, help="Use subset of data")
    
    # Flags
    parser.add_argument("--dry-run", action="store_true", help="Test without training")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def collate_fn(examples, tokenizer):
    """Collate function for DataLoader."""
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    masks = torch.stack([e["mask"] for e in examples])
    
    # Tokenize captions
    captions = [e["caption"] for e in examples]
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    return {
        "pixel_values": pixel_values,
        "masks": masks,
        "input_ids": inputs.input_ids,
    }


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line args
    if args.output_dir:
        config["checkpointing"]["output_dir"] = args.output_dir
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.max_steps:
        config["training"]["max_train_steps"] = args.max_steps
    if args.batch_size:
        config["training"]["train_batch_size"] = args.batch_size
    if args.seed:
        config["training"]["seed"] = args.seed
    
    # Setup output directory
    output_dir = Path(config["checkpointing"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup accelerator
    accelerator_project_config = ProjectConfiguration(
        project_dir=str(output_dir),
        logging_dir=str(output_dir / "logs"),
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        mixed_precision=config["training"]["mixed_precision"],
        log_with=config["logging"]["report_to"],
        project_config=accelerator_project_config,
    )
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    
    # Set seed for reproducibility
    if config["training"]["seed"] is not None:
        set_seed(config["training"]["seed"])
    
    # Load tokenizer and text encoder
    logger.info("Loading tokenizer and text encoder...")
    tokenizer = CLIPTokenizer.from_pretrained(
        config["model"]["pretrained_model_name_or_path"],
        subfolder="tokenizer",
    )
    text_encoder = CLIPTextModel.from_pretrained(
        config["model"]["pretrained_model_name_or_path"],
        subfolder="text_encoder",
    )
    text_encoder.requires_grad_(False)
    
    # Load VAE
    logger.info("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        config["model"]["pretrained_model_name_or_path"],
        subfolder="vae",
    )
    vae.requires_grad_(False)
    
    # VAE memory optimization
    vae.enable_slicing()  # Process in slices to save VRAM
    vae.enable_tiling()   # Process large images in tiles
    
    # Load UNet
    logger.info("Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(
        config["model"]["pretrained_model_name_or_path"],
        subfolder="unet",
    )
    
    # Configure LoRA
    logger.info("Configuring LoRA...")
    lora_config = LoraConfig(
        r=config["model"]["lora_rank"],
        lora_alpha=config["model"]["lora_alpha"],
        init_lora_weights="gaussian",
        target_modules=config["model"]["target_modules"],
        lora_dropout=config["model"].get("lora_dropout", 0.0),
    )
    unet = get_peft_model(unet, lora_config)
    
    # Print trainable parameters
    unet.print_trainable_parameters()
    
    # Load noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        config["model"]["pretrained_model_name_or_path"],
        subfolder="scheduler",
    )
    
    # Enable gradient checkpointing
    if config["training"]["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()
    
    # Enable xformers if available
    if config["training"].get("enable_xformers", False):
        try:
            unet.enable_xformers_memory_efficient_attention()
            logger.info("xformers enabled")
        except Exception as e:
            logger.warning(f"xformers not available: {e}")
    
    # Move models to device
    weight_dtype = torch.float16 if config["training"]["mixed_precision"] == "fp16" else torch.float32
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # Setup optimizer (8-bit Adam for VRAM savings if available)
    use_8bit_adam = config["training"].get("use_8bit_adam", False)
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
            logger.info("Using 8-bit AdamW optimizer for VRAM savings")
        except ImportError:
            logger.warning("bitsandbytes not found, falling back to standard AdamW")
            optimizer_cls = torch.optim.AdamW
    else:
        optimizer_cls = torch.optim.AdamW
    
    lora_layers = list(filter(lambda p: p.requires_grad, unet.parameters()))
    optimizer = optimizer_cls(
        lora_layers,
        lr=config["training"]["learning_rate"],
        betas=(config["training"]["adam_beta1"], config["training"]["adam_beta2"]),
        weight_decay=config["training"]["adam_weight_decay"],
        eps=config["training"]["adam_epsilon"],
    )
    
    # Setup dataset
    logger.info("Loading dataset...")
    train_dataset = InpaintingDataset(
        data_root=Path(config["dataset"]["train_data_dir"]).parent,
        split="train",
        resolution=config["dataset"]["resolution"],
        center_crop=config["dataset"]["center_crop"],
        random_flip=config["dataset"]["random_flip"],
        quality_threshold=config["dataset"]["quality_threshold"],
        max_samples=args.subset,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["training"]["train_batch_size"],
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, tokenizer),
        num_workers=config["dataset"]["dataloader_num_workers"],
    )
    
    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config["training"]["gradient_accumulation_steps"]
    )
    max_train_steps = config["training"]["max_train_steps"]
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    
    logger.info(f"Dataset size: {len(train_dataset)}")
    logger.info(f"Batch size: {config['training']['train_batch_size']}")
    logger.info(f"Gradient accumulation: {config['training']['gradient_accumulation_steps']}")
    logger.info(f"Effective batch size: {config['training']['train_batch_size'] * config['training']['gradient_accumulation_steps']}")
    logger.info(f"Num epochs: {num_train_epochs}")
    logger.info(f"Max train steps: {max_train_steps}")
    
    # Setup learning rate scheduler
    lr_scheduler = get_scheduler(
        config["training"]["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=config["training"]["lr_warmup_steps"],
        num_training_steps=max_train_steps,
    )
    
    # Prepare with accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # Dry run mode
    if args.dry_run:
        logger.info("=== DRY RUN MODE ===")
        logger.info("Testing data loading...")
        batch = next(iter(train_dataloader))
        logger.info(f"Batch keys: {batch.keys()}")
        logger.info(f"pixel_values shape: {batch['pixel_values'].shape}")
        logger.info(f"masks shape: {batch['masks'].shape}")
        logger.info(f"input_ids shape: {batch['input_ids'].shape}")
        logger.info("=== DRY RUN COMPLETE ===")
        return
    
    # Training loop
    logger.info("***** Starting training *****")
    global_step = 0
    initial_global_step = 0
    
    # Resume from checkpoint if specified
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            logger.info(f"Resuming from checkpoint: {resume_path}")
            accelerator.load_state(str(resume_path))
            
            # Try to restore global_step from training_state.json
            state_file = resume_path.parent / "training_state.json" if "checkpoint" in resume_path.name else resume_path / "training_state.json"
            if not state_file.exists():
                state_file = output_dir / "training_state.json"
            
            if state_file.exists():
                with open(state_file, "r") as f:
                    training_state = json.load(f)
                global_step = training_state.get("global_step", 0)
                initial_global_step = global_step
                logger.info(f"Resumed at global_step={global_step}")
            else:
                # Try to infer from checkpoint name
                try:
                    global_step = int(resume_path.name.split("-")[-1])
                    initial_global_step = global_step
                    logger.info(f"Inferred global_step={global_step} from checkpoint name")
                except ValueError:
                    logger.warning("Could not determine global_step, starting from 0")
        else:
            logger.warning(f"Resume path not found: {resume_path}")
    
    progress_bar = tqdm(
        range(initial_global_step, max_train_steps),
        desc="Training",
        initial=initial_global_step,
        disable=not accelerator.is_local_main_process,
    )
    
    for epoch in range(num_train_epochs):
        unet.train()
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Get latents
                latents = vae.encode(
                    batch["pixel_values"].to(weight_dtype)
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Get mask and masked image latents
                mask = batch["masks"].to(weight_dtype)
                masked_image = batch["pixel_values"].to(weight_dtype) * (1 - mask)
                masked_image_latents = vae.encode(masked_image).latent_dist.sample()
                masked_image_latents = masked_image_latents * vae.config.scaling_factor
                
                # Resize mask to latent size
                mask = F.interpolate(
                    mask, 
                    size=(latents.shape[2], latents.shape[3]),
                    mode="nearest",
                )
                
                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Sample timesteps
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device
                ).long()
                
                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Concatenate latents for inpainting UNet
                # UNet expects: [noisy_latents, mask, masked_image_latents]
                latent_model_input = torch.cat(
                    [noisy_latents, mask, masked_image_latents], dim=1
                )
                
                # Get text embeddings
                encoder_hidden_states = text_encoder(
                    batch["input_ids"].to(accelerator.device)
                )[0].to(weight_dtype)
                
                # Predict noise
                noise_pred = unet(
                    latent_model_input,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample
                
                # Calculate loss
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                
                # Backprop
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), config["training"]["max_grad_norm"])
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # Log metrics
                accelerator.log(
                    {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]},
                    step=global_step,
                )
                
                if global_step % config["checkpointing"]["checkpointing_steps"] == 0:
                    if accelerator.is_main_process:
                        save_path = output_dir / f"checkpoint-{global_step}"
                        accelerator.save_state(str(save_path))
                        
                        # Save training state metadata
                        training_state = {
                            "global_step": global_step,
                            "epoch": epoch,
                            "loss": loss.detach().item(),
                            "learning_rate": lr_scheduler.get_last_lr()[0],
                            "checkpoint_path": str(save_path),
                        }
                        state_file = output_dir / "training_state.json"
                        with open(state_file, "w") as f:
                            json.dump(training_state, f, indent=2)
                        
                        logger.info(f"Saved checkpoint to {save_path}")
                
                # Validation
                if global_step % config["validation"]["validation_steps"] == 0:
                    if accelerator.is_main_process:
                        logger.info("Running validation...")
                        generate_validation_samples(
                            accelerator, unet, vae, text_encoder, tokenizer,
                            noise_scheduler, config, output_dir, global_step, weight_dtype
                        )
            
            if global_step >= max_train_steps:
                break
        
        if global_step >= max_train_steps:
            break
    
    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet_unwrapped = accelerator.unwrap_model(unet)
        unet_unwrapped.save_pretrained(output_dir)
        
        # Save final training state
        final_state = {
            "global_step": global_step,
            "status": "completed",
            "total_steps": max_train_steps,
        }
        with open(output_dir / "training_state.json", "w") as f:
            json.dump(final_state, f, indent=2)
        
        logger.info(f"Final model saved to {output_dir}")
    
    accelerator.end_training()
    logger.info("Training complete!")


def generate_validation_samples(
    accelerator, unet, vae, text_encoder, tokenizer,
    noise_scheduler, config, output_dir, global_step, weight_dtype
):
    """Generate validation samples during training using real validation data."""
    try:
        # Create a simple pipeline for inference
        pipeline = StableDiffusionInpaintPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=accelerator.unwrap_model(unet),
            scheduler=noise_scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        pipeline.set_progress_bar_config(disable=True)
        
        # Load validation examples
        val_dir = output_dir / "validation_samples"
        val_dir.mkdir(exist_ok=True)
        
        # Helper to load dataset samples
        val_dataset_path = Path(config["dataset"]["train_data_dir"]).parent / "validation"
        val_dataset = InpaintingDataset(
            data_root=Path(config["dataset"]["train_data_dir"]).parent,
            split="validation",
            resolution=config["dataset"]["resolution"],
            max_samples=4
        )
        
        generator = torch.Generator(device=accelerator.device).manual_seed(42)
        
        logger.info(f"Generating {len(val_dataset)} validation samples...")
        
        for i in range(min(4, len(val_dataset))):
            sample = val_dataset[i]
            
            # Prepare inputs (convert back from tensor for pipeline)
            # Tensor is [-1, 1], convert to PIL [0, 255]
            image_tensor = sample["pixel_values"]
            image_np = ((image_tensor.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
            val_image = Image.fromarray(image_np)
            
            # Mask tensor is [1, H, W], convert to PIL L
            mask_tensor = sample["mask"]
            mask_np = (mask_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
            val_mask = Image.fromarray(mask_np, mode="L")
            
            prompt = sample["caption"]
            
            with torch.autocast("cuda"):
                result = pipeline(
                    prompt=prompt,
                    image=val_image,
                    mask_image=val_mask,
                    num_inference_steps=20,
                    generator=generator,
                ).images[0]
            
            # Save grid: Original | Mask | Result
            w, h = val_image.size
            grid = Image.new("RGB", (w * 3, h))
            grid.paste(val_image, (0, 0))
            grid.paste(val_mask.convert("RGB"), (w, 0))
            grid.paste(result, (w * 2, 0))
            
            grid.save(val_dir / f"step_{global_step}_sample_{i}.png")
        
        logger.info(f"Saved validation samples to {val_dir}")
        
    except Exception as e:
        logger.warning(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
