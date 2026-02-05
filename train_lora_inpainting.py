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

from dataset_fix import WallInpaintingDataset, WallInpaintingCollator, dominant_color_to_pil
from dataset_inpainting import prepare_mask_and_masked_image
from validation_utils import (
    ValidationVisualizer, 
    ValidationSample, 
    compute_color_fidelity_metrics,
    tensor_to_pil,
    mask_tensor_to_pil,
    depth_tensor_to_pil,
    create_segment_overlay,
)
from ip_adapter_utils import IPAdapterImageEncoder, freeze_ip_adapter_weights

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
    """Collate function for DataLoader with reference image support."""
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    masks = torch.stack([e["mask"] for e in examples])
    
    # Reference images (solid color from dominant color extraction)
    reference_images = torch.stack([e["reference_image"] for e in examples]) if "reference_image" in examples[0] else None
    dominant_colors = torch.stack([e["dominant_color"] for e in examples]) if "dominant_color" in examples[0] else None
    
    # Tokenize captions (empty for Zero-Prompt strategy)
    captions = [e.get("caption", "") for e in examples]
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    batch = {
        "pixel_values": pixel_values,
        "masks": masks,
        "input_ids": inputs.input_ids,
    }
    
    if reference_images is not None:
        batch["reference_images"] = reference_images
    if dominant_colors is not None:
        batch["dominant_colors"] = dominant_colors
    
    return batch


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
    
    # Define weight dtype early (needed for IP-Adapter encoder)
    weight_dtype = torch.float16 if config["training"]["mixed_precision"] == "fp16" else torch.float32
    
    # ========== IP-ADAPTER INTEGRATION ==========
    # Load IP-Adapter for reference-based training (following Paint-by-Example approach)
    ip_adapter_config = config.get("ip_adapter", {})
    use_ip_adapter = ip_adapter_config.get("enabled", False)
    ip_adapter_encoder = None
    
    if use_ip_adapter:
        logger.info("Setting up IP-Adapter for training...")
        
        # Create temporary pipeline to load IP-Adapter weights into UNet
        from diffusers import StableDiffusionInpaintPipeline
        temp_pipe = StableDiffusionInpaintPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=noise_scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        
        # Load IP-Adapter into UNet
        temp_pipe.load_ip_adapter(
            ip_adapter_config.get("model_id", "h94/IP-Adapter"),
            subfolder="models",
            weight_name=ip_adapter_config.get("weight_name", "ip-adapter-plus_sd15.bin"),
        )
        temp_pipe.set_ip_adapter_scale(ip_adapter_config.get("scale", 1.0))
        
        # Get the modified UNet with IP-Adapter
        unet = temp_pipe.unet
        
        # Freeze IP-Adapter weights (Sequential Training approach)
        if ip_adapter_config.get("freeze_weights", True):
            freeze_ip_adapter_weights(unet)
            logger.info("IP-Adapter weights frozen - LoRA will learn to collaborate")
        
        # Create image encoder for reference images
        ip_adapter_encoder = IPAdapterImageEncoder(
            device=str(accelerator.device),
            dtype=weight_dtype,
        )
        ip_adapter_encoder.load()
        
        logger.info(f"IP-Adapter enabled with scale={ip_adapter_config.get('scale', 1.0)}")
        
        # Clean up temp pipeline (keep UNet)
        del temp_pipe
        gc.collect()
        torch.cuda.empty_cache()
    else:
        logger.info("IP-Adapter disabled - training without reference conditioning")
    # ========== END IP-ADAPTER INTEGRATION ==========
    
    # Move models to device (weight_dtype already defined above)
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
    
    # Setup dataset with FIXED dominant color extraction
    # Read reference config (with defaults for backward compatibility)
    ref_config = config.get("reference", {})
    logger.info("Loading dataset with hybrid color strategy...")
    logger.info(f"  Random color prob: {ref_config.get('random_color_prob', 0.0):.0%}")
    logger.info(f"  Mask erosion: {ref_config.get('mask_erosion_size', 0)}px")
    
    train_dataset = WallInpaintingDataset(
        data_root=Path(config["dataset"]["train_data_dir"]).parent,
        split="train",
        resolution=config["dataset"]["resolution"],
        center_crop=config["dataset"]["center_crop"],
        random_flip=config["dataset"]["random_flip"],
        quality_threshold=config["dataset"]["quality_threshold"],
        max_samples=args.subset,
        # Reference image generation config
        color_extraction_method=ref_config.get("color_extraction_method", "median"),
        random_color_prob=ref_config.get("random_color_prob", 0.0),
        color_jitter_prob=ref_config.get("color_jitter_prob", 0.0),
        color_jitter_range=ref_config.get("color_jitter_range", 10.0),
        add_reference_texture=ref_config.get("add_reference_texture", True),
        texture_noise_std=ref_config.get("texture_noise_std", 8.0),
        add_lighting_gradient=ref_config.get("add_lighting_gradient", True),
        mask_erosion_size=ref_config.get("mask_erosion_size", 0),
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
                # Check for unconditional training mode (Zero-Prompt strategy)
                if config["training"].get("unconditional_training", False):
                    # Use pre-computed unconditional embeddings
                    encoder_hidden_states = get_unconditional_embedding(
                        tokenizer, text_encoder, accelerator.device, weight_dtype
                    ).expand(bsz, -1, -1)
                else:
                    # Standard text-conditioned training
                    encoder_hidden_states = text_encoder(
                        batch["input_ids"].to(accelerator.device)
                    )[0].to(weight_dtype)
                
                # Predict noise
                # If IP-Adapter is enabled, encode reference images and pass to UNet
                added_cond_kwargs = None
                if use_ip_adapter and ip_adapter_encoder is not None:
                    # Get reference images from batch
                    if "reference_images" in batch:
                        ref_images = batch["reference_images"].to(accelerator.device, dtype=weight_dtype)
                        # Encode reference images
                        image_embeds = ip_adapter_encoder.encode_batch(ref_images)
                        added_cond_kwargs = {"image_embeds": image_embeds}
                
                noise_pred = unet(
                    latent_model_input,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample
                
                # Calculate loss (noise prediction loss)
                noise_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                
                # Optional: Color consistency loss
                color_loss_config = config.get("color_loss", {})
                use_color_loss = color_loss_config.get("enabled", False)
                color_loss = torch.tensor(0.0, device=accelerator.device)
                
                if use_color_loss and global_step >= color_loss_config.get("start_step", 0):
                    # Decode predicted clean latents to compute color loss
                    # x0_pred = (noisy_latents - sqrt(1-alpha) * noise_pred) / sqrt(alpha)
                    alpha_t = noise_scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(noisy_latents.device)
                    x0_pred = (noisy_latents - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
                    
                    # Decode to image space (with gradient)
                    with torch.cuda.amp.autocast(enabled=False):
                        decoded = vae.decode(x0_pred.float() / vae.config.scaling_factor).sample
                    
                    # Compute color loss on masked region (LAB space approximation)
                    if "dominant_colors" in batch:
                        target_colors = batch["dominant_colors"].to(accelerator.device).float() / 255.0  # [B, 3] in [0,1]
                        target_colors = target_colors * 2 - 1  # [0,1] -> [-1,1]
                        
                        # Expand to match image size
                        target_expanded = target_colors.view(-1, 3, 1, 1).expand_as(decoded)
                        
                        # Apply mask
                        mask_expanded = F.interpolate(batch["masks"], size=decoded.shape[-2:], mode="nearest")
                        masked_decoded = decoded * mask_expanded
                        masked_target = target_expanded * mask_expanded
                        
                        # Color loss in masked region
                        color_loss = F.mse_loss(masked_decoded, masked_target)
                
                # Total loss
                color_weight = color_loss_config.get("weight", 0.1)
                loss = noise_loss + color_weight * color_loss
                
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


def get_unconditional_embedding(tokenizer, text_encoder, device, dtype):
    """
    Generate the unconditional embedding (empty prompt).
    This is used for "Zero-Prompt" training to force reliance on image conditioning.
    """
    uncond_tokens = tokenizer(
        [""],  # Empty prompt
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(
            uncond_tokens.input_ids.to(device)
        )[0].to(dtype)
    return uncond_embeddings  # Shape: [1, 77, 768]


def generate_validation_samples(
    accelerator, unet, vae, text_encoder, tokenizer,
    noise_scheduler, config, output_dir, global_step, weight_dtype
):
    """
    Generate comprehensive validation with configurable samples and full input/output grids.
    
    Creates multi-column grids: Source | Reference | Mask | Segment | Depth | Output
    Computes color fidelity metrics in LAB/HSV space.
    """
    try:
        num_samples = config["validation"].get("num_validation_samples", 50)
        
        # Initialize visualizer
        visualizer = ValidationVisualizer(
            output_dir=output_dir,
            resolution=config["dataset"]["resolution"],
            num_samples=num_samples,
        )
        
        # Create pipeline for inference
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
        
        # Load validation dataset with FIXED dominant color extraction
        val_dataset = WallInpaintingDataset(
            data_root=Path(config["dataset"]["train_data_dir"]).parent,
            split="validation",
            resolution=config["dataset"]["resolution"],
            max_samples=num_samples,
            color_extraction_method="median",
            add_reference_texture=True,
        )
        
        generator = torch.Generator(device=accelerator.device).manual_seed(42)
        samples = []
        all_metrics = []
        
        actual_samples = min(num_samples, len(val_dataset))
        logger.info(f"Generating {actual_samples} validation samples...")
        
        for i in range(actual_samples):
            sample_data = val_dataset[i]
            
            # Convert tensors to PIL
            source_image = tensor_to_pil(sample_data["pixel_values"])
            mask_image = mask_tensor_to_pil(sample_data["mask"])
            prompt = sample_data.get("caption", "")
            
            # FIXED: Use solid color reference from dominant color extraction
            dominant_color = tuple(sample_data["dominant_color"].tolist())
            reference_image = dominant_color_to_pil(dominant_color, size=(512, 512))
            
            # Depth map placeholder (add depth estimation if available)
            depth_map = None
            
            # Run inference with empty prompt for Zero-Prompt strategy
            inference_prompt = "" if config["training"].get("unconditional_training", False) else prompt
            with torch.autocast("cuda"):
                result = pipeline(
                    prompt=inference_prompt,
                    image=source_image,
                    mask_image=mask_image,
                    num_inference_steps=20,
                    generator=generator,
                ).images[0]
            
            # Create segment overlay
            segment_overlay = create_segment_overlay(source_image, mask_image)
            
            # Compute color fidelity metrics (reference is solid color now)
            metrics = compute_color_fidelity_metrics(reference_image, result, mask_image)
            all_metrics.append(metrics)
            
            # Create sample object
            samples.append(ValidationSample(
                source_image=source_image,
                reference_image=reference_image,
                mask=mask_image,
                segment_overlay=segment_overlay,
                depth_map=depth_map,
                model_output=result,
                prompt=prompt,
                sample_id=i,
            ))
        
        # Create comparison sheets (10 samples per sheet)
        saved_paths = visualizer.create_comparison_sheet(samples, global_step, samples_per_sheet=10)
        
        # Aggregate metrics
        if all_metrics:
            avg_metrics = {
                f"val/{k}_mean": float(np.mean([m[k] for m in all_metrics]))
                for k in all_metrics[0].keys()
            }
            
            # Log to accelerator
            accelerator.log(avg_metrics, step=global_step)
            
            # Save metrics file
            visualizer.log_metrics(samples, global_step, avg_metrics)
            
            logger.info(
                f"Color metrics: LAB={avg_metrics.get('val/lab_distance_mean', 0):.2f}, "
                f"Hue={avg_metrics.get('val/hue_error_mean', 0):.2f}"
            )
        
        logger.info(f"Saved {len(saved_paths)} validation sheets to {visualizer.output_dir}")
        
    except Exception as e:
        logger.warning(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
