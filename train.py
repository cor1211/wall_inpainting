
import os
import argparse
import logging
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm

# Fix for Windows OMP/CUDA issues
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

# Import our custom modules
from dataset.wall_paint_dataset import WallPaintDataset

# Hybrid Loader
from models.wall_recoloring_pipeline import get_wall_recoloring_pipeline

def validate_and_save(accelerator, unet, controlnet, vae, text_encoder, tokenizer, image_encoder, val_dataloader, global_step, weight_dtype, output_dir):
    """
    Validation step: Generate images using the trained UNet.
    
    Validation follows the same pipeline as inference:
    - Input: Random noise latents
    - Conditions: Source image (ControlNet), Color reference (IP-Adapter), Mask (Inpainting)
    - Output: Generated image with new wall color
    """
    if not accelerator.is_local_main_process:
        return
        
    unet.eval()
    print(f"\nRunning Validation at Step {global_step}...")
    
    val_dir = os.path.join(output_dir, "validation_images")
    os.makedirs(val_dir, exist_ok=True)
    
    # Prepare Scheduler (DDIM for Inference)
    from diffusers import DDIMScheduler
    resolution = 512
    device = accelerator.device
    
    # Use DDIMScheduler for faster inference
    val_scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-inpainting", subfolder="scheduler")
    
    try:
        batch = next(iter(val_dataloader))
    except StopIteration:
        return

    # Use max 2 samples for validation
    if batch['color_patches'].shape[0] > 2:
        for k in batch:
            if isinstance(batch[k], torch.Tensor) or isinstance(batch[k], list):
                 batch[k] = batch[k][:2]
    
    bsz = batch['color_patches'].shape[0]
    
    with torch.no_grad():
        # 1. Prepare Latents (Random Noise - Start from scratch)
        latents = torch.randn(
            (bsz, 4, resolution // 8, resolution // 8),
            device=device,
            dtype=weight_dtype
        )
        
        # 2. Text Embeddings
        inputs = tokenizer(
            batch["prompts"], 
            max_length=tokenizer.model_max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        encoder_hidden_states = text_encoder(inputs.input_ids.to(device))[0]
        
        # 3. ControlNet Condition: Use SOURCE image (old wall) for structure preservation
        # conditional_images is already preprocessed (depth/canny) from dataset
        control_images = batch["conditional_images"].to(device, dtype=weight_dtype)
        # Convert to [-1, 1] if needed (ControlNet expects normalized)
        if control_images.max() <= 1.0:
            control_images = control_images * 2.0 - 1.0
        
        # 4. IP-Adapter: Use COLOR REFERENCE (new color) for color transfer
        pixel_values_ip = batch["color_patches"].to(device, dtype=weight_dtype)
        
        # Resize to 224x224 for CLIP (if not already)
        if pixel_values_ip.shape[2] != 224:
            pixel_values_ip = F.interpolate(pixel_values_ip, size=(224, 224), mode="bilinear", align_corners=False)
        
        # Normalize for CLIP Vision Encoder
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device, dtype=weight_dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device, dtype=weight_dtype).view(1, 3, 1, 1)
        pixel_values_ip = (pixel_values_ip - mean) / std
        # Encode color reference to image embeddings (IP-Adapter Plus uses hidden states)
        image_encoder_output = image_encoder(pixel_values_ip, output_hidden_states=True)
        image_embeds = image_encoder_output.hidden_states[-2] # (B, SeqLen, 1280)
        added_cond_kwargs = {"image_embeds": image_embeds}

        # 5. Inpainting Setup: Use SOURCE image (old wall) for masked source
        # This preserves the non-wall regions while generating new wall color
        masked_source_pixel = batch["masked_sources"].to(device, dtype=weight_dtype) * 2.0 - 1.0
        masked_latents = vae.encode(masked_source_pixel).latent_dist.sample() * vae.config.scaling_factor
        
        # Mask: Resize to latent size
        mask = batch["mask"].to(device, dtype=weight_dtype)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # Add channel dimension if needed
        mask_latents = F.interpolate(mask, size=latents.shape[2:], mode="nearest")
        
        # Set timesteps for denoising
        val_scheduler.set_timesteps(20)
        timesteps = val_scheduler.timesteps
        
        # Denoising loop
        with accelerator.autocast():
            for t in timesteps:
                # Concatenate Inpainting inputs: [noisy_latents, mask, masked_source_latents]
                unet_input = torch.cat([latents, mask_latents, masked_latents], dim=1)
                
                # ControlNet: Preserve structure from source image
                down_block_res_samples, mid_block_res_sample = controlnet(
                    latents,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=control_images,
                    return_dict=False,
                )
                
                # UNet: Predict noise with all conditions
                noise_pred = unet(
                    unet_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    added_cond_kwargs=added_cond_kwargs
                ).sample
                
                # Denoising step
                latents = val_scheduler.step(noise_pred, t, latents).prev_sample
            
        # Decode latents to image
        latents = 1 / vae.config.scaling_factor * latents
        latents = latents.to(weight_dtype)
        image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        
        # Save validation images
        from PIL import Image as PILImage
        import numpy as np
        for i in range(bsz):
            img = (image[i] * 255).astype(np.uint8)
            PILImage.fromarray(img).save(os.path.join(val_dir, f"step_{global_step}_sample_{i}.png"))
            
    print(f"Saved validation images to {val_dir}")
    unet.train()


def main():
    parser = argparse.ArgumentParser(description="Train Wall Recoloring Hybrid Model")
    parser.add_argument("--data_json", type=str, default="dataset_v2/train/metadata.jsonl")
    parser.add_argument("--validation_json", type=str, default="dataset_v2/validation/metadata.jsonl")
    parser.add_argument("--output_dir", type=str, default="output/wall_recolor_v1")
    parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-inpainting")
    parser.add_argument("--controlnet_model", type=str, default="lllyasviel/control_v11f1p_sd15_depth")
    parser.add_argument("--ip_adapter_scale", type=float, default=0.6)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--enable_xformers", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    
    args = parser.parse_args()
    
    # 1. Accelerator Setup
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, "logs"))
    # Handle "none" string from CLI
    log_with = args.report_to
    if log_with == "none":
        log_with = None
        
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=log_with,
        project_config=accelerator_project_config,
    )
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    if accelerator.is_local_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            
    set_seed(args.seed)
    
    # 2. Load Models
    # We load components using our loader logic but we need access to individual modules for training
    # The pipeline loader returns a pipeline, which extracts these.
    # To facilitate training, we'll load them directly or extract them.
    # Let's load the full pipeline then extract UNet for LoRA training.
    
    print("Loading Hybrid Pipeline...")
    # Using 'cpu' initially to avoid VRAM spike before we choose what to move
    pipeline = get_wall_recoloring_pipeline(
        base_model_path=args.base_model,
        controlnet_path=args.controlnet_model,
        ip_adapter_scale=args.ip_adapter_scale,
        device="cpu"
    )
    
    # Extract components
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    vae = pipeline.vae
    unet = pipeline.unet
    controlnet = pipeline.controlnet
    image_encoder = pipeline.image_encoder # CLIP Vision
    
    # Noise Scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler")
    
    # 3. Freeze Models & Setup LoRA
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.requires_grad_(False)
    image_encoder.requires_grad_(False)
    
    # UNet: Enable Gradients? 
    # For robust adaptation, we usually fine-tune UNet with LoRA.
    # We will use PEFT for LoRA.
    from peft import LoraConfig, get_peft_model
    
    unet.requires_grad_(False) # Freeze base UNet
    
    target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    
    # 3.1. Enable Gradient Checkpointing
    unet.enable_gradient_checkpointing()
    
    # 3.2. Ensure all other params are frozen (double check IP-Adapter)
    # peft handles LoRA, but we want to make sure the IP-Adapter weights 
    # (which are inside UNet attn processors) are frozen if we are NOT fine-tuning them.
    # By default, unet.requires_grad_(False) froze everything, then get_peft_model unfroze LoRA.
    # So IP-Adapter weights should be safe.
    # But let's verify:
    for name, param in unet.named_parameters():
        if "processor" in name and "lora" not in name:
             param.requires_grad = False

    
    # Enable xformers
    if args.enable_xformers:
        unet.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()
        
    # Optimizer
    # Optimizer: Standard AdamW for stability on Windows
    # bitsandbytes often causes DLL issues (WinError 1114)
    optimizer_cls = torch.optim.AdamW
    print("Using standard AdamW optimizer.")

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-2
    )
    

    
    # 4. Dataset
    # Use Depth for ControlNet (better structure preservation)
    train_dataset = WallPaintDataset(
        data_json=args.data_json,
        image_size=args.resolution,
        reconstruction_ratio=0.0,  # Always use target (new color) as GT, not source
        use_depth=True,  # Use depth map for ControlNet
        use_canny=False,
        random_flip=True
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        collate_fn=train_dataset.collate_fn,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False
    )

    # Learning Rate Scheduler (Must be after DataLoader to know length)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps,
    )

    # Validation Set
    val_dataset = WallPaintDataset(
        data_json=args.validation_json,
        image_size=args.resolution,
        reconstruction_ratio=0.0,  # Always use target (new color) as GT
        use_depth=True,  # Use depth map for ControlNet
        use_canny=False,
        random_flip=False  # No augmentation for validation
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=2,
        collate_fn=val_dataset.collate_fn,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False
    )

    
    # Prepare with Accelerator
    # Only prepare UNet, Optimizer, TrainDataloader, LR Scheduler. 
    # ValDataloader is manually iterated.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # Iterate
    global_step = 0
    
    # Move frozen models to device
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # Note: IP-Adapter weights are inside UNet now (in attn processors).
    # Since we loaded the pipeline, the IP-Adapter weights were injected into UNet.
    # Wait, get_peft_model wraps the UNet.
    # IP-Adapter modifies UNet's attention processors.
    # When we wrap with PEFT, does it preserve the custom IP-Adapter attention processors?
    # This is TRICKY.
    # Standard diffusers load_ip_adapter REPLACES attention processors.
    # PEFT wraps layers.
    # If we apply PEFT *after* loading IP-Adapter, PEFT usually wraps the Linear layers *inside* the attention blocks.
    # So provided IP-Adapter uses standard Linear layers in its custom attention, it might work.
    # BUT IP-Adapter AttnProcessor is custom.
    # A safer bet: Only LoRA fine-tune the standard self-attn/cross-attn layers, not the new IP-Adapter layers (unless we want to train them).
    # Since we passed target_modules=["to_k", ...], PEFT will find Linear layers with those names.
    # In standard UNet, those exist.
    # In IP-Adapter, the cross-attention to image is NEW.
    # So LoRA will apply to text-cross-attn and self-attn.
    # The IP-Adapter weights (image-cross-attn) will remain frozen (as per `unet.requires_grad_(False)`).
    # This is exactly "Phase 2" logic: Train UNet (LoRA) to coordinate, keep Adapters frozen.
    
    print("Starting Training...")
    print("=" * 80)
    print("TRAINING STRATEGY:")
    print("  - Input (noisy): TARGET image (new wall color) - This is the key!")
    print("  - Conditions:")
    print("    * ControlNet: SOURCE image (old wall) for structure preservation")
    print("    * IP-Adapter: COLOR REFERENCE (new color) for color transfer")
    print("    * Masked Source: SOURCE image (old wall) for inpainting context")
    print("  - Target (GT): TARGET image (new wall color)")
    print("=" * 80)
    
    for epoch in range(args.num_train_epochs):
        unet.train()
        progress_bar = tqdm(train_dataloader, disable=not accelerator.is_local_main_process)
        
        for batch in progress_bar:
            with accelerator.accumulate(unet):
                # ====================================================================
                # PHASE 1: PREPARE TARGET LATENTS (Ground Truth)
                # ====================================================================
                # Dataset returns: source, target, mask, color_patches, 
                #                  conditional_images, masked_sources, prompts
                # 
                # Training Strategy: Use TARGET image (new wall color) as GT
                # This teaches the model to generate new colors, not reconstruct old ones
                
                target_pixel_values = batch["target"].to(dtype=weight_dtype)  # [B, 3, H, W] (0-1)
                target_pixel_values_normalized = target_pixel_values * 2.0 - 1.0  # [-1, 1] for VAE
                
                # Encode TARGET image to latent space (this is our GT)
                with torch.no_grad():
                    target_latents = vae.encode(target_pixel_values_normalized).latent_dist.sample()
                    target_latents = target_latents * vae.config.scaling_factor  # [B, 4, 64, 64]
                
                # ====================================================================
                # PHASE 2: ADD NOISE TO TARGET (Diffusion Forward Process)
                # ====================================================================
                # Sample random noise and timestep
                noise = torch.randn_like(target_latents)  # ε ~ N(0, I) - This is the GT for loss
                bsz = target_latents.shape[0]
                timesteps = torch.randint(
                    0, 
                    noise_scheduler.config.num_train_timesteps, 
                    (bsz,), 
                    device=target_latents.device
                ).long()
                
                # Add noise to TARGET latents: z_t = √(α_t) * z_0 + √(1-α_t) * ε
                noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)
                
                # ====================================================================
                # PHASE 3: PREPARE CONDITIONS
                # ====================================================================
                
                # A. Text Embeddings
                inputs = tokenizer(
                    batch["prompts"], 
                    max_length=tokenizer.model_max_length, 
                    padding="max_length", 
                    truncation=True, 
                    return_tensors="pt"
                )
                encoder_hidden_states = text_encoder(inputs.input_ids.to(accelerator.device))[0]
                
                # B. ControlNet Condition: Use SOURCE image (old wall) for structure
                # conditional_images is depth map (or canny) preprocessed from SOURCE image
                control_images = batch["conditional_images"].to(dtype=weight_dtype)  # [B, 3, H, W] (0-1)
                # Convert to [-1, 1] for ControlNet
                control_images_normalized = control_images * 2.0 - 1.0
                
                # ControlNet forward: Preserve structure from source image
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,  # Use noisy TARGET latents
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=control_images_normalized,  # Depth/canny from SOURCE
                    return_dict=False,
                )
                
                # C. IP-Adapter Condition: Use COLOR REFERENCE (new color) for color transfer
                # color_patches is already [B, 3, 224, 224] (0-1) from dataset
                pixel_values_ip = batch["color_patches"].to(dtype=weight_dtype)
                
                # Ensure size is 224x224 (should already be from dataset)
                if pixel_values_ip.shape[2] != 224 or pixel_values_ip.shape[3] != 224:
                    pixel_values_ip = F.interpolate(
                        pixel_values_ip, 
                        size=(224, 224), 
                        mode="bilinear", 
                        align_corners=False
                    )
                
                # Normalize for CLIP Vision Encoder
                mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(
                    device=accelerator.device, 
                    dtype=weight_dtype
                ).view(1, 3, 1, 1)
                std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(
                    device=accelerator.device, 
                    dtype=weight_dtype
                ).view(1, 3, 1, 1)
                pixel_values_ip_normalized = (pixel_values_ip - mean) / std
                
                # Encode color reference to image embeddings
                # IP-Adapter Plus expects hidden states from penultimate layer, NOT pooled embeds
                image_encoder_output = image_encoder(pixel_values_ip_normalized, output_hidden_states=True)
                image_embeds = image_encoder_output.hidden_states[-2]  # [B, 257, 1280]
                added_cond_kwargs = {"image_embeds": image_embeds}
                
                # ====================================================================
                # PHASE 4: PREPARE INPAINTING INPUTS
                # ====================================================================
                # For inpainting, we need: [noisy_latents, mask, masked_source_latents]
                # 
                # Important: Use SOURCE image (old wall) for masked_source, not target!
                # This preserves the non-wall regions while generating new wall color
                
                masked_source_pixel = batch["masked_sources"].to(dtype=weight_dtype)  # [B, 3, H, W] (0-1)
                masked_source_pixel_normalized = masked_source_pixel * 2.0 - 1.0  # [-1, 1]
                
                with torch.no_grad():
                    masked_latents = vae.encode(masked_source_pixel_normalized).latent_dist.sample()
                    masked_latents = masked_latents * vae.config.scaling_factor  # [B, 4, 64, 64]
                
                # Resize mask to latent size
                mask = batch["mask"].to(dtype=weight_dtype)  # [B, 1, H, W] (0-1)
                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)  # Add channel dim if needed
                mask_latents = F.interpolate(
                    mask, 
                    size=target_latents.shape[2:], 
                    mode="nearest"
                )  # [B, 1, 64, 64]
                
                # Concatenate inpainting inputs: [noisy_latents, mask, masked_source_latents]
                # Total channels: 4 + 1 + 4 = 9
                unet_input = torch.cat([noisy_latents, mask_latents, masked_latents], dim=1)
                
                # ====================================================================
                # PHASE 5: UNET PREDICTION
                # ====================================================================
                # UNet predicts the noise that was added to TARGET latents
                # Conditions guide it to generate new color while preserving structure
                
                noise_pred = unet(
                    unet_input,  # [B, 9, 64, 64]
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,  # Text embeddings
                    down_block_additional_residuals=down_block_res_samples,  # ControlNet (structure)
                    mid_block_additional_residual=mid_block_res_sample,  # ControlNet (structure)
                    added_cond_kwargs=added_cond_kwargs  # IP-Adapter (color)
                ).sample
                
                # ====================================================================
                # PHASE 6: LOSS COMPUTATION
                # ====================================================================
                # Loss: MSE between predicted noise and actual noise
                # The model learns to predict what noise was added to TARGET image
                # This teaches it to generate new colors (target) from conditions
                
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                
                # Backward pass
                accelerator.backward(loss)
                
                # Gradient clipping for stability
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            
            # Logging
            if global_step % 20 == 0:
                logs = {
                    "loss": loss.detach().item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                logging.info(f"Step {global_step}, Loss: {loss.item():.4f}")
                 
            # Validation and checkpointing
            if global_step > 0 and global_step % 50 == 0:
                validate_and_save(
                    accelerator, unet, controlnet, vae, text_encoder, tokenizer, 
                    image_encoder, val_dataloader, global_step, weight_dtype, args.output_dir
                )
                
                # Save checkpoint
                if accelerator.is_main_process:
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(checkpoint_dir)
                    logging.info(f"Saved checkpoint to {checkpoint_dir}")

            global_step += 1
            
    # Final validation and checkpoint
    if accelerator.is_main_process:
        validate_and_save(
            accelerator, unet, controlnet, vae, text_encoder, tokenizer, 
            image_encoder, val_dataloader, global_step, weight_dtype, args.output_dir
        )
        
        # Save final checkpoint
        final_checkpoint_dir = os.path.join(args.output_dir, "checkpoint-final")
        accelerator.save_state(final_checkpoint_dir)
        logging.info(f"Saved final checkpoint to {final_checkpoint_dir}")

    print("Training Complete.")
    accelerator.end_training()

if __name__ == "__main__":
    main()
