
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
    """
    if not accelerator.is_local_main_process:
        return
        
    unet.eval()
    print(f"\nRunning Validation at Step {global_step}...")
    
    # We need to construct a pipeline manually or use the components
    # Easier to just run the manual generation loop for a few samples
    # Or instantiate a temporary pipeline?
    # Instantiating pipeline is cleaner but heavy.
    # Let's run manual loop for 1 batch.
    
    val_dir = os.path.join(output_dir, "validation_images")
    os.makedirs(val_dir, exist_ok=True)
    
    # Prepare Scheduler (DDIM for Inference)
    from diffusers import DDIMScheduler
    resolution = 512
    device = accelerator.device
    
    # We need a scheduler config. Use DDIMScheduler compatible with SD1.5
    # The training used DDPMScheduler. 
    val_scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-inpainting", subfolder="scheduler")
    
    try:
        batch = next(iter(val_dataloader))
    except StopIteration:
        return

    # Use max 2 samples
    if batch['color_patches'].shape[0] > 2:
        for k in batch:
            if isinstance(batch[k], torch.Tensor) or isinstance(batch[k], list):
                 batch[k] = batch[k][:2]
    
    bsz = batch['color_patches'].shape[0]
    
    # 1. Prepare Latents (Noise)
    # UNet inpainting has 9 channels, but base latents are 4.
    latents = torch.randn(
        (bsz, 4, resolution // 8, resolution // 8),
        device=device,
        dtype=weight_dtype
    )
    
    # 2. Text Embeds
    with torch.no_grad():
        inputs = tokenizer(batch["prompts"], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
        encoder_hidden_states = text_encoder(inputs.input_ids.to(device))[0]
        
        # 3. ControlNet Cond
        control_images = batch["conditional_images"].to(device, dtype=weight_dtype)
        
        # 4. IP-Adapter Embeds
        pixel_values_ip = batch["color_patches"].to(device, dtype=weight_dtype)
        
        # Resize to 224x224 for CLIP
        pixel_values_ip = F.interpolate(pixel_values_ip, size=(224, 224), mode="bilinear", align_corners=False)
        
        # Norm
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device, dtype=weight_dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device, dtype=weight_dtype).view(1, 3, 1, 1)
        pixel_values_ip = (pixel_values_ip - mean) / std
        image_embeds = image_encoder(pixel_values_ip).image_embeds
        added_cond_kwargs = {"image_embeds": image_embeds}

        # 5. Inpainting Setup
        
        # Masked Source
        masked_source_pixel = batch["masked_sources"].to(device, dtype=weight_dtype) * 2.0 - 1.0
        masked_latents = vae.encode(masked_source_pixel).latent_dist.sample() * vae.config.scaling_factor
        
        # Mask
        mask = batch["masks"].to(device, dtype=weight_dtype).unsqueeze(1)
        mask_latents = F.interpolate(mask, size=latents.shape[2:], mode="nearest")
        
        val_scheduler.set_timesteps(20)
        timesteps = val_scheduler.timesteps
        
        # Use autocast for mixed precision inference logic
        with accelerator.autocast():
            for t in timesteps:
                # Expand latents for conditional/unconditional (here supervised, so just 1)
                latent_model_input = latents
                
                # Concatenate Inpainting inputs
                unet_input = torch.cat([latent_model_input, mask_latents, masked_latents], dim=1)
                
                # ControlNet
                down_block_res_samples, mid_block_res_sample = controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=control_images,
                    return_dict=False,
                )
                
                # UNet
                noise_pred = unet(
                    unet_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    added_cond_kwargs=added_cond_kwargs
                ).sample
                
                # Step
                latents = val_scheduler.step(noise_pred, t, latents).prev_sample
            
        # Decode
        latents = 1 / vae.config.scaling_factor * latents
        latents = latents.to(weight_dtype)
        image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        
        # Save
        from PIL import Image
        import numpy as np
        for i in range(bsz):
            img = (image[i] * 255).astype(np.uint8)
            Image.fromarray(img).save(os.path.join(val_dir, f"step_{global_step}_sample_{i}.png"))
            
    print(f"Saved validation images to {val_dir}")
    unet.train()


def main():
    parser = argparse.ArgumentParser(description="Train Wall Recoloring Hybrid Model")
    parser.add_argument("--data_json", type=str, default="dataset_v2/train/metadata.jsonl")
    parser.add_argument("--validation_json", type=str, default="dataset_v2/validation/metadata.jsonl")
    parser.add_argument("--output_dir", type=str, default="output/wall_recolor_v1")
    parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-inpainting")
    parser.add_argument("--controlnet_model", type=str, default="lllyasviel/control_v11p_sd15_canny")
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
    train_dataset = WallPaintDataset(
        data_json=args.data_json,
        image_size=args.resolution,
        reconstruction_ratio=0.5
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

    # Validation Set
    val_dataset = WallPaintDataset(
        data_json=args.validation_json,
        image_size=args.resolution,
        reconstruction_ratio=0.5
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
    # Only prepare UNet, Optimizer, TrainDataloader. 
    # ValDataloader is manually iterated.
    unet, optimizer, train_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader
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
    
    for epoch in range(args.num_train_epochs):
        unet.train()
        for batch in tqdm(train_dataloader, disable=not accelerator.is_local_main_process):
            with accelerator.accumulate(unet):
                # 1. Inputs
                # batch keys: ['color_patches', 'masked_sources', 'targets', 'conditional_images', 'masks', 'prompts']
                
                # Convert images to latent space
                # Inputs are already normalized to [0,1] floating point in Dataset?
                # Check Dataset: .float() / 255.0. Yes.
                # VAE expects usually [-1, 1].
                # Let's normalize.
                
                # Target (GT) -> Latents
                pixel_values = batch["targets"].to(dtype=weight_dtype) * 2.0 - 1.0
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # 2. Conditioning
                
                # A. Text
                inputs = tokenizer(batch["prompts"], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
                encoder_hidden_states = text_encoder(inputs.input_ids.to(accelerator.device))[0]
                
                # B. ControlNet (Source Image)
                control_images = batch["conditional_images"].to(dtype=weight_dtype) # [0, 1] is fine for ControlNet
                # Get ControlNet output
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=control_images,
                    return_dict=False,
                )
                
                # C. IP-Adapter (Color Patch)
                # The IP-Adapter logic is embedded in the UNet. We need to pass the image embeddings.
                # In diffusers `StableDiffusionPipeline`, `ip_adapter_image` creates `image_embeds`.
                # We need to manually extract image embeds and pass to UNet `added_cond_kwargs` or `encoder_hidden_states`?
                # Diffusers implementation of IP-Adapter stores embeds in `unet.encoder_hidden_states` via concatenation?
                # No, standard IP-Adapter passes `image_embeds` to the attention processor.
                # BUT `unet` forward() signature doesn't standardly accept `image_embeds`.
                # The `StableDiffusionPipeline` hacks this via `added_cond_kwargs` for SDXL, or by modifying attention processors.
                # When we called `pipe.load_ip_adapter`, it replaced attention processors.
                # These processors EXPECT `image_embeds` passed implicitly or via a side channel?
                # Actually, diffusers implementation usually expects `added_cond_kwargs` with `image_embeds`.
                
                # Let's encode image
                # CLIPVision expects preprocessed images. Dataset gives raw tensors.
                # We need a CLIPImageProcessor equivalent.
                # Simple resize/norm usually sufficient or use `CLIPImageProcessor` if strictly needed.
                # For now assume dataset `color_patches` (0-1) are close enough, just normalize to CLIP expected.
                # CLIP mean/std: [0.481, 0.457, 0.408], [0.268, 0.261, 0.275]
                # Dataset output is [0, 1].
                from transformers import CLIPImageProcessor
                clip_image_processor = CLIPImageProcessor()
                # We need to convert back to list of numpy or use tensors?
                # Processor accepts tensors too.
                # batch["color_patches"] is (B, 3, H, W).
                # This might slow down training.
                # Let's assume we pass raw tensors to image_encoder if it accepts it.
                # CLIPVisionModel expects pixel_values.
                
                pixel_values_ip = batch["color_patches"].to(dtype=weight_dtype) 
                
                # Resize to 224x224 for CLIP
                pixel_values_ip = F.interpolate(pixel_values_ip, size=(224, 224), mode="bilinear", align_corners=False)
                
                # Need proper normalization for CLIP
                # (val - mean) / std
                mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device=accelerator.device, dtype=weight_dtype).view(1, 3, 1, 1)
                std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device=accelerator.device, dtype=weight_dtype).view(1, 3, 1, 1)
                pixel_values_ip = (pixel_values_ip - mean) / std
                
                image_embeds = image_encoder(pixel_values_ip).image_embeds # (B, 1024)
                
                # Correct way for modern Diffusers (>=0.27):
                # IP-Adapter expects `added_cond_kwargs={"image_embeds": ...}`
                # IMPORTANT: Do NOT wrap in list if only one adapter.
                added_cond_kwargs = {"image_embeds": image_embeds} 
                
                # 3. Model Prediction
                # SD Inpainting expects concatenated input: [noisy_latents, mask, masked_image_latents]
                # Encode Masked Source
                masked_source_pixel = batch["masked_sources"].to(dtype=weight_dtype) * 2.0 - 1.0
                masked_latents = vae.encode(masked_source_pixel).latent_dist.sample() * vae.config.scaling_factor
                
                # Resize Mask to latent size
                mask = batch["masks"].to(dtype=weight_dtype).unsqueeze(1) # (B, 1, H, W)
                mask_latents = F.interpolate(mask, size=latents.shape[2:], mode="nearest")
                
                # Concatenate
                unet_input = torch.cat([noisy_latents, mask_latents, masked_latents], dim=1)
                
                # Predict
                noise_pred = unet(
                    unet_input,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    added_cond_kwargs=added_cond_kwargs
                ).sample
                
                # 4. Loss
                # Masked Loss?
                # Only punish loss in wall region.
                # mask is 1 for wall, 0 for background.
                # Actually, target contains background (from original).
                # We want to reconstruct everything, but focus on Wall?
                # Simple MSE is fine, or weighted.
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none")
                loss = loss.mean()
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            if global_step % 20 == 0: # Check more frequently for short run
                 logging.info(f"Step {global_step}, Loss: {loss.item()}")
                 
            # Validation every 50 steps (or less for testing)
            if global_step > 0 and global_step % 50 == 0:
                 validate_and_save(accelerator, unet, controlnet, vae, text_encoder, tokenizer, image_encoder, val_dataloader, global_step, weight_dtype, args.output_dir)

            global_step += 1
            
    # Final validation
    validate_and_save(accelerator, unet, controlnet, vae, text_encoder, tokenizer, image_encoder, val_dataloader, global_step, weight_dtype, args.output_dir)

    print("Training Complete.")
    accelerator.save_state(args.output_dir)

if __name__ == "__main__":
    main()
