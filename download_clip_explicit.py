
import os
import sys
from huggingface_hub import snapshot_download

model_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
print(f"Downloading {model_id}...")
print("Optimizing download: Fetching ONLY safetensors and config (skipping redundant .bin files)...")

# Only download transformers-compatible safetensors and configs
# Exclude open_clip specific files and pytorch binaries to save bandwidth
local_dir = snapshot_download(
    repo_id=model_id, 
    allow_patterns=[
        "config.json",
        "*.json", 
        "*.txt", 
        "model.safetensors"  # Only download this one weight file
    ],
    resume_download=True
)

print(f"\nDownload completed! Model stored in: {local_dir}")
