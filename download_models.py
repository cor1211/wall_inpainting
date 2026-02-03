"""
Model Download Script for AI Interior Wall Re-skinning

Downloads all required model weights:
- FastSAM-x.pt for segmentation
- Pre-caches HuggingFace models for faster first run

Usage:
    python download_models.py
    python download_models.py --all  # Download all including HF models
"""

import os
import sys
import argparse
from pathlib import Path
import urllib.request
import hashlib
from tqdm import tqdm


# Import config
try:
    from config import (
        MODELS_DIR, 
        config,
    )
except ImportError:
    # Fallback if config not available
    MODELS_DIR = Path(__file__).parent / "models"
    MODELS_DIR.mkdir(exist_ok=True)


# Model definitions
MODELS = {
    "fastsam": {
        "filename": "FastSAM-x.pt",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.1.0/FastSAM-x.pt",
        "size_mb": 138,
        "sha256": None,  # Optional checksum
    },
    "fastsam-s": {
        "filename": "FastSAM-s.pt", 
        "url": "https://github.com/ultralytics/assets/releases/download/v8.1.0/FastSAM-s.pt",
        "size_mb": 23,
        "sha256": None,
    },
}

# HuggingFace models to pre-cache
HF_MODELS = [
    "runwayml/stable-diffusion-inpainting",
    "lllyasviel/control_v11f1p_sd15_depth",
    "h94/IP-Adapter",
    "Intel/dpt-large",
    "facebook/mask2former-swin-base-ade-semantic",
]


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads."""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, dest_path: Path, desc: str = None) -> bool:
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from
        dest_path: Destination file path
        desc: Description for progress bar
        
    Returns:
        True if successful, False otherwise
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    if desc is None:
        desc = dest_path.name
    
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as pbar:
            urllib.request.urlretrieve(url, dest_path, reporthook=pbar.update_to)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def verify_checksum(file_path: Path, expected_sha256: str) -> bool:
    """Verify file SHA256 checksum."""
    if expected_sha256 is None:
        return True
        
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    actual = sha256_hash.hexdigest()
    if actual != expected_sha256:
        print(f"Checksum mismatch!")
        print(f"  Expected: {expected_sha256}")
        print(f"  Got:      {actual}")
        return False
    return True


def download_fastsam(model_key: str = "fastsam") -> bool:
    """Download FastSAM model."""
    model_info = MODELS.get(model_key)
    if not model_info:
        print(f"Unknown model: {model_key}")
        return False
    
    dest_path = MODELS_DIR / model_info["filename"]
    
    if dest_path.exists():
        print(f"✓ {model_info['filename']} already exists ({dest_path})")
        return True
    
    print(f"\nDownloading {model_info['filename']} (~{model_info['size_mb']}MB)...")
    
    success = download_file(
        model_info["url"],
        dest_path,
        model_info["filename"]
    )
    
    if success and model_info.get("sha256"):
        if not verify_checksum(dest_path, model_info["sha256"]):
            dest_path.unlink()
            return False
    
    if success:
        print(f"✓ Downloaded to {dest_path}")
    
    return success


def download_huggingface_models():
    """Pre-download and cache HuggingFace models."""
    print("\nPre-caching HuggingFace models...")
    print("This may take a while on first run.\n")
    
    try:
        from huggingface_hub import snapshot_download
        
        for model_id in HF_MODELS:
            print(f"Caching: {model_id}")
            try:
                snapshot_download(
                    repo_id=model_id,
                    local_files_only=False,
                    resume_download=True,
                )
                print(f"  ✓ {model_id}")
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                
    except ImportError:
        print("huggingface_hub not installed. Skipping HF model caching.")
        print("Install with: pip install huggingface-hub")


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✓ CUDA available: {device_name} ({vram:.1f}GB VRAM)")
            return True
        else:
            print("⚠ CUDA not available. Pipeline will run on CPU (slower).")
            return False
    except ImportError:
        print("⚠ PyTorch not installed.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download required models for AI Wall Re-skinning"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Download all models including HuggingFace caching"
    )
    parser.add_argument(
        "--fastsam-only",
        action="store_true",
        help="Only download FastSAM model"
    )
    parser.add_argument(
        "--check",
        action="store_true", 
        help="Check system requirements"
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Download smaller FastSAM-s model instead of FastSAM-x"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AI Interior Wall Re-skinning - Model Downloader")
    print("=" * 60)
    
    # Check system
    if args.check:
        print("\nSystem Check:")
        check_cuda()
        
        # Check models
        print("\nModel Status:")
        for key, info in MODELS.items():
            path = MODELS_DIR / info["filename"]
            status = "✓ Found" if path.exists() else "✗ Not found"
            print(f"  {info['filename']}: {status}")
        return
    
    # Create models directory
    MODELS_DIR.mkdir(exist_ok=True)
    
    # Download FastSAM
    model_key = "fastsam-s" if args.small else "fastsam"
    success = download_fastsam(model_key)
    
    if not success:
        print("\n✗ Failed to download FastSAM model.")
        sys.exit(1)
    
    # Download HuggingFace models if requested
    if args.all and not args.fastsam_only:
        download_huggingface_models()
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    
    # Show next steps
    print("\nNext steps:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Run a test: python main.py --source <image> --color '200,180,160'")


if __name__ == "__main__":
    main()
