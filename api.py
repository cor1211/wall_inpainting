"""
FastAPI Web Service for AI Interior Wall Re-skinning

REST API endpoints:
- POST /process - Full wall re-skinning (upload source + reference)
- POST /segment - Wall mask extraction only
- POST /process-color - Re-skin with solid color  
- GET /health - Health check

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000
    
    # With auto-reload for development
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import uuid
import time
from pathlib import Path
from typing import Optional, Tuple
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np

# Local imports
try:
    from config import config, OUTPUT_DIR
    from segmentation import WallSegmenter
    from pipeline import WallReskinPipeline, create_solid_color_reference
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all modules are in the same directory.")
    raise


# Global instances (lazy loaded)
_segmenter: Optional[WallSegmenter] = None
_pipeline: Optional[WallReskinPipeline] = None


def get_segmenter() -> WallSegmenter:
    """Get or create segmenter instance."""
    global _segmenter
    if _segmenter is None:
        _segmenter = WallSegmenter()
    return _segmenter


def get_pipeline() -> WallReskinPipeline:
    """Get or create pipeline instance with LoRA if configured."""
    global _pipeline
    if _pipeline is None:
        # Check if LoRA is enabled in config
        lora_path = None
        lora_scale = 1.0
        if config.lora.enabled:
            from pathlib import Path
            lora_dir = Path(config.lora.default_path)
            if lora_dir.exists() and (lora_dir / "adapter_config.json").exists():
                lora_path = str(lora_dir)
                lora_scale = config.lora.default_scale
                print(f"LoRA enabled: {lora_path} (scale={lora_scale})")
            else:
                print(f"LoRA path not found or invalid: {lora_dir}, loading without LoRA")
        
        _pipeline = WallReskinPipeline(
            lora_path=lora_path,
            lora_scale=lora_scale,
        )
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "temp").mkdir(exist_ok=True)
    print("API server starting...")
    yield
    # Shutdown
    print("API server shutting down...")


# Create FastAPI app
app = FastAPI(
    title="AI Wall Re-skinning API",
    description="Change wall colors and textures in interior images using AI",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Helper Functions ==============

def validate_image(file: UploadFile) -> Image.Image:
    """Validate and load uploaded image."""
    # Check file extension
    allowed = (".jpg", ".jpeg", ".png", ".webp")
    if not file.filename.lower().endswith(allowed):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed}"
        )
    
    # Check file size
    max_size = config.api.max_file_size_mb * 1024 * 1024
    file.file.seek(0, 2)  # Seek to end
    size = file.file.tell()
    file.file.seek(0)  # Seek back to start
    
    if size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {config.api.max_file_size_mb}MB"
        )
    
    # Load image
    try:
        image = Image.open(file.file).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load image: {str(e)}"
        )


def image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """Convert PIL Image to bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return buffer.getvalue()


def parse_color(color_str: str) -> Tuple[int, int, int]:
    """Parse color string 'R,G,B' to tuple."""
    try:
        parts = color_str.split(",")
        if len(parts) != 3:
            raise ValueError("Invalid format")
        r, g, b = map(int, parts)
        if not all(0 <= v <= 255 for v in (r, g, b)):
            raise ValueError("Values must be 0-255")
        return (r, g, b)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid color format. Use 'R,G,B' (e.g., '200,180,160')"
        )


# ============== API Endpoints ==============

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    import torch
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "AI Wall Re-skinning API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "GET - Health check",
            "/segment": "POST - Extract wall mask",
            "/process": "POST - Full wall re-skinning",
            "/process-color": "POST - Re-skin with solid color",
        }
    }


@app.post("/segment")
async def segment_wall(
    image: UploadFile = File(..., description="Interior image to segment"),
    strategy: str = Form("semantic", description="Strategy: semantic, clip, heuristic, auto"),
    include_ceiling: bool = Form(False, description="Include ceiling in mask"),
):
    """
    Extract wall mask from an interior image.
    
    Returns binary mask as PNG image.
    """
    # Load and validate image
    pil_image = validate_image(image)
    
    # Save temporarily for processing
    temp_path = OUTPUT_DIR / "temp" / f"{uuid.uuid4()}.png"
    pil_image.save(temp_path)
    
    try:
        # Get segmenter and process
        segmenter = get_segmenter()
        mask = segmenter.get_wall_mask(
            temp_path,
            strategy=strategy,
            include_ceiling=include_ceiling,
            return_pil=True,
        )
        
        # Return mask as PNG
        return StreamingResponse(
            io.BytesIO(image_to_bytes(mask, "PNG")),
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=mask.png"}
        )
    finally:
        # Cleanup temp file
        if temp_path.exists():
            temp_path.unlink()


@app.post("/process")
async def process_image(
    source: UploadFile = File(..., description="Source interior image"),
    reference: UploadFile = File(..., description="Reference color/texture image"),
    mask: Optional[UploadFile] = File(None, description="Optional custom mask"),
    strategy: str = Form("semantic", description="Segmentation strategy"),
    steps: int = Form(30, description="Number of inference steps"),
    controlnet_scale: float = Form(0.8, description="ControlNet scale (0-1)"),
    ip_scale: float = Form(1.0, description="IP-Adapter scale (1.0 for maximum color transfer)"),
    seed: Optional[int] = Form(None, description="Random seed"),
):
    """
    Full wall re-skinning pipeline.
    
    Upload source image and reference image.
    Optionally provide custom mask.
    """
    start_time = time.time()
    
    # Load images
    source_img = validate_image(source)
    reference_img = validate_image(reference)
    
    # Get or generate mask
    if mask:
        mask_img = Image.open(mask.file).convert("L")
    else:
        # Save source temporarily for segmentation
        temp_path = OUTPUT_DIR / "temp" / f"{uuid.uuid4()}.png"
        source_img.save(temp_path)
        
        try:
            segmenter = get_segmenter()
            mask_img = segmenter.get_wall_mask(
                temp_path,
                strategy=strategy,
                return_pil=True,
            )
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    # Check mask has content
    mask_array = np.array(mask_img)
    if mask_array.sum() == 0:
        raise HTTPException(
            status_code=422,
            detail="No wall detected in image. Try different strategy or provide custom mask."
        )
    
    # Process with pipeline
    pipeline = get_pipeline()
    result = pipeline.process(
        source_image=source_img,
        mask_image=mask_img,
        reference_image=reference_img,
        num_inference_steps=steps,
        controlnet_conditioning_scale=controlnet_scale,
        ip_adapter_scale=ip_scale,
        seed=seed,
    )
    
    elapsed = time.time() - start_time
    
    # Return result with timing header
    return StreamingResponse(
        io.BytesIO(image_to_bytes(result, "PNG")),
        media_type="image/png",
        headers={
            "Content-Disposition": "attachment; filename=result.png",
            "X-Processing-Time": f"{elapsed:.2f}s"
        }
    )


@app.post("/process-color")
async def process_with_color(
    source: UploadFile = File(..., description="Source interior image"),
    color: str = Form(..., description="Target color as 'R,G,B'"),
    mask: Optional[UploadFile] = File(None, description="Optional custom mask"),
    strategy: str = Form("semantic", description="Segmentation strategy"),
    steps: int = Form(30, description="Number of inference steps"),
    controlnet_scale: float = Form(0.8, description="ControlNet scale"),
    ip_scale: float = Form(1.0, description="IP-Adapter scale (1.0 for maximum color transfer)"),
    seed: Optional[int] = Form(None, description="Random seed"),
):
    """
    Re-skin wall with a solid color.
    
    Simpler endpoint - just provide source image and target color.
    """
    start_time = time.time()
    
    # Parse color
    rgb_color = parse_color(color)
    
    # Load source image
    source_img = validate_image(source)
    
    # Create solid color reference
    reference_img = create_solid_color_reference(rgb_color)
    
    # Get or generate mask
    if mask:
        mask_img = Image.open(mask.file).convert("L")
    else:
        temp_path = OUTPUT_DIR / "temp" / f"{uuid.uuid4()}.png"
        source_img.save(temp_path)
        
        try:
            segmenter = get_segmenter()
            mask_img = segmenter.get_wall_mask(
                temp_path,
                strategy=strategy,
                return_pil=True,
            )
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    # Check mask
    mask_array = np.array(mask_img)
    if mask_array.sum() == 0:
        raise HTTPException(
            status_code=422,
            detail="No wall detected in image."
        )
    
    # Process
    pipeline = get_pipeline()
    result = pipeline.process(
        source_image=source_img,
        mask_image=mask_img,
        reference_image=reference_img,
        num_inference_steps=steps,
        controlnet_conditioning_scale=controlnet_scale,
        ip_adapter_scale=ip_scale,
        seed=seed,
    )
    
    elapsed = time.time() - start_time
    
    return StreamingResponse(
        io.BytesIO(image_to_bytes(result, "PNG")),
        media_type="image/png",
        headers={
            "Content-Disposition": "attachment; filename=result.png",
            "X-Processing-Time": f"{elapsed:.2f}s"
        }
    )


# ============== Error Handlers ==============

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Handle uncaught exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if config.debug else "An unexpected error occurred"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host=config.api.host,
        port=config.api.port,
        reload=True,
    )
