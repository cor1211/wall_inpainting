"""
Test Suite for Pipeline Module

Tests the wall re-skinning pipeline components.

Usage:
    pytest tests/test_pipeline.py -v
    python tests/test_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from PIL import Image

try:
    from pipeline import (
        WallReskinPipeline,
        create_solid_color_reference,
        create_gradient_reference,
    )
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False


class TestReferenceCreation:
    """Test reference image creation utilities."""
    
    @pytest.mark.skipif(not PIPELINE_AVAILABLE, reason="Pipeline not available")
    def test_solid_color_reference(self):
        """Test solid color reference creation."""
        color = (255, 200, 150)
        ref = create_solid_color_reference(color)
        
        assert isinstance(ref, Image.Image)
        assert ref.size == (224, 224)
        assert ref.mode == "RGB"
        
        # Check color
        pixels = list(ref.getdata())
        assert pixels[0] == color
    
    @pytest.mark.skipif(not PIPELINE_AVAILABLE, reason="Pipeline not available")
    def test_solid_color_custom_size(self):
        """Test solid color with custom size."""
        ref = create_solid_color_reference((100, 100, 100), size=(512, 512))
        assert ref.size == (512, 512)
    
    @pytest.mark.skipif(not PIPELINE_AVAILABLE, reason="Pipeline not available")
    def test_gradient_reference_vertical(self):
        """Test vertical gradient creation."""
        ref = create_gradient_reference(
            (255, 0, 0),  # Red
            (0, 0, 255),  # Blue
            direction="vertical"
        )
        
        assert isinstance(ref, Image.Image)
        assert ref.size == (224, 224)
        
        # Check gradient (top should be more red)
        arr = np.array(ref)
        assert arr[0, 0, 0] > arr[-1, 0, 0]  # Red decreases
        assert arr[0, 0, 2] < arr[-1, 0, 2]  # Blue increases
    
    @pytest.mark.skipif(not PIPELINE_AVAILABLE, reason="Pipeline not available")
    def test_gradient_reference_horizontal(self):
        """Test horizontal gradient creation."""
        ref = create_gradient_reference(
            (255, 255, 255),  # White
            (0, 0, 0),  # Black
            direction="horizontal"
        )
        
        arr = np.array(ref)
        # Left should be lighter than right
        assert arr[0, 0].sum() > arr[0, -1].sum()


class TestPipelineInit:
    """Test pipeline initialization."""
    
    @pytest.mark.skipif(not PIPELINE_AVAILABLE, reason="Pipeline not available")
    def test_pipeline_init(self):
        """Test basic pipeline initialization (without loading models)."""
        pipeline = WallReskinPipeline()
        assert pipeline.device in ["cuda", "cpu"]
        assert pipeline._pipe is None  # Lazy loading
        assert pipeline._depth_estimator is None


class TestPipelineConfig:
    """Test pipeline configuration."""
    
    @pytest.mark.skipif(not PIPELINE_AVAILABLE, reason="Pipeline not available")
    def test_default_models(self):
        """Test default model IDs are set."""
        pipeline = WallReskinPipeline()
        assert "stable-diffusion-inpainting" in pipeline.base_model_id
        assert "control" in pipeline.controlnet_model_id.lower()


def run_manual_tests():
    """Run tests without pytest."""
    print("Running pipeline tests...")
    
    if not PIPELINE_AVAILABLE:
        print("Pipeline module not available")
        return
    
    # Test reference image creation
    print("\n1. Testing solid color reference...")
    ref = create_solid_color_reference((200, 180, 160))
    print(f"   Created: {ref.size}, mode={ref.mode}")
    
    print("\n2. Testing gradient reference...")
    ref = create_gradient_reference((255, 0, 0), (0, 0, 255))
    print(f"   Created: {ref.size}")
    
    print("\n3. Testing pipeline init...")
    try:
        pipeline = WallReskinPipeline()
        print(f"   Device: {pipeline.device}")
        print(f"   Models loaded: {pipeline._pipe is not None}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\nTests complete!")


if __name__ == "__main__":
    run_manual_tests()
