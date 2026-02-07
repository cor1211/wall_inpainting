"""
Color utilities for wall recoloring dataset.

Functions to create color reference patches for IP-Adapter.
"""

import numpy as np
from PIL import Image
from typing import Tuple, Union
import random


# Common wall colors for augmentation
WALL_COLORS = [
    ("Beige", (245, 245, 220)),
    ("Cream", (255, 253, 208)),
    ("Off-White", (250, 249, 246)),
    ("Light Gray", (211, 211, 211)),
    ("Sage Green", (188, 238, 104)),
    ("Pale Blue", (173, 216, 230)),
    ("Terracotta", (226, 114, 91)),
    ("Charcoal", (54, 69, 79)),
    ("Navy Blue", (0, 0, 128)),
    ("Dusty Rose", (220, 174, 150)),
    ("Lavender", (230, 230, 250)),
    ("Mint", (189, 252, 201)),
    ("Peach", (255, 229, 180)),
    ("Sky Blue", (135, 206, 235)),
    ("Teal", (0, 128, 128)),
    ("Mustard", (255, 219, 88)),
    ("Olive", (128, 128, 0)),
    ("Coral", (255, 127, 80)),
    ("Salmon", (250, 128, 114)),
    ("Taupe", (72, 60, 50)),
    ("Warm White", (255, 248, 220)),
    ("Cool Gray", (140, 146, 172)),
]


def sample_random_color() -> Tuple[str, Tuple[int, int, int]]:
    """
    Sample a random wall color from the predefined palette.
    
    Returns:
        tuple: (color_name, (r, g, b))
    """
    return random.choice(WALL_COLORS)


def create_color_patch(
    rgb: Union[Tuple[int, int, int], list],
    size: Union[int, Tuple[int, int]] = 512,
    add_texture: bool = True,
    texture_noise_std: float = 8.0,
    add_gradient: bool = True,
) -> np.ndarray:
    """
    Create a color reference patch for IP-Adapter.
    
    Pure solid colors can break CLIP feature extraction (not enough variance),
    so we add subtle noise and gradient to give the encoder something to work with.
    
    Args:
        rgb: RGB color tuple or list (0-255)
        size: Output size. If int, creates square image. If tuple, (width, height)
        add_texture: If True, add Gaussian noise for texture
        texture_noise_std: Standard deviation of noise (default: 8.0)
        add_gradient: If True, add subtle vertical gradient to simulate lighting
    
    Returns:
        numpy array of shape (H, W, 3) with dtype uint8
    
    Example:
        >>> color_patch = create_color_patch((128, 0, 32), size=512)
        >>> color_patch.shape  # (512, 512, 3)
        >>> # Convert to PIL Image if needed:
        >>> img = Image.fromarray(color_patch)
    """
    # Normalize size
    if isinstance(size, int):
        size = (size, size)
    width, height = size
    
    # Ensure rgb is tuple
    if isinstance(rgb, list):
        rgb = tuple(rgb)
    
    # Create base solid color image
    img = Image.new("RGB", (width, height), rgb)
    arr = np.array(img, dtype=np.float32)
    
    # Add texture (noise)
    if add_texture:
        noise = np.random.normal(0, texture_noise_std, arr.shape)
        arr = arr + noise
    
    # Add gradient (lighting variation)
    if add_gradient:
        # Vertical gradient: slightly darker at top, brighter at bottom
        gradient = np.linspace(0.95, 1.05, height).reshape(-1, 1, 1)
        gradient = np.tile(gradient, (1, width, 3))
        arr = arr * gradient
    
    # Clip to valid range and convert to uint8
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    
    return arr


def create_color_patch_pil(
    rgb: Union[Tuple[int, int, int], list],
    size: Union[int, Tuple[int, int]] = 512,
    **kwargs
) -> Image.Image:
    """
    Create a color reference patch and return as PIL Image.
    
    Args:
        rgb: RGB color tuple or list (0-255)
        size: Output size
        **kwargs: Additional arguments passed to create_color_patch
    
    Returns:
        PIL Image
    
    Example:
        >>> img = create_color_patch_pil((128, 0, 32), size=224)
    """
    arr = create_color_patch(rgb, size=size, **kwargs)
    return Image.fromarray(arr)


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """
    Convert hex color string to RGB tuple.
    
    Args:
        hex_color: Hex color string (e.g., "#FF0000" or "FF0000")
    
    Returns:
        RGB tuple (0-255)
    
    Example:
        >>> rgb = hex_to_rgb("#FF0000")  # (255, 0, 0)
        >>> rgb = hex_to_rgb("FF0000")   # (255, 0, 0)
    """
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return rgb


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """
    Convert RGB tuple to hex color string.
    
    Args:
        rgb: RGB tuple (0-255)
    
    Returns:
        Hex color string (e.g., "#FF0000")
    
    Example:
        >>> hex_color = rgb_to_hex((255, 0, 0))  # "#FF0000"
    """
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
