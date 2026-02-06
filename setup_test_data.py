
import os
import shutil
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

def create_dummy_data():
    """Create dummy source data for testing."""
    root = Path("test_raw_data")
    if root.exists():
        shutil.rmtree(root)
    
    root.mkdir()
    (root / "images").mkdir()
    (root / "masks").mkdir()
    
    # Create 5 source images
    for i in range(5):
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        img[:, :] = (200, 200, 200) # Gray wall
        cv2.circle(img, (128, 128), 50, (100, 100, 100), -1) 
        
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[50:200, 50:200] = 255
        
        Image.fromarray(img).save(root / "images" / f"img_{i}.png")
        Image.fromarray(mask).save(root / "masks" / f"img_{i}.png")
    
    print("Created test_raw_data")
    return root

if __name__ == "__main__":
    create_dummy_data()
