import os
import shutil
import cv2
import numpy as np
import json
from PIL import Image
from pathlib import Path
# from dataset.wall_paint_dataset import WallPaintDataset

def create_dummy_data():
    """Create dummy source data for testing."""
    root = Path("test_dataset_pairs")
    if root.exists():
        shutil.rmtree(root)
    
    root.mkdir()
    (root / "images").mkdir()
    (root / "masks").mkdir()
    
    # Create 1 source image
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:, :] = (200, 200, 200) # Gray wall
    cv2.circle(img, (128, 128), 50, (100, 100, 100), -1) # Shadow
    
    # Create 1 mask
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[50:200, 50:200] = 255
    
    Image.fromarray(img).save(root / "images" / "test_img.png")
    Image.fromarray(mask).save(root / "masks" / "test_img.png")
    
    return root

def test_pipeline():
    print("1. Creating dummy data...")
    input_dir = create_dummy_data()
    output_dir = Path("test_output_pairs")
    
    if output_dir.exists():
        shutil.rmtree(output_dir)
        
    print("2. Running prepare_dataset_v2.py...")
    # Simulate command line run
    from prepare_dataset_v2 import DatasetPipeline, DatasetStats
    
    pipeline = DatasetPipeline(
        colors_per_image=2,
        val_ratio=0.5,
        min_quality=0.0, # disable checks for dummy
        min_surface_ratio=0.0
    )
    
    pipeline.run(
        input_dir=input_dir,
        output_dir=output_dir,
        use_existing_masks=True,
        segment_new=False
    )
    
    print("3. Verifying output structure...")
    train_meta = output_dir / "train" / "metadata.jsonl"
    val_meta = output_dir / "validation" / "metadata.jsonl"
    
    has_train = train_meta.exists()
    has_val = val_meta.exists()
    
    print(f"Train meta exists: {has_train}")
    print(f"Val meta exists: {has_val}")
    
    if has_train:
        meta_file = train_meta
    elif has_val:
        meta_file = val_meta
    else:
        print("No metadata generated!")
        return

    # Check content
    with open(meta_file, 'r') as f:
        line = f.readline()
        data = json.loads(line)
        print("Sample metadata keys:", data.keys())
        if 'source_path' in data and 'target_path' in data:
            print("Metadata structure correct (Has source/target path)")
        else:
            print("Metadata structure incorrect!")
            return

    '''
    print("4. Testing Dataset Loader (SKIPPED due to missing torch)...")
    # dataset = WallPaintDataset(
    #     data_json=str(meta_file),
    #     image_size=256
    # )
    
    # sample = dataset[0]
    # print("Dataset item keys:", sample.keys())
    
    # if 'conditional_image' in sample and 'target' in sample:
    #     print("Dataset returns correct keys.")
    #     print('Target shape:', sample['target'].shape)
    #     print('Conditional shape:', sample['conditional_image'].shape)
    # else:
    #     print("Dataset missing keys!")
    '''

if __name__ == "__main__":
    test_pipeline()
