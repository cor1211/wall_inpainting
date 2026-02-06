import numpy as np
import cv2
import os
import sys

# Add current dir to path
sys.path.append(os.getcwd())

try:
    from dataset.color_transforms import apply_color_shift
    print("Import successful")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_transform():
    # Create a dummy image (100x100), gray wall with some texture
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    # Gradient to simulate lighting/texture
    for i in range(100):
        val = 100 + i  # 100 to 200
        image[i, :, :] = val
    
    # Create a mask (center square)
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[25:75, 25:75] = 1.0
    
    # Target color: Red (255, 0, 0)
    target_rgb = (255, 0, 0)
    
    # Apply transform
    output = apply_color_shift(image, mask, target_rgb)
    
    # Check center pixel
    center_pixel_orig = image[50, 50] # Should be around (150, 150, 150)
    center_pixel_new = output[50, 50]
    
    print(f"Original center: {center_pixel_orig}")
    print(f"New center: {center_pixel_new}")
    
    # Check outside pixel (should be unchanged)
    corner_pixel_orig = image[0, 0]
    corner_pixel_new = output[0, 0]
    
    if np.array_equal(corner_pixel_orig, corner_pixel_new):
        print("Outside mask: Unchanged (Pass)")
    else:
        print("Outside mask: Changed (Fail)")

    if center_pixel_new[0] > center_pixel_new[1] and center_pixel_new[0] > center_pixel_new[2]:
         print("Color shift: Red dominant (Pass)")
    else:
         print("Color shift: Failed")
         
    # Check Luminance preservation roughly
    # Convert back to Lab
    orig_lab = cv2.cvtColor(np.array([[center_pixel_orig]]), cv2.COLOR_RGB2LAB)[0,0]
    new_lab = cv2.cvtColor(np.array([[center_pixel_new]]), cv2.COLOR_RGB2LAB)[0,0]
    
    # PREVIOUSLY: We wanted L to be equal (strict preservation)
    # NOW: We want L to move towards target L, but keep relative texture
    
    print(f"Original L: {orig_lab[0]}, New L: {new_lab[0]}")
    
    if int(new_lab[0]) != int(orig_lab[0]):
        print("Luminance changed to match target (Pass)")
    else:
        print("Luminance stayed exact (Warning: Old Logic?)")

if __name__ == "__main__":
    try:
        test_transform()
        print("Test finished")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
