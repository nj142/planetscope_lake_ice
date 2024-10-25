import rasterio
import numpy as np
from PIL import Image
from PIL import ImageFile
import time

Image.MAX_IMAGE_PIXELS = None  # Remove size limit
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Handle truncated images

def mask_png_with_tiff(tiff_path, png_path, output_path):
    """
    Mask a PNG image using zero values from a TIFF file.
    
    Parameters:
    tiff_path (str): Path to the TIFF file with zero as null values
    png_path (str): Path to the PNG file to be masked
    output_path (str): Path where the masked PNG will be saved
    """
    # Read the TIFF file
    with rasterio.open(tiff_path) as src:
        tiff_data = src.read(1)  # Read the first band
        
        # Create mask where TIFF is non-zero
        mask = tiff_data != 0
        
        # Print mask statistics
        valid_pixels = np.sum(mask)
        total_pixels = mask.size
        print(f"\nMask statistics:")
        print(f"Valid pixels: {valid_pixels} ({valid_pixels/total_pixels*100:.2f}%)")
        print(f"Null pixels: {total_pixels-valid_pixels} ({(total_pixels-valid_pixels)/total_pixels*100:.2f}%)")
    
    # Read the PNG file
    png_img = Image.open(png_path)
    png_array = np.array(png_img)
    
    # Check if dimensions match
    if tiff_data.shape != png_array.shape[:2]:
        raise ValueError(f"Image dimensions don't match: TIFF is {tiff_data.shape}, PNG is {png_array.shape[:2]}")
    
    # Create masked version of PNG
    # Expand mask to match PNG dimensions (if PNG has multiple channels)
    if len(png_array.shape) == 3:
        mask = np.expand_dims(mask, axis=2)
        mask = np.repeat(mask, png_array.shape[2], axis=2)
    
    # Apply mask
    masked_png = png_array.copy()
    masked_png[~mask] = 0  # Set non-valid pixels to black
    
    # Save the result
    masked_image = Image.fromarray(masked_png)
    masked_image.save(output_path)
    
    """# Save mask for visualization (optional)
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    mask_path = output_path.rsplit('.', 1)[0] + '_mask.png'
    mask_image.save(mask_path)"""

    print(f"Cropped PNG saved to {output_path}")
    
    return "Masked PNG saved successfully! Also saved mask visualization."

# Use raw strings (r) for Windows paths to handle backslashes
tiff_path = r"D:\Testing\Image2\20230511_212249_22_24bb_3B_Visual.tif"
png_path = r"D:\Testing\Image2\labels_colour\20230511_212249_22_24bb_3B_Visual.jpg-mask.png"
output_path = r"D:\Testing\Image2\masked.png"

result = mask_png_with_tiff(tiff_path, png_path, output_path)
print(result)