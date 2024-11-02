import os
import rasterio
import numpy as np
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None  # Remove size limit
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Handle truncated images

def process_and_mask_images(png_dir, tiff_dir, output_dir):
    """
    Process all matching PNG and TIFF files in the specified directories and save the masked PNG files.
    
    Parameters:
    png_dir (str): Directory containing PNG files to be masked
    tiff_dir (str): Directory containing TIFF files used for masking
    output_dir (str): Directory where masked PNG files will be saved
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for png_filename in os.listdir(png_dir):
        if png_filename.endswith(".png"):
            base_name = png_filename.replace("_mask.png", "")
            tiff_filename = base_name + ".tif"
            
            tiff_path = os.path.join(tiff_dir, tiff_filename)
            png_path = os.path.join(png_dir, png_filename)
            output_path = os.path.join(output_dir, png_filename)
            
            if os.path.exists(tiff_path):
                print(f"Processing: {png_filename} with {tiff_filename}")
                
                # Read the TIFF file
                with rasterio.open(tiff_path) as src:
                    tiff_data = src.read(1)
                    mask = tiff_data != 0
                    
                    # Print mask statistics
                    valid_pixels = np.sum(mask)
                    total_pixels = mask.size
                    print(f"\nMask statistics for {png_path}:")
                    print(f"Valid pixels: {valid_pixels} ({valid_pixels / total_pixels * 100:.2f}%)")
                    print(f"Null pixels: {total_pixels - valid_pixels} ({(total_pixels - valid_pixels) / total_pixels * 100:.2f}%)")
                
                # Read the PNG file
                png_img = Image.open(png_path)
                png_array = np.array(png_img)
                
                # Check if dimensions match
                if tiff_data.shape != png_array.shape[:2]:
                    raise ValueError(f"Image dimensions don't match: TIFF is {tiff_data.shape}, PNG is {png_array.shape[:2]}")
                
                # Expand mask if PNG has multiple channels
                if len(png_array.shape) == 3:
                    mask = np.expand_dims(mask, axis=2)
                    mask = np.repeat(mask, png_array.shape[2], axis=2)
                
                # Apply mask
                masked_png = png_array.copy()
                masked_png[~mask] = 0
                
                # Save the result
                masked_image = Image.fromarray(masked_png)
                masked_image.save(output_path)
                print(f"Masked PNG saved to {output_path}")
            else:
                print(f"Corresponding TIFF file not found for {png_filename}")


if __name__ == "__main__":
    png_dir = r"D:\planetscope_lake_ice\Data_TEST\3 - Download Labelbox masks here\Lake_Ice_Breakup_2023_YKD_RGB_psscene_visual\labels_categorical"
    tiff_dir = r"D:\planetscope_lake_ice\Data_TEST\1 - Download your Planet RGB orders here\Lake_Ice_Breakup_2023_YKD_RGB_psscene_visual\PSScene"
    output_dir = r"D:\planetscope_lake_ice\Data_TEST\3 - Download Labelbox masks here\Lake_Ice_Breakup_2023_YKD_RGB_psscene_visual\clipped_masks"

    process_and_mask_images(png_dir, tiff_dir, output_dir)
