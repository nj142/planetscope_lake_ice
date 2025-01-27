import os
import rasterio
import numpy as np
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None  # Remove size limit
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Handle truncated images

def process_and_mask_images(png_dir, tiff_dir, output_dir):
    """
    Process all PNG files in the specified directory by finding and applying 
    corresponding TIFF masks from a given root directory.
    
    Parameters:
    png_dir (str): Directory containing PNG files to be masked
    tiff_dir (str): Root directory to search for TIFF files
    output_dir (str): Directory where masked PNG files will be saved
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each PNG file in the directory
    for png_filename in os.listdir(png_dir):
        if png_filename.endswith(".png"):
            # Construct full path for PNG
            png_path = os.path.join(png_dir, png_filename)
            
            # Extract base name (remove '_mask.png')
            base_name = os.path.splitext(png_filename)[0].replace('_mask', '')
            
            # Search for matching TIFF file
            matching_tiff = None
            for root, dirs, files in os.walk(tiff_dir):
                for file in files:
                    if file.endswith('.tif') or file.endswith('.tiff'):
                        # Check if the file contains the base name
                        if base_name in file:
                            matching_tiff = os.path.join(root, file)
                            break
                if matching_tiff:
                    break
            
            # If no matching TIFF found, skip this file
            if not matching_tiff:
                print(f"No matching TIFF file found for {png_filename}")
                continue
            
            print(f"Processing: {png_filename} with {os.path.basename(matching_tiff)}")
            
            # Read the TIFF file
            with rasterio.open(matching_tiff) as src:
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
                print(f"WARNING: Dimension mismatch for {png_filename}")
                print(f"TIFF dimensions: {tiff_data.shape}")
                print(f"PNG dimensions: {png_array.shape[:2]}")
                continue
            
            # Expand mask if PNG has multiple channels
            if len(png_array.shape) == 3:
                mask = np.expand_dims(mask, axis=2)
                mask = np.repeat(mask, png_array.shape[2], axis=2)
            
            # Apply mask
            masked_png = png_array.copy()
            masked_png[~mask] = 0
            
            # Prepare output path
            output_path = os.path.join(output_dir, png_filename)
            
            # Save the result
            masked_image = Image.fromarray(masked_png)
            masked_image.save(output_path)
            print(f"Masked PNG saved to {output_path}")

if __name__ == "__main__":
    png_dir = r"D:\planetscope_lake_ice\Data\3 - Download Labelbox masks here\YKD_YF_2023\labels_categorical"
    tiff_dir = r"D:\planetscope_lake_ice\Data\1 - Download your Planet RGB orders here"
    output_dir = r"D:\planetscope_lake_ice\Data\3 - Download Labelbox masks here\YKD_YF_2023\clipped_masks"

    process_and_mask_images(png_dir, tiff_dir, output_dir)