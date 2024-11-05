import os
import rasterio
import numpy as np
from PIL import Image
import shutil

def process_udm_folder(udm_folder, rgb_folder, output_folder):
    # Ensure the main output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Define colors for different masks
    masks_config = {
        "snow": {
            "band": 2,
            "color": (0, 255, 255, 255)    # Cyan with alpha
        },
        "cloud_shadow": {
            "band": 3,
            "color": (128, 0, 128, 255)    # Dark purple with alpha
        },
        "light_haze": {
            "band": 4,
            "color": (144, 238, 144, 255)  # Light green with alpha
        },
        "heavy_haze": {
            "band": 5,
            "color": (0, 128, 0, 255)      # Mid-green with alpha
        },
        "cloud": {
            "band": 6,
            "color": (0, 100, 0, 255)      # Dark green with alpha
        }
    }

    # Loop through UDM files
    for filename in os.listdir(udm_folder):
        if filename.endswith('_udm2.tif'):
            udm_path = os.path.join(udm_folder, filename)
            base_name = filename.split('_udm2')[0]
            
            # Copy RGB image to main output directory if it exists
            rgb_filename = f"{base_name}_Visual.jpg"
            rgb_path = os.path.join(rgb_folder, rgb_filename)
            if os.path.exists(rgb_path):
                rgb_output_path = os.path.join(output_folder, rgb_filename)
                shutil.copy2(rgb_path, rgb_output_path)
                print(f"Copied RGB image to {rgb_output_path}")
            else:
                print(f"Warning: RGB file {rgb_filename} not found")
            
            # Create masks folder next to RGB file
            masks_folder = os.path.join(output_folder, base_name)
            os.makedirs(masks_folder, exist_ok=True)
            
            # Read UDM file and process each mask type
            with rasterio.open(udm_path) as src:
                # Process each mask type
                for mask_name, config in masks_config.items():
                    # Create empty RGBA image for this mask (fully transparent)
                    mask_img = np.zeros((src.height, src.width, 4), dtype=np.uint8)
                    
                    # Read the specific band
                    band_data = src.read(config["band"])
                    
                    # Apply color where mask is present (value == 1)
                    # This sets both the color and alpha channel
                    mask_positions = band_data == 1
                    for i in range(4):  # RGBA channels
                        mask_img[mask_positions, i] = config["color"][i]
                    
                    # Convert to PIL Image with alpha channel
                    mask_pil = Image.fromarray(mask_img, mode='RGBA')
                    
                    # Save the mask as PNG (to preserve transparency)
                    output_file = os.path.join(masks_folder, f"{base_name}_{mask_name}.png")
                    mask_pil.save(output_file, format='PNG')
                    print(f"Saved {mask_name} mask to {output_file}")


if __name__ == "__main__":

    # Define folder paths
    udm_folder = r"D:\Training\YKD_SR\Lake_Ice_Breakup_2023_YKD_psscene_analytic_sr_udm2\PSScene"  # UDM files path
    rgb_folder = r"D:\Training\YKD_RGB\JPGs"  # RGB files path
    output_folder = r"D:\Testing\UDM_multi"  # Output folder path

    # Run the function
    process_udm_folder(udm_folder, rgb_folder, output_folder)