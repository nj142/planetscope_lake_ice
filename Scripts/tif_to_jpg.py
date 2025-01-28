import os
from PIL import Image
import numpy as np
import rasterio

# SCRIPT ORDER #1 - CONVERT TIFFs to JPGs

def convert_tif_to_jpg(tif_path, jpg_path, band_config):
    """
    Convert a multi-band TIF to a 3-band RGB jpg using rasterio and Pillow
    
    Parameters:
    tif_path (str): Path to input TIF file
    jpg_path (str): Path to output JPG file
    band_config (dict): Dictionary containing band names and their indices
                       e.g., {'red': 1, 'green': 2, 'blue': 3}
    """
    try:
        with rasterio.open(tif_path) as src:
            # Read the RGB bands using the configured indices
            red = src.read(band_config['red'])
            green = src.read(band_config['green'])
            blue = src.read(band_config['blue'])

            # Stack the bands into a single RGB image
            rgb_image = np.stack((red, green, blue), axis=-1)

            # Convert to 8-bit (0-255) if necessary
            if rgb_image.dtype != np.uint8:
                rgb_image = (rgb_image / rgb_image.max() * 255).astype(np.uint8)

            # Convert to a Pillow image
            rgb_img = Image.fromarray(rgb_image)

            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(jpg_path), exist_ok=True)

            # Save as jpg
            rgb_img.save(jpg_path)
            print(f"Saved jpg: {jpg_path}")
            return True
    except Exception as e:
        print(f"Error converting {tif_path}: {e}")
        return False

def process_collection(base_dir, collection_name, output_base_dir, keyword, band_config):
    """
    Process a single collection and convert matching TIFs to JPGs
    
    Parameters:
    base_dir (str): Base directory containing collections
    collection_name (str): Name of the collection to process
    output_base_dir (str): Base directory for output JPGs
    keyword (str): Keyword to filter TIF files
    band_config (dict): Dictionary containing band names and their indices
    """
    collection_path = os.path.join(base_dir, collection_name)
    psscene_path = os.path.join(collection_path, "PSScene")
    
    # Create output directory for this collection
    output_collection_path = os.path.join(output_base_dir, collection_name)
    
    converted_count = 0
    
    if not os.path.exists(psscene_path):
        print(f"Warning: PSScene folder not found in {collection_path}")
        return converted_count

    # Process each TIF file in the PSScene folder
    for file_name in os.listdir(psscene_path):
        if (file_name.endswith(".tif") and 
            not file_name.endswith("udm2.tif") and 
            keyword.lower() in file_name.lower()):
            
            tif_path = os.path.join(psscene_path, file_name)
            jpg_name = os.path.splitext(file_name)[0] + ".jpg"
            jpg_path = os.path.join(output_collection_path, jpg_name)
            
            if convert_tif_to_jpg(tif_path, jpg_path, band_config):
                converted_count += 1

    return converted_count

if __name__ == "__main__":
    """Main function to process collections and convert TIFs to JPGs"""
    # Base directory containing the collections
    base_directory = r"D:\planetscope_lake_ice\Data_TEST\1 - Download your Planet RGB orders here"
    
    # Get the parent directory of the base directory (Data_TEST)
    parent_dir = os.path.dirname(base_directory)
    
    # Output directory for JPGs (parallel to input directory)
    output_base = os.path.join(parent_dir, "2 - RGB JPGs for labeling will be saved here")
    
    # List of collections to process
    collections = [
        "Lake_Ice_Breakup_2023_YF_RGB_psscene_visual"
        # Add more collections as needed
    ]
    
    # Keyword to filter TIF files - you can change this if you need to use this script to process SR images for some reason to read UDM or SR
    keyword = "Visual"
    
    # Band configuration dictionary
    # Modify these indices based on your TIF file band order. Note SR images are BGRNIR!
    band_config = {
        'red': 1,    # Red band index
        'green': 2,  # Green band index
        'blue': 3    # Blue band index
    }
    
    # Create output base directory if it doesn't exist
    os.makedirs(output_base, exist_ok=True)
    
    total_converted = 0
    
    # Process each collection
    for collection in collections:
        print(f"\nProcessing collection: {collection}")
        converted = process_collection(base_directory, collection, output_base, keyword, band_config)
        total_converted += converted
        print(f"Converted {converted} files from {collection}")
    
    print(f"\nTotal files converted: {total_converted} across {len(collections)} collections")