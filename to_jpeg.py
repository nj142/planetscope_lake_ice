import os
from PIL import Image
import numpy as np
import rasterio

# Function to perform histogram stretching on an image
def histogram_stretch(image_array):
    """Apply histogram stretching to enhance image contrast."""
    # Calculate the minimum and maximum values of the image
    min_val = np.min(image_array)
    max_val = np.max(image_array)

    # Stretch the image
    stretched = (image_array - min_val) / (max_val - min_val) * 255
    stretched = np.clip(stretched, 0, 255).astype(np.uint8)
    
    return stretched

# Function to convert a 4-band TIF to a 3-band RGB JPEG using rasterio and Pillow
def convert_tif_to_jpeg(tif_path, jpeg_path):
    try:
        with rasterio.open(tif_path) as src:
            # Read the RGB bands (1, 2, 3)
            red = src.read(1)  # Band 1 - Red
            green = src.read(2)  # Band 2 - Green
            blue = src.read(3)  # Band 3 - Blue

            # Apply histogram stretching to each band
            red_stretched = histogram_stretch(red)
            green_stretched = histogram_stretch(green)
            blue_stretched = histogram_stretch(blue)

            # Stack the bands into a single RGB image
            rgb_image = np.stack((red_stretched, green_stretched, blue_stretched), axis=-1)

            # Convert to a Pillow image
            rgb_img = Image.fromarray(rgb_image)

            # Save as JPEG
            rgb_img.save(jpeg_path, format='JPEG')
            print(f"Saved JPEG: {jpeg_path}")
    except Exception as e:
        print(f"Error converting {tif_path}: {e}")

# Function to convert all relevant TIF files to JPEG in a given PlanetScope collection
def planetscope_collection_to_jpeg(tif_folder, jpeg_folder):
    converted_count = 0

    # Process each TIF file in the folder
    for file_name in os.listdir(tif_folder):
        if file_name.endswith(".tif") and not file_name.endswith("udm2.tif"):  # Skip udm2 files
            tif_path = os.path.join(tif_folder, file_name)
            jpeg_name = os.path.splitext(file_name)[0] + ".jpeg"
            jpeg_path = os.path.join(jpeg_folder, jpeg_name)
            convert_tif_to_jpeg(tif_path, jpeg_path)
            converted_count += 1

    return converted_count

# Function to process all collections in a study site
def process_study_site(study_site_path):
    num_collections = 0
    num_converted = 0

    # JPEGs folder created for each study site
    jpeg_folder = os.path.join(study_site_path, "JPEGs")  
    if not os.path.exists(jpeg_folder):
        os.makedirs(jpeg_folder)

    # Iterate through each downloaded collection of TIFFs in the study site folder
    for dir_name in os.listdir(study_site_path):
            collection_path = os.path.join(study_site_path, dir_name)

            # Check if it's a downloaded _psscene_ directory and not a ZIP file or your JPEGs folder
            if os.path.isdir(collection_path) and "_psscene_" in dir_name:  
                tif_folder = os.path.join(collection_path, "PSScene")  # Path to the TIF files (since each order stores TIFFs in PSScene)
                
                num_converted += planetscope_collection_to_jpeg(tif_folder, jpeg_folder)
                num_collections += 1

    print(f"Processed {num_collections} collections, converted {num_converted} files in {study_site_path}")
    return num_collections, num_converted

################### MAIN CODE ###################

def main():
    """ This program is for converting your PlanetScope GEOTIFFs to JPEGs for use in LabelBox"""
    
    base_directory = "D:/Training/"  # Root directory
    study_sites = ["YF","YKD"]
    
    total_num_collections = 0
    total_num_converted_files = 0
    
    for study_site in study_sites:
        study_site_path = os.path.join(base_directory, study_site)
        
        num_collections, num_converted_files = process_study_site(study_site_path)
        total_num_collections += num_collections
        total_num_converted_files += num_converted_files
    
    print(f"\nConverted {total_num_converted_files} PlanetScope TIFs to JPEG across {len(study_sites)} study sites.")

if __name__ == "__main__":
    main()