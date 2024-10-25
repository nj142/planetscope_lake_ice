import os
from PIL import Image
import numpy as np
import rasterio

# Function to perform contrast stretching on an image
def contrast_stretch(array):
    lower_bound = np.percentile(array, 2)
    upper_bound = np.percentile(array, 98)
    clipped_array = np.clip(array, lower_bound, upper_bound)
    return ((clipped_array - lower_bound) / (upper_bound - lower_bound) * 255).astype(np.uint8)


# Function to convert a 4-band TIF to a 3-band RGB jpg using rasterio and Pillow
def convert_tif_to_jpg(tif_path, jpg_path):
    try:
        with rasterio.open(tif_path) as src:
            # Read the RGB bands (1, 2, 3)
            red = src.read(1)  # Band 1 - Red
            green = src.read(2)  # Band 2 - Green
            blue = src.read(3)  # Band 3 - Blue

            # Apply contrast stretching to each band
            red_stretched = contrast_stretch(red)
            green_stretched = contrast_stretch(green)
            blue_stretched = contrast_stretch(blue)

            # Stack the bands into a single RGB image
            rgb_image = np.stack((red, green, blue), axis=-1)

            # Convert to a Pillow image
            rgb_img = Image.fromarray(rgb_image)

            # Save as jpg
            rgb_img.save(jpg_path)
            print(f"Saved jpg: {jpg_path}")
    except Exception as e:
        print(f"Error converting {tif_path}: {e}")

# Function to convert all relevant TIF files to jpg in a given PlanetScope collection
def planetscope_collection_to_jpg(tif_folder, jpg_folder):
    converted_count = 0

    # Process each TIF file in the folder
    for file_name in os.listdir(tif_folder):
        if file_name.endswith(".tif") and not file_name.endswith("udm2.tif"):  # Skip udm2 files
            tif_path = os.path.join(tif_folder, file_name)
            jpg_name = os.path.splitext(file_name)[0] + ".jpg"
            jpg_path = os.path.join(jpg_folder, jpg_name)
            convert_tif_to_jpg(tif_path, jpg_path)
            converted_count += 1

    return converted_count

# Function to process all collections in a study site
def process_study_site(study_site_path):
    num_collections = 0
    num_converted = 0

    # jpgs folder created for each study site
    jpg_folder = os.path.join(study_site_path, "JPGs")  
    if not os.path.exists(jpg_folder):
        os.makedirs(jpg_folder)

    # Iterate through each downloaded collection of TIFFs in the study site folder
    for dir_name in os.listdir(study_site_path):
            collection_path = os.path.join(study_site_path, dir_name)

            # Check if it's a downloaded _psscene_ directory and not a ZIP file or your jpgs folder
            if os.path.isdir(collection_path) and "_psscene_" in dir_name:  
                tif_folder = os.path.join(collection_path, "PSScene")  # Path to the TIF files (since each order stores TIFFs in PSScene)
                
                num_converted += planetscope_collection_to_jpg(tif_folder, jpg_folder)
                num_collections += 1

    print(f"Processed {num_collections} collections, converted {num_converted} files in {study_site_path}")
    return num_collections, num_converted

################### MAIN CODE ###################

def main():
    """ This program is for converting your PlanetScope GEOTIFFs to jpgs for use in labeling"""
    
    base_directory = "D:/Training/"  # Root directory
    study_sites = ["YKD_RGB"]
    
    total_num_collections = 0
    total_num_converted_files = 0
    
    for study_site in study_sites:
        study_site_path = os.path.join(base_directory, study_site)
        
        num_collections, num_converted_files = process_study_site(study_site_path)
        total_num_collections += num_collections
        total_num_converted_files += num_converted_files
    
    print(f"\nConverted {total_num_converted_files} PlanetScope TIFs to jpg across {len(study_sites)} study sites.")

if __name__ == "__main__":
    main()
