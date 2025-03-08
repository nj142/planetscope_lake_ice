import os
import glob
import traceback
from PIL import Image
import numpy as np
import rasterio

def convert_tif_to_jpg(tif_path, jpg_path):
    """
    Convert a 4-band (B, G, R, NIR) TIF to a 3-band RGB jpg using rasterio and Pillow
    
    Parameters:
    tif_path (str): Path to input TIF file
    jpg_path (str): Path to output JPG file
    """
    try:
        # Print detailed information about the file being processed
        print(f"Processing file: {tif_path}")
        
        with rasterio.open(tif_path) as src:
            # Print raster details to help diagnose issues
            print(f"  File info: {src.count} bands, dtype: {src.dtypes[0]}, shape: {src.shape}")
            
            # Make sure we have at least 3 bands for RGB
            if src.count < 3:
                print(f"  Error: Not enough bands ({src.count}). Need at least 3 for RGB.")
                return False
            
            # Read the B G R bands (assuming 4-band BGRNIR format)
            blue = src.read(1)
            green = src.read(2)
            red = src.read(3)
            
            # Print value ranges to help with scaling
            print(f"  Value ranges - R: [{red.min()}-{red.max()}], G: [{green.min()}-{green.max()}], B: [{blue.min()}-{blue.max()}]")
            
            # Stack the bands into a single RGB image (R, G, B order for PIL)
            rgb_image = np.stack((red, green, blue), axis=-1)
            
            # Print RGB array info
            print(f"  RGB array shape: {rgb_image.shape}, dtype: {rgb_image.dtype}")

            # Convert to 8-bit (0-255) if necessary
            if rgb_image.dtype != np.uint8:
                # PlanetScope-specific scaling: DN values typically range from 0-4000
                # Scale to 0-255 with a bit of brightness enhancement
                print(f"  Applying PlanetScope-specific scaling (DN to 8-bit)")
                # Scale values assuming max DN around 4000, but clip at 255
                # Using 4000 as the denominator instead of 10000 to enhance brightness
                rgb_image = np.clip(rgb_image * (255.0 / 4000.0), 0, 255).astype(np.uint8)
                
                # Print the new value range after scaling
                print(f"  After scaling - min: {rgb_image.min()}, max: {rgb_image.max()}")

            # Convert to a Pillow image
            print("  Converting to PIL image")
            rgb_img = Image.fromarray(rgb_image)

            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(jpg_path), exist_ok=True)

            # Save as jpg
            print(f"  Saving to: {jpg_path}")
            rgb_img.save(jpg_path)
            print(f"  Successfully saved jpg")
            return True
    except Exception as e:
        print(f"Error converting {tif_path}: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        return False

def process_collection(base_dir, collection_name, output_base_dir):
    """
    Process a single collection and convert _AnalyticMS_SR.tif files to JPGs
    
    Parameters:
    base_dir (str): Base directory containing collections
    collection_name (str): Name of the collection to process
    output_base_dir (str): Base directory for output JPGs
    """
    # Create full path to the collection
    collection_path = os.path.join(base_dir, collection_name)
    
    # Create output directory for this collection
    output_collection_path = os.path.join(output_base_dir, collection_name)
    os.makedirs(output_collection_path, exist_ok=True)
    
    # Print the path we're searching to help with debugging
    print(f"Searching for TIFs in: {collection_path}")
    
    # Use glob to find all TIF files matching the pattern in any subfolder
    pattern = os.path.join(collection_path, "**", "*_AnalyticMS_SR.tif")
    tif_files = glob.glob(pattern, recursive=True)
    
    # Print the number of files found to help with debugging
    print(f"Found {len(tif_files)} TIF files matching the pattern")
    
    # If we didn't find any files, try an alternative pattern
    if len(tif_files) == 0:
        print("No files found matching the specific pattern. Trying alternatives...")
        pattern = os.path.join(collection_path, "**", "*.tif")
        all_tifs = glob.glob(pattern, recursive=True)
        print(f"Found {len(all_tifs)} total TIF files")
        
        # Print some example filenames
        if all_tifs:
            print("Example TIF filenames:")
            for i, tif in enumerate(all_tifs[:5]):
                print(f"  {i+1}. {os.path.basename(tif)}")
    
    converted_count = 0
    
    for tif_path in tif_files:
        # Skip only files that specifically end with _udm2.tif
        if tif_path.endswith("_udm2.tif"):
            print(f"Skipping UDM2 file: {os.path.basename(tif_path)}")
            continue
        
        # Skip any files that are in a zip file
        if ".zip" in tif_path:
            print(f"Skipping file in zip: {os.path.basename(tif_path)}")
            continue
            
        # Extract base filename for the jpg
        file_name = os.path.basename(tif_path)
        jpg_name = os.path.splitext(file_name)[0] + ".jpg"
        jpg_path = os.path.join(output_collection_path, jpg_name)
        
        print(f"Attempting to convert: {file_name}")
        if convert_tif_to_jpg(tif_path, jpg_path):
            converted_count += 1
            print(f"Successfully converted file {converted_count}: {file_name}")
        else:
            print(f"Failed to convert: {file_name}")

    return converted_count

def main():
    """Main function to process collections and convert TIFs to JPGs"""
    # Base directory containing the collections
    base_directory = r"D:\planetscope_lake_ice\Data (Unclassified)\3.5 - Freeze Up Time Series Input (Planet downloads)"
    
    # Output directory for JPGs
    output_base = r"D:\planetscope_lake_ice\Data (Unclassified)\4 - Freeze Up Time Series Output\JPEGs for Labeling"
    
    # List of collections to process
    collections = [
        "Lake_Ice_Freezeup_2019_2020_YKD_psscene_analytic_sr_udm2"
        # Add more collections as needed
    ]
    
    # Create output base directory if it doesn't exist
    os.makedirs(output_base, exist_ok=True)
    
    # Check if the base directory exists
    if not os.path.exists(base_directory):
        print(f"Error: Base directory does not exist: {base_directory}")
        return
    
    # Process each collection
    total_converted = 0
    for collection in collections:
        collection_path = os.path.join(base_directory, collection)
        if not os.path.exists(collection_path):
            print(f"Warning: Collection directory does not exist: {collection_path}")
            continue
            
        print(f"\nProcessing collection: {collection}")
        converted = process_collection(base_directory, collection, output_base)
        total_converted += converted
        print(f"Converted {converted} files from {collection}")
    
    print(f"\nTotal files converted: {total_converted} across {len(collections)} collections")

if __name__ == "__main__":
    main()