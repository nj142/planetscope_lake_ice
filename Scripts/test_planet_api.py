import os
import requests
from pathlib import Path
import json
from datetime import datetime
import sys
import yaml
import base64

def get_auth_header(api_key):
    """Create authorization header using Planet API key."""
    if not api_key:
        raise ValueError("Please provide a valid API key from the config file")
    
    # Create the proper Basic auth header by encoding the API key
    encoded_key = base64.b64encode(api_key.encode('utf-8')).decode('utf-8')
    return {'Authorization': f'Basic {encoded_key}'}

def get_item_id_from_filename(filename):
    """Extract item ID from Planet filename."""
    print(f"Processing filename: {filename}")
    return filename.split('_AnalyticMS_SR')[0]

def get_rgb_product_for_item(item_id, auth_header):
    """Get the RGB product information for a given item ID."""
    print(f"Searching for RGB product for item ID: {item_id}")
    
    # Include item type in the URL
    search_url = f'https://api.planet.com/data/v1/item-types/PSScene/items/{item_id}/assets'
    
    print(f"Making request to: {search_url}")
    response = requests.get(search_url, headers=auth_header)
    print(f"Response status code: {response.status_code}")
    
    if response.status_code == 404:
        # Try alternative item types if the first one fails
        alternative_types = ['PSScene4Band', 'PSScene8Band']
        for item_type in alternative_types:
            alt_url = f'https://api.planet.com/data/v1/item-types/{item_type}/items/{item_id}/assets'
            print(f"Trying alternative URL: {alt_url}")
            response = requests.get(alt_url, headers=auth_header)
            if response.status_code == 200:
                break
    
    if response.status_code != 200:
        raise Exception(f"Failed to get assets for item {item_id}. Status code: {response.status_code}. Response: {response.text}")
    
    assets = response.json()
    print(f"Available assets: {list(assets.keys())}")
    
    # Check for visual product with different possible keys
    visual_keys = ['visual', 'visual_rgb', 'rgb']
    available_visual_key = None
    for key in visual_keys:
        if key in assets:
            available_visual_key = key
            break
    
    if not available_visual_key:
        raise Exception(f"No RGB/Visual product available for item {item_id}. Available assets: {list(assets.keys())}")
    
    return assets[available_visual_key]

def activate_and_download(product_info, auth_header, output_path):
    """Activate asset if needed and download when ready."""
    print(f"Starting download process for: {output_path}")
    
    # Check if activation is needed
    if product_info['status'] != 'active':
        print("Product needs activation...")
        activation_url = product_info['_links']['activate']
        response = requests.post(activation_url, headers=auth_header)
        if response.status_code != 204 and response.status_code != 202:
            raise Exception(f"Failed to activate asset: {response.text}")
        print("Activation request sent successfully")
        
        # Wait for activation
        print("Waiting for activation...")
        max_attempts = 30
        attempt = 0
        while attempt < max_attempts:
            status_response = requests.get(activation_url, headers=auth_header)
            if status_response.json()['status'] == 'active':
                print("Asset activated successfully")
                break
            print(f"Activation in progress... (attempt {attempt + 1}/{max_attempts})")
            attempt += 1
            import time
            time.sleep(5)
    
    # Download the file
    print(f"Downloading from URL: {product_info['location']}")
    response = requests.get(product_info['location'], headers=auth_header, stream=True)
    
    if response.status_code != 200:
        raise Exception(f"Failed to download file: {response.text}")
    
    print(f"Writing file to: {output_path}")
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download completed successfully")

def process_folder(input_folder, output_folder, api_key):
    """Process all AnalyticMS_SR TIFFs in a folder and download corresponding RGB versions."""
    print("\n=== Starting Planet Labs Download Script ===")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    
    # Validate input folder exists
    if not os.path.exists(input_folder):
        raise ValueError(f"Input folder does not exist: {input_folder}")
    
    auth_header = get_auth_header(api_key)
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all AnalyticMS_SR TIFF files
    analytic_files = list(Path(input_folder).glob('*_AnalyticMS_SR*.tif*'))
    print(f"\nFound {len(analytic_files)} AnalyticMS_SR TIFF files")
    
    if not analytic_files:
        print("No AnalyticMS_SR TIFF files found in the input folder!")
        return
    
    # Process each AnalyticMS_SR TIFF
    for file in analytic_files:
        try:
            print(f"\nProcessing file: {file.name}")
            
            # Get item ID from filename
            item_id = get_item_id_from_filename(file.name)
            print(f"Extracted item ID: {item_id}")
            
            # Get RGB product information
            rgb_product = get_rgb_product_for_item(item_id, auth_header)
            
            # Create output filename
            output_path = Path(output_folder) / f"{item_id}_visual.tiff"
            
            # Download RGB version
            activate_and_download(rgb_product, auth_header, output_path)
            
            print(f"Successfully processed: {file.name}")
            
        except Exception as e:
            print(f"Error processing {file.name}: {str(e)}")
            print(f"Full error: {sys.exc_info()}")

if __name__ == "__main__":
    try:
        # Load API key from YAML config
        with open(r"D:\planetscope_lake_ice\planet.yaml", 'r') as file:
            config = yaml.safe_load(file)
        api_key = config['api_key']
        
        # Set input and output folders
        INPUT_FOLDER = r"D:\planetscope_lake_ice\Data_TEST\4- Planet SR TIFFs from API\Lake_Ice_Breakup_2021_YKD_psscene_analytic_sr_udm2\PSScene"
        OUTPUT_FOLDER = r"D:\planetscope_lake_ice\Data_TEST\2 - RGB JPGs for labeling will be saved here\Lake_Ice_Breakup_2021_YKD_RGB_psscene_visual"
        
        # Run the processing
        process_folder(INPUT_FOLDER, OUTPUT_FOLDER, api_key)
        
    except Exception as e:
        print(f"\nScript failed with error: {str(e)}")
        print(f"Full error: {sys.exc_info()}")
    
    print("\n=== Script Execution Completed ===")