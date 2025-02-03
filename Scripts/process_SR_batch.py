import os
import glob
from clip_ALPOD_to_SR_extent import clip_vector_to_raster
from mask_clouds_and_classify_ice import create_masked_raster, classify_ice_cover
from calculate_ice_cover_statistics_per_lake import update_lake_statistics

def process_sr_image(sr_path, config):
    """
    Process a single SR image.
    """
    try:
        base_name = os.path.basename(sr_path)
        print(f"\nProcessing {base_name}...")
        
        # Create temporary directory for this image.
        temp_dir = os.path.join(config['temp_dir'], os.path.splitext(base_name)[0])
        os.makedirs(temp_dir, exist_ok=True)
        
        # 1. Clip vector to UDM valid data extent (using our fast ogr2ogr with coordinate transformation)
        print("Clipping lakes to valid UDM data extent (excluding padded areas)...")
        clipped_vectors = os.path.join(temp_dir, "clipped_lakes.shp")
        
        # Find corresponding UDM file.
        # Replace _AnalyticMS_SR with _udm2 in the file name.
        udm_name = base_name.replace('_AnalyticMS_SR.tif', '_udm2.tif')
        udm_path = os.path.join(config['psscene_dir'], udm_name)
        if not os.path.exists(udm_path):
            print(f"Could not find UDM file: {udm_path}")
            print("Checking directory contents...")
            for f in os.listdir(config['psscene_dir']):
                if 'udm' in f.lower():
                    print(f"Found UDM file: {f}")
            raise ValueError(f"UDM file not found for {base_name}")
            
        print(f"Found UDM file: {udm_name}")
        
        # Use the UDM file (band 8) to clip the vector.
        features_kept = clip_vector_to_raster(
            config['vector_path'],
            udm_path,
            clipped_vectors
        )
        print(f"Lakes kept: {features_kept}")
        
        # 2. Create masked raster (using the original SR image and the UDM file as before)
        print("Removing unusable data...")
        masked_dir = os.path.join(config['output_dir'], os.path.splitext(base_name)[0])
        os.makedirs(masked_dir, exist_ok=True)
        
        masked_path = os.path.join(masked_dir, "masked.tif")
        create_masked_raster(
            sr_path,
            udm_path,
            clipped_vectors,
            config['mask_bands'],
            config['keep_bands'],
            masked_path
        )
        print(f"Masked {', '.join(['band ' + str(b) for b in config['keep_bands']])} saved to {masked_dir}")
        
        # 3. Classify ice cover
        print("Classifying ice cover...")
        classified_path = os.path.join(masked_dir, "classified.tif")
        classify_ice_cover(
            masked_path,
            config['thresholds'],
            classified_path
        )
        class_labels = [f"{i+1} = {name}" for i, name in enumerate(config['thresholds'].keys())]
        print(f"Categorical classified ice mask saved with {', '.join(class_labels)}")
        
        # 4. Update lake statistics
        print("Calculating lake statistics...")
        update_lake_statistics(
            config['vector_path'],
            classified_path,
            os.path.splitext(base_name)[0],
            config['min_clear_percent']
        )
        
        print("Processing completed successfully!")
        
    except Exception as e:
        print(f"Error processing {sr_path}: {str(e)}")

def main():
    # Configuration
    config = {
        'psscene_dir': r"D:\planetscope_lake_ice\Data\10 - Time Series Output\Time_Series_Test_Lake_Ice_Freezeup_2024_YKD_psscene_analytic_sr_udm2\PSScene",
        'vector_path': r"D:\planetscope_lake_ice\Data\8 - Download ALPOD data here\ALPODlakes.shp",
        'output_dir': r"D:\planetscope_lake_ice\Data\10 - Time Series Output\Usable Ice Cover Data",
        'temp_dir': r"D:\planetscope_lake_ice\Data\10 - Time Series Output\SR Extent Lake Shapefiles",
        
        # Bands to mask out (e.g. cloud and cloud shadow)
        'mask_bands': [3, 6],
        
        # Bands to keep from SR image
        'keep_bands': [3],
        
        # Classification thresholds:
        # Each key will be assigned a numeric class (1, 2, 3, …).
        'thresholds': {
            'Ice': (950, 3800),
            'Snow': (3800, float('inf')),
            'Water': (float('-inf'), 950)
        },
        
        # Minimum percentage of clear pixels required
        'min_clear_percent': 50
    }
    
    # Create output directories.
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['temp_dir'], exist_ok=True)
    
    # Get all SR files.
    sr_files = glob.glob(os.path.join(config['psscene_dir'], '*_SR.tif'))
    if not sr_files:
        print(f"No SR files found in {config['psscene_dir']}")
        return
    
    print(f"Found {len(sr_files)} SR files to process")
    
    # Process each SR image.
    for sr_file in sr_files:
        process_sr_image(sr_file, config)
    
    print("\nAll processing completed!")

if __name__ == "__main__":
    main()