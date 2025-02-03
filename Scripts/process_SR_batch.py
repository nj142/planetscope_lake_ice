import os
import glob
from clip_ALPOD_to_SR_extent import clip_vector_to_raster
from mask_clouds_and_classify_ice import create_masked_raster, classify_ice_cover
from calculate_ice_cover_statistics_per_lake import update_lake_statistics

def process_sr_image(sr_path, config, site):
    """
    Process a single SR image.
    """
    try:
        base_name = os.path.basename(sr_path)
        print(f"\nProcessing {base_name}...")
        
        # Create site-specific paths
        site_rasters_dir = os.path.join(config['output_rasters_dir'], site)
        site_shapefiles_dir = os.path.join(config['output_shapefiles_dir'], site)
        
        # Create temporary directory for this image's shapefiles
        temp_dir = os.path.join(site_shapefiles_dir, os.path.splitext(base_name)[0])
        os.makedirs(temp_dir, exist_ok=True)
        
        # 1. Clip vector to UDM valid data extent
        print("Clipping lakes to valid UDM data extent (excluding padded areas)...")
        clipped_vectors = os.path.join(temp_dir, "clipped_lakes.shp")
        
        # Find corresponding UDM file
        udm_name = base_name.replace('_AnalyticMS_SR.tif', '_udm2.tif')
        udm_path = os.path.join(os.path.dirname(sr_path), udm_name)
        if not os.path.exists(udm_path):
            print(f"Could not find UDM file: {udm_path}")
            print("Checking directory contents...")
            for f in os.listdir(os.path.dirname(sr_path)):
                if 'udm' in f.lower():
                    print(f"Found UDM file: {f}")
            raise ValueError(f"UDM file not found for {base_name}")
            
        print(f"Found UDM file: {udm_name}")
        
        # Use the UDM file (band 8) to clip the vector
        features_kept = clip_vector_to_raster(
            config['vector_path'],
            udm_path,
            clipped_vectors
        )
        print(f"Lakes kept: {features_kept}")
        
        # 2. Create masked raster
        print("Removing unusable data...")
        site_masked_dir = os.path.join(site_rasters_dir, 'Masked')
        os.makedirs(site_masked_dir, exist_ok=True)
        masked_path = os.path.join(site_masked_dir, base_name)
        create_masked_raster(
            sr_path,
            udm_path,
            clipped_vectors,
            config['mask_bands'],
            config['keep_bands'],
            masked_path
        )
        print(f"Masked {', '.join(['band ' + str(b) for b in config['keep_bands']])} saved to {masked_path}")
        
        # 3. Classify ice cover
        print("Classifying ice cover...")
        site_classified_dir = os.path.join(site_rasters_dir, 'Classified')
        os.makedirs(site_classified_dir, exist_ok=True)
        classified_path = os.path.join(site_classified_dir, base_name)
        classify_ice_cover(
            masked_path,
            config['thresholds'],
            classified_path
        )
        class_labels = [f"{i+1} = {name}" for i, name in enumerate(config['thresholds'].keys())]
        print(f"Categorical classified ice mask saved with {', '.join(class_labels)}")
        
        print("Processing completed successfully!")
        
    except Exception as e:
        print(f"Error processing {sr_path}: {str(e)}")

def main():
    # Input folders to process
    input_folders = {
        'YF': [
            r"D:\planetscope_lake_ice\Data\4 - Planet SR TIFFs from API\Lake_Ice_Breakup_2017_YF_psscene_analytic_sr_udm2\PSScene",
            r"D:\planetscope_lake_ice\Data\4 - Planet SR TIFFs from API\Lake_Ice_Breakup_2019_YF_psscene_analytic_sr_udm2\PSScene",
            r"D:\planetscope_lake_ice\Data\4 - Planet SR TIFFs from API\Lake_Ice_Breakup_2021_YF_psscene_analytic_sr_udm2\PSScene",
            r"D:\planetscope_lake_ice\Data\4 - Planet SR TIFFs from API\Lake_Ice_Breakup_2023_YF_psscene_analytic_sr_udm2\PSScene",
        ],
        'YKD': [
            r"D:\planetscope_lake_ice\Data\4 - Planet SR TIFFs from API\Lake_Ice_Breakup_2017_YKD_psscene_analytic_sr_udm2\PSScene",
            r"D:\planetscope_lake_ice\Data\4 - Planet SR TIFFs from API\Lake_Ice_Breakup_2019_YKD_psscene_analytic_sr_udm2\PSScene",
            r"D:\planetscope_lake_ice\Data\4 - Planet SR TIFFs from API\Lake_Ice_Breakup_2021_YKD_psscene_analytic_sr_udm2\PSScene",
            r"D:\planetscope_lake_ice\Data\4 - Planet SR TIFFs from API\Lake_Ice_Breakup_2023_YKD_psscene_analytic_sr_udm2\PSScene"
        ]
    }
    
    # Configuration
    config = {
        'vector_path': r"D:\planetscope_lake_ice\Data\8 - Download ALPOD data here\ALPODlakes.shp",
        'output_rasters_dir': r"D:\planetscope_lake_ice\Data\11 - Break Up Time Series Output\Rasters",
        'output_shapefiles_dir': r"D:\planetscope_lake_ice\Data\11 - Break Up Time Series Output\Shapefiles",
        'mask_bands': [3, 6],
        'keep_bands': [3],
        'thresholds': {
            'Ice': (950, 3800),
            'Snow': (3800, float('inf')),
            'Water': (float('-inf'), 950)
        },
        'min_clear_percent': 50
    }
    
    # Create base output directories
    os.makedirs(config['output_rasters_dir'], exist_ok=True)
    os.makedirs(config['output_shapefiles_dir'], exist_ok=True)
    
    # Create site-specific directories
    for site in input_folders.keys():
        site_rasters_dir = os.path.join(config['output_rasters_dir'], site)
        site_shapefiles_dir = os.path.join(config['output_shapefiles_dir'], site)
        os.makedirs(site_rasters_dir, exist_ok=True)
        os.makedirs(site_shapefiles_dir, exist_ok=True)
    
    # Process all input folders
    total_files = 0
    for site, folders in input_folders.items():
        for folder in folders:
            print(f"\nProcessing {site} folder: {folder}")
            sr_files = glob.glob(os.path.join(folder, '*_SR.tif'))
            if not sr_files:
                print(f"No SR files found in {folder}")
                continue
            
            print(f"Found {len(sr_files)} SR files to process")
            total_files += len(sr_files)
            
            # Process each SR image
            for sr_file in sr_files:
                process_sr_image(sr_file, config, site)
    
    print(f"\nAll processing completed! Processed {total_files} files")

if __name__ == "__main__":
    main()