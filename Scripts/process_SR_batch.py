import os
import glob
from clip_ALPOD_to_SR_extent import clip_vector_to_raster
from mask_clouds_and_classify_ice import create_masked_raster, classify_ice_cover
from calculate_ice_cover_statistics_per_lake import calculate_lake_statistics

def process_sr_image(sr_path, config, site):
    """
    Processes a single Surface Reflectance (SR) image for lake ice analysis:

    1. Clip ALPOD vector data from the entire state to just those lakes completely within the extent of the current SR image.

    2. Clip the current SR raster to only those pixels usable in our classification analysis. (This saves tons of file space.)
        a. Remove any unneeded bands (we throw out everything but band 3 of the SR image-- all we need for red thresholding.)
        b. Rasterize the clipped ALPOD lakes vector from step 1. All pixels falling outside a lake pixel are saved as -9999 (nodata.) 
        c. Exclude any pixels marked as cloud/haze/shadow in the corresponding bands of the Planet Usable Data Mask (UDM) file by setting to -9999.
        
        This leaves only cloud-free, haze-free, shadow-free lake pixels in our final single band raster, saved to the study site's
        "Masked" folder.

    3. Classify ice cover on the trimmed raster image from step 2.
        (All remaining pixels are assumed to be valid, so we use red threshold on all pixels not marked -9999 nodata.)

        The classified masks are saved to the study site's "Classified" folder.
    
    4. Run lake statistics on the classified rasters.  Exclude lakes with too few % valid pixels.

    """
    try:
        # -------------------------------------------------------------------------
        # 0. Find corresponding Planet's Usable Data Mask (UDM) for given SR image. 
        #    Also initialize output folders / files.
        # ------------------------------------------------------------------------- 
        # "base_name" is just the root name of each image.  Used to find corresponding UDM mask for SR images
        base_name = os.path.basename(sr_path)
        print(f"\nProcessing {base_name}...")
        
        site_rasters_dir = os.path.join(config['output_rasters_dir'], site)
        site_shapefiles_dir = os.path.join(config['output_shapefiles_dir'], site)      
        temp_dir = os.path.join(site_shapefiles_dir, os.path.splitext(base_name)[0])
        os.makedirs(temp_dir, exist_ok=True)
        clipped_vector_path = os.path.join(temp_dir, "clipped_lakes.shp")

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
                
        # -------------------------------------------------------------------------
        # 1. Clip the given large ALPOD vector dataset to just the SR image's extent.
        #     Save the clipped vector to the clipped_vector_path.
        # ------------------------------------------------------------------------- 
        print("Clipping lakes to valid UDM data extent (excluding padded areas)...")

        # Use the UDM file to clip the vector
        features_kept = clip_vector_to_raster(
            config['vector_path'],
            udm_path,
            clipped_vector_path
        )
        print(f"Lakes kept: {features_kept}")
        
        # -------------------------------------------------------------------------
        # 2. Mask out the haze/cloud/shadow layers (or whichever "mask_bands" are selected)
        #      and save just the pixels from the red band (or whichever "keep_bands" are selected)
        #      which are contained within the vector outlines from step 1, used like a cookie cutter.
        # ------------------------------------------------------------------------- 
        print("Removing unusable data...")
        site_masked_dir = os.path.join(site_rasters_dir, 'Masked')
        os.makedirs(site_masked_dir, exist_ok=True)
        masked_path = os.path.join(site_masked_dir, base_name)
        create_masked_raster(
            sr_path,
            udm_path,
            clipped_vector_path,
            config['mask_bands'],
            config['keep_bands'],
            masked_path
        )
        print(f"Masked {', '.join(['band ' + str(b) for b in config['keep_bands']])} saved to {masked_path}")
        
        # -------------------------------------------------------------------------
        # 3. Classify Ice, Snow, and Water (or whatever given classes are) using band
        #     thresholding on the "keep" band.  (For our cases, this is red band thresholding.)
        # ------------------------------------------------------------------------- 
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
        
        # -------------------------------------------------------------------------
        # 4. Calculate lake statistics for the lake, and save to the final CSV file.
        # ------------------------------------------------------------------------- 
        print("Calculating lake statistics...")
        try:
            base_name = os.path.basename(sr_path)
            print(f"\nProcessing {base_name} for lake statistics...")
            sr_name = os.path.splitext(base_name)[0]
            
            # Construct the file path for the clipped vector.
            clipped_vector_path = os.path.join(config['output_shapefiles_dir'], site, sr_name, "clipped_lakes.shp")
            if not os.path.exists(clipped_vector_path):
                raise ValueError(f"Clipped vector file not found: {clipped_vector_path}")
            
            # Construct the file path for the classified raster.
            classified_path = os.path.join(config['output_rasters_dir'], site, "Classified", base_name)
            if not os.path.exists(classified_path):
                raise ValueError(f"Classified raster file not found: {classified_path}")
            
            # Temporarily override the vector path in the config with the clipped vector.
            original_vector = config['vector_path']
            config['vector_path'] = clipped_vector_path
            
            # Calculate lake statistics for this SR image.
            calculate_lake_statistics(classified_path, sr_name, config)
            
            # Restore the original vector path.
            config['vector_path'] = original_vector
        
        except Exception as e:
            print(f"Error processing {sr_path}: {str(e)}")
            
    except Exception as e:
        print(f"Error processing {sr_path}: {str(e)}")

def main():
    # Input folders to process (You can just unzip your PlanetScope downloads in a project folder and copy the filepath here.)  Use the dictionary below to organize study sites.  
    # I recommend downloading imagery in batches by year if doing time series.  Make sure your input folders contain both the UDM masks and the SR image files.
    input_folders = {
        'YF': [
            r"D:\planetscope_lake_ice\Data (Unclassified)\1 - Break Up Time Series Input\Lake_Ice_Breakup_2023_YF_psscene_analytic_sr_udm2\PSScene",
            r"D:\planetscope_lake_ice\Data (Unclassified)\1 - Break Up Time Series Input\Lake_Ice_Breakup_2017_YF_psscene_analytic_sr_udm2\PSScene",
            r"D:\planetscope_lake_ice\Data (Unclassified)\1 - Break Up Time Series Input\Lake_Ice_Breakup_2019_YF_psscene_analytic_sr_udm2\PSScene",
            r"D:\planetscope_lake_ice\Data (Unclassified)\1 - Break Up Time Series Input\Lake_Ice_Breakup_2021_YF_psscene_analytic_sr_udm2\PSScene"
            ],
        'YKD': [
            r"D:\planetscope_lake_ice\Data (Unclassified)\1 - Break Up Time Series Input\Lake_Ice_Breakup_2019_YKD_psscene_analytic_sr_udm2\PSScene",
            r"D:\planetscope_lake_ice\Data (Unclassified)\1 - Break Up Time Series Input\Lake_Ice_Breakup_2021_YKD_psscene_analytic_sr_udm2\PSScene",
            r"D:\planetscope_lake_ice\Data (Unclassified)\1 - Break Up Time Series Input\Lake_Ice_Breakup_2023_YKD_psscene_analytic_sr_udm2\PSScene",
            r"D:\planetscope_lake_ice\Data (Unclassified)\1 - Break Up Time Series Input\Lake_Ice_Breakup_2017_YKD_psscene_analytic_sr_udm2\PSScene"
            ]
    }
    
    # Change config if needed for other use case
    config = {
        'vector_path': r"D:\planetscope_lake_ice\Data (Validation)\8 - Download ALPOD data here\ALPODlakes.shp",
        'output_rasters_dir': r"D:\planetscope_lake_ice\Data (Unclassified)\2 - Break Up Time Series Output\Rasters",
        'output_shapefiles_dir': r"D:\planetscope_lake_ice\Data (Unclassified)\2 - Break Up Time Series Output\Shapefiles",
        'time_series_csv': r"D:\planetscope_lake_ice\Data (Unclassified)\2 - Break Up Time Series Output\lake_statistics.csv",
        'mask_bands': [3, 4, 6],
        'keep_bands': [3],
        'thresholds': {
            'Ice': (950, 3800),
            'Snow': (3800, float('inf')),
            'Water': (float('-inf'), 950)
        },
        'min_clear_percent': 30
    }
    
   # Create necessary output directories.
    os.makedirs(config['output_rasters_dir'], exist_ok=True)
    os.makedirs(config['output_shapefiles_dir'], exist_ok=True)
    csv_dir = os.path.dirname(config['time_series_csv'])
    os.makedirs(csv_dir, exist_ok=True)
    
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
            
            for sr_file in sr_files:
                process_sr_image(sr_file, config, site)
    
    print(f"\nAll processing completed! Processed {total_files} files")

if __name__ == "__main__":
    main()