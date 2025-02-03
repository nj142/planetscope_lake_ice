import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from numba import jit

def create_masked_raster(sr_path, udm_path, lake_vectors_path, mask_bands, keep_bands, output_path):
    """
    Create binary mask from lakes and clouds, then apply to red band.
    """
    try:
        print(f"\nProcessing surface reflectance image: {os.path.basename(sr_path)}")
        print(f"Reading red band (band {keep_bands[0]})")
        
        # Open SR image to get metadata and red band
        with rasterio.open(sr_path) as src:
            sr_meta = src.meta.copy()
            red_band = src.read(keep_bands[0])  # Assuming red band is specified in keep_bands
            
        print(f"Creating lake mask from: {os.path.basename(lake_vectors_path)}")
        # Create lakes mask (1 inside lakes, 0 outside)
        lakes_gdf = gpd.read_file(lake_vectors_path).to_crs(sr_meta['crs'])
        lake_mask = rasterize(
            shapes=[(geom, 1) for geom in lakes_gdf.geometry],
            out_shape=(sr_meta['height'], sr_meta['width']),
            transform=sr_meta['transform'],
            fill=0,
            dtype='uint8'
        )

        print(f"Reading cloud mask bands {mask_bands} from: {os.path.basename(udm_path)}")
        # Create cloud mask (1 for clear, 0 for cloudy)
        with rasterio.open(udm_path) as udm:
            cloud_mask = np.ones((sr_meta['height'], sr_meta['width']), dtype='uint8')
            for band_idx in mask_bands:
                cloud_mask &= (udm.read(band_idx) == 0)

        print("Applying combined lake and cloud mask to red band")
        
        @jit(nopython=True)
        def apply_masks(red_band, lake_mask, cloud_mask):
            """
            Numba-accelerated mask application using NumPy arrays.
            """
            output = np.full_like(red_band, -9999, dtype=np.float32)
            height, width = red_band.shape
            for i in range(height):
                for j in range(width):
                    if lake_mask[i, j] and cloud_mask[i, j]:
                        output[i, j] = red_band[i, j]
            return output
        
        # Ensure inputs are NumPy arrays with correct types
        red_band = np.ascontiguousarray(red_band, dtype=np.float32)
        lake_mask = np.ascontiguousarray(lake_mask, dtype=np.uint8)
        cloud_mask = np.ascontiguousarray(cloud_mask, dtype=np.uint8)
        
        # Apply masks using accelerated function
        red_band = apply_masks(red_band, lake_mask, cloud_mask)
        
        # Update metadata for output
        sr_meta.update({
            "count": 1,
            "dtype": 'float32',
            "nodata": -9999
        })

        # Save masked red band
        with rasterio.open(output_path, "w", **sr_meta) as dest:
            dest.write(red_band.astype('float32'), 1)
            
        print(f"Saved masked red band to: {output_path}")
        return output_path

    except Exception as e:
        print(f"Error in create_masked_raster: {str(e)}")
        raise

@jit(nopython=True)
def classify_pixels(data, thresholds_list):
    """
    Efficient pixel classification using Numba.
    """
    out = np.full(data.shape, -9999, dtype=np.float32)
    height, width = data.shape
    n_classes = len(thresholds_list)
    
    for i in range(height):
        for j in range(width):
            val = data[i, j]
            if val != -9999:
                for class_idx in range(n_classes):
                    low, high = thresholds_list[class_idx]
                    if low <= val < high:
                        out[i, j] = class_idx + 1  # Add 1 to match 1-based class numbering
                        break
    return out

def classify_ice_cover(input_path, thresholds, output_path):
    """
    Classify ice cover using value ranges.
    """
    print(f"\nClassifying ice cover in: {os.path.basename(input_path)}")
    print("Using classification scheme:")
    for class_name, (low, high) in thresholds.items():
        print(f"  {class_name}: values from {low} to {high}")
    
    with rasterio.open(input_path) as src:
        red_band = src.read(1)
        meta = src.meta.copy()
    
    print("Applying classification to valid pixels...")
    # Convert thresholds to list of tuples
    thresholds_list = [v for v in thresholds.values()]
    
    # Classify valid pixels
    classified = classify_pixels(red_band, np.array(thresholds_list, dtype=np.float32))
    
    # Save classification
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(classified.astype('float32'), 1)
    
    print("Output classes:")
    for idx, (class_name, _) in enumerate(thresholds.items(), start=1):
        print(f"  Class {idx}: {class_name}")
    print(f"Saved classified raster to: {output_path}")
    
    return output_path