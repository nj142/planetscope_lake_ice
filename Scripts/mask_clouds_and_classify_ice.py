import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from numba import jit

def create_mask_rasters(planetscope_image_path, udm_path, lake_vectors_path, mask_bands, keep_band, masked_path, lake_id_path):
    try:
        # Open SR image to get metadata and specified band
        with rasterio.open(planetscope_image_path) as src:
            image_meta = src.meta.copy()
            # Read the specified band number and squeeze to 2D
            band_data = src.read(keep_band).squeeze()  # This removes the singleton dimension
    
        # Create lakes mask with lake IDs
        lakes_gdf = gpd.read_file(lake_vectors_path).to_crs(image_meta['crs'])
        
        # Ensure 'id' field exists in the GeoDataFrame
        if 'id' not in lakes_gdf.columns:
            print("     Warning: 'id' field not found in lake vector data. Using index as ID.")
            lakes_gdf['id'] = lakes_gdf.index + 1  # Add 1 to avoid 0 which indicates outside lakes
            
        lake_id_dtype = 'uint32'
        
        # Create a raster with lake IDs (pixel value = lake ID, 0 outside lakes)
        lake_id_raster = rasterize(
            shapes=[(geom, id) for geom, id in zip(lakes_gdf.geometry, lakes_gdf['id'])],
            out_shape=(image_meta['height'], image_meta['width']),
            transform=image_meta['transform'],
            fill=0,
            dtype=lake_id_dtype
        )
        
        # Create a binary lake mask (1 inside any lake, 0 outside)
        lake_mask = (lake_id_raster > 0).astype('uint8')

        print(f"     Reading cloud mask bands {mask_bands} from: {os.path.basename(udm_path)}")
        # Create cloud mask (1 for clear, 0 for cloudy)
        with rasterio.open(udm_path) as udm:
            cloud_mask = np.ones((image_meta['height'], image_meta['width']), dtype='uint8')
            for band_idx in mask_bands:
                cloud_mask &= (udm.read(band_idx) == 0)
                
        # Save the lake ID raster
        lake_id_meta = image_meta.copy()
        lake_id_meta.update({
            "count": 1,
            "dtype": lake_id_dtype,
            "nodata": 0
        })
        
        with rasterio.open(lake_id_path, "w", **lake_id_meta) as dest:
            dest.write(lake_id_raster, 1)
        print(f"     Saved lake ID raster.")

        print("     Applying combined lake and cloud mask to red band")
        
        @jit(nopython=True)
        def apply_masks(band_data, lake_mask, cloud_mask):
            """
            Numba-accelerated mask application using NumPy arrays.
            """
            output = np.full_like(band_data, -9999, dtype=np.float32)
            height, width = band_data.shape
            for i in range(height):
                for j in range(width):
                    if lake_mask[i, j] and cloud_mask[i, j]:
                        output[i, j] = band_data[i, j]
            return output
        
        # Ensure inputs are NumPy arrays with correct types
        band_data = np.ascontiguousarray(band_data, dtype=np.float32)
        lake_mask = np.ascontiguousarray(lake_mask, dtype=np.uint8)
        cloud_mask = np.ascontiguousarray(cloud_mask, dtype=np.uint8)
        
        # Apply masks using accelerated function
        masked_band = apply_masks(band_data, lake_mask, cloud_mask)
        
        # Update metadata for output
        masked_meta = image_meta.copy()
        masked_meta.update({
            "count": 1,
            "dtype": 'float32',
            "nodata": -9999
        })

        # Save masked band
        with rasterio.open(masked_path, "w", **masked_meta) as dest:
            dest.write(masked_band.astype('float32'), 1)
            
        print(f"     Saved masked raster.")
        return masked_path, lake_id_path

    except Exception as e:
        print(f"\n     ######################\n     Error in create_masked_raster: {str(e)}\n     ######################\n")
        raise
    
@jit(nopython=True)
def classify_pixels(data, thresholds_list):
    """
    Efficient pixel classification using Numba. Separate function for numba efficiency.
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

def classify_ice_cover(input_path, thresholds, classified_path):
    """
    Classify ice cover in an input image based on value ranges and save the resulting classification.
    Output is a categorical raster mask with values equivalent to their class 

    Parameters:
    - input_path (str): Path to the input surface reflectance image, which contains the red band for classification.
    - thresholds (dict): Dictionary where each key is a class name (e.g., 'ice', 'water') and the value is a tuple defining the 
      range of values for that class (low, high).
    - classified_path (str): Path where the classified ice cover raster will be saved.
    """

    print(f"     Classifying ice cover")
    print("     Using classification scheme:")
    for class_name, (low, high) in thresholds.items():
        print(f"       {class_name}: values from {low} to {high}")
    
    with rasterio.open(input_path) as src:
        keep_band = src.read(1)
        meta = src.meta.copy()
    
    print("     Applying classification to valid pixels...")
    # Convert thresholds to list of tuples
    thresholds_list = [v for v in thresholds.values()]
    
    # Classify valid pixels
    classified = classify_pixels(keep_band, np.array(thresholds_list, dtype=np.float32))
    
    # Save classification
    with rasterio.open(classified_path, "w", **meta) as dst:
        dst.write(classified.astype('float32'), 1)
    
    print("     Output classes:")
    for idx, (class_name, _) in enumerate(thresholds.items(), start=1):
        print(f"       Class {idx}: {class_name}")
    
    return classified_path