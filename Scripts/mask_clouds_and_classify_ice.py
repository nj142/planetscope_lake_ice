
import rasterio
import numpy as np
import dask.array as da
from rasterio.enums import Resampling
import os
from dask import delayed
import dask


def calculate_output_rasters(sr_image_path, udm_path, udm_mask_bands, sr_keep_bands, 
                       output_sr_path, output_udm_path, output_classified_path):
    """
    Creates three output rasters from PlanetScope SR and UDM data using Dask and NumPy vectorization:
    1. Single-band SR raster (keep only specified band)
    2. Cloud mask from UDM bands
    3. Classified ice/snow/water mask
    
    Args:
        sr_image_path: Path to input SR image
        udm_path: Path to input UDM file
        udm_mask_bands: List of UDM band numbers to use for cloud masking
        sr_keep_bands: List of SR band numbers to keep (should be [3] for red band)
        output_vector_path: Path to clipped vector file (not used in this function)
        output_sr_path: Output path for single-band SR raster
        output_udm_path: Output path for cloud mask
        output_classified_path: Output path for classified mask
    """
    
    # Ensure output directories exist
    for output_path in [output_sr_path, output_udm_path, output_classified_path]:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # ────────────────────────────────────────────────────────────────────────────────
    # Load data with Dask for memory-efficient processing
    # ────────────────────────────────────────────────────────────────────────────────
    print("Loading raster data with Dask...")
    
    # Read SR and UDM data as Dask arrays
    with rasterio.open(sr_image_path) as src_sr:
        sr_profile = src_sr.profile.copy()
        # Load red band as Dask array with optimal chunking
        red_band_da = da.from_array(src_sr.read(sr_keep_bands[0]), 
                                   chunks=(1024, 1024))  # Adjust chunk size as needed
    
    with rasterio.open(udm_path) as src_udm:
        udm_profile = src_udm.profile.copy()
        # Load UDM bands as Dask arrays
        udm_band8_da = da.from_array(src_udm.read(8), chunks=(1024, 1024))
        udm_mask_bands_da = [da.from_array(src_udm.read(band_num), chunks=(1024, 1024)) 
                            for band_num in udm_mask_bands]
    
    # ────────────────────────────────────────────────────────────────────────────────
    # Section 1: Create single-band SR raster 
    # ────────────────────────────────────────────────────────────────────────────────
    print("Creating single red band raster...")
    
    with rasterio.open(sr_image_path) as src:
        profile = src.profile.copy()
        profile.update({
            'count': 1,
            'dtype': 'uint16',
            'compress': 'lzw'
        })
        
        # Read entire band into memory (only if you have enough RAM)
        red_data = src.read(sr_keep_bands[0])
        
        with rasterio.open(output_sr_path, 'w', **profile) as dst:
            dst.write(red_data, 1)
    
    # ────────────────────────────────────────────────────────────────────────────────
    # Section 2: Create cloud mask from UDM bands (Vectorized with Dask)
    # ────────────────────────────────────────────────────────────────────────────────
    print("Creating cloud mask from UDM...")
    
    # Extract nodata mask (bit 0 of band 8) - vectorized
    nodata_mask_da = (udm_band8_da & 1) > 0
    
    # Combine UDM mask bands for cloud detection - vectorized
    cloud_mask_da = da.zeros_like(udm_band8_da, dtype=bool)
    for band_da in udm_mask_bands_da:
        cloud_mask_da = da.logical_or(cloud_mask_da, band_da > 0)
    
    # Create final cloud mask with proper values - vectorized
    final_cloud_mask_da = da.where(nodata_mask_da, 255,  # nodata = 255
                                  da.where(cloud_mask_da, 1, 0))  # cloud = 1, clear = 0
    final_cloud_mask_da = final_cloud_mask_da.astype(np.uint8)
    
    # Create profile for cloud mask output
    profile_udm = udm_profile.copy()
    profile_udm.update({
        'count': 1,
        'dtype': 'uint8',
        'compress': 'lzw',
        'nodata': None
    })
    
    # Write cloud mask
    with rasterio.open(output_udm_path, 'w', **profile_udm) as dst_udm:
        dst_udm.write(final_cloud_mask_da.compute(), 1)
    
    print(f"Cloud mask saved to: {output_udm_path}")
    
    # ────────────────────────────────────────────────────────────────────────────────
    # Section 3: Create classified ice/snow/water mask (Vectorized with Dask)
    # ────────────────────────────────────────────────────────────────────────────────
    print("Creating classified ice/snow/water mask...")
    
    # Apply pixel reflectance thresholds - fully vectorized
    water_mask_da = red_band_da < 950
    ice_mask_da = (red_band_da >= 950) & (red_band_da < 3800)
    snow_mask_da = red_band_da >= 3800
    
    # Create classified mask - vectorized with da.where for multiple conditions
    classified_mask_da = da.where(nodata_mask_da, 255,  # nodata = 255
                                 da.where(final_cloud_mask_da == 1, 0,  # clouds = 0 (or change to keep classification)
                                         da.where(water_mask_da, 3,  # water = 3
                                                 da.where(ice_mask_da, 1,  # ice = 1
                                                         da.where(snow_mask_da, 2, 0)))))  # snow = 2, default = 0
    
    classified_mask_da = classified_mask_da.astype(np.uint8)
    
    # Create profile for classified output
    profile_classified = sr_profile.copy()
    profile_classified.update({
        'count': 1,
        'dtype': 'uint8',
        'compress': 'lzw',
        'nodata': None
    })
    
    # Write classified mask
    with rasterio.open(output_classified_path, 'w', **profile_classified) as dst_classified:
        dst_classified.write(classified_mask_da.compute(), 1)
    
    print(f"Classified mask saved to: {output_classified_path}")
    print("All raster outputs created successfully.\n")
    
    return True
