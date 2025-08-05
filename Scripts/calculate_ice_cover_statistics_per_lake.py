import os
import json
import csv
import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from collections import defaultdict
import time
import pandas as pd

def calculate_lake_statistics(classified_raster_path, image_core_name, output_path, study_site, lake_vector_shapefile):
    """
    Calculate ice/snow/water statistics for each lake in the image using rasterization approach.
    
    Parameters:
    - classified_raster_path: Path to the classified ice/snow/water raster (uint8)
    - image_core_name: Core name of the image (e.g., "20200415_222212_87_1060")
    - output_path: Base output directory
    - study_site: Study site identifier
    - lake_vector_shapefile: Path to the clipped lakes shapefile for this image
    - config: Configuration dictionary
    """
    
    print(f"    Loading classified raster and lake vectors...")
    
    # Load the classified raster
    with rasterio.open(classified_raster_path) as src:
        classified_data = src.read(1)  # Assuming single band
        transform = src.transform
        crs = src.crs
    
    # Load the clipped lake vectors
    lakes_gdf = gpd.read_file(lake_vector_shapefile)
    
    if len(lakes_gdf) == 0:
        print(f"    No lakes found in clipped shapefile for {image_core_name}")
        return
    
    print(f"    Processing {len(lakes_gdf)} lakes...")
    print(f"    Lakes CRS: {lakes_gdf.crs}")
    print(f"    Raster CRS: {crs}")
    
    # Reproject lakes to match the raster CRS if needed
    if lakes_gdf.crs != crs:
        print(f"    Reprojecting lakes from {lakes_gdf.crs} to {crs}...")
        lakes_gdf = lakes_gdf.to_crs(crs)
        print(f"    Reprojection complete")
    

    # Get lake ID column
    id_column = 'id'
    
    # Prepare geometries and IDs for rasterization
    # Format: [(geometry, lake_id), ...]
    shapes = [(geom, lake_id) for geom, lake_id in zip(lakes_gdf.geometry, lakes_gdf[id_column])]
    
    print(f"    Rasterizing lakes to ID raster...")
    start_time = time.time()
    
    # Rasterize lakes to create lake ID raster
    # Use uint32 to handle lake IDs safely
    lake_id_raster = rasterize(
        shapes,
        out_shape=classified_data.shape,
        transform=transform,
        fill=0,  # Background value (no lake)
        dtype=np.uint32,
        all_touched=False  # Don't include pixels that are partially covered
    )
    
    raster_time = time.time() - start_time
    print(f"    Rasterization completed in {raster_time:.2f} seconds")
    
    # Get unique lake IDs present in the raster (excluding background=0)
    unique_lake_ids = np.unique(lake_id_raster[lake_id_raster > 0])
    
    print(f"    Calculating statistics for {len(unique_lake_ids)} lakes...")
    
    # Create output directory
    output_csv_dir = os.path.join(output_path, "Lake Time Series CSVs")
    os.makedirs(output_csv_dir, exist_ok=True)
    
    # Process each lake and write/append to individual CSV files
    for lake_id in unique_lake_ids:
        # Create mask for this lake
        lake_mask = (lake_id_raster == lake_id)
        
        # Extract classified values for this lake
        lake_classified_values = classified_data[lake_mask]
        
        # Calculate histogram (count of each class)
        unique_vals, counts = np.unique(lake_classified_values, return_counts=True)
        
        histogram = {}
        for val, count in zip(unique_vals, counts):
            histogram[str(val)] = int(count)
        
        # Get the corresponding lake geometry as GeoJSON
        lake_row = lakes_gdf[lakes_gdf[id_column] == lake_id].iloc[0]
        lake_geom = lake_row.geometry
        
        # Convert to GeoJSON format
        geojson_geom = {
            "type": lake_geom.geom_type,
            "coordinates": list(lake_geom.exterior.coords) if lake_geom.geom_type == "Polygon" else list(lake_geom.coords)
        }
        
        # Create record for this observation
        record = {
            'histogram': str(histogram).replace("'", '"'),  # JSON-compatible string
            'id': int(lake_id),
            'image_name': image_core_name,
            'study_site': study_site,
            'unix_time': extract_unix_time_from_image_name(image_core_name),
            '.geo': json.dumps(geojson_geom)
        }
        
        # Define CSV file path for this lake
        output_csv_path = os.path.join(output_csv_dir, f"{lake_id}_ice_snow.csv")
        
        # Check if file exists to determine if we need to write header
        file_exists = os.path.exists(output_csv_path)
        
        # Write or append to CSV
        with open(output_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['histogram', 'id', 'image_name', 'study_site', 'unix_time', '.geo']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header only if file is new
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(record)
    
    print(f"    Successfully processed {len(unique_lake_ids)} lakes and saved/appended to individual CSV files")


def extract_unix_time_from_image_name(image_core_name):
    """
    Extract unix timestamp from PlanetScope image name.
    Example: "20200415_222212_87_1060" -> extract date/time and convert to unix timestamp
    """
    try:
        date_part = image_core_name.split('_')[0]  # "20200415"
        time_part = image_core_name.split('_')[1]  # "222212"
        
        # Parse date and time
        from datetime import datetime
        dt_string = f"{date_part}_{time_part}"
        dt = datetime.strptime(dt_string, "%Y%m%d_%H%M%S")
        
        # Convert to unix timestamp
        unix_time = int(dt.timestamp())
        return unix_time
        
    except (ValueError, IndexError):
        # If parsing fails, return a placeholder or extract from other metadata
        print(f"    Warning: Could not extract timestamp from {image_core_name}")
        return 0