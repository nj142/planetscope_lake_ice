import os
import numpy as np
import pandas as pd
from osgeo import gdal, ogr

def calculate_lake_statistics(classified_path, sr_name, config):
    """
    For each lake in the ALPOD vector file, calculate statistics from the classified raster.
    
    Uses the ALPOD shapefile specified in config['vector_path'] and writes (or merges)
    results into the time series CSV given in config['time_series_csv'].
    
    The ALPOD vector is assumed to have the attributes:
       - "id" for lake id,
       - "area" for lake area,
       - "perimeter" for lake perimeter.
    
    For each lake, this function calculates:
       - total pixels within the lake (via rasterizing the lake polygon),
       - usable pixels (pixels not equal to the raster nodata value),
       - clear percent (usable pixels / total pixels * 100), and,
       - for each class (as defined by config['thresholds']):
             * pixel count and
             * percent cover relative to usable pixels.
             
    Detailed class statistics are only computed if the clear percent meets or exceeds config['min_clear_percent'].
    
    Parameters:
      classified_path: Path to the classified raster.
      sr_name: An identifier for the SR image (used as a prefix for output columns).
      config: Dictionary with keys including:
              'vector_path'      : Path to the ALPOD shapefile.
              'time_series_csv'  : Full file path to the output CSV.
              'min_clear_percent': Minimum percent of usable pixels to compute detailed stats.
              'thresholds'       : A dictionary whose keys are the class names (e.g., 'Ice', 'Snow', 'Water').
    """
    # Open the classified raster.
    ds = gdal.Open(classified_path)
    if ds is None:
        raise IOError(f"Could not open classified raster: {classified_path}")
    band = ds.GetRasterBand(1)
    classified_data = band.ReadAsArray()
    nodata = band.GetNoDataValue()
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    ds = None

    # Open the vector (ALPOD lakes) from the config.
    vector_path = config['vector_path']
    vector_ds = ogr.Open(vector_path, 0)
    if vector_ds is None:
        raise IOError(f"Could not open vector file: {vector_path}")
    layer = vector_ds.GetLayer()

    # Build class mapping from the keys in config['thresholds'].
    # Numeric codes are assigned in order: 1 = first key, 2 = second key, etc.
    classes = list(config['thresholds'].keys())
    class_mapping = {i+1: classes[i].lower() for i in range(len(classes))}

    lake_stats = {}
    for feat in layer:
        # Get lake ID from attribute "id"; fallback to FID if missing.
        lake_id = feat.GetField("id")
        if lake_id is None:
            lake_id = feat.GetFID()
        # Read lake area and perimeter from the attribute table.
        lake_area = feat.GetField("area")
        lake_perimeter = feat.GetField("perimeter")

        geom = feat.GetGeometryRef()
        if geom is None:
            continue

        # Create an in-memory raster mask matching the classified raster.
        cols = classified_data.shape[1]
        rows = classified_data.shape[0]
        mem_driver = gdal.GetDriverByName("MEM")
        mem_raster = mem_driver.Create('', cols, rows, 1, gdal.GDT_Byte)
        mem_raster.SetGeoTransform(geotransform)
        mem_raster.SetProjection(projection)
        mem_band = mem_raster.GetRasterBand(1)
        mem_band.Fill(0)

        # Create a temporary in-memory vector for just this lake.
        mem_vector_driver = ogr.GetDriverByName("Memory")
        mem_vector_ds = mem_vector_driver.CreateDataSource("temp")
        mem_layer = mem_vector_ds.CreateLayer("temp", srs=layer.GetSpatialRef(), geom_type=ogr.wkbPolygon)
        mem_feat = ogr.Feature(mem_layer.GetLayerDefn())
        mem_feat.SetGeometry(geom.Clone())
        mem_layer.CreateFeature(mem_feat)
        mem_feat = None

        # Rasterize the lake polygon into the in-memory raster.
        gdal.RasterizeLayer(mem_raster, [1], mem_layer, burn_values=[1])
        lake_mask = mem_raster.GetRasterBand(1).ReadAsArray().astype(bool)
        mem_raster = None
        mem_vector_ds = None

        total_pixels = int(np.sum(lake_mask))
        if total_pixels == 0:
            continue

        # Extract pixels from the classified raster that fall within the lake.
        lake_pixels = classified_data[lake_mask]
        # Usable pixels are those not equal to nodata.
        usable_pixels = lake_pixels[lake_pixels != nodata]
        usable_count = int(len(usable_pixels))
        clear_percent = (usable_count / total_pixels) * 100

        min_clear = config.get('min_clear_percent', 50)
        class_counts = {}
        if clear_percent >= min_clear:
            unique, counts = np.unique(usable_pixels, return_counts=True)
            for cls, cnt in zip(unique, counts):
                cls_int = int(cls)
                if cls_int in class_mapping:
                    class_counts[cls_int] = int(cnt)
        else:
            # If the clear percentage is too low, record zeros for all classes.
            for cls in class_mapping:
                class_counts[cls] = 0

        lake_stats[lake_id] = {
            'area': lake_area,
            'perimeter': lake_perimeter,
            'total_pixels': total_pixels,
            'usable_pixels': usable_count,
            'clear_percent': clear_percent,
            'class_counts': class_counts
        }
    vector_ds = None

    # Build a DataFrame from the computed statistics.
    rows_list = []
    for lake_id, stats in lake_stats.items():
        row = {
            'lake_id': lake_id,
            'area': stats['area'],
            'perimeter': stats['perimeter'],
            f"{sr_name}_total_pixels": stats['total_pixels'],
            f"{sr_name}_usable_pixels": stats['usable_pixels'],
            f"{sr_name}_clear_percent": stats['clear_percent']
        }
        for cls, cls_name in class_mapping.items():
            cnt = stats['class_counts'].get(cls, 0)
            row[f"{sr_name}_{cls_name}_pixels"] = cnt
            if stats['usable_pixels'] > 0:
                row[f"{sr_name}_{cls_name}_percent"] = (cnt / stats['usable_pixels']) * 100
            else:
                row[f"{sr_name}_{cls_name}_percent"] = 0
        rows_list.append(row)
    df = pd.DataFrame(rows_list)

    # Merge with the existing time series CSV if it exists.
    output_csv = config['time_series_csv']
    if os.path.exists(output_csv):
        df_existing = pd.read_csv(output_csv)
        # Drop duplicate constant columns (area, perimeter) from the new df if already present.
        for col in ['area', 'perimeter']:
            if col in df.columns and col in df_existing.columns:
                df = df.drop(columns=[col])
        df_merged = pd.merge(df_existing, df, on="lake_id", how="outer")
        df_merged.to_csv(output_csv, index=False)
    else:
        df.to_csv(output_csv, index=False)
    print(f"Lake statistics updated and saved to {output_csv}")

# Example main block for testing the stats function independently.
if __name__ == "__main__":
    config = {
        'vector_path': r"D:\planetscope_lake_ice\Data (Validation)\8 - Download ALPOD data here\ALPODlakes.shp",
        'time_series_csv': r"D:\planetscope_lake_ice\Data (Unclassified)\2 - Break Up Time Series Output\lake_statistics.csv",
        'min_clear_percent': 30,
        'thresholds': {
            'Ice': (950, 3800),
            'Snow': (3800, float('inf')),
            'Water': (float('-inf'), 950)
        }
    }
    classified_path = r"path_to_classified_raster.tif"  # Replace with your classified raster path
    sr_name = "example_SR"  # Identifier (e.g., a date or unique scene ID)
    calculate_lake_statistics(classified_path, sr_name, config)
