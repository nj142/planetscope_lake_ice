import os
import subprocess
import numpy as np
from osgeo import gdal, ogr, osr

def clip_vector_to_raster(vector_path, udm_path, output_path):
    """
    Clips a vector shapefile (e.g., lakes) to include only features that are completely within
    the valid data area of a UDM 2.1 mask. The UDM’s band 8 is interpreted so that pixels where
    bit 0 equals 1 are considered "no data" (padding) and all other pixels are considered real data.
    The valid-data area is polygonized and then transformed to the vector’s coordinate system
    for filtering.
    
    To avoid creating an extremely long WKT string (which can trigger Windows error 206),
    the valid-data polygon is simplified before being used in the SQL query.
    
    Parameters:
      vector_path: Path to the input shapefile.
      udm_path: Path to the UDM 2.1 mask file (PlanetScope UDM) whose band 8 is used.
      output_path: Path for the output clipped shapefile.
      
    Returns:
      features_kept: The number of features (lakes) kept.
    """
    # -------------------------------------------------------------------------
    # 1. Open the UDM file and extract band 8 (PlanetScope UDM)
    # -------------------------------------------------------------------------
    udm_ds = gdal.Open(udm_path)
    if udm_ds is None:
        raise ValueError(f"Could not open UDM file: {udm_path}")

    # Note: In GDAL bands are 1-indexed. Here we assume band 8 contains the UDM bit mask.
    band8 = udm_ds.GetRasterBand(8)
    if band8 is None:
        raise ValueError("UDM file does not contain band 8.")
        
    # Read band 8 as an array.
    band8_array = band8.ReadAsArray()
    if band8_array is None:
        raise ValueError("Could not read data from band 8 of the UDM file.")

    # Create a binary mask: valid data = 1 when bit 0 is NOT set (i.e. (value & 1)==0),
    # otherwise 0. (Bit 0==1 indicates black fill or padding.)
    valid_mask = np.where((band8_array & 1) == 0, 1, 0)

    # -------------------------------------------------------------------------
    # 2. Polygonize the valid data area from the binary mask
    # -------------------------------------------------------------------------
    # Create an in‑memory raster to hold the valid mask.
    mem_driver = gdal.GetDriverByName('MEM')
    mem_ds = mem_driver.Create('', udm_ds.RasterXSize, udm_ds.RasterYSize, 1, gdal.GDT_Byte)
    mem_ds.SetGeoTransform(udm_ds.GetGeoTransform())
    mem_ds.SetProjection(udm_ds.GetProjection())
    mem_band = mem_ds.GetRasterBand(1)
    mem_band.WriteArray(valid_mask)
    
    # Create an in‑memory vector layer to store the polygonized valid areas.
    mem_vector_driver = ogr.GetDriverByName('Memory')
    mem_vector_ds = mem_vector_driver.CreateDataSource('out')
    udm_srs = osr.SpatialReference()
    udm_srs.ImportFromWkt(udm_ds.GetProjection())
    poly_layer = mem_vector_ds.CreateLayer('poly', srs=udm_srs)
    
    # Add an attribute field to record the pixel value.
    field_defn = ogr.FieldDefn('value', ogr.OFTInteger)
    poly_layer.CreateField(field_defn)
    
    # Polygonize the binary mask.
    gdal.Polygonize(mem_band, None, poly_layer, 0, [], callback=None)
    
    # Union all polygons that represent valid data (value==1).
    valid_poly = None
    for feature in poly_layer:
        val = feature.GetField('value')
        if val == 1:
            geom = feature.GetGeometryRef().Clone()
            if valid_poly is None:
                valid_poly = geom
            else:
                valid_poly = valid_poly.Union(geom)
    
    if valid_poly is None:
        raise ValueError("No valid data area found in the UDM mask.")

    # -------------------------------------------------------------------------
    # 3. Simplify the valid-data polygon to avoid extremely long WKT strings.
    # -------------------------------------------------------------------------
    # Use a tolerance equal to roughly 10 pixels.
    gt = udm_ds.GetGeoTransform()
    pixel_size = abs(gt[1])
    tolerance = pixel_size * 10
    simplified_poly = valid_poly.Simplify(tolerance)
    if simplified_poly is None:
        simplified_poly = valid_poly  # fallback if simplify returns None

    # -------------------------------------------------------------------------
    # 4. Transform the valid-data polygon from UDM SRS to the vector layer’s SRS
    # -------------------------------------------------------------------------
    vector_ds = ogr.Open(vector_path)
    if vector_ds is None:
        raise ValueError(f"Could not open vector: {vector_path}")
    vector_layer = vector_ds.GetLayer()
    vector_srs = vector_layer.GetSpatialRef()
    if vector_srs is None:
        raise ValueError("Vector has no spatial reference defined.")
    
    # Create coordinate transformation from UDM SRS to vector SRS.
    transform = osr.CoordinateTransformation(udm_srs, vector_srs)
    simplified_poly.Transform(transform)
    
    # Export the simplified valid-data polygon to WKT.
    valid_poly_wkt = simplified_poly.ExportToWkt()

    # Try to get the authority code (e.g., EPSG code) for the vector SRS.
    epsg = vector_srs.GetAuthorityCode(None)
    if not epsg:
        epsg = 4326  # fallback to EPSG:4326 if not defined

    # -------------------------------------------------------------------------
    # 5. Build the SQL query using the valid-data polygon WKT
    # -------------------------------------------------------------------------
    # We assume the layer name is the same as the shapefile’s base name.
    base_name = os.path.splitext(os.path.basename(vector_path))[0]
    sql = (
        f"SELECT * FROM \"{base_name}\" "
        f"WHERE ST_Within(geometry, ST_GeomFromText('{valid_poly_wkt}', {epsg}))"
    )

    # -------------------------------------------------------------------------
    # 6. Run ogr2ogr to clip the vector features
    # -------------------------------------------------------------------------
    command = [
        "ogr2ogr",
        "-f", "ESRI Shapefile",
        output_path,
        vector_path,
        "-dialect", "sqlite",
        "-sql", sql,
        "-overwrite"
    ]
    
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ogr2ogr command failed: {e}")
    
    # Open the output shapefile to count how many features were kept.
    ds_out = ogr.Open(output_path)
    if ds_out is None:
        raise ValueError(f"Could not open output shapefile: {output_path}")
    out_layer = ds_out.GetLayer(0)
    features_kept = out_layer.GetFeatureCount()
    ds_out = None  # close dataset

    # Clean up the UDM dataset.
    udm_ds = None

    return features_kept
