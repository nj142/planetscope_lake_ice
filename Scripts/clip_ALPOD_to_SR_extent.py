import os
import subprocess
import json
from osgeo import ogr, osr
import sys
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, mapping

def extract_geospatial_info_from_xml(xml_file_path):
    """
    Parse the PlanetScope XML metadata to get:
      - 'geometry'   : GeoJSON-like polygon footprint
      - 'epsg_code'  : integer EPSG code for projection
    """
    tree = ET.parse(xml_file_path)
    ns = {
        'gml': 'http://www.opengis.net/gml',
        'ps':  'http://schemas.planet.com/ps/v1/planet_product_metadata_geocorrected_level'
    }

    # pull the outerBoundaryIs coordinates
    coords_text = tree.find(
        './/gml:outerBoundaryIs//gml:coordinates',
        namespaces=ns
    ).text.strip()
    coords = [tuple(map(float, xy.split(','))) for xy in coords_text.split()]
    poly  = Polygon(coords)
    print(f"     Extracted polygon with {len(coords)} vertices")

    # pull the EPSG code
    epsg_code = int(tree.find('.//ps:epsgCode', namespaces=ns).text)
    return {
        'geometry': mapping(poly),
        'epsg_code': epsg_code
    }


def clip_vector_with_geometry(vector_path, geometry, output_path):
    """
    Clips a shapefile so only features entirely within ‘geometry’ are kept.
    Uses ogr2ogr + SQLite ‘ST_Within’ for maximum speed.
    
    Parameters:
      vector_path (str): input .shp
      geometry (dict): GeoJSON {type:"Polygon",coordinates:…} in EPSG:4326
      output_path (str): output .shp
    Returns:
      int: number of features kept
    """
    # build OGR geometry from GeoJSON & assign WGS84
    geom = ogr.CreateGeometryFromJson(json.dumps(geometry))
    srs4326 = osr.SpatialReference()
    srs4326.ImportFromEPSG(4326)
    srs4326.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    geom.AssignSpatialReference(srs4326)

    # open input to get its SRS and layer name
    src_ds = ogr.Open(vector_path)
    src_lyr = src_ds.GetLayer()
    vec_srs = src_lyr.GetSpatialRef()
    vec_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    layer_name = src_lyr.GetName()

    # transform footprint into the vector’s SRS (if needed)
    if not vec_srs.IsSame(srs4326):
        tr = osr.CoordinateTransformation(srs4326, vec_srs)
        geom.Transform(tr)

    # grab its EPSG code
    auth_code = vec_srs.GetAuthorityCode(None)
    epsg = int(auth_code) if auth_code else 4326

    # turn it into WKT for SQL
    wkt = geom.ExportToWkt()

    # build and run the ogr2ogr + SQLite query
    sql = (
        f"SELECT * FROM {layer_name} "
        f"WHERE ST_Within(Geometry, GeomFromText('{wkt}', {epsg}))"
    )
    subprocess.check_call([
        "ogr2ogr",
        "-f", "ESRI Shapefile",
        output_path,
        vector_path,
        "-dialect", "SQLite",
        "-sql", sql
    ])

    # count how many made it through
    out_ds = ogr.Open(output_path)
    count = out_ds.GetLayer(0).GetFeatureCount()
    src_ds, out_ds = None, None
    print(f"     Output shapefile contains {count} lakes")
    return count