import os
import glob
from osgeo import gdal, ogr, osr
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from scipy import stats

def reproject_to_alaska_albers(input_raster, output_raster):
    """
    Reproject input raster to Alaska Albers projection.
    Returns True if successful, False if failed
    """
    alaska_albers = osr.SpatialReference()
    alaska_albers.ImportFromEPSG(3338)  # Alaska Albers
    
    try:
        src_ds = gdal.Open(input_raster)
        src_proj = src_ds.GetProjection()
        src_srs = osr.SpatialReference()
        src_srs.ImportFromWkt(src_proj)
        
        if src_srs.IsSame(alaska_albers):
            return True  # Already in correct projection
            
        gdal.Warp(output_raster, 
                  src_ds, 
                  dstSRS='EPSG:3338',
                  resampleAlg=gdal.GRA_NearestNeighbour)
        return True
        
    except Exception as e:
        print(f"Error reprojecting {input_raster}: {str(e)}")
        return False

def clip_raster_to_shapefile(input_raster, shapefile, output_raster):
    """
    Clip input raster to shapefile extent.
    Returns True if there's overlap, False if no overlap
    """
    try:
        # Check for overlap
        raster = gdal.Open(input_raster)
        shp_ds = ogr.Open(shapefile)
        shp_layer = shp_ds.GetLayer()
        
        # Get raster and shapefile extents
        raster_geotransform = raster.GetGeoTransform()
        raster_minx = raster_geotransform[0]
        raster_maxx = raster_minx + raster_geotransform[1] * raster.RasterXSize
        raster_miny = raster_geotransform[3] + raster_geotransform[5] * raster.RasterYSize
        raster_maxy = raster_geotransform[3]
        
        shp_extent = shp_layer.GetExtent()
        shp_minx, shp_maxx, shp_miny, shp_maxy = shp_extent
        
        # Check for overlap
        if (raster_minx > shp_maxx or raster_maxx < shp_minx or
            raster_miny > shp_maxy or raster_maxy < shp_miny):
            return False
            
        # Clip raster
        gdal.Warp(output_raster,
                  input_raster,
                  cutlineDSName=shapefile,
                  cropToCutline=True,
                  dstNodata=None)
        return True
        
    except Exception as e:
        print(f"Error clipping {input_raster}: {str(e)}")
        return False

def classify_ice_cover(input_raster, output_raster, thresholds, class_labels):
    """
    Classify ice cover based on red band values using provided thresholds
    """
    try:
        ds = gdal.Open(input_raster)
        red_band = ds.GetRasterBand(3)  # Red band (3rd band)
        red_data = red_band.ReadAsArray()
        
        # Create classification array
        classified = np.zeros_like(red_data, dtype=np.uint8)
        
        # Apply thresholds
        classified[(red_data >= thresholds['Ice'][0]) & (red_data < thresholds['Ice'][1])] = 1
        classified[red_data >= thresholds['Snow'][0]] = 2
        classified[(red_data >= thresholds['Water'][0]) & (red_data < thresholds['Water'][1])] = 3
        
        # Save classified raster
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(output_raster, 
                             ds.RasterXSize, 
                             ds.RasterYSize, 
                             1, 
                             gdal.GDT_Byte)
        
        out_ds.SetGeoTransform(ds.GetGeoTransform())
        out_ds.SetProjection(ds.GetProjection())
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(classified)
        
        # Set color table
        colors = gdal.ColorTable()
        colors.SetColorEntry(1, (0, 255, 255, 255))    # Ice - Cyan
        colors.SetColorEntry(2, (255, 255, 255, 255))  # Snow - White
        colors.SetColorEntry(3, (0, 0, 255, 255))      # Water - Blue
        out_band.SetColorTable(colors)
        
        out_band.FlushCache()
        out_ds = None
        
        return True
        
    except Exception as e:
        print(f"Error classifying {input_raster}: {str(e)}")
        return False

def analyze_time_series(mask_files, output_plot):
    """
    Generate time series analysis of ice, snow, and water coverage
    """
    try:
        dates = []
        ice_coverage = []
        snow_coverage = []
        water_coverage = []
        
        # Sort files by date (assuming date is in filename)
        mask_files.sort()
        
        for mask_file in mask_files:
            # Extract date from filename (adjust pattern as needed)
            date_str = os.path.basename(mask_file).split('_')[0]  # Adjust split pattern
            date = datetime.strptime(date_str, '%Y%m%d')  # Adjust date format
            
            ds = gdal.Open(mask_file)
            data = ds.GetRasterBand(1).ReadAsArray()
            
            total_pixels = np.sum(data > 0)  # Non-zero pixels
            ice_pixels = np.sum(data == 1)
            snow_pixels = np.sum(data == 2)
            water_pixels = np.sum(data == 3)
            
            # Calculate percentages
            dates.append(date)
            ice_coverage.append((ice_pixels / total_pixels) * 100)
            snow_coverage.append((snow_pixels / total_pixels) * 100)
            water_coverage.append((water_pixels / total_pixels) * 100)
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        # Plot coverage lines
        plt.plot(dates, ice_coverage, 'c-', label='Ice cover', marker='o')
        plt.plot(dates, snow_coverage, 'k-', label='Snow on ice', marker='o')
        plt.plot(dates, water_coverage, 'b-', label='Water', marker='o')
        
        # Calculate and plot trends
        x = np.array([(d - dates[0]).days for d in dates])
        
        # Ice trend
        ice_slope, ice_intercept, _, _, _ = stats.linregress(x, ice_coverage)
        ice_trend = ice_slope * x + ice_intercept
        plt.plot(dates, ice_trend, 'c--', alpha=0.5, label='Ice trend')
        
        # Snow trend
        snow_slope, snow_intercept, _, _, _ = stats.linregress(x, snow_coverage)
        snow_trend = snow_slope * x + snow_intercept
        plt.plot(dates, snow_trend, 'k--', alpha=0.5, label='Snow trend')
        
        plt.xlabel('Date')
        plt.ylabel('Coverage (%)')
        plt.title('Lake Ice and Snow Coverage Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(output_plot)
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"Error analyzing time series: {str(e)}")
        return False

if __name__ == "__main__":
    # Input parameters
    image_dir = r"D:\planetscope_lake_ice\Data_TEST\10 - Lake Ice Freezeup\Time_Series_Test_Lake_Ice_Freezeup_2024_YKD_psscene_analytic_sr_udm2\PSScene\*_SR.tif"
    shapefile_path = r"D:\planetscope_lake_ice\Data_TEST\10 - Lake Ice Freezeup\Lake Shapefile\time_series_lake_shapefile.shp"
    masks_dir = r"D:\planetscope_lake_ice\Data_TEST\10 - Lake Ice Freezeup\Masks"
    temp_dir = r"D:\planetscope_lake_ice\Data_TEST\10 - Lake Ice Freezeup\Temporary FIles"
    output_plot = r"D:\planetscope_lake_ice\Data_TEST\10 - Lake Ice Freezeup\Output Plot\time_series.png"
    
    # Classification parameters
    thresholds = {
        'Ice': (950, 3800),
        'Snow': (3800, float('inf')),
        'Water': (float('-inf'), 950)
    }
    
    class_labels = {
        1: "Ice cover",
        2: "Snow on ice",
        3: "Water"
    }
    
    # Create output directories if they don't exist
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Process each image
    image_files = glob.glob(image_dir)
    mask_files = []
    
    for img_file in image_files:
        base_name = os.path.basename(img_file)
        name_without_ext = os.path.splitext(base_name)[0]
        
        # Reprojected file path
        proj_file = os.path.join(temp_dir, f"{name_without_ext}_proj.tif")
        
        # Reproject to Alaska Albers
        if reproject_to_alaska_albers(img_file, proj_file):
            # Clipped file path
            clip_file = os.path.join(temp_dir, f"{name_without_ext}_clip.tif")
            
            # Clip to shapefile
            if clip_raster_to_shapefile(proj_file, shapefile_path, clip_file):
                # Classified mask path
                mask_file = os.path.join(masks_dir, f"{name_without_ext}_mask.tif")
                
                # Classify ice cover
                if classify_ice_cover(clip_file, mask_file, thresholds, class_labels):
                    mask_files.append(mask_file)
                else:
                    print(f"Failed to classify {img_file}")
            else:
                print(f"No overlap between {img_file} and shapefile")
        else:
            print(f"Failed to reproject {img_file}")
    
    # Generate time series analysis
    if mask_files:
        if analyze_time_series(mask_files, output_plot):
            print("Time series analysis completed successfully")
        else:
            print("Failed to generate time series analysis")
    else:
        print("No valid masked files to analyze")