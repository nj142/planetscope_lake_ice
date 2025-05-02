import os
import time
import calendar
from datetime import datetime
import numpy as np
import netCDF4 as nc
from osgeo import gdal, ogr

def add_observation(nc_path, lake_id, timestamp, study_site, stats, image_name):
    """
    Append a single lake observation to an existing NetCDF file, initializing
    or appending variable-length arrays as needed.
    """
    with nc.Dataset(nc_path, 'a') as ds:
        lake_ids = ds.variables['lake_id'][:]

        if lake_id in lake_ids:
            idx = int(np.where(lake_ids == lake_id)[0][0])
            ds.variables['obs_count'][idx] += 1
            ds.variables['unix_time'][idx] = np.append(ds.variables['unix_time'][idx], timestamp)
            ds.variables['usable_pixels'][idx] = np.append(
                ds.variables['usable_pixels'][idx], stats['usable_pixels'])
            ds.variables['clear_percent'][idx] = np.append(
                ds.variables['clear_percent'][idx], stats['clear_percent'])
            for cls, cnt in stats['class_counts'].items():
                pct = (cnt / stats['usable_pixels'] * 100) if stats['usable_pixels'] > 0 else 0
                ds.variables[f'{cls}_pixels'][idx] = np.append(
                    ds.variables[f'{cls}_pixels'][idx], cnt)
                ds.variables[f'{cls}_percent'][idx] = np.append(
                    ds.variables[f'{cls}_percent'][idx], pct)
            old_csv = ds.variables['image_names_csv'][idx] or ''
            ds.variables['image_names_csv'][idx] = (
                f"{old_csv},{image_name}" if old_csv else image_name
            )
        else:
            idx = len(ds.dimensions['lake'])
            ds.variables['lake_id'][idx]      = lake_id
            ds.variables['area'][idx]         = stats['area']
            ds.variables['perimeter'][idx]    = stats['perimeter']
            ds.variables['total_pixels'][idx] = stats['total_pixels']
            ds.variables['study_site'][idx]   = study_site
            ds.variables['obs_count'][idx]    = 1
            ds.variables['unix_time'][idx]     = np.array([timestamp], dtype='int32')
            ds.variables['usable_pixels'][idx] = np.array([stats['usable_pixels']], dtype='int32')
            ds.variables['clear_percent'][idx] = np.array([stats['clear_percent']], dtype='float32')
            for cls, cnt in stats['class_counts'].items():
                pct = (cnt / stats['usable_pixels'] * 100) if stats['usable_pixels'] > 0 else 0
                ds.variables[f'{cls}_pixels'][idx]  = np.array([cnt], dtype='int32')
                ds.variables[f'{cls}_percent'][idx] = np.array([pct], dtype='float32')
            ds.variables['image_names_csv'][idx] = image_name


def calculate_lake_statistics(lake_id_mask_path, classified_path, img_name,
                               netcdf_path, study_site,
                               lake_vector_path, config):
    """
    Compute per-lake ice/snow/water statistics from a classified raster
    and append observations directly into an existing NetCDF file.
    """
    # Parse image timestamp from filename YYYYMMDD_HHMMSS_...
    basename = os.path.basename(img_name)
    try:
        date_part, time_part, *_ = basename.split("_")
        # Interpret filename timestamp as local time (this fixes 4 hour UTC offset for some reason)
        dt_local = datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")
        image_timestamp = int(dt_local.timestamp())
    except Exception:
        image_timestamp = 0

    # Open classified raster and read data array + nodata
    ds_class = gdal.Open(classified_path)
    band_class = ds_class.GetRasterBand(1)
    classified_data = band_class.ReadAsArray()
    nodata = band_class.GetNoDataValue()

    # Open lake ID mask raster and read data array
    ds_mask = gdal.Open(lake_id_mask_path)
    mask_band = ds_mask.GetRasterBand(1)
    lake_id_data = mask_band.ReadAsArray()

    if classified_data.shape != lake_id_data.shape:
        raise ValueError("Classified raster and lake ID mask have different dimensions")

    lake_attrs = {}
    if lake_vector_path:
        drv = ogr.Open(lake_vector_path, 0)
        if drv:
            layer = drv.GetLayer()
            fld_map = config.get('vector_fields', {})
            id_fld = fld_map.get('id', 'id')
            area_fld = fld_map.get('area', 'area')
            perim_fld = fld_map.get('perimeter', 'perimeter')
            for feat in layer:
                lid = feat.GetField(id_fld)
                lake_attrs[lid] = {
                    'area': feat.GetField(area_fld) or 0,
                    'perimeter': feat.GetField(perim_fld) or 0
                }

    code_map = {
        i+1: cls_name.lower()
        for i, cls_name in enumerate(config['thresholds'].keys())
    }

    unique_ids = np.unique(lake_id_data)
    unique_ids = unique_ids[unique_ids > 0]

    for lid in unique_ids:
        mask = (lake_id_data == lid)
        total_pix = int(mask.sum())

        with nc.Dataset(netcdf_path, 'r') as ds_check:
            lake_ids_nc = ds_check.variables['lake_id'][:]
            if lid in lake_ids_nc:
                idx0 = int(np.where(lake_ids_nc == lid)[0][0])
                existing_tot = int(ds_check.variables['total_pixels'][idx0])
                if existing_tot != total_pix:
                    print(f"ERROR: Total pixels mismatch for lake {lid}: current {total_pix}, existing {existing_tot}")
                total_pix = existing_tot

        vals = classified_data[mask]
        if nodata is not None:
            vals = vals[(vals != nodata) & (vals != 0)]
        else:
            vals = vals[vals != 0]

        class_counts = {
            name: int(np.count_nonzero(vals == code))
            for code, name in code_map.items()
        }

        usable_count = sum(class_counts.values())
        clear_pct = (usable_count / total_pix * 100) if total_pix > 0 else 0

        stats = {
            'area':          lake_attrs.get(lid, {}).get('area', 0),
            'perimeter':     lake_attrs.get(lid, {}).get('perimeter', 0),
            'total_pixels':  total_pix,
            'usable_pixels': usable_count,
            'clear_percent': clear_pct,
            'class_counts':  class_counts
        }

        add_observation(
            nc_path    = netcdf_path,
            lake_id    = lid,
            timestamp  = image_timestamp,
            study_site = study_site,
            stats      = stats,
            image_name = basename
        )

    ds_class = None
    ds_mask  = None

    print(f"Processed {len(unique_ids)} lakes and updated: {netcdf_path}")
