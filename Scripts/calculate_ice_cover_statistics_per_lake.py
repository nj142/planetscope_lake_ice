import numpy as np
import netCDF4 as nc
from osgeo import gdal, ogr
import os, time


def add_observation(nc_path, lake_id, timestamp, study_site, stats, image_name):
    """
    Append a single lake observation to an existing NetCDF file, initializing
    or appending variable-length arrays as needed.
    """
    with nc.Dataset(nc_path, 'a') as ds:
        lake_ids = ds.variables['lake_id'][:]
        if lake_id in lake_ids:
            # Existing lake: append to VLEN arrays
            idx = int(np.where(lake_ids == lake_id)[0][0])
            ds.variables['obs_count'][idx] += 1
            # Append to unix_time
            ds.variables['unix_time'][idx] = np.append(
                ds.variables['unix_time'][idx], timestamp)
            # Append to usable_pixels and clear_percent
            ds.variables['usable_pixels'][idx] = np.append(
                ds.variables['usable_pixels'][idx], stats['usable_pixels'])
            ds.variables['clear_percent'][idx] = np.append(
                ds.variables['clear_percent'][idx], stats['clear_percent'])
            # Append class-specific counts and percents
            for cls in stats['class_counts']:
                cnt = stats['class_counts'][cls]
                pct = (cnt / stats['usable_pixels'] * 100) if stats['usable_pixels'] > 0 else 0
                ds.variables[f'{cls}_pixels'][idx] = np.append(
                    ds.variables[f'{cls}_pixels'][idx], cnt)
                ds.variables[f'{cls}_percent'][idx] = np.append(
                    ds.variables[f'{cls}_percent'][idx], pct)
            # Update comma-separated image list
            old = ds.variables['image_names_csv'][idx] or ''
            ds.variables['image_names_csv'][idx] = f"{old},{image_name}" if old else image_name

        else:
            # New lake: initialize all variables at this index
            idx = len(ds.dimensions['lake'])
            ds.variables['lake_id'][idx]      = lake_id
            ds.variables['area'][idx]         = stats['area']
            ds.variables['perimeter'][idx]    = stats['perimeter']
            ds.variables['total_pixels'][idx] = stats['total_pixels']
            ds.variables['study_site'][idx]   = study_site
            ds.variables['obs_count'][idx]    = 1
            # Initialize VLEN arrays with a single entry (matching defined dtype)
            ds.variables['unix_time'][idx]     = np.array([int(timestamp)], dtype='int32')
            ds.variables['usable_pixels'][idx] = np.array([stats['usable_pixels']], dtype='int32')
            ds.variables['clear_percent'][idx] = np.array([stats['clear_percent']], dtype='float32')
            for cls in stats['class_counts']:
                cnt = stats['class_counts'][cls]
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
    # Open classified raster
    ds_class = gdal.Open(classified_path)
    if ds_class is None:
        raise IOError(f"Could not open classified raster: {classified_path}")
    band_class = ds_class.GetRasterBand(1)
    classified_data = band_class.ReadAsArray()
    nodata = band_class.GetNoDataValue()

    # Open lake ID mask raster
    ds_mask = gdal.Open(lake_id_mask_path)
    if ds_mask is None:
        raise IOError(f"Could not open lake ID mask: {lake_id_mask_path}")
    mask_band = ds_mask.GetRasterBand(1)
    lake_id_data = mask_band.ReadAsArray()

    # Check dimensions match
    if classified_data.shape != lake_id_data.shape:
        raise ValueError("Classified raster and lake ID mask have different dimensions")

    # Load vector attributes if provided
    lake_attrs = {}
    if lake_vector_path:
        drv = ogr.Open(lake_vector_path, 0)
        if drv:
            layer = drv.GetLayer()
            fields = config.get('vector_fields', {})
            id_fld = fields.get('id', 'id')
            area_fld = fields.get('area', 'area')
            perim_fld = fields.get('perimeter', 'perimeter')
            for feat in layer:
                lid = feat.GetField(id_fld)
                lake_attrs[lid] = {
                    'area': feat.GetField(area_fld) or 0,
                    'perimeter': feat.GetField(perim_fld) or 0
                }
            drv = None

    # Map integer codes to class names
    classes = list(config['thresholds'].keys())
    code_map = {i+1: classes[i].lower() for i in range(len(classes))}

    # Unique lake IDs (skip 0)
    unique_ids = np.unique(lake_id_data)
    unique_ids = unique_ids[unique_ids > 0]

    current_ts = time.time()

    for lid in unique_ids:
        mask = (lake_id_data == lid)
        total_pix = int(mask.sum())

        # Values inside lake mask
        vals = classified_data[mask]
        if nodata is not None:
            usable_vals = vals[vals != nodata]
        else:
            usable_vals = vals
        usable_count = int(len(usable_vals))
        clear_pct = (usable_count / total_pix * 100) if total_pix > 0 else 0

        # Compute class counts
        class_counts = {}
        if clear_pct >= config.get('min_clear_percent', 50):
            uniq_vals, cnts = np.unique(usable_vals, return_counts=True)
            for code, cnt in zip(uniq_vals, cnts):
                name = code_map.get(int(code))
                if name:
                    class_counts[name] = int(cnt)
        # Ensure every class appears, default to zero
        for name in code_map.values():
            class_counts.setdefault(name, 0)

        stats = {
            'area': lake_attrs.get(lid, {}).get('area', 0),
            'perimeter': lake_attrs.get(lid, {}).get('perimeter', 0),
            'total_pixels': total_pix,
            'usable_pixels': usable_count,
            'clear_percent': clear_pct,
            'class_counts': class_counts
        }

        add_observation(
            nc_path    = netcdf_path,
            lake_id    = lid,
            timestamp  = current_ts,
            study_site = study_site,
            stats      = stats,
            image_name = img_name
        )

    # Cleanup
    ds_class = None
    ds_mask  = None

    print(f"Processed {len(unique_ids)} lakes and updated: {netcdf_path}")
