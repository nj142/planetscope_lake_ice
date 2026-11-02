"""
Sentinel_DCC.py — MPI + multiprocessing Random Forest classification
                      for Sentinel-2 imagery on the DCC (multi-site)

WHAT THE SCRIPT DOES
--------------------
Iterates through every study site listed in `SITES` and, for each one:

  1. Splits that site's image folders evenly across MPI ranks (one rank
     per CPU task in run.sh — ranks live on the supercomputer's compute
     nodes).
  2. For every image, reads all FEATURE_BANDS + SCL on the 10 m grid,
     runs OmniCloudMask on (Red, Green, NIR) to produce a cloud
     classification, and copies the whole stack into POSIX shared memory.
  3. Splits the lakes inside that image's footprint evenly across worker
     processes (16 workers/rank by default — set in run.sh).
  4. Runs the same ordered list of per-lake steps on every lake — see
     `process_lake()` below.  STEP 3 builds a binary union of:
         (SCL ∈ SCL_INVALID_CODES)  OR  (OCM ∈ {Thick Cloud, Thin Cloud})
     so any pixel flagged by EITHER mask is excluded from RF input.
  5. Writes one CSV per site:
         {OUTPUT_DIR}/{SITE_NAME}_freezeup_lake_obs.csv
     with columns:
         lake_id, dataset, year, datetime, date,
         ice_pixels, water_pixels, ice_fraction,
         cloud_pixels, nodata_pixels, filename

DEPENDENCIES
------------
This script needs OmniCloudMask in addition to the original deps:

    pip install omnicloudmask

OmniCloudMask pulls in PyTorch as a transitive dependency.  On the DCC,
inference auto-selects the best available device (CUDA > MPS > CPU) — no
config needed for GPU vs. CPU.  Override via OCM_DEVICE if you want to
force one.

USAGE
-----
    mpirun -n 8 python Sentinel_DCC.py
    mpirun -n 8 python Sentinel_DCC.py --nimages 10 --workers 30
    mpirun -n 8 python Sentinel_DCC.py --sites YF NS

LAYOUT OF THIS FILE
-------------------
The script has two sections separated by a clearly-marked divider:

  1. USER-EDITABLE SECTION (just below this docstring)
       Site list, band configuration, OCM settings, and the per-lake
       processing steps.  Edit these to point at different data, swap
       algorithms, or add masks.  No HPC knowledge required.

  2. HPC MACHINERY (below the divider)
       MPI work distribution, shared memory, file discovery, per-image
       progress CSVs, timing instrumentation, OCM driver.  You should
       not need to edit anything here.

OUTPUT
------
    {OUTPUT_DIR}/progress/{SITE}/{image_folder}.csv  per-image, written live
    {OUTPUT_DIR}/{SITE}_freezeup_lake_obs.csv        final per-site CSV
"""

import os
import sys
import re
import glob
import time
import json
import csv
import shutil
import argparse
import random
import tempfile
import subprocess
import multiprocessing as mp
import multiprocessing.shared_memory as shm_mod
from collections import defaultdict
from multiprocessing import resource_tracker
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import transform_bounds
from rasterio.enums import Resampling
from rasterio.transform import Affine, rowcol
import geopandas as gpd
from shapely.geometry import Polygon, mapping
from osgeo import ogr, osr
osr.UseExceptions()
ogr.UseExceptions()

try:
    from mpi4py import MPI
    COMM    = MPI.COMM_WORLD
    RANK    = COMM.Get_rank()
    SIZE    = COMM.Get_size()
    HAS_MPI = True

except ImportError:
    print("WARNING: mpi4py not found — running serial (rank 0 of 1)")
    COMM, RANK, SIZE, HAS_MPI = None, 0, 1, False


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                                                                      ║
# ║                      USER-EDITABLE SECTION                           ║
# ║                                                                      ║
# ║  Five things to edit in this section:                                ║
# ║    1. SITES            · per-site paths and dataset names            ║
# ║    2. BAND CONFIG      · which bands and SCL/OCM codes to use        ║
# ║    3. OCM CONFIG       · OmniCloudMask runtime settings              ║
# ║    4. process_lake()   · the ordered STEPs run on every lake         ║
# ║    5. OUTPUT_COLUMNS   · which columns appear in the output CSV      ║
# ║                                                                      ║
# ║  Everything below the END-OF-USER-SECTION divider is HPC machinery   ║
# ║  (MPI, shared memory, work distribution, OCM driver) and does not    ║
# ║  need editing.                                                       ║
# ║                                                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝

# -----------------------------------------------------------------------
# SITES
# -----------------------------------------------------------------------
# One entry per study site.  The script iterates over these in order;
# for each site it runs the full MPI pipeline and writes a single CSV
# named {SITE['name']}_freezeup_lake_obs.csv in OUTPUT_DIR.
#
# Required keys per site:
#   name        unique short label (used in filenames + console logs)
#   dataset     value written to the "dataset" column of the output CSV
#   s2_folder   folder containing one sub-folder per S-2 image
#   lake_mask   shapefile of lake polygons covering the site
#   rf_model    trained RF .joblib with model + label_encoder + features
#
# Limit which sites run via --sites on the command line.
SITES = [
    {
        "name":       "YKD",
        "dataset":    "YKD_50x50",
        "s2_folder":  "/work/nj142/S2/YKD",
        "lake_mask":  "/work/nj142/ALPOD_Tiles/YKD_ALPOD/YKD_ALPOD.shp",
        "rf_model":   "/work/nj142/S2/Models/YKD/YKD_freezeup_RFmodel.joblib",
    },
    {
        "name":       "YF",
        "dataset":    "YF_50x50",
        "s2_folder":  "/work/nj142/S2/YF",
        "lake_mask":  "/work/nj142/ALPOD_Tiles/YF_ALPOD/YF_ALPOD.shp",
        "rf_model":   "/work/nj142/S2/Models/YF/YF_freezeup_RFmodel.joblib",
    },
    {
        "name":       "NS",
        "dataset":    "NS_50x50",
        "s2_folder":  "/work/nj142/S2/NS",
        "lake_mask":  "/work/nj142/ALPOD_Tiles/NS_ALPOD/NS_ALPOD.shp",
        "rf_model":   "/work/nj142/S2/Models/NS/NS_freezeup_RFmodel.joblib",
    },
]

# Where to write per-site outputs and per-image progress CSVs.
OUTPUT_DIR = "/hpc/home/nj142/Output"


# -----------------------------------------------------------------------
# BAND CONFIGURATION
# -----------------------------------------------------------------------
# Feature bands present in each S-2 image folder, named exactly as is.
# Must include B04 (Red), B03 (Green), and B08 (NIR) — these are the
# bands OmniCloudMask consumes.
FEATURE_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]

# 10 m reference band — every other band is resampled to its grid, and
# its bounding box is used as the image footprint.  (B02 is always 10 m.)
REF_BAND = "B02"

# Which bands feed OmniCloudMask.  Must be members of FEATURE_BANDS.
OCM_RED   = "B04"   # Red    (665 nm)
OCM_GREEN = "B03"   # Green  (560 nm)
OCM_NIR   = "B08"   # NIR    (842 nm)

# Sentinel-2 SCL codes considered invalid in STEP 3 (clouds, shadows,
# saturation, no-data, ...).
#   0=no data  1=saturated  2=dark area  3=cloud shadow
#   7=unclassified  8=cloud medium  9=cloud high  10=thin cirrus
SCL_INVALID_CODES = np.array([0, 1, 2, 3, 7, 8, 9, 10], dtype=np.uint16)

# OmniCloudMask classes considered cloud:
#   0 = Clear  1 = Thick Cloud  2 = Thin Cloud  3 = Cloud Shadow
# We exclude 1 and 2 (per project spec) — cloud shadow is left to SCL.
OCM_CLOUD_CODES = np.array([1, 2], dtype=np.uint16)

# Column in the lake mask shapefile holding each lake's unique integer ID.
LAKE_ID_COL = "id"


# -----------------------------------------------------------------------
# OMNICLOUDMASK CONFIGURATION
# -----------------------------------------------------------------------
# inference_device: None → auto (CUDA > MPS > CPU).  Set to "cpu",
# "cuda", or "mps" to force.  Auto is fine for the DCC.
OCM_DEVICE        = None
OCM_PATCH_SIZE    = 1000     # tile size for the U-Net forward pass
OCM_PATCH_OVERLAP = 300      # overlap between tiles
OCM_BATCH_SIZE    = 1        # bump up on GPU nodes for throughput
OCM_DTYPE         = "fp32"   # use "bf16" on Ampere+ GPUs


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                                                                      ║
# ║                       OUTPUT CSV LAYOUT                              ║
# ║                                                                      ║
# ║  One row is written per lake per image.  Columns come from two       ║
# ║  places:                                                             ║
# ║                                                                      ║
# ║    PER-IMAGE columns  (filled in by HPC machinery — same value for   ║
# ║                        every lake in a given image):                 ║
# ║      dataset           SITES[i]['dataset']  (e.g. "YF_50x50")        ║
# ║      year              YYYY parsed from the image folder name        ║
# ║      datetime          M/D/YYYY H:MM         (UTC, hour from folder  ║
# ║                                              if present, else 0)     ║
# ║      date              M/D/YYYY                                      ║
# ║      filename          S-2 image folder name                         ║
# ║                                                                      ║
# ║    PER-LAKE columns   (returned from STEP 7 of process_lake — every  ║
# ║                        key in that dict becomes a CSV column):       ║
# ║      lake_id           unique lake ID from the lake mask             ║
# ║      ice_pixels        ice pixel count from STEP 6                   ║
# ║      water_pixels      water pixel count from STEP 6                 ║
# ║      ice_fraction      ice / (ice + water)  (0 if denom is 0)        ║
# ║      cloud_pixels      pixels in the lake masked by SCL∪OCM          ║
# ║      nodata_pixels     non-cloud pixels in the lake whose feature    ║
# ║                        bands were all instrument-fill                ║
# ║                                                                      ║
# ║  Header + sample row:                                                ║
# ║    lake_id,dataset,year,datetime,date,                               ║
# ║      ice_pixels,water_pixels,ice_fraction,                           ║
# ║      cloud_pixels,nodata_pixels,filename                             ║
# ║    665429,YF_50x50,2019,9/5/2019 0:00,9/5/2019,                      ║
# ║      0,562,0,0,0,S2A_6WWU_20190905_0_L2A                             ║
# ║                                                                      ║
# ║  To add a column:                                                    ║
# ║    1. Compute its value in process_lake() (in any STEP).             ║
# ║    2. Add it to the dict returned in STEP 7.                         ║
# ║    3. Add its name to OUTPUT_COLUMNS at the end of this section.     ║
# ║                                                                      ║
# ║  To remove a column: delete its name from OUTPUT_COLUMNS.  (You can  ║
# ║  leave it in the STEP 7 dict — extras are silently dropped.)         ║
# ║                                                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                                                                      ║
# ║                   PER-LAKE PROCESSING STEPS                          ║
# ║                                                                      ║
# ║  Each worker calls `process_lake()` ONCE PER LAKE assigned to it.    ║
# ║  Steps run in the order they appear below — change, reorder, add,    ║
# ║  or remove steps to alter the analysis.                              ║
# ║                                                                      ║
# ║  Common edits:                                                       ║
# ║    · Replace STEP 6 (Random Forest) with a band threshold, NDWI,     ║
# ║      or any other classifier — just set ice_pixels & water_pixels.   ║
# ║    · Add a new STEP that ANDs an extra mask into valid_mask          ║
# ║      (e.g. terrain mask, water-body type, NDWI threshold).           ║
# ║    · Reorder steps if the data dependencies still hold.              ║
# ║    · Add a new column in STEP 7 (then add it to OUTPUT_COLUMNS).     ║
# ║                                                                      ║
# ║  band_data layout (set up for you by the HPC machinery):             ║
# ║    shape : (len(FEATURE_BANDS) + 2, rows, cols)   dtype: uint16      ║
# ║    slices 0 .. len(FEATURE_BANDS)-1   feature bands in order         ║
# ║    slice  len(FEATURE_BANDS)          SCL band                       ║
# ║    slice  len(FEATURE_BANDS) + 1      OmniCloudMask classification   ║
# ║                                                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝

def process_lake(lake_id, lake_geom, band_data, transform, nodata):
    """
    Run the full per-lake analysis on shared raster data.

    Parameters
    ----------
    lake_id    : int                       unique lake identifier
    lake_geom  : shapely.geometry.Polygon  polygon in the raster's CRS
    band_data  : np.ndarray, uint16        shape (n_feature_bands + 2, rows, cols)
                                           — second-to-last slice is SCL,
                                             last slice is OCM cloud class
    transform  : rasterio.transform.Affine raster-to-world transform
    nodata     : int or None               instrument fill value (None if not set)

    Returns
    -------
    dict — one entry per per-lake CSV column.  See STEP 7 below and
    OUTPUT_COLUMNS at the bottom of this section for the column list.
    """
    n_feature  = len(FEATURE_BANDS)
    rows, cols = band_data.shape[1], band_data.shape[2]

    # Defaults for the per-lake CSV fields.  Later steps overwrite these
    # if any pixels survive the masks.
    ice_pixels    = 0
    water_pixels  = 0
    cloud_pixels  = 0
    nodata_pixels = 0

    # ┌────────────────────────────────────────────────────────────────┐
    # │  STEP 1  ·  COMPUTE PIXEL-SPACE WINDOW AROUND THIS LAKE        │
    # └────────────────────────────────────────────────────────────────┘
    # Translate the lake polygon's geographic bounds into pixel
    # row/col indices, giving us a small rectangular window that
    # contains the lake.  Every later step works only on this window
    # — not the whole 10 980 × 10 980 image — which is the dominant
    # per-lake speedup.
    minx, miny, maxx, maxy = lake_geom.bounds
    row_top, col_left  = rowcol(transform, minx, maxy)
    row_bot, col_right = rowcol(transform, maxx, miny)

    row_off = max(0,    int(row_top))
    col_off = max(0,    int(col_left))
    row_end = min(rows, int(row_bot)   + 1)
    col_end = min(cols, int(col_right) + 1)

    # If the lake bbox doesn't overlap the image, skip STEPs 2–6 and
    # fall through to STEP 7 with all counters at 0.
    if row_off < row_end and col_off < col_end:
        win_rows, win_cols = row_end - row_off, col_end - col_off
        win_transform      = transform * Affine.translation(col_off, row_off)

        # ┌────────────────────────────────────────────────────────────────┐
        # │  STEP 2  ·  BURN LAKE POLYGON TO A BOOLEAN PIXEL MASK          │
        # └────────────────────────────────────────────────────────────────┘
        # rasterio.features.geometry_mask returns True for pixels INSIDE
        # the polygon (because invert=True).  We rasterize only into the
        # window, not the whole image.
        inside_mask = geometry_mask(
            [lake_geom],
            transform=win_transform,
            invert=True,
            out_shape=(win_rows, win_cols),
        )

        # ┌────────────────────────────────────────────────────────────────┐
        # │  STEP 3  ·  CLOUD MASK = SCL ∪ OmniCloudMask                   │
        # └────────────────────────────────────────────────────────────────┘
        # Per-pixel binary union of the two cloud sources:
        #   · SCL flags clouds, shadows, saturation, and no-data pixels.
        #   · OCM independently flags Thick / Thin cloud (codes 1, 2).
        # Any pixel flagged by EITHER mask is treated as cloud and is
        # excluded from the RF input.  cloud_pixels is the count of
        # such pixels INSIDE the lake.
        scl_window   = band_data[n_feature,     row_off:row_end, col_off:col_end]
        ocm_window   = band_data[n_feature + 1, row_off:row_end, col_off:col_end]
        scl_invalid  = np.isin(scl_window, SCL_INVALID_CODES)
        ocm_cloud    = np.isin(ocm_window, OCM_CLOUD_CODES)
        cloud_window = scl_invalid | ocm_cloud

        cloud_pixels = int(np.sum(inside_mask & cloud_window))

        # ┌────────────────────────────────────────────────────────────────┐
        # │  STEP 4  ·  NO-DATA MASK (instrument fill in feature bands)    │
        # └────────────────────────────────────────────────────────────────┘
        # Some image edges have all-zero (or all-nodata) pixels even
        # though SCL didn't flag them.  We track these separately so
        # they're reported as nodata_pixels rather than cloud_pixels.
        feat_window   = band_data[:n_feature, row_off:row_end, col_off:col_end]
        nodata_val    = nodata if nodata is not None else 0
        nodata_window = np.all(feat_window == nodata_val, axis=0)

        # Pixels inside the lake that survived the cloud mask:
        not_cloud     = inside_mask & ~cloud_window
        nodata_pixels = int(np.sum(not_cloud & nodata_window))

        # ┌────────────────────────────────────────────────────────────────┐
        # │  STEP 5  ·  EXTRACT BAND VALUES FOR SURVIVING PIXELS           │
        # └────────────────────────────────────────────────────────────────┘
        # Pixels that go into the RF: inside the lake, not cloud, not
        # nodata.  The model was trained on FEATURE_BANDS + SCL — we
        # do NOT pass the OCM layer to the model.
        valid_mask     = not_cloud & ~nodata_window
        n_model_bands  = n_feature + 1   # FEATURE_BANDS + SCL

        # ┌────────────────────────────────────────────────────────────────┐
        # │  STEP 6  ·  CLASSIFY EACH PIXEL WITH RANDOM FOREST             │
        # └────────────────────────────────────────────────────────────────┘
        # Hand the (bands × pixels) matrix to the trained RF model and
        # tally ice vs. water predictions.  Replace the body of this block
        # with another classifier (band thresholding, NDWI, deep learning,
        # ...) — just set ice_pixels and water_pixels.
        if np.any(valid_mask):
            model_block = band_data[:n_model_bands, row_off:row_end, col_off:col_end]
            pixel_data  = model_block[:, valid_mask]

            if pixel_data.size > 0:
                package      = RF_MODEL_PACKAGE
                model        = package["model"]
                le           = package["label_encoder"]
                feature_cols = package["feature_columns"]

                pixels_np = pixel_data.T

                if pixels_np.shape[1] != len(feature_cols):
                    raise ValueError(
                        f"Band count mismatch: stack has {pixels_np.shape[1]} "
                        f"bands but model expects {len(feature_cols)} "
                        f"features {feature_cols}"
                    )

                pixels_df   = pd.DataFrame(pixels_np, columns=feature_cols)
                predictions = model.predict(pixels_df)

                ice_idx   = int(np.where(le.classes_ == "ice")[0][0])
                water_idx = int(np.where(le.classes_ == "water")[0][0])

                ice_pixels   = int(np.sum(predictions == ice_idx))
                water_pixels = int(np.sum(predictions == water_idx))

    # ┌────────────────────────────────────────────────────────────────┐
    # │  STEP 7  ·  BUILD OUTPUT ROW (per-lake CSV columns)            │
    # └────────────────────────────────────────────────────────────────┘
    # Each key returned here becomes a per-lake column in the CSV.
    # Per-image columns (dataset, year, datetime, ...) are added
    # automatically by the HPC machinery — don't include them here.
    # If you add a key, also add its name to OUTPUT_COLUMNS below.
    total_classified = ice_pixels + water_pixels
    if total_classified > 0:
        ice_fraction = round(ice_pixels / total_classified, 4)
    else:
        ice_fraction = 0

    return {
        "lake_id":       lake_id,
        "ice_pixels":    ice_pixels,
        "water_pixels":  water_pixels,
        "ice_fraction":  ice_fraction,
        "cloud_pixels":  cloud_pixels,
        "nodata_pixels": nodata_pixels,
    }


# -----------------------------------------------------------------------
# OUTPUT CSV COLUMN ORDER
# -----------------------------------------------------------------------
# Final list of column names written to the CSV, in the order they will
# appear.  Per-image columns (filled in by HPC machinery) and per-lake
# columns from STEP 7 of process_lake are interleaved to match the
# requested time-series schema.
OUTPUT_COLUMNS = [
    "lake_id",        # per-lake  · unique lake ID
    "dataset",        # per-image · SITES[i]['dataset']
    "year",           # per-image · YYYY from the image folder name
    "datetime",       # per-image · M/D/YYYY H:MM (UTC)
    "date",           # per-image · M/D/YYYY
    "ice_pixels",     # per-lake  · RF "ice" predictions
    "water_pixels",   # per-lake  · RF "water" predictions
    "ice_fraction",   # per-lake  · ice / (ice + water)
    "cloud_pixels",   # per-lake  · pixels in lake masked by SCL ∪ OCM
    "nodata_pixels",  # per-lake  · non-cloud lake pixels with no band data
    "filename",       # per-image · S2 image folder name
]


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                                                                      ║
# ║                    END OF USER-EDITABLE SECTION                      ║
# ║                                                                      ║
# ║  Below: HPC machinery — MPI ranks, multiprocessing pool, POSIX       ║
# ║  shared memory, ogr2ogr lake clipping, file discovery, progress      ║
# ║  CSVs, OmniCloudMask driver, timing report.  You shouldn't need to   ║
# ║  edit anything here.                                                 ║
# ║                                                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝


# ---------------------------------------------------------------------------
# DERIVED PATHS / GLOBALS  (re-set per site by main())
# ---------------------------------------------------------------------------

CURRENT_SITE     = None        # dict from SITES — set per site by main()
RF_MODEL_PACKAGE = None        # populated by load_rf_model() before pool fork


def _site_progress_dir(site):
    return os.path.join(OUTPUT_DIR, "progress", site["name"])


def _site_output_path(site):
    return os.path.join(OUTPUT_DIR, f"{site['name']}_freezeup_lake_obs.csv")


# ---------------------------------------------------------------------------
# TIMING INFRASTRUCTURE
# ---------------------------------------------------------------------------

TIMINGS: defaultdict = defaultdict(float)


def _tick(key: str, elapsed: float) -> None:
    TIMINGS[key] += elapsed


def _timed(key: str):
    """Context manager — records wall time of the enclosed block into TIMINGS."""
    class _Ctx:
        def __enter__(self):
            self._t0 = time.perf_counter()
            return self
        def __exit__(self, *_):
            _tick(key, time.perf_counter() - self._t0)
    return _Ctx()


_TIMING_LABELS = {
    "footprint_extract":   "Raster footprint extraction (B02)",
    "clip_vector":         "ogr2ogr lake clip",
    "read_clipped_shp":    "Read clipped shapefile (gpd)",
    "raster_read":         "Band read + resample to 10 m (rasterio/GDAL)",
    "ocm_predict":         "OmniCloudMask inference",
    "shm_alloc":           "Shared-memory alloc + copy",
    "geom_project":        "Lake CRS reproject",
    "build_jobs":          "Build worker job list",
    "pool_map":            "pool.map wall time (all lakes)",
    "shm_cleanup":         "Shared-memory cleanup",
    "worker_shm_attach":   "[worker] shm attach/detach",
    "worker_process_lake": "[worker] process_lake() total",
    "load_rf_model":       "Load RF model (joblib)",
    "write_progress_csv":  "Write per-image progress CSVs",
    "write_csv":           "Write final per-site CSVs",
}


def _print_timing_report(all_timings: dict, total_wall: float) -> None:
    sorted_items    = sorted(all_timings.items(), key=lambda kv: kv[1], reverse=True)
    total_accounted = sum(v for _, v in sorted_items)
    col_w           = 46
    print(f"\n{'='*72}", flush=True)
    print(f"  WALL-TIME BREAKDOWN  (summed across all {SIZE} rank(s))", flush=True)
    print(f"{'='*72}", flush=True)
    print(f"  {'Stage':<{col_w}} {'Seconds':>10}  {'% of sum':>9}", flush=True)
    print(f"  {'-'*col_w} {'-'*10}  {'-'*9}", flush=True)
    for key, secs in sorted_items:
        label = _TIMING_LABELS.get(key, key)
        pct   = 100.0 * secs / total_accounted if total_accounted > 0 else 0.0
        print(f"  {label:<{col_w}} {secs:>10.2f}s  {pct:>8.1f}%", flush=True)
    print(f"  {'-'*col_w} {'-'*10}  {'-'*9}", flush=True)
    print(f"  {'Sum of timed stages':<{col_w}} {total_accounted:>10.2f}s", flush=True)
    print(f"  {'Total script wall time':<{col_w}} {total_wall:>10.2f}s", flush=True)
    print(f"{'='*72}\n", flush=True)


# ---------------------------------------------------------------------------
# OMNICLOUDMASK DRIVER
# ---------------------------------------------------------------------------

_OCM_PREDICT = None     # cached predict_from_array import (per process)


def _get_ocm_predict():
    """Lazy import of omnicloudmask.predict_from_array (per process)."""
    global _OCM_PREDICT
    if _OCM_PREDICT is None:
        try:
            from omnicloudmask import predict_from_array
        except ImportError as exc:
            raise ImportError(
                "OmniCloudMask is not installed in this Python environment. "
                "Install it with:\n"
                "    pip install omnicloudmask\n"
                "(it pulls in PyTorch as a dependency)."
            ) from exc
        _OCM_PREDICT = predict_from_array
    return _OCM_PREDICT


def compute_ocm_mask(band_stack: np.ndarray) -> np.ndarray:
    """
    Run OmniCloudMask on the (Red, Green, NIR) channels of band_stack.

    Parameters
    ----------
    band_stack : (len(FEATURE_BANDS) + 1, H, W) uint16
                 feature bands followed by SCL — same layout as
                 read_band_stack() returns.

    Returns
    -------
    (H, W) uint16 array of class labels:
        0 = Clear, 1 = Thick Cloud, 2 = Thin Cloud, 3 = Cloud Shadow
    """
    predict_from_array = _get_ocm_predict()

    red_idx   = FEATURE_BANDS.index(OCM_RED)
    green_idx = FEATURE_BANDS.index(OCM_GREEN)
    nir_idx   = FEATURE_BANDS.index(OCM_NIR)

    # OCM expects (3, H, W) — Red, Green, NIR.  It accepts uint16 raw
    # reflectance values and normalizes internally.
    rgb_nir = np.stack(
        [band_stack[red_idx], band_stack[green_idx], band_stack[nir_idx]],
        axis=0,
    )

    kwargs = dict(
        patch_size    = OCM_PATCH_SIZE,
        patch_overlap = OCM_PATCH_OVERLAP,
        batch_size    = OCM_BATCH_SIZE,
    )
    if OCM_DEVICE is not None:
        kwargs["inference_device"] = OCM_DEVICE
    if OCM_DTYPE is not None:
        kwargs["inference_dtype"] = OCM_DTYPE

    prediction = predict_from_array(rgb_nir, **kwargs)

    # Result is (1, H, W) — squeeze to (H, W).
    if prediction.ndim == 3:
        prediction = prediction[0]
    return prediction.astype(np.uint16)


def _verify_ocm_importable():
    """Rank-0 startup check so failures happen before image processing."""
    try:
        _get_ocm_predict()
    except ImportError as exc:
        print(f"\n[rank 0] FATAL: {exc}\n", flush=True)
        if HAS_MPI:
            COMM.Abort(1)
        sys.exit(1)


# ---------------------------------------------------------------------------
# RF MODEL LOADING
# ---------------------------------------------------------------------------

def load_rf_model(rf_model_path: str) -> None:
    """
    Load the trained RF package into RF_MODEL_PACKAGE before the
    multiprocessing pool is forked.  Workers inherit the model via
    copy-on-write — no re-loading per worker.

    n_jobs is forced to 1 to avoid contention with the worker pool.
    """
    global RF_MODEL_PACKAGE
    t0 = time.perf_counter()
    if not os.path.isfile(rf_model_path):
        raise FileNotFoundError(f"RF model not found: {rf_model_path}")
    package = joblib.load(rf_model_path)
    package["model"].set_params(n_jobs=1)
    RF_MODEL_PACKAGE = package
    print(f"[rank {RANK}] Loaded RF model  <-  {rf_model_path}", flush=True)
    _tick("load_rf_model", time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# MULTIPROCESSING WORKER (thin wrapper around process_lake)
# ---------------------------------------------------------------------------

def _rf_worker(args: tuple) -> tuple:
    """
    Multiprocessing worker — handles the shared-memory dance and then
    calls the user-defined process_lake() exactly once for one lake.

    Shared memory layout
    --------------------
    dtype : uint16
    shape : (len(FEATURE_BANDS) + 2, rows, cols)
      slices 0 .. len(FEATURE_BANDS)-1   feature bands in FEATURE_BANDS order
      slice  len(FEATURE_BANDS)          SCL (cast to uint16)
      slice  len(FEATURE_BANDS) + 1      OmniCloudMask classification

    Returns
    -------
    (lake_id, lake_row, worker_timings)
        lake_row        dict from process_lake() STEP 7, or None on error
        worker_timings  dict mapping timing-key -> seconds
    """
    (lake_id, geom_wkt, transform_tuple, shm_name,
     raster_shape, raster_dtype_str, nodata) = args

    wtimings: dict = defaultdict(float)
    existing_shm   = None

    try:
        # Attach to shared memory
        t = time.perf_counter()
        existing_shm = shm_mod.SharedMemory(name=shm_name)
        # Prevent the resource tracker from trying to unlink a block
        # owned by the main process (only the main needs to clean up).
        try:
            resource_tracker.unregister(f"/{shm_name}", "shared_memory")
        except (KeyError, ValueError):
            pass
        band_data = np.ndarray(
            raster_shape,
            dtype=np.dtype(raster_dtype_str),
            buffer=existing_shm.buf,
        )
        wtimings["worker_shm_attach"] += time.perf_counter() - t

        transform = Affine(*transform_tuple)
        from shapely import wkt as shapely_wkt
        geom = shapely_wkt.loads(geom_wkt)

        # Run the user-defined per-lake pipeline
        t = time.perf_counter()
        lake_row = process_lake(int(lake_id), geom, band_data, transform, nodata)
        wtimings["worker_process_lake"] += time.perf_counter() - t

        existing_shm.close()
        return (int(lake_id), lake_row, dict(wtimings))

    except Exception as exc:
        if existing_shm is not None:
            try: existing_shm.close()
            except Exception: pass
        print(f"[worker] lake {lake_id} error: {exc}", flush=True)
        return (int(lake_id), None, dict(wtimings))


# ---------------------------------------------------------------------------
# RASTER + VECTOR HELPERS
# ---------------------------------------------------------------------------

def get_footprint_from_band(band_path: str) -> tuple:
    """
    Open the reference band and return its footprint (in WGS84) plus
    metadata used by the rest of the pipeline.

    Returns
    -------
    footprint_4326, crs, transform, height, width, nodata
    """
    with rasterio.open(band_path) as src:
        crs       = src.crs
        transform = src.transform
        height    = src.height
        width     = src.width
        nodata    = src.nodata
        bounds    = src.bounds

    l, b, r, t = transform_bounds(crs, "EPSG:4326", *bounds)
    return (mapping(Polygon([(l, b), (r, b), (r, t), (l, t)])),
            crs, transform, height, width, nodata)


def read_band_stack(img_folder: str, height: int, width: int) -> np.ndarray:
    """
    Read FEATURE_BANDS + SCL from *img_folder*, resampled to (height, width)
    on the 10 m grid using nearest-neighbour (matches training pipeline).

    Returns shape (len(FEATURE_BANDS) + 1, H, W) uint16.  The OCM layer
    is added later by process_image() after compute_ocm_mask() runs.
    """
    all_bands = FEATURE_BANDS + ["SCL"]
    stack     = np.empty((len(all_bands), height, width), dtype=np.uint16)

    for i, bname in enumerate(all_bands):
        bpath = os.path.join(img_folder, f"{bname}.tif")
        if not os.path.isfile(bpath):
            raise FileNotFoundError(f"Missing band file: {bpath}")
        with rasterio.open(bpath) as src:
            stack[i] = src.read(
                1,
                out_shape=(height, width),
                resampling=Resampling.nearest,
            ).astype(np.uint16)

    return stack


def clip_vector_with_geometry(vector_path: str, geometry: dict,
                               output_path: str) -> int:
    """
    Clip *vector_path* to features entirely within *geometry* (WGS84 dict).
    Writes a new shapefile to *output_path*; returns its feature count.
    """
    geom    = ogr.CreateGeometryFromJson(json.dumps(geometry))
    srs4326 = osr.SpatialReference()
    srs4326.ImportFromEPSG(4326)
    srs4326.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    geom.AssignSpatialReference(srs4326)

    src_ds     = ogr.Open(vector_path)
    src_lyr    = src_ds.GetLayer()
    vec_srs    = src_lyr.GetSpatialRef()
    vec_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    layer_name = src_lyr.GetName()

    if not vec_srs.IsSame(srs4326):
        geom.Transform(osr.CoordinateTransformation(srs4326, vec_srs))

    auth_code = vec_srs.GetAuthorityCode(None)
    epsg      = int(auth_code) if auth_code else 4326
    sql       = (
        f"SELECT * FROM {layer_name} "
        f"WHERE ST_Within(Geometry, GeomFromText('{geom.ExportToWkt()}', {epsg}))"
    )
    subprocess.check_call([
        "ogr2ogr", "-f", "ESRI Shapefile",
        output_path, vector_path,
        "-dialect", "SQLite", "-sql", sql,
    ])

    out_ds = ogr.Open(output_path)
    count  = out_ds.GetLayer(0).GetFeatureCount()
    src_ds = None
    out_ds = None
    return count


_FOLDER_DATE_RE     = re.compile(r'(\d{8})')
_FOLDER_DATETIME_RE = re.compile(r'(\d{8}T\d{6})')


def _parse_folder_datetime(folder_name: str) -> datetime | None:
    """
    Try to parse a UTC datetime out of a Sentinel-2 image folder name.

    Supports two patterns:
      · YYYYMMDDTHHMMSS  (full L1C/L2A SAFE-style timestamp)
      · YYYYMMDD         (date-only — time defaults to 00:00:00 UTC)
    """
    m = _FOLDER_DATETIME_RE.search(folder_name)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            pass
    m = _FOLDER_DATE_RE.search(folder_name)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            pass
    return None


def _format_datetime_str(dt: datetime) -> str:
    """Format like '9/5/2019 4:00' (no leading zeros, portable)."""
    return f"{dt.month}/{dt.day}/{dt.year} {dt.hour}:{dt.minute:02d}"


def _format_date_str(dt: datetime) -> str:
    return f"{dt.month}/{dt.day}/{dt.year}"


def _cleanup_shp(shp_path: str):
    base = os.path.splitext(shp_path)[0]
    for ext in (".shp", ".dbf", ".shx", ".prj", ".cpg", ".sbn", ".sbx", ".shp.xml"):
        try: os.remove(base + ext)
        except FileNotFoundError: pass


# ---------------------------------------------------------------------------
# PROGRESS / RESUME HELPERS  (per-site)
# ---------------------------------------------------------------------------

def get_progress_path(site: dict, folder_name: str) -> str:
    return os.path.join(_site_progress_dir(site), f"{folder_name}.csv")


def image_already_done(site: dict, folder_name: str) -> bool:
    return os.path.isfile(get_progress_path(site, folder_name))


def write_progress_csv(rows: list, site: dict, folder_name: str) -> str:
    """Write per-image results immediately after pool.map() returns."""
    path = get_progress_path(site, folder_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore", restval="",
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def combine_progress_csvs(site: dict, output_path: str) -> None:
    """Merge every per-image progress CSV for this site into one CSV."""
    progress_files = sorted(glob.glob(
        os.path.join(_site_progress_dir(site), "*.csv")
    ))
    if not progress_files:
        print(
            f"[rank 0] WARNING: no progress CSVs found for {site['name']}.",
            flush=True,
        )
        return

    all_rows = []
    for pf in progress_files:
        try:
            with open(pf, newline="") as f:
                all_rows.extend(list(csv.DictReader(f)))
        except Exception as exc:
            print(f"[rank 0] WARNING: could not read {pf}: {exc}", flush=True)

    all_rows.sort(key=lambda r: (
        r.get("year", ""),
        r.get("filename", ""),
        int(r.get("lake_id") or 0),
    ))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore", restval="",
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(
        f"[rank 0] {site['name']}: combined {len(progress_files)} progress "
        f"files ({len(all_rows)} rows) -> {output_path}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# FILE DISCOVERY (rank 0 only)
# ---------------------------------------------------------------------------

def discover_files(site: dict, n_per_site: int | None, seed: int = 42) -> list:
    """
    Scan site['s2_folder'] for valid image folders (each must contain
    every FEATURE_BANDS tif plus SCL.tif).  Optionally subsample
    *n_per_site*.

    Images that already have a progress CSV are skipped automatically
    so resumed runs don't reprocess completed work.
    """
    rng        = random.Random(seed)
    s2_folder  = site["s2_folder"]
    site_name  = site["name"]

    if not os.path.isdir(s2_folder):
        print(f"[rank 0] WARNING: {s2_folder} not found — skipping site "
              f"{site_name}", flush=True)
        return []

    candidates = []
    for img_folder in sorted(glob.glob(os.path.join(s2_folder, "*"))):
        if not os.path.isdir(img_folder):
            continue
        folder_name = os.path.basename(img_folder)

        if not os.path.isfile(os.path.join(img_folder, f"{REF_BAND}.tif")):
            continue

        missing = [
            b for b in FEATURE_BANDS + ["SCL"]
            if not os.path.isfile(os.path.join(img_folder, f"{b}.tif"))
        ]
        if missing:
            print(
                f"[rank 0] WARNING: {site_name}/{folder_name} missing bands "
                f"{missing} — skipping",
                flush=True,
            )
            continue

        candidates.append((img_folder, folder_name))

    if n_per_site is None:
        sample = candidates
        print(f"[rank 0] {site_name}: using all {len(sample)} images",
              flush=True)
    else:
        sample = rng.sample(candidates, min(n_per_site, len(candidates)))
        print(
            f"[rank 0] {site_name}: {len(candidates)} images available, "
            f"sampling {len(sample)}",
            flush=True,
        )

    records, n_skipped = [], 0
    for (img_folder, folder_name) in sample:
        if image_already_done(site, folder_name):
            n_skipped += 1
            continue
        records.append({
            "path":        img_folder,
            "folder_name": folder_name,
        })

    if n_skipped:
        print(
            f"[rank 0] {site_name}: skipping {n_skipped} already-completed "
            f"image(s) (progress CSVs found in {_site_progress_dir(site)})",
            flush=True,
        )

    return records


# ---------------------------------------------------------------------------
# WORK DISTRIBUTION
# ---------------------------------------------------------------------------

def scatter_work(records: list) -> list:
    if not HAS_MPI:
        return records or []
    if RANK == 0:
        records = records or []
        chunks  = [[] for _ in range(SIZE)]
        for i, rec in enumerate(records):
            chunks[i % SIZE].append(rec)
        print("\n[rank 0] Work distribution:", flush=True)
        for r, chunk in enumerate(chunks):
            print(f"         rank {r:>2}: {len(chunk):>3} images", flush=True)
        print("", flush=True)
    else:
        chunks = None
    return COMM.scatter(chunks, root=0)


# ---------------------------------------------------------------------------
# PER-IMAGE PROCESSING
# ---------------------------------------------------------------------------

def _build_per_image_fields(site: dict, folder_name: str) -> dict:
    """Compute the static per-image columns once per image."""
    dt = _parse_folder_datetime(folder_name)
    if dt is None:
        # Couldn't parse a date — leave year/datetime/date blank.
        return {
            "dataset":  site["dataset"],
            "year":     "",
            "datetime": "",
            "date":     "",
            "filename": folder_name,
        }
    return {
        "dataset":  site["dataset"],
        "year":     str(dt.year),
        "datetime": _format_datetime_str(dt),
        "date":     _format_date_str(dt),
        "filename": folder_name,
    }


def process_image(rec: dict, site: dict, pool: mp.Pool, tmp_dir: str) -> list:
    """
    Full S2 pipeline for one image folder.  Returns a list of row dicts
    and writes a per-image progress CSV immediately so partial results
    survive a job kill.
    """
    img_folder  = rec["path"]
    folder_name = rec["folder_name"]
    site_name   = site["name"]
    t0          = time.perf_counter()

    print(f"\n[rank {RANK}] START {site_name}/{folder_name}", flush=True)

    base_row = _build_per_image_fields(site, folder_name)

    def err(msg, read_t=None):
        elapsed = time.perf_counter() - t0
        print(
            f"[rank {RANK}] ERROR {site_name}/{folder_name}: {msg}  "
            f"elapsed={elapsed:.2f}s",
            flush=True,
        )
        # Image-level failure: write an empty progress CSV (just the
        # header) so resume logic skips this image next time.
        t_csv = time.perf_counter()
        write_progress_csv([], site, folder_name)
        _tick("write_progress_csv", time.perf_counter() - t_csv)
        return []

    # 1. Footprint from the 10 m reference band
    ref_band_path = os.path.join(img_folder, f"{REF_BAND}.tif")
    try:
        with _timed("footprint_extract"):
            footprint, crs, transform, height, width, nodata = \
                get_footprint_from_band(ref_band_path)
    except Exception as exc:
        return err(f"Footprint extraction failed: {exc}")

    # 2. Clip the lake mask to the image footprint
    clip_out = os.path.join(
        tmp_dir, f"clip_{site_name}_r{RANK}_{folder_name}.shp"
    )
    try:
        with _timed("clip_vector"):
            n_lakes = clip_vector_with_geometry(
                site["lake_mask"], footprint, clip_out
            )
    except Exception as exc:
        return err(f"clip_vector_with_geometry failed: {exc}")

    if n_lakes == 0:
        elapsed = time.perf_counter() - t0
        print(
            f"[rank {RANK}] DONE  {site_name}/{folder_name}  "
            f"no lakes within footprint  elapsed={elapsed:.2f}s",
            flush=True,
        )
        _cleanup_shp(clip_out)
        t_csv = time.perf_counter()
        write_progress_csv([], site, folder_name)
        _tick("write_progress_csv", time.perf_counter() - t_csv)
        return []

    # 3. Read clipped lakes
    try:
        with _timed("read_clipped_shp"):
            lakes_gdf = gpd.read_file(clip_out)
    except Exception as exc:
        _cleanup_shp(clip_out)
        return err(f"Could not read clipped shapefile: {exc}")
    _cleanup_shp(clip_out)

    # 4. Read all feature bands + SCL, resampled to 10 m
    try:
        with _timed("raster_read"):
            base_stack = read_band_stack(img_folder, height, width)
    except Exception as exc:
        return err(f"Band read/resample failed: {exc}")

    # 5. Run OmniCloudMask on (Red, Green, NIR) and append as extra layer.
    #    Result shape: (len(FEATURE_BANDS) + 2, H, W) — feature bands,
    #    then SCL, then OCM cloud classification.
    try:
        with _timed("ocm_predict"):
            ocm_layer = compute_ocm_mask(base_stack)   # (H, W) uint16
        band_stack = np.empty(
            (base_stack.shape[0] + 1, height, width), dtype=np.uint16
        )
        band_stack[:base_stack.shape[0]] = base_stack
        band_stack[-1]                   = ocm_layer
        del base_stack, ocm_layer
    except Exception as exc:
        return err(f"OmniCloudMask inference failed: {exc}")

    read_time = time.perf_counter() - t0

    # 6. Copy band stack into POSIX shared memory
    raster_shape = band_stack.shape
    raster_dtype = str(band_stack.dtype)
    shm_name     = f"rf_s2_{site_name}_r{RANK}_{folder_name}"

    try:
        with _timed("shm_alloc"):
            image_shm  = shm_mod.SharedMemory(
                name=shm_name, create=True, size=band_stack.nbytes
            )
            shared_arr = np.ndarray(band_stack.shape, dtype=band_stack.dtype,
                                    buffer=image_shm.buf)
            shared_arr[:] = band_stack
            del band_stack, shared_arr
    except Exception as exc:
        return err(f"Shared memory allocation failed: {exc}")

    transform_tuple = (
        transform.a, transform.b, transform.c,
        transform.d, transform.e, transform.f,
    )

    # 7. Reproject lakes to raster CRS, build per-lake job list
    with _timed("geom_project"):
        lakes_proj = lakes_gdf.to_crs(crs)

    with _timed("build_jobs"):
        jobs = [
            (int(lake_row[LAKE_ID_COL]),
             lake_row.geometry.wkt,
             transform_tuple,
             shm_name,
             raster_shape,
             raster_dtype,
             nodata)
            for _, lake_row in lakes_proj.iterrows()
        ]

    n_workers_actual = pool._processes
    chunksize        = max(1, len(jobs) // max(1, n_workers_actual))

    print(
        f"[rank {RANK}]   distributing {n_lakes} lakes to {n_workers_actual} "
        f"workers (chunksize={chunksize}  read+ocm={read_time:.2f}s)",
        flush=True,
    )

    # 8. Dispatch lakes to the worker pool
    t_rf = time.perf_counter()
    with _timed("pool_map"):
        raw_results = pool.map(_rf_worker, jobs, chunksize=chunksize)
    rf_time = time.perf_counter() - t_rf

    results = []
    for lake_id, lake_row, wtimings in raw_results:
        for k, v in wtimings.items():
            _tick(k, v)
        results.append((lake_id, lake_row))

    # 9. Release shared memory
    with _timed("shm_cleanup"):
        try:
            image_shm.close()
            image_shm.unlink()
        except Exception as exc:
            print(
                f"[rank {RANK}] WARNING: shared memory cleanup failed: {exc}",
                flush=True,
            )

    elapsed_total = time.perf_counter() - t0

    print(
        f"[rank {RANK}] DONE  {site_name}/{folder_name}  "
        f"{n_lakes} lakes classified  "
        f"read+ocm={read_time:.2f}s  rf={rf_time:.2f}s  "
        f"elapsed={elapsed_total:.2f}s",
        flush=True,
    )

    rows = []
    for lake_id, lake_row in results:
        if lake_row is None:
            # Worker exception — emit a row with lake_id and zeros so
            # the lake isn't silently dropped from the time series.
            rows.append({
                **base_row,
                "lake_id":       lake_id,
                "ice_pixels":    0,
                "water_pixels":  0,
                "ice_fraction":  0,
                "cloud_pixels":  0,
                "nodata_pixels": 0,
            })
        else:
            rows.append({**base_row, **lake_row})

    # 10. Write per-image progress CSV immediately
    t_csv = time.perf_counter()
    csv_path = write_progress_csv(rows, site, folder_name)
    _tick("write_progress_csv", time.perf_counter() - t_csv)
    print(f"[rank {RANK}] PROGRESS  {site_name}/{folder_name}  -> {csv_path}",
          flush=True)

    return rows


# ---------------------------------------------------------------------------
# PER-SITE DRIVER
# ---------------------------------------------------------------------------

def run_site(site: dict, n_sample: int | None, n_workers: int,
             tmp_dir: str) -> None:
    """Run the full pipeline for a single site (called once per site)."""
    site_name = site["name"]
    global CURRENT_SITE
    CURRENT_SITE = site

    if RANK == 0:
        print(f"\n{'#'*72}", flush=True)
        print(f"#  SITE: {site_name}   dataset='{site['dataset']}'", flush=True)
        print(f"#  S2 folder : {site['s2_folder']}", flush=True)
        print(f"#  Lake mask : {site['lake_mask']}", flush=True)
        print(f"#  RF model  : {site['rf_model']}", flush=True)
        print(f"#  Output    : {_site_output_path(site)}", flush=True)
        print(f"{'#'*72}", flush=True)

        if not os.path.isfile(site["lake_mask"]):
            print(f"[rank 0] WARNING: lake mask not found: "
                  f"{site['lake_mask']} — skipping site",
                  flush=True)
            records = []
        elif not os.path.isfile(site["rf_model"]):
            print(f"[rank 0] WARNING: RF model not found: "
                  f"{site['rf_model']} — skipping site",
                  flush=True)
            records = []
        else:
            records = discover_files(site, n_sample)
            print(
                f"[rank 0] {site_name}: {len(records)} image(s) remaining "
                f"to process (completed images already skipped)",
                flush=True,
            )
    else:
        records = None

    # If rank 0 decided there's nothing to do for this site, broadcast a
    # sentinel so all ranks return early uniformly.
    if HAS_MPI:
        skip_flag = COMM.bcast(
            (RANK == 0 and (records is None or len(records) == 0)),
            root=0,
        )
    else:
        skip_flag = (records is None or len(records) == 0)

    if skip_flag:
        # Even if there are no new images, rank 0 still rebuilds the
        # final CSV from any existing progress files (handles resumed
        # runs where every image was already done).
        if RANK == 0:
            t_csv = time.perf_counter()
            combine_progress_csvs(site, _site_output_path(site))
            _tick("write_csv", time.perf_counter() - t_csv)
        if HAS_MPI:
            COMM.Barrier()
        return

    local_records = scatter_work(records)

    # Load THIS site's RF model before forking the pool — workers
    # inherit the model via copy-on-write.
    load_rf_model(site["rf_model"])

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=n_workers) as pool:
        for rec in local_records:
            process_image(rec, site, pool, tmp_dir)

    # Make sure every rank has finished before rank 0 combines outputs.
    if HAS_MPI:
        COMM.Barrier()

    if RANK == 0:
        t_csv = time.perf_counter()
        combine_progress_csvs(site, _site_output_path(site))
        _tick("write_csv", time.perf_counter() - t_csv)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    if RANK == 0:
        parser = argparse.ArgumentParser(
            description="RF benchmark S2 — MPI + multiprocessing + OCM, "
                        "multi-site"
        )
        parser.add_argument("--nimages", type=int, default=None,
                            help="Images to sample per site (default: all)")
        parser.add_argument("--workers", type=int,
                            default=max(1, mp.cpu_count() - 1),
                            help="Pool workers per rank (default: cpu_count - 1)")
        parser.add_argument("--sites", nargs="+", default=None,
                            help="Subset of site names to run (default: all "
                                 "sites in SITES)")
        args = parser.parse_args()
        cfg  = {
            "nimages": args.nimages,
            "workers": args.workers,
            "sites":   args.sites,
        }
    else:
        cfg = None

    if HAS_MPI:
        cfg = COMM.bcast(cfg, root=0)

    n_sample, n_workers = cfg["nimages"], cfg["workers"]
    sites_filter        = cfg["sites"]

    if sites_filter is None:
        sites_to_run = list(SITES)
    else:
        wanted = set(sites_filter)
        sites_to_run = [s for s in SITES if s["name"] in wanted]
        unknown = wanted - {s["name"] for s in SITES}
        if RANK == 0 and unknown:
            print(f"[rank 0] WARNING: unknown site(s) ignored: "
                  f"{sorted(unknown)}", flush=True)

    if RANK == 0 and not sites_to_run:
        print("[rank 0] FATAL: no sites to run.", flush=True)
        if HAS_MPI:
            COMM.Abort(1)
        sys.exit(1)

    tmp_dir = tempfile.mkdtemp(prefix=f"rf_s2_rank{RANK}_")

    if RANK == 0:
        global_start = time.time()
        n_label = str(n_sample) if n_sample is not None else "ALL"
        print(f"\n{'='*72}", flush=True)
        print(f"  Sentinel_DCC.py  |  {SIZE} MPI ranks  |  {n_workers} workers/rank",
              flush=True)
        print(f"  Sites      : {[s['name'] for s in sites_to_run]}", flush=True)
        print(f"  Images/site: {n_label}", flush=True)
        print(f"  Feat. bands: {FEATURE_BANDS}", flush=True)
        print(f"  SCL excl.  : {SCL_INVALID_CODES.tolist()}", flush=True)
        print(f"  OCM excl.  : {OCM_CLOUD_CODES.tolist()} "
              f"(1=Thick Cloud, 2=Thin Cloud)", flush=True)
        print(f"  OCM device : {OCM_DEVICE if OCM_DEVICE else 'auto'}",
              flush=True)
        print(f"  Output dir : {OUTPUT_DIR}", flush=True)
        print(f"{'='*72}\n", flush=True)

        # Verify OCM is importable up front so we fail fast rather than
        # mid-run.  (Workers will import lazily on first use.)
        _verify_ocm_importable()
    else:
        global_start = None

    # ── Run each site sequentially ────────────────────────────────────
    for site in sites_to_run:
        try:
            run_site(site, n_sample, n_workers, tmp_dir)
        except Exception as exc:
            print(f"[rank {RANK}] SITE {site['name']} FAILED: {exc}", flush=True)
            if HAS_MPI:
                COMM.Barrier()

    shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Gather timings + final report on rank 0 ──────────────────────
    if HAS_MPI:
        all_timings_nested = COMM.gather(dict(TIMINGS), root=0)
    else:
        all_timings_nested = [dict(TIMINGS)]

    if RANK == 0:
        total_elapsed = time.time() - global_start

        combined: defaultdict = defaultdict(float)
        for rank_timings in all_timings_nested:
            for k, v in rank_timings.items():
                combined[k] += v

        print(f"\n{'='*72}", flush=True)
        print(f"  Total wall time : {total_elapsed:.2f}s  "
              f"({total_elapsed/60:.1f} min)", flush=True)
        print(f"  Per-site outputs:", flush=True)
        for site in sites_to_run:
            out = _site_output_path(site)
            exists = "✓" if os.path.isfile(out) else "✗"
            print(f"    {exists}  {out}", flush=True)
        print(f"{'='*72}\n", flush=True)

        _print_timing_report(combined, total_elapsed)


if __name__ == "__main__":
    mp.freeze_support()
    main()