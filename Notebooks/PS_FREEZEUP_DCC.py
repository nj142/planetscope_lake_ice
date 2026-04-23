"""
rf_benchmark.py - MPI + multiprocessing Random Forest classification benchmark
                   for PlanetScope imagery on the Duke Compute Cluster (DCC)

Parallelism:
  Outer (MPI)               images distributed across nodes/ranks
  Inner (multiprocessing)   per-lake RF jobs within each rank

Per-image workflow:
  1. Locate the co-sited XML metadata file and parse the Planet footprint polygon
  2. Clip the study site's ALPOD lake shapefile to that footprint
     (ST_Within via ogr2ogr - lakes entirely inside the valid image area only)
  3. Read the clipped lakes into memory and clean up the temp shapefile
  4. Read all bands from the TIF into POSIX shared memory as a single uint16
     array of shape (n_bands, height, width).  One copy per rank; workers map
     the same physical pages.
  5. Each worker:
       a. Computes the lake polygon's pixel-space bounding box (row/col window)
       b. Burns the lake polygon to a boolean mask ONLY within that window
       c. Extracts surviving pixel columns from the feature bands in the window
       d. Runs RF.predict()
     All masking is therefore per-lake, not per-image - only the tiny window
     around each lake is touched, not the full raster.
  6. Shared memory released after pool.map() returns for that image.
  7. Results written to a per-image progress CSV immediately on completion.
     On restart, images with an existing progress CSV are skipped automatically.
  8. Rank 0 gathers all results, combines per-image CSVs into one master CSV,
     and writes timing stats.

Progress / resume behaviour:
  Per-image CSVs are written to OUTPUT_DIR/progress/{core_name}.csv as soon
  as pool.map() returns for that image.  On the next run, rank 0 reads that
  directory before scattering work and drops already-finished images, so no
  rank wastes a slot on them.  The final master CSV is assembled by combining
  every file in the progress directory.

Memory layout per rank:
  Main process RAM  : full band stack written into a POSIX shared memory block
  Shared memory     : one block per image, named "rf_rank<R>_<core_name>"
                      lives in /dev/shm (tmpfs), counted against node RAM,
                      NOT disk.  Workers only read - copy-on-write never fires.
  Worker process RAM: boolean lake mask + pixel slice for the lake's bounding-box
                      window only - all local, freed when the worker function returns.

Timing:
  Module-level TIMINGS dict (defaultdict(float)) accumulates wall seconds per
  stage on every rank.  Worker timings are returned in the result tuple and
  folded into TIMINGS by the main process.  Rank 0 gathers and sums all ranks,
  then prints a sorted breakdown table.

Directory layout expected on the DCC:
  /work/nj142/<SITE>/<YEAR_FOLDER>/<IMAGE>.tif
  /work/nj142/<SITE>/<YEAR_FOLDER>/<IMAGE_PREFIX>_*.xml
  /work/nj142/RF_Models/<SITE>_planet_training_RFmodel.joblib
  /work/nj142/ALPOD_Tiles/<SITE>_ALPOD/*.shp

Arguments:
    --datasets  subset of [NS_50x50, YF_50x50, YKD_50x50]  (default: all three)
    --nimages   images to sample per study site (default: ALL images)
    --workers   Pool workers per rank    (default: cpu_count - 1)

Output:
    /hpc/home/nj142/Output/progress/{core_name}.csv   (per-image, written live)
    /hpc/home/nj142/Output/rf_benchmark_<TIMESTAMP>.csv  (final combined)

"""

import os
import sys
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
import xml.etree.ElementTree as ET
from datetime import datetime
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import Affine, rowcol
import geopandas as gpd
from shapely.geometry import Polygon, mapping
from osgeo import ogr, osr
osr.UseExceptions()
ogr.UseExceptions()

# MPI setup
try:
    from mpi4py import MPI
    COMM    = MPI.COMM_WORLD
    RANK    = COMM.Get_rank()
    SIZE    = COMM.Get_size()
    HAS_MPI = True
except ImportError:
    print("WARNING: mpi4py not found - running serial (rank 0 of 1)")
    COMM    = None
    RANK    = 0
    SIZE    = 1
    HAS_MPI = False


# -----------------------------------------------------------------
# TIMING INFRASTRUCTURE
# -----------------------------------------------------------------

TIMINGS: defaultdict = defaultdict(float)


def _tick(key: str, elapsed: float) -> None:
    """Add *elapsed* seconds to the named timing bucket."""
    TIMINGS[key] += elapsed


def _timed(key: str):
    """
    Context manager - records wall time of the enclosed block into TIMINGS.

    Usage::

        with _timed("raster_read"):
            data = src.read()
    """
    class _Ctx:
        def __enter__(self):
            self._t0 = time.perf_counter()
            return self
        def __exit__(self, *_):
            _tick(key, time.perf_counter() - self._t0)
    return _Ctx()


_TIMING_LABELS = {
    # main-process stages (per image, accumulated across all images)
    "xml_parse":            "XML metadata parse",
    "clip_vector":          "ogr2ogr lake clip",
    "read_clipped_shp":     "Read clipped shapefile (gpd)",
    "raster_read":          "Raster read (rasterio)",
    "shm_alloc":            "Shared-memory alloc + copy",
    "geom_project":         "Lake CRS reproject",
    "build_jobs":           "Build worker job list",
    "pool_map":             "pool.map wall time (all lakes)",
    "shm_cleanup":          "Shared-memory cleanup",
    # worker-side stages (aggregated from all lakes on this rank)
    "worker_shm_attach":    "[worker] shm attach/detach",
    "worker_bbox_calc":     "[worker] Lake pixel bounding box",
    "worker_mask_burn":     "[worker] Polygon mask burn (windowed)",
    "worker_pixel_extract": "[worker] Pixel extraction",
    "worker_rf_classify":   "[worker] RF classify_lake_rf",
    # one-shot stages
    "load_rf_models":       "Load RF models (joblib)",
    "write_progress_csv":   "Write per-image progress CSVs",
    "write_csv":            "Write final combined CSV",
}


def _print_timing_report(all_timings: dict, total_wall: float) -> None:
    """Print a sorted wall-time breakdown table to stdout (rank 0 only)."""
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


# -----------------------------------------------------------------
# RF MODEL LOADING
# -----------------------------------------------------------------
import joblib

RF_MODELS_DIR = "/work/nj142/RF_Models"
RF_MODELS: dict = {}


def load_rf_models(datasets: list):
    """
    Load one RF model per dataset into RF_MODELS before the Pool is forked.
    Workers inherit the model objects via copy-on-write - no re-loading needed.
    Forces n_jobs=1 to avoid conflicts with the multiprocessing pool.
    """
    t0 = time.perf_counter()
    for ds in datasets:
        site       = ds.split("_")[0]
        model_path = os.path.join(
            RF_MODELS_DIR, f"{site}_planet_training_RFmodel.joblib"
        )
        if not os.path.isfile(model_path):
            print(f"[rank {RANK}] WARNING: RF model not found: {model_path}", flush=True)
            continue
        package = joblib.load(model_path)
        package["model"].set_params(n_jobs=1)
        RF_MODELS[ds] = package
        print(f"[rank {RANK}] Loaded RF model for {ds}  <-  {model_path}", flush=True)
    _tick("load_rf_models", time.perf_counter() - t0)


# -----------------------------------------------------------------
# RF CLASSIFICATION
# -----------------------------------------------------------------

def classify_lake_rf(pixel_data: np.ndarray, nodata=None, dataset: str = "") -> tuple:
    """
    Run RF prediction on a (n_bands, n_valid_pixels) array.

    Pixels are already extracted from the lake's bounding-box window by the
    worker before being passed here.  The optional nodata check below removes
    any residual all-zero columns (instrument fill value) as a safety net.

    Returns (ice_pixel_count, water_pixel_count).
    """
    package = RF_MODELS.get(dataset)
    if package is None:
        return 0, 0

    model        = package["model"]
    le           = package["label_encoder"]
    feature_cols = package["feature_columns"]

    if pixel_data.size == 0:
        return 0, 0

    if nodata is not None:
        valid_mask = ~np.all(pixel_data == nodata, axis=0)
        pixels_np  = pixel_data[:, valid_mask].T
    else:
        pixels_np  = pixel_data.T

    if pixels_np.size == 0:
        return 0, 0

    if pixels_np.shape[1] != len(feature_cols):
        raise ValueError(
            f"Band count mismatch: raster has {pixels_np.shape[1]} bands "
            f"but model expects {len(feature_cols)} features {feature_cols}"
        )

    pixels_df   = pd.DataFrame(pixels_np, columns=feature_cols)
    predictions = model.predict(pixels_df)

    ice_idx   = int(np.where(le.classes_ == "ice")[0][0])
    water_idx = int(np.where(le.classes_ == "water")[0][0])

    return int(np.sum(predictions == ice_idx)), int(np.sum(predictions == water_idx))


# -----------------------------------------------------------------
# MULTIPROCESSING WORKER
# -----------------------------------------------------------------

def _rf_worker(args: tuple) -> tuple:
    """
    Worker function for a single lake within one PlanetScope image.

    Shared memory layout
    --------------------
    dtype : matches source raster (typically uint16)
    shape : (n_bands, rows, cols)

    Processing steps (all masking is WINDOWED to the lake's bounding box)
    -----------------------------------------------------------------------
    1. Attach to the named shared memory block (read-only view).
    2. Compute the lake polygon's bounding box in pixel-space (row/col window).
       This is a tiny arithmetic operation - no array scans.
    3. Burn the lake polygon to a boolean mask using geometry_mask(), but only
       for the window sub-array, not the full image.
    4. Slice out the feature band columns for surviving pixels within the window.
    5. Run classify_lake_rf().

    Why windowing matters
    ---------------------
    Without windowing, every lake runs geometry_mask() across the full image
    (e.g. 4000 x 4000 = ~16 M pixels for a PS tile).  With windowing those
    arrays shrink to the lake's bounding box - typically a few hundred to a
    few thousand pixels - cutting per-lake masking time by 3-4 orders of
    magnitude and making this the dominant speedup.

    Returns
    -------
    (lake_id, ice, water, worker_timings)
        worker_timings  dict mapping timing-key -> seconds
    """
    (lake_id, geom_wkt, transform_tuple, shm_name,
     raster_shape, raster_dtype_str, nodata, dataset) = args

    wtimings: dict = defaultdict(float)

    try:
        # attach to shared memory
        t = time.perf_counter()
        existing_shm = shm_mod.SharedMemory(name=shm_name)
        # Prevent the resource tracker from trying to unlink a block owned by
        # the main process (it only needs to be cleaned up once, by the main).
        try:
            resource_tracker.unregister(f"/{shm_name}", "shared_memory")
        except (KeyError, ValueError):
            pass
        dtype = np.dtype(raster_dtype_str)
        data  = np.ndarray(raster_shape, dtype=dtype, buffer=existing_shm.buf)
        wtimings["worker_shm_attach"] += time.perf_counter() - t

        n_bands, rows, cols = raster_shape

        transform = Affine(*transform_tuple)

        from shapely import wkt as shapely_wkt
        geom = shapely_wkt.loads(geom_wkt)

        # compute pixel-space bounding box of the lake polygon
        # All masking will be performed only within this window, not the full
        # image, which is the key performance improvement over the original.
        t = time.perf_counter()
        minx, miny, maxx, maxy = geom.bounds

        # rowcol() maps (x, y) geographic coords -> (row, col) pixel indices.
        # Top-left of bbox  = (minx, maxy);  bottom-right = (maxx, miny).
        row_top, col_left  = rowcol(transform, minx, maxy)
        row_bot, col_right = rowcol(transform, maxx, miny)

        # rowcol can return floats; cast and add 1 to make the right/bottom
        # inclusive, then clamp to the valid raster extent.
        row_off = max(0, int(row_top))
        col_off = max(0, int(col_left))
        row_end = min(rows, int(row_bot) + 1)
        col_end = min(cols, int(col_right) + 1)
        wtimings["worker_bbox_calc"] += time.perf_counter() - t

        if row_off >= row_end or col_off >= col_end:
            # Lake bbox is entirely outside the raster extent - nothing to do.
            existing_shm.close()
            return (int(lake_id), 0, 0, dict(wtimings))

        win_rows = row_end - row_off
        win_cols = col_end - col_off

        # Shift the affine transform to the top-left of the window so that
        # geometry_mask() burns into the correctly-positioned sub-array.
        win_transform = transform * Affine.translation(col_off, row_off)

        # burn polygon mask (windowed)
        t = time.perf_counter()
        inside_mask = geometry_mask(
            [geom], transform=win_transform,
            invert=True, out_shape=(win_rows, win_cols),
        )
        wtimings["worker_mask_burn"] += time.perf_counter() - t

        # extract pixel columns for all bands within the window.
        # planet has no cloud mask - inside_mask is the only filter applied.
        t = time.perf_counter()
        pixel_data = data[:, row_off:row_end, col_off:col_end][:, inside_mask]
        wtimings["worker_pixel_extract"] += time.perf_counter() - t

        existing_shm.close()

        # RF classification
        t = time.perf_counter()
        ice, water = classify_lake_rf(pixel_data, nodata, dataset)
        wtimings["worker_rf_classify"] += time.perf_counter() - t

        return (int(lake_id), int(ice), int(water), dict(wtimings))

    except Exception as exc:
        try:
            existing_shm.close()
        except Exception:
            pass
        print(f"[worker] lake {lake_id} error: {exc}", flush=True)
        return (int(lake_id), -1, -1, dict(wtimings))


# -----------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------

def extract_geospatial_info_from_xml(xml_file_path: str) -> dict:
    tree = ET.parse(xml_file_path)
    ns = {
        'gml': 'http://www.opengis.net/gml',
        'ps':  'http://schemas.planet.com/ps/v1/planet_product_metadata_geocorrected_level',
    }
    coords_text = tree.find(
        './/gml:outerBoundaryIs//gml:coordinates', namespaces=ns
    ).text.strip()
    coords    = [tuple(map(float, xy.split(','))) for xy in coords_text.split()]
    poly      = Polygon(coords)
    epsg_code = int(tree.find('.//ps:epsgCode', namespaces=ns).text)
    return {'geometry': mapping(poly), 'epsg_code': epsg_code}


def clip_vector_with_geometry(vector_path: str, geometry: dict,
                               output_path: str) -> int:
    """
    Clip *vector_path* to features entirely within *geometry* (WGS84 dict).
    Writes an ESRI Shapefile to *output_path*.
    Returns the feature count of the clipped layer.
    """
    geom = ogr.CreateGeometryFromJson(json.dumps(geometry))
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
        tr = osr.CoordinateTransformation(srs4326, vec_srs)
        geom.Transform(tr)

    auth_code = vec_srs.GetAuthorityCode(None)
    epsg      = int(auth_code) if auth_code else 4326
    wkt       = geom.ExportToWkt()
    sql       = (
        f"SELECT * FROM {layer_name} "
        f"WHERE ST_Within(Geometry, GeomFromText('{wkt}', {epsg}))"
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


def extract_unix_time_from_image_name(image_core_name: str) -> int:
    try:
        parts = image_core_name.split('_')
        dt    = datetime.strptime(f"{parts[0]}_{parts[1]}", "%Y%m%d_%H%M%S")
        return int(dt.timestamp())
    except (ValueError, IndexError):
        return 0


def _cleanup_shp(shp_path: str):
    base = os.path.splitext(shp_path)[0]
    for ext in (".shp", ".dbf", ".shx", ".prj", ".cpg", ".sbn", ".sbx", ".shp.xml"):
        try:
            os.remove(base + ext)
        except FileNotFoundError:
            pass


# -----------------------------------------------------------------
# PATHS AND DATASET CONFIGURATION
# -----------------------------------------------------------------

WORK_ROOT    = "/work/nj142"
ALPOD_ROOT   = "/work/nj142/ALPOD_Tiles"
OUTPUT_DIR   = "/hpc/home/nj142/Output"
PROGRESS_DIR = os.path.join(OUTPUT_DIR, "progress")
ALL_DATASETS = ["NS_50x50", "YF_50x50", "YKD_50x50"]

ALPOD_DIRS = {
    "NS_50x50":  os.path.join(ALPOD_ROOT, "NS_ALPOD"),
    "YF_50x50":  os.path.join(ALPOD_ROOT, "YF_ALPOD"),
    "YKD_50x50": os.path.join(ALPOD_ROOT, "YKD_ALPOD"),
}

LAKE_ID_COL = "id"

CSV_FIELDS = [
    "rank", "dataset", "year_folder", "year", "filename", "unix_timestamp",
    "lake_id", "ice_pixels", "water_pixels",
    "n_lakes_in_image", "n_valid_pixels", "read_time_s", "rf_time_s",
    "error",
]


def find_alpod_shapefile(dataset: str) -> str:
    alpod_dir = ALPOD_DIRS.get(dataset)
    if not alpod_dir or not os.path.isdir(alpod_dir):
        raise FileNotFoundError(f"ALPOD dir not found for {dataset}: {alpod_dir}")
    matches = glob.glob(os.path.join(alpod_dir, "*.shp"))
    if not matches:
        raise FileNotFoundError(f"No .shp found in {alpod_dir}")
    return sorted(matches)[0]


# -----------------------------------------------------------------
# PROGRESS / RESUME HELPERS
# -----------------------------------------------------------------

def get_progress_path(core_name: str) -> str:
    """Return the path for the per-image progress CSV."""
    return os.path.join(PROGRESS_DIR, f"{core_name}.csv")


def image_already_done(core_name: str) -> bool:
    """True if a progress CSV already exists for this image."""
    return os.path.isfile(get_progress_path(core_name))


def write_progress_csv(rows: list, core_name: str) -> str:
    """
    Write per-image results to PROGRESS_DIR/{core_name}.csv immediately
    after pool.map() returns.  This is the crash-safe intermediate record.

    Returns the path written.
    """
    path = get_progress_path(core_name)
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def combine_progress_csvs(output_path: str):
    """
    Read every CSV in PROGRESS_DIR, sort by dataset / year / filename / lake,
    and write the merged master CSV to *output_path*.
    """
    progress_files = sorted(glob.glob(os.path.join(PROGRESS_DIR, "*.csv")))
    if not progress_files:
        print("[rank 0] WARNING: no progress CSVs found to combine.", flush=True)
        return

    all_rows = []
    for pf in progress_files:
        try:
            with open(pf, newline="") as f:
                reader = csv.DictReader(f)
                all_rows.extend(list(reader))
        except Exception as exc:
            print(f"[rank 0] WARNING: could not read {pf}: {exc}", flush=True)

    all_rows.sort(key=lambda r: (
        r.get("dataset", ""),
        r.get("year_folder", ""),
        r.get("filename", ""),
        int(r.get("lake_id") or 0),
    ))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(
        f"[rank 0] Combined {len(progress_files)} progress files "
        f"({len(all_rows)} rows) -> {output_path}",
        flush=True,
    )


# -----------------------------------------------------------------
# FILE DISCOVERY (rank 0 only)
# -----------------------------------------------------------------

def discover_files(datasets: list, n_per_site: int | None, seed: int = 42) -> list:
    """
    Scan WORK_ROOT/<site>/Freezeup_*/ for valid TIF images (excluding UDM files).
    Optionally subsample *n_per_site* images per dataset.

    Images that already have a progress CSV in PROGRESS_DIR are logged and
    excluded from the returned list so no rank wastes a slot on them.
    """
    rng     = random.Random(seed)
    records = []

    for ds in datasets:
        ds_root = os.path.join(WORK_ROOT, ds)
        if not os.path.isdir(ds_root):
            print(f"[rank 0] WARNING: {ds_root} not found - skipping", flush=True)
            continue

        all_candidates = []
        for yd in sorted(glob.glob(os.path.join(ds_root, "Freezeup_*"))):
            if not os.path.isdir(yd):
                continue
            folder_name = os.path.basename(yd)
            year_str    = folder_name.split("_")[-1]

            for tif in sorted(glob.glob(os.path.join(yd, "*.tif"))):
                filename = os.path.basename(tif)
                if "UDM" in filename.upper():
                    continue

                parts  = filename.split("_")
                prefix = "_".join(parts[:4])

                xml_candidates = glob.glob(os.path.join(yd, f"{prefix}_*.xml"))
                core_name      = os.path.splitext(filename)[0]
                xml_path       = xml_candidates[0] if xml_candidates else None

                all_candidates.append(
                    (tif, folder_name, year_str, filename, core_name, xml_path)
                )

        if n_per_site is None:
            sample = all_candidates
            print(f"[rank 0] {ds}: using all {len(sample)} images", flush=True)
        else:
            sample = rng.sample(all_candidates, min(n_per_site, len(all_candidates)))
            print(
                f"[rank 0] {ds}: {len(all_candidates)} images available, "
                f"sampling {len(sample)}",
                flush=True,
            )

        # RESUME: skip images that already have a progress CSV
        n_skipped = 0
        for (tif, folder_name, year_str, filename, core_name, xml_path) in sample:
            if image_already_done(core_name):
                n_skipped += 1
                continue
            records.append({
                "path":        tif,
                "xml_path":    xml_path,
                "dataset":     ds,
                "year_folder": folder_name,
                "year":        year_str,
                "filename":    filename,
                "core_name":   core_name,
            })

        if n_skipped:
            print(
                f"[rank 0] {ds}: skipping {n_skipped} already-completed image(s) "
                f"(progress CSVs found in {PROGRESS_DIR})",
                flush=True,
            )

    return records


# -----------------------------------------------------------------
# WORK DISTRIBUTION
# -----------------------------------------------------------------

def scatter_work(records: list) -> list:
    if not HAS_MPI:
        return records
    if RANK == 0:
        chunks = [[] for _ in range(SIZE)]
        for i, rec in enumerate(records):
            chunks[i % SIZE].append(rec)
        print(f"\n[rank 0] Work distribution:", flush=True)
        for r, chunk in enumerate(chunks):
            print(f"         rank {r:>2}: {len(chunk):>3} images", flush=True)
        print("", flush=True)
    else:
        chunks = None
    return COMM.scatter(chunks, root=0)


# -----------------------------------------------------------------
# PER-IMAGE PROCESSING
# -----------------------------------------------------------------

def process_image(rec: dict, alpod_shapefiles: dict,
                  pool: mp.Pool, tmp_dir: str) -> list:
    """
    Full PlanetScope pipeline for one image TIF.

    Returns a list of row dicts (one per lake) for the output CSV, and
    immediately writes a per-image progress CSV to PROGRESS_DIR so that
    results are preserved even if the job is killed before completion.
    """
    path      = rec["path"]
    xml_path  = rec["xml_path"]
    dataset   = rec["dataset"]
    core_name = rec["core_name"]
    t0        = time.perf_counter()

    print(
        f"\n[rank {RANK}] START {dataset}/{rec['year_folder']}/{rec['filename']}",
        flush=True,
    )

    base_row = {
        "rank":           RANK,
        "dataset":        dataset,
        "year_folder":    rec["year_folder"],
        "year":           rec["year"],
        "filename":       rec["filename"],
        "unix_timestamp": extract_unix_time_from_image_name(core_name),
        "error":          "",
    }

    def err(msg, read_t=None):
        elapsed = time.perf_counter() - t0
        print(
            f"[rank {RANK}] ERROR {rec['filename']}: {msg}  elapsed={elapsed:.2f}s",
            flush=True,
        )
        rows = [{**base_row,
                 "lake_id": None, "ice_pixels": -1, "water_pixels": -1,
                 "n_lakes_in_image": 0, "n_valid_pixels": -1,
                 "read_time_s": round(read_t or elapsed, 4),
                 "rf_time_s": -1,
                 "error": msg}]
        # Write the error row to the progress CSV so the image is not
        # retried endlessly - operator can delete the file to force a retry.
        t_csv = time.perf_counter()
        write_progress_csv(rows, core_name)
        _tick("write_progress_csv", time.perf_counter() - t_csv)
        return rows

    # 1. Parse XML footprint
    if not xml_path or not os.path.isfile(xml_path):
        return err("XML metadata file not found")
    try:
        with _timed("xml_parse"):
            geo_info  = extract_geospatial_info_from_xml(xml_path)
            footprint = geo_info["geometry"]
    except Exception as exc:
        return err(f"XML parse failed: {exc}")

    # 2. Clip ALPOD lakes to image footprint
    alpod_shp = alpod_shapefiles.get(dataset)
    if not alpod_shp:
        return err(f"No ALPOD shapefile registered for dataset {dataset}")

    clip_out = os.path.join(tmp_dir, f"clip_r{RANK}_{core_name}.shp")
    try:
        with _timed("clip_vector"):
            n_lakes = clip_vector_with_geometry(alpod_shp, footprint, clip_out)
    except Exception as exc:
        return err(f"clip_vector_with_geometry failed: {exc}")

    if n_lakes == 0:
        elapsed = time.perf_counter() - t0
        print(
            f"[rank {RANK}] DONE  {rec['filename']}  "
            f"no lakes entirely within footprint  elapsed={elapsed:.2f}s",
            flush=True,
        )
        _cleanup_shp(clip_out)
        # Write an empty progress CSV (just the header) to mark this image done.
        t_csv = time.perf_counter()
        write_progress_csv([], core_name)
        _tick("write_progress_csv", time.perf_counter() - t_csv)
        return []

    # 3. Read clipped lakes into memory
    try:
        with _timed("read_clipped_shp"):
            lakes_gdf = gpd.read_file(clip_out)
    except Exception as exc:
        _cleanup_shp(clip_out)
        return err(f"Could not read clipped shapefile: {exc}")
    _cleanup_shp(clip_out)

    # 4. Read raster bands into main process RAM
    try:
        with _timed("raster_read"):
            with rasterio.open(path) as src:
                nodata    = src.nodata
                transform = src.transform
                crs       = src.crs
                data      = src.read()   # shape: (n_bands, rows, cols)
    except Exception as exc:
        return err(f"Raster read failed: {exc}")

    read_time = time.perf_counter() - t0

    # 5. Copy band stack into POSIX shared memory
    raster_shape = data.shape
    raster_dtype = str(data.dtype)
    shm_name     = f"rf_rank{RANK}_{core_name}"

    try:
        with _timed("shm_alloc"):
            image_shm  = shm_mod.SharedMemory(
                name=shm_name, create=True, size=data.nbytes
            )
            shared_arr = np.ndarray(data.shape, dtype=data.dtype, buffer=image_shm.buf)
            shared_arr[:] = data
            del data, shared_arr     # free main-process copy
    except Exception as exc:
        return err(f"Shared memory allocation failed: {exc}")

    transform_tuple = (
        transform.a, transform.b, transform.c,
        transform.d, transform.e, transform.f,
    )

    # 6. Reproject lakes to raster CRS and build per-lake job list
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
             nodata,
             dataset)
            for _, lake_row in lakes_proj.iterrows()
        ]

    n_workers_actual = pool._processes
    chunksize        = max(1, len(jobs) // max(1, n_workers_actual))

    print(
        f"[rank {RANK}]   distributing {n_lakes} lakes to {n_workers_actual} workers "
        f"(chunksize={chunksize}  read={read_time:.2f}s)",
        flush=True,
    )

    # 7. Dispatch lakes to worker pool
    t_rf = time.perf_counter()
    with _timed("pool_map"):
        raw_results = pool.map(_rf_worker, jobs, chunksize=chunksize)
    rf_time = time.perf_counter() - t_rf

    # Unpack and fold worker-side timings into this rank's TIMINGS
    results = []
    for item in raw_results:
        lake_id, ice, water = item[0], item[1], item[2]
        wtimings = item[3] if len(item) > 3 else {}
        for k, v in wtimings.items():
            _tick(k, v)
        results.append((lake_id, ice, water))

    # 8. Release shared memory
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
        f"[rank {RANK}] DONE  {dataset}/{rec['year_folder']}/{rec['filename']}  "
        f"{n_lakes} lakes classified  "
        f"read={read_time:.2f}s  rf={rf_time:.2f}s  elapsed={elapsed_total:.2f}s",
        flush=True,
    )

    rows = [
        {**base_row,
         "lake_id":          lake_id,
         "ice_pixels":       ice,
         "water_pixels":     water,
         "n_lakes_in_image": n_lakes,
         "n_valid_pixels":   (ice + water) if (ice >= 0 and water >= 0) else -1,
         "read_time_s":      round(read_time, 4),
         "rf_time_s":        round(rf_time,   4)}
        for (lake_id, ice, water) in results
    ]

    # 9. Write per-image progress CSV immediately
    # This is the crash-safe record.  If the job is killed before the final
    # combine step, all rows up to this point are already on disk.
    t_csv = time.perf_counter()
    csv_path = write_progress_csv(rows, core_name)
    _tick("write_progress_csv", time.perf_counter() - t_csv)
    print(
        f"[rank {RANK}] PROGRESS  {core_name}  -> {csv_path}",
        flush=True,
    )

    return rows


# -----------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------

def main():
    if RANK == 0:
        parser = argparse.ArgumentParser(
            description="RF benchmark - MPI + multiprocessing"
        )
        parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS,
                            help=f"Datasets to process (default: {ALL_DATASETS})")
        parser.add_argument("--nimages", type=int, default=None,
                            help="Images to sample per study site (default: all)")
        parser.add_argument("--workers", type=int,
                            default=max(1, mp.cpu_count() - 1),
                            help="Pool workers per rank (default: cpu_count - 1)")
        args = parser.parse_args()
        cfg  = {
            "datasets": [d for d in args.datasets if d in ALL_DATASETS],
            "nimages":  args.nimages,
            "workers":  args.workers,
        }
    else:
        cfg = None

    if HAS_MPI:
        cfg = COMM.bcast(cfg, root=0)

    datasets  = cfg["datasets"]
    n_sample  = cfg["nimages"]
    n_workers = cfg["workers"]

    # Locate ALPOD shapefiles
    alpod_shapefiles = {}
    for ds in ALL_DATASETS:
        try:
            alpod_shapefiles[ds] = find_alpod_shapefile(ds)
            if RANK == 0:
                print(f"[rank 0] ALPOD  {ds:12s}  ->  {alpod_shapefiles[ds]}",
                      flush=True)
        except FileNotFoundError as exc:
            if RANK == 0:
                print(f"[rank 0] WARNING: {exc}", flush=True)

    tmp_dir = tempfile.mkdtemp(prefix=f"rf_rank{RANK}_")

    if RANK == 0:
        global_start = time.time()
        n_label = str(n_sample) if n_sample is not None else "ALL"
        print(f"\n{'='*72}", flush=True)
        print(f"  rf_benchmark.py  |  {SIZE} MPI ranks  |  {n_workers} workers/rank",
              flush=True)
        print(f"  Datasets  : {datasets}", flush=True)
        print(f"  Images    : {n_label} per site", flush=True)
        print(f"  Progress  : {PROGRESS_DIR}", flush=True)
        print(f"{'='*72}\n", flush=True)

        records     = discover_files(datasets, n_sample)
        n_remaining = len(records)
        print(
            f"[rank 0] {n_remaining} image(s) remaining to process "
            f"(completed images already skipped)",
            flush=True,
        )
        if not records:
            print(
                "[rank 0] All images already complete.  "
                "Combining progress CSVs into final output...",
                flush=True,
            )
            timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(OUTPUT_DIR, f"rf_benchmark_{timestamp}.csv")
            t_csv = time.perf_counter()
            combine_progress_csvs(output_path)
            _tick("write_csv", time.perf_counter() - t_csv)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            (COMM.Abort(0) if HAS_MPI else sys.exit(0))
    else:
        records      = None
        global_start = None

    local_records = scatter_work(records)

    # Load RF models before forking the pool so workers inherit via copy-on-write
    load_rf_models(datasets)

    ctx        = mp.get_context("fork")
    local_rows = []
    with ctx.Pool(processes=n_workers) as pool:
        for rec in local_records:
            local_rows.extend(
                process_image(rec, alpod_shapefiles, pool, tmp_dir)
            )

    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Gather timings from all ranks (rows already written to disk per-image)
    if HAS_MPI:
        all_timings_nested = COMM.gather(dict(TIMINGS), root=0)
    else:
        all_timings_nested = [dict(TIMINGS)]

    if RANK == 0:
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"rf_benchmark_{timestamp}.csv")

        # Combine ALL progress CSVs (including ones from previous runs)
        t_csv = time.perf_counter()
        combine_progress_csvs(output_path)
        _tick("write_csv", time.perf_counter() - t_csv)

        total_elapsed = time.time() - global_start

        # Sum timings across ranks
        combined: defaultdict = defaultdict(float)
        for rank_timings in all_timings_nested:
            for k, v in rank_timings.items():
                combined[k] += v

        print(f"\n{'='*72}", flush=True)
        print(f"  Total wall time : {total_elapsed:.2f}s  "
              f"({total_elapsed/60:.1f} min)", flush=True)
        print(f"  Output          : {output_path}", flush=True)
        print(f"{'='*72}\n", flush=True)

        _print_timing_report(combined, total_elapsed)


if __name__ == "__main__":
    mp.freeze_support()
    main()