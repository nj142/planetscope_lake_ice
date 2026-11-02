"""
rf_benchmark.py — MPI + multiprocessing Random Forest classification benchmark

Parallelism:
  Outer (MPI)               images distributed across nodes/ranks
  Inner (multiprocessing)   per-lake RF jobs within each rank

Per-image workflow:
  1. Locate the co-sited XML metadata file and parse the Planet footprint polygon
  2. Clip the study site's ALPOD lake shapefile to that footprint
     (ST_Within via ogr2ogr — lakes entirely inside the valid image area only)
  3. Read the clipped lakes into memory and clean up the temp shapefile
  4. Place the full raster into POSIX shared memory (one copy per rank).
     Workers receive only the lake geometry and a handle to that shared block —
     no pixel arrays are pickled through pipes.
  5. Each worker reconstructs a numpy view of the shared raster, burns its lake
     polygon to a boolean mask, extracts the pixel slice, and runs RF.predict().
  6. The shared memory block is released after pool.map() returns for that image.
  7. Rank 0 gathers all results and writes one CSV.

Memory layout per rank:
  Main process RAM  : full raster array written into a POSIX shared memory block
  Shared memory     : one block per image, named "rf_rank<R>_<core_name>"
                      lives in /dev/shm (tmpfs), counted against the node's RAM
                      budget, NOT on disk. All worker processes on the same rank
                      map the same physical pages — no copies until a worker
                      writes (copy-on-write), and workers only read from it.
  Worker process RAM: boolean mask (rows*cols bytes) + pixel slice
                      (bands * n_lake_pixels * dtype bytes) — both local,
                      freed when the worker function returns.

Timing:
  Each major stage is timed with time.perf_counter().  Timings are accumulated
  in the module-level TIMINGS dict (a defaultdict(float)) on every rank.
  Worker-side timings (mask burn, pixel extract, RF classify, shm attach) are
  returned alongside the classification result and folded into TIMINGS by the
  main process.  At script end, rank 0 gathers TIMINGS from all ranks, sums
  them, and prints a sorted wall-time breakdown table.

Usage:
    mpirun -n 8 python rf_benchmark.py
    mpirun -n 8 python rf_benchmark.py --datasets NS_50x50 YF_50x50 --nimages 10 --workers 30

Arguments:
    --datasets  subset of [NS_50x50, YF_50x50, YKD_50x50]  (default: all three)
    --nimages   images to sample per study site (default: ALL images)
    --workers   Pool workers per rank            (default: cpu_count - 1)

Output:
    /hpc/home/nj142/Output/rf_benchmark_<TIMESTAMP>.csv
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
import geopandas as gpd
from shapely.geometry import Polygon, mapping, shape
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
    print("WARNING: mpi4py not found — running serial (rank 0 of 1)")
    COMM    = None
    RANK    = 0
    SIZE    = 1
    HAS_MPI = False

# ---------------------------------------------------------------------------
# TIMING INFRASTRUCTURE
# ---------------------------------------------------------------------------
# Module-level accumulator — keyed by stage name, value is cumulative wall
# seconds spent in that stage on this rank.  Workers that run inside
# pool.map() cannot write here directly (fork gives them a private copy);
# they return their timings in the result tuple and the main process folds
# them in after pool.map() returns.
TIMINGS: defaultdict = defaultdict(float)


def _tick(key: str, elapsed: float) -> None:
    """Add *elapsed* seconds to the named timing bucket."""
    TIMINGS[key] += elapsed


def _timed(key: str):
    """
    Context manager — records wall time of the enclosed block into TIMINGS.

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


# Human-readable labels for every timing key
_TIMING_LABELS = {
    # --- main-process stages (per image, accumulated over all images) ---
    "xml_parse":           "XML metadata parse",
    "clip_vector":         "ogr2ogr lake clip",
    "read_clipped_shp":    "Read clipped shapefile (gpd)",
    "raster_read":         "Raster read (rasterio)",
    "shm_alloc":           "Shared-memory alloc + copy",
    "geom_project":        "Lake CRS reproject",
    "build_jobs":          "Build worker job list",
    "pool_map":            "pool.map wall time (all lakes)",
    "shm_cleanup":         "Shared-memory cleanup",
    # --- worker-side stages (aggregated from all lakes on this rank) ---
    "worker_shm_attach":   "[worker] shm attach/detach",
    "worker_mask_burn":    "[worker] Polygon mask burn",
    "worker_pixel_extract":"[worker] Pixel extraction",
    "worker_rf_classify":  "[worker] RF classify_lake_rf",
    # --- one-shot stages ---
    "load_rf_models":      "Load RF models (joblib)",
    "write_csv":           "Write output CSV",
}


def _print_timing_report(all_timings: dict, total_wall: float) -> None:
    """
    Print a sorted wall-time breakdown table to stdout.

    *all_timings* is the sum across all MPI ranks (computed by rank 0).
    """
    # Sort descending by cumulative seconds
    sorted_items = sorted(all_timings.items(), key=lambda kv: kv[1], reverse=True)
    total_accounted = sum(v for _, v in sorted_items)

    col_w = 38
    print(f"\n{'='*68}", flush=True)
    print(f"  WALL-TIME BREAKDOWN  (summed across all {SIZE} rank(s))", flush=True)
    print(f"{'='*68}", flush=True)
    header = f"  {'Stage':<{col_w}} {'Seconds':>10}  {'% of sum':>9}"
    print(header, flush=True)
    print(f"  {'-'*col_w} {'-'*10}  {'-'*9}", flush=True)
    for key, secs in sorted_items:
        label   = _TIMING_LABELS.get(key, key)
        pct     = 100.0 * secs / total_accounted if total_accounted > 0 else 0.0
        print(f"  {label:<{col_w}} {secs:>10.2f}s  {pct:>8.1f}%", flush=True)
    print(f"  {'-'*col_w} {'-'*10}  {'-'*9}", flush=True)
    print(f"  {'Sum of timed stages':<{col_w}} {total_accounted:>10.2f}s", flush=True)
    print(f"  {'Total script wall time':<{col_w}} {total_wall:>10.2f}s", flush=True)
    print(f"{'='*68}\n", flush=True)


# ---------------------------------------------------------------------------
# RF MODEL LOADING
# ---------------------------------------------------------------------------
import joblib

RF_MODELS_DIR = "/work/nj142/RF_Models"
RF_MODELS: dict = {}


def load_rf_models(datasets: list):
    """
    Load one RF model per dataset into RF_MODELS before the Pool is forked.
    Workers inherit the model objects via copy-on-write — no re-loading needed.
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


# ---------------------------------------------------------------------------
# RF CLASSIFICATION
# ---------------------------------------------------------------------------

def classify_lake_rf(pixel_data: np.ndarray, nodata=None, dataset: str = "") -> tuple:
    """
    Run RF prediction on a (bands, n_lake_pixels) array.
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


# ---------------------------------------------------------------------------
# MULTIPROCESSING WORKER
# ---------------------------------------------------------------------------

def _rf_worker(args: tuple) -> tuple:
    """
    Worker function for a single lake.

    Returns
    -------
    (lake_id, ice, water, worker_timings)
        worker_timings is a dict mapping timing-key -> seconds for the stages
        executed inside this worker call. The main process folds these into its
        TIMINGS dict after pool.map() returns.

    Timing keys emitted:
      worker_shm_attach    — SharedMemory.name lookup + numpy view setup
      worker_mask_burn     — geometry_mask polygon rasterisation
      worker_pixel_extract — boolean-index pixel slice from shared array
      worker_rf_classify   — classify_lake_rf() call
    """
    (lake_id, geom_wkt, transform_tuple, shm_name,
     raster_shape, raster_dtype_str, nodata, dataset) = args

    wtimings: dict = defaultdict(float)

    try:
        # -- attach to shared memory ------------------------------------------
        t = time.perf_counter()
        existing_shm = shm_mod.SharedMemory(name=shm_name)
        try:
            resource_tracker.unregister(f"/{shm_name}", "shared_memory")
        except (KeyError, ValueError):
            pass
        dtype = np.dtype(raster_dtype_str)
        data  = np.ndarray(raster_shape, dtype=dtype, buffer=existing_shm.buf)
        wtimings["worker_shm_attach"] += time.perf_counter() - t

        n_bands, rows, cols = raster_shape

        from rasterio.transform import Affine
        transform = Affine(*transform_tuple)

        from shapely import wkt as shapely_wkt
        geom = shapely_wkt.loads(geom_wkt)

        # -- burn polygon mask ------------------------------------------------
        t = time.perf_counter()
        inside_mask = geometry_mask(
            [geom], transform=transform,
            invert=True,
            out_shape=(rows, cols)
        )
        wtimings["worker_mask_burn"] += time.perf_counter() - t

        # -- extract pixel slice ----------------------------------------------
        t = time.perf_counter()
        pixel_data = data[:, inside_mask]   # local copy (non-contiguous)
        wtimings["worker_pixel_extract"] += time.perf_counter() - t

        existing_shm.close()

        # -- RF classification ------------------------------------------------
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


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# ALPOD SHAPEFILE DISCOVERY
# ---------------------------------------------------------------------------

WORK_ROOT    = "/work/nj142"
ALPOD_ROOT   = "/work/nj142/ALPOD_Tiles"
OUTPUT_DIR   = "/hpc/home/nj142/Output"
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
    "n_lakes_in_image", "read_time_s", "rf_time_s",
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


# ---------------------------------------------------------------------------
# FILE DISCOVERY (rank 0 only)
# ---------------------------------------------------------------------------

def discover_files(datasets: list, n_per_site: int | None, seed: int = 42) -> list:
    rng     = random.Random(seed)
    records = []
    for ds in datasets:
        ds_root = os.path.join(WORK_ROOT, ds)
        if not os.path.isdir(ds_root):
            print(f"[rank 0] WARNING: {ds_root} not found — skipping", flush=True)
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

                core_name = os.path.splitext(filename)[0]
                xml_path  = xml_candidates[0] if xml_candidates else None

                all_candidates.append((tif, folder_name, year_str, filename, core_name, xml_path))

        if n_per_site is None:
            sample = all_candidates
            print(f"[rank 0] {ds}: using all {len(sample)} images", flush=True)
        else:
            sample = rng.sample(all_candidates, min(n_per_site, len(all_candidates)))
            print(f"[rank 0] {ds}: {len(all_candidates)} images available, "
                  f"sampling {len(sample)}", flush=True)

        for (tif, folder_name, year_str, filename, core_name, xml_path) in sample:
            records.append({
                "path":        tif,
                "xml_path":    xml_path,
                "dataset":     ds,
                "year_folder": folder_name,
                "year":        year_str,
                "filename":    filename,
                "core_name":   core_name,
            })
    return records


# ---------------------------------------------------------------------------
# WORK DISTRIBUTION
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# PER-IMAGE PROCESSING
# ---------------------------------------------------------------------------

def process_image(rec: dict, alpod_shapefiles: dict,
                  pool: mp.Pool, tmp_dir: str) -> list:
    """
    Full pipeline for one PlanetScope image.

    Each major stage is wrapped with _timed() so elapsed seconds accumulate
    in the module-level TIMINGS dict.  Worker-side timings are returned in the
    result tuple from pool.map() and folded in here.
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
        print(f"[rank {RANK}] ERROR {rec['filename']}: {msg}  "
              f"elapsed={elapsed:.2f}s", flush=True)
        return [{**base_row,
                 "lake_id": None, "ice_pixels": -1, "water_pixels": -1,
                 "n_lakes_in_image": 0,
                 "read_time_s": round(read_t or elapsed, 4),
                 "rf_time_s": -1,
                 "error": msg}]

    # 1. Parse XML footprint -----------------------------------------------
    if not xml_path or not os.path.isfile(xml_path):
        return err("XML metadata file not found")
    try:
        with _timed("xml_parse"):
            geo_info  = extract_geospatial_info_from_xml(xml_path)
            footprint = geo_info["geometry"]
    except Exception as exc:
        return err(f"XML parse failed: {exc}")

    # 2. Clip ALPOD lakes to footprint ------------------------------------
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
        print(f"[rank {RANK}] DONE  {rec['filename']}  "
              f"no lakes entirely within footprint  elapsed={elapsed:.2f}s",
              flush=True)
        _cleanup_shp(clip_out)
        return []

    # 3. Read clipped lakes into memory ------------------------------------
    try:
        with _timed("read_clipped_shp"):
            lakes_gdf = gpd.read_file(clip_out)
    except Exception as exc:
        _cleanup_shp(clip_out)
        return err(f"Could not read clipped shapefile: {exc}")
    _cleanup_shp(clip_out)

    # 4. Read raster into main process RAM --------------------------------
    try:
        with _timed("raster_read"):
            with rasterio.open(path) as src:
                nodata    = src.nodata
                transform = src.transform
                crs       = src.crs
                data      = src.read()   # (bands, rows, cols)
    except Exception as exc:
        return err(f"Raster read failed: {exc}")

    read_time           = time.perf_counter() - t0
    n_bands, rows, cols = data.shape

    # 5. Copy raster into POSIX shared memory -----------------------------
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
            del data, shared_arr
    except Exception as exc:
        return err(f"Shared memory allocation failed: {exc}")

    transform_tuple = (
        transform.a, transform.b, transform.c,
        transform.d, transform.e, transform.f,
    )

    # 6. Project lakes and build per-lake job list -------------------------
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

    # 7. Dispatch to pool — workers return (lake_id, ice, water, wtimings) --
    t_rf = time.perf_counter()
    with _timed("pool_map"):
        raw_results = pool.map(_rf_worker, jobs, chunksize=chunksize)
    rf_time = time.perf_counter() - t_rf

    # Unpack results and fold worker-side timings into this rank's TIMINGS
    results = []
    for item in raw_results:
        lake_id, ice, water = item[0], item[1], item[2]
        wtimings = item[3] if len(item) > 3 else {}
        for k, v in wtimings.items():
            _tick(k, v)
        results.append((lake_id, ice, water))

    # 8. Release shared memory --------------------------------------------
    with _timed("shm_cleanup"):
        try:
            image_shm.close()
            image_shm.unlink()
        except Exception as exc:
            print(f"[rank {RANK}] WARNING: shared memory cleanup failed: {exc}",
                  flush=True)

    elapsed_total = time.perf_counter() - t0

    print(
        f"[rank {RANK}] DONE  {dataset}/{rec['year_folder']}/{rec['filename']}  "
        f"{n_lakes} lakes classified  "
        f"read={read_time:.2f}s  rf={rf_time:.2f}s  elapsed={elapsed_total:.2f}s",
        flush=True,
    )

    return [
        {**base_row,
         "lake_id":          lake_id,
         "ice_pixels":       ice,
         "water_pixels":     water,
         "n_lakes_in_image": n_lakes,
         "read_time_s":      round(read_time, 4),
         "rf_time_s":        round(rf_time,   4)}
        for (lake_id, ice, water) in results
    ]


def _cleanup_shp(shp_path: str):
    base = os.path.splitext(shp_path)[0]
    for ext in (".shp", ".dbf", ".shx", ".prj", ".cpg", ".sbn", ".sbx", ".shp.xml"):
        try:
            os.remove(base + ext)
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# CSV WRITE (rank 0 only)
# ---------------------------------------------------------------------------

def write_csv(rows: list, output_path: str):
    with _timed("write_csv"):
        rows_sorted = sorted(
            rows,
            key=lambda r: (r.get("dataset",""), r.get("year_folder",""),
                           r.get("filename",""), r.get("lake_id") or 0)
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows_sorted)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    if RANK == 0:
        parser = argparse.ArgumentParser(description="RF benchmark — MPI + multiprocessing")
        parser.add_argument("--datasets",  nargs="+", default=ALL_DATASETS)
        parser.add_argument("--nimages",   type=int,  default=None,
                            help="Images to sample per study site (default: all)")
        parser.add_argument("--workers",   type=int,
                            default=max(1, mp.cpu_count() - 1),
                            help="Pool workers per rank (default: cpu_count - 1)")
        args = parser.parse_args()
        cfg  = {"datasets": [d for d in args.datasets if d in ALL_DATASETS],
                "nimages": args.nimages, "workers": args.workers}
    else:
        cfg = None

    if HAS_MPI:
        cfg = COMM.bcast(cfg, root=0)

    datasets  = cfg["datasets"]
    n_sample  = cfg["nimages"]
    n_workers = cfg["workers"]

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
        print(f"\n{'='*68}", flush=True)
        print(f"  rf_benchmark.py  |  {SIZE} MPI ranks  |  {n_workers} workers/rank",
              flush=True)
        print(f"  Datasets  : {datasets}", flush=True)
        print(f"  Images    : {n_label} per site", flush=True)
        print(f"{'='*68}\n", flush=True)

        records = discover_files(datasets, n_sample)
        print(f"[rank 0] {len(records)} images selected total", flush=True)
        if not records:
            print("[rank 0] Nothing to process — check --datasets and paths.")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            (COMM.Abort(0) if HAS_MPI else sys.exit(0))
    else:
        records      = None
        global_start = None

    local_records = scatter_work(records)

    # Load RF models before forking the pool so workers inherit them via
    # copy-on-write — no disk reads inside workers.
    load_rf_models(datasets)

    # Fork context lets workers inherit RF_MODELS without using pickle.
    ctx        = mp.get_context("fork")
    local_rows = []
    with ctx.Pool(processes=n_workers) as pool:
        for rec in local_records:
            local_rows.extend(
                process_image(rec, alpod_shapefiles, pool, tmp_dir)
            )

    shutil.rmtree(tmp_dir, ignore_errors=True)

    # -----------------------------------------------------------------------
    # Gather results AND timings from all ranks
    # -----------------------------------------------------------------------
    if HAS_MPI:
        all_rows_nested    = COMM.gather(local_rows, root=0)
        all_timings_nested = COMM.gather(dict(TIMINGS), root=0)
    else:
        all_rows_nested    = [local_rows]
        all_timings_nested = [dict(TIMINGS)]

    if RANK == 0:
        all_rows = [row for rank_rows in all_rows_nested for row in rank_rows]

        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"rf_benchmark_{timestamp}.csv")
        write_csv(all_rows, output_path)

        total_elapsed = time.time() - global_start

        # Sum timings across ranks (write_csv ran only on rank 0, already in TIMINGS)
        combined: defaultdict = defaultdict(float)
        for rank_timings in all_timings_nested:
            for k, v in rank_timings.items():
                combined[k] += v
        # write_csv is only in rank-0's TIMINGS, already included above

        print(f"\n{'='*68}", flush=True)
        print(f"  Rows written    : {len(all_rows)}", flush=True)
        print(f"  Total wall time : {total_elapsed:.2f}s  "
              f"({total_elapsed/60:.1f} min)", flush=True)
        print(f"  Output          : {output_path}", flush=True)
        print(f"{'='*68}\n", flush=True)

        _print_timing_report(combined, total_elapsed)


if __name__ == "__main__":
    mp.freeze_support()
    main()
