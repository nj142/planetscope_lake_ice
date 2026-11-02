import os
import re
import glob
import shutil
import random
import cv2
import rasterio
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

study_site = "NS"  # Change for other sites
src_base = (
    Path(r"E:\planetscope_lake_ice\Data\Input\Study Sites - Manual ALPOD Data")
    / f"{study_site} 50x50 km"
)

# pattern can be "Freezeup_" or "Breakup_" (or anything else)
pattern = "Freezeup_"

dst_base = (
    Path(r"E:\planetscope_lake_ice\Data\Input\Machine Learning - Planet Samples for RF")
    / study_site
)
dst_water = dst_base / "_WATER"
dst_ice = dst_base / "_ICE"

for d in [dst_water, dst_ice]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------


def count_scenes(folder):
    """Count distinct scene sets by counting analytic_4b_sr.tif files."""
    return len(list(Path(folder).glob("*_analytic_4b_sr.tif")))


def load_planet_rgb(sr_path, max_side=1200):
    """Load 4-band analytic Planet image and build scaled BGR image for OpenCV.
    Planet band order: B=1, G=2, R=3, NIR=4. OpenCV expects BGR, so read [1,2,3] directly."""
    with rasterio.open(sr_path) as src:
        h, w = src.height, src.width
        scale = min(max_side / h, max_side / w, 1.0)
        out_h, out_w = max(1, int(h * scale)), max(1, int(w * scale))

        data = src.read(
            indexes=[1, 2, 3],  # B, G, R → already BGR for OpenCV
            out_shape=(3, out_h, out_w),
            resampling=rasterio.enums.Resampling.average,
        )

    bgr = np.moveaxis(data, 0, -1).astype(np.float32)  # (h, w, 3)

    # Percentile contrast stretch per band
    for i in range(3):
        bnd = bgr[..., i]
        valid = bnd[bnd > 0]
        if valid.size > 0:
            p2, p98 = np.percentile(valid, (2, 98))
            if p98 > p2:
                bgr[..., i] = np.clip((bnd - p2) / (p98 - p2), 0, 1)

    return (bgr * 255).clip(0, 255).astype(np.uint8)


def find_companion_files(sr_path):
    """Return [analytic, udm2, xml] if all present, else None."""
    base = sr_path.replace("_analytic_4b_sr.tif", "")
    udm = base + "_udm2.tif"
    xml1 = base + "_analytic_4b_xml.xml"
    xml2 = base + ".xml"
    xml = xml1 if os.path.exists(xml1) else xml2 if os.path.exists(xml2) else None

    if all(os.path.exists(p) for p in [sr_path, udm]) and xml:
        return [sr_path, udm, xml]
    return None


# ---------------------------------------------------------------------
# MAIN LABELING LOOP
# ---------------------------------------------------------------------


def main():
    sr_files = list(Path(src_base).rglob("*_analytic_4b_sr.tif"))
    sr_files = [str(f) for f in sr_files if f.is_file() and pattern in f.as_posix()]

    if not sr_files:
        print(f"No {pattern} analytic_4b_sr.tif files found.")
        return

    random.shuffle(sr_files)
    print(f"Found {len(sr_files)} candidate {pattern} scenes, randomized order.")
    print("Press I = full ice, W = full water, S = skip, Q = quit")

    cv2.namedWindow("Planet Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Planet Preview", 1200, 960)

    for idx, sr_path in enumerate(sr_files, start=1):
        fileset = find_companion_files(sr_path)
        if not fileset:
            print(f"[{idx}] Missing companion files for {os.path.basename(sr_path)} → skipped.")
            continue

        try:
            img = load_planet_rgb(sr_path)
        except Exception as e:
            print(f"[{idx}] Error reading {sr_path}: {e}")
            continue

        n_ice = count_scenes(dst_ice)
        n_water = count_scenes(dst_water)

        display = cv2.resize(img, (min(img.shape[1], 1200), min(img.shape[0], 900)))

        # Header bar (two rows)
        header_h = 60
        header = np.full((header_h, display.shape[1], 3), 40, dtype=np.uint8)

        # Row 1 — file info + progress
        row1 = f"[{idx}/{len(sr_files)}]  {os.path.basename(sr_path)}"
        cv2.putText(header, row1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)

        # Row 2 — counters (coloured)
        counter_text_ice = f"ICE: {n_ice}"
        counter_text_water = f"WATER: {n_water}"
        keys_text = "I = ice    W = water    S = skip    Q = quit"

        cv2.putText(header, counter_text_ice,   (10,  50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 220, 255), 2)
        cv2.putText(header, counter_text_water, (130, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200,  80), 2)
        cv2.putText(header, keys_text,          (280, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        combined = np.vstack([header, display])
        cv2.imshow("Planet Preview", combined)

        key = cv2.waitKey(0) & 0xFF
        if key in [ord("q"), 27]:
            print("Quitting labeling.")
            break
        elif key == ord("s"):
            print(f"[{idx}] Skipped {os.path.basename(sr_path)}")
            continue
        elif key == ord("i"):
            dest, label = dst_ice, "ICE"
        elif key == ord("w"):
            dest, label = dst_water, "WATER"
        else:
            print("Invalid key — use I / W / S / Q")
            continue

        for f in fileset:
            shutil.copy2(f, dest)

        n_ice = count_scenes(dst_ice)
        n_water = count_scenes(dst_water)
        print(f"[{idx}] Copied → {label} ({len(fileset)} files)  |  ICE: {n_ice}  WATER: {n_water}")

    cv2.destroyAllWindows()
    print("Classification session finished.")


if __name__ == "__main__":
    main()