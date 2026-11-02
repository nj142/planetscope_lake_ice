"""
Cloud Cover Classifier for Planet Scenes (Freezeup folders)
============================================================
Keys during classification:
    X = Not Cloudy   C = Cloudy   V = Very Cloudy   B = Back   Q = Quit

HOW IT WORKS
------------
1. Run the script. A background thread immediately starts converting EVERY
   Freezeup TIF across all three study sites into JPEGs saved in Temp_JPEGs/.
   Progress prints to the terminal every 5 seconds.

2. You can leave the script running between sessions -- it keeps converting.

3. Press ENTER in the terminal when ready to label. Already-converted JPEGs
   load instantly; any not yet done are converted on the fly so you never wait.

4. One CSV per study site, updated after every keypress. Resumable.

5. When a site is fully classified, its temp JPEGs are deleted automatically.

Dependencies:  pip install opencv-python rasterio numpy pandas
"""

import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio
import rasterio.enums

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(r"E:\planetscope_lake_ice\Data\Input\Study Sites - Manual ALPOD Data")

STUDY_SITES    = ["YKD", "NS", "YF"]
TEMP_JPEG_DIR  = BASE_DIR / "Temp_JPEGs"
OUTPUT_DIR     = BASE_DIR

JPEG_QUALITY   = 80
MAX_W, MAX_H   = 1200, 900
PREFETCH_AHEAD = 8

LABEL_MAP = {
    ord("x"): "Not Cloudy",
    ord("c"): "Cloudy",
    ord("v"): "Very Cloudy",
}
LABEL_COLORS = {
    "Not Cloudy":  (100, 240, 100),
    "Cloudy":      (80,  200, 255),
    "Very Cloudy": (80,   80, 255),
}

# =============================================================================
# FILE DISCOVERY
# =============================================================================

def find_freezeup_tifs(site_dir: Path) -> list:
    results = []
    for tif in sorted(site_dir.rglob("*.tif")):
        if "udm" in tif.name.lower():
            continue
        if any("freezeup" in part.lower() for part in tif.parts):
            results.append(tif)
    return results


def jpeg_cache_path(tif: Path) -> Path:
    return TEMP_JPEG_DIR / f"{tif.parent.name}__{tif.stem}.jpg"


# =============================================================================
# IMAGE CONVERSION
# =============================================================================

def tif_to_jpeg(tif: Path, out: Path) -> bool:
    try:
        with rasterio.open(tif) as src:
            h, w = src.height, src.width
            scale = min(MAX_W / w, MAX_H / h, 1.0)
            ow, oh = max(1, int(w * scale)), max(1, int(h * scale))
            data = src.read(
                indexes=[3, 2, 1],
                out_shape=(3, oh, ow),
                resampling=rasterio.enums.Resampling.average,
            )
        rgb = np.moveaxis(data, 0, -1).astype(np.float32)
        for i in range(3):
            ch = rgb[..., i]
            valid = ch[ch > 0]
            if valid.size:
                p2, p98 = np.percentile(valid, (2, 98))
                if p98 > p2:
                    rgb[..., i] = np.clip((ch - p2) / (p98 - p2), 0, 1)
        bgr = cv2.cvtColor((rgb * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out), bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return True
    except Exception as exc:
        print(f"    [JPEG error] {tif.name}: {exc}")
        return False


# =============================================================================
# GLOBAL BACKGROUND PRE-CONVERTER  <-- NEW
# Starts on launch, converts every TIF across all sites, skips already-done.
# =============================================================================

class GlobalConverter:
    def __init__(self, all_tifs: list):
        self.total   = len(all_tifs)
        self.done    = 0      # newly converted this run
        self.skipped = 0      # already existed on disk
        self.errors  = 0
        self.finished = False
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._tifs   = all_tifs
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="global-converter"
        )
        self._thread.start()

    def _run(self):
        TEMP_JPEG_DIR.mkdir(parents=True, exist_ok=True)
        for tif in self._tifs:
            if self._stop.is_set():
                break
            jp = jpeg_cache_path(tif)
            if jp.exists():
                with self._lock:
                    self.skipped += 1
            else:
                ok = tif_to_jpeg(tif, jp)
                with self._lock:
                    if ok:
                        self.done += 1
                    else:
                        self.errors += 1
        self.finished = True

    def progress(self):
        with self._lock:
            return self.done, self.skipped, self.errors, self.total

    def stop(self):
        self._stop.set()


def progress_monitor(converter: GlobalConverter):
    """Prints a live line every 5 s until conversion finishes."""
    while not converter.finished:
        done, skipped, errors, total = converter.progress()
        ready = done + skipped
        pct = 100.0 * ready / total if total else 100.0
        print(
            f"\r  [converting]  {ready}/{total} ({pct:.1f}%)"
            f"  new={done}  cached={skipped}  errors={errors}   ",
            end="", flush=True,
        )
        time.sleep(5)
    done, skipped, errors, total = converter.progress()
    print(
        f"\r  [converting]  DONE  {done + skipped}/{total} ready"
        f"  (new={done}  cached={skipped}  errors={errors})          "
    )


# =============================================================================
# SESSION PREFETCHER  (used during labeling -- just reads existing JPEGs fast)
# Falls back to inline conversion if global converter hasn't reached a file yet.
# =============================================================================

class SessionPrefetcher:
    _SENTINEL = object()

    def __init__(self, tif_list: list):
        self._q      = queue.Queue(maxsize=PREFETCH_AHEAD)
        self._stop   = threading.Event()
        self._tifs   = tif_list
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="session-prefetch"
        )
        self._thread.start()

    def _run(self):
        for tif in self._tifs:
            if self._stop.is_set():
                break
            jp = jpeg_cache_path(tif)
            if not jp.exists():
                if not tif_to_jpeg(tif, jp):
                    continue
            self._q.put((tif, jp))
        self._q.put(self._SENTINEL)

    def get(self, timeout: float = 120):
        try:
            item = self._q.get(timeout=timeout)
            return None if item is self._SENTINEL else item
        except queue.Empty:
            return None

    def stop(self):
        self._stop.set()


# =============================================================================
# CSV HELPERS
# =============================================================================

CSV_COLUMNS = ["Study_Site", "Season_Folder", "Filename", "Filepath", "Classification"]


def load_csv(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame(columns=CSV_COLUMNS)


def save_csv(csv_path: Path, df: pd.DataFrame):
    df.to_csv(csv_path, index=False)


def done_filepaths(df: pd.DataFrame) -> set:
    return set(df["Filepath"].astype(str))


# =============================================================================
# DISPLAY
# =============================================================================

HEADER_H = 70


def build_frame(img, tif: Path, site: str,
                display_idx: int, grand_total: int, current_label) -> np.ndarray:
    header = np.full((HEADER_H, img.shape[1], 3), 28, dtype=np.uint8)
    r1 = f"[{display_idx}/{grand_total}]  {site}  /  {tif.parent.name}  /  {tif.name}"
    cv2.putText(header, r1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (210, 210, 210), 1, cv2.LINE_AA)
    guide = [
        ("X=Not Cloudy",    LABEL_COLORS["Not Cloudy"]),
        ("  C=Cloudy",      LABEL_COLORS["Cloudy"]),
        ("  V=Very Cloudy", LABEL_COLORS["Very Cloudy"]),
        ("  B=Back",        (160, 160, 255)),
        ("  Q=Quit",        (130, 130, 130)),
    ]
    x = 10
    for text, color in guide:
        cv2.putText(header, text, (x, 52), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, color, 1, cv2.LINE_AA)
        x += cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0][0]
    if current_label:
        bc = LABEL_COLORS.get(current_label, (180, 180, 180))
        bt = f" {current_label} "
        (bw, bh), _ = cv2.getTextSize(bt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        bx = img.shape[1] - bw - 14
        cv2.rectangle(header, (bx - 4, 26), (bx + bw + 4, 32 + bh), bc, -1)
        cv2.putText(header, bt, (bx, 30 + bh), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (15, 15, 15), 2, cv2.LINE_AA)
    return np.vstack([header, img])


# =============================================================================
# PER-SITE LABELING SESSION
# =============================================================================

def classify_site(site_name: str, site_dir: Path):
    csv_path    = OUTPUT_DIR / f"{site_name}_cloud_classifications.csv"
    df          = load_csv(csv_path)
    done_set    = done_filepaths(df)
    all_tifs    = find_freezeup_tifs(site_dir)
    grand_total = len(all_tifs)

    if not all_tifs:
        print(f"  No Freezeup TIF files found for {site_name} -- skipping.")
        return

    remaining = [t for t in all_tifs if str(t) not in done_set]

    print(f"\n  {'='*62}")
    print(f"  Site: {site_name}   Total: {grand_total}   "
          f"Done: {len(done_set)}   Remaining: {len(remaining)}")
    print(f"  CSV:  {csv_path}")
    print(f"  {'='*62}")

    if not remaining:
        print("  All images already classified -- nothing to do.")
        return

    cv2.namedWindow("Cloud Classifier", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Cloud Classifier", 1260, 1010)

    prefetcher = SessionPrefetcher(remaining)
    shown, labels, cursor = [], [], 0
    quit_requested = False

    while True:
        if cursor >= len(shown):
            item = prefetcher.get(timeout=120)
            if item is None:
                print("  All remaining images processed!")
                break
            shown.append(item)
            labels.append(None)

        tif, jpeg_path = shown[cursor]

        if not jpeg_path.exists():
            print(f"  Missing JPEG for {tif.name} -- skipping.")
            cursor += 1
            continue
        img = cv2.imread(str(jpeg_path))
        if img is None:
            print(f"  Could not read JPEG for {tif.name} -- skipping.")
            cursor += 1
            continue

        n_session_done = sum(1 for i, lbl in enumerate(labels)
                             if lbl is not None and i < cursor)
        display_idx = len(done_set) + n_session_done + 1

        cv2.imshow("Cloud Classifier",
                   build_frame(img, tif, site_name, display_idx, grand_total, labels[cursor]))
        key = cv2.waitKey(0) & 0xFF

        if key in [ord("q"), 27]:
            print("  Quit -- progress saved.")
            quit_requested = True
            break

        elif key == ord("b"):
            if labels[cursor] is not None:
                labels[cursor] = None
                df = df[df["Filepath"] != str(tif)]
                save_csv(csv_path, df)
            if cursor > 0:
                cursor -= 1
                prev_tif = shown[cursor][0]
                if labels[cursor] is not None:
                    labels[cursor] = None
                    df = df[df["Filepath"] != str(prev_tif)]
                    save_csv(csv_path, df)
            else:
                print("  Already at the first image of this session.")

        elif key in LABEL_MAP:
            label = LABEL_MAP[key]
            labels[cursor] = label
            df = df[df["Filepath"] != str(tif)]
            df = pd.concat([df, pd.DataFrame([{
                "Study_Site":     site_name,
                "Season_Folder":  tif.parent.name,
                "Filename":       tif.name,
                "Filepath":       str(tif),
                "Classification": label,
            }])], ignore_index=True)
            save_csv(csv_path, df)
            total_done = len(done_set) + sum(1 for lbl in labels if lbl is not None)
            print(f"  [{total_done}/{grand_total}]  {label:<12}  {tif.name}")
            cursor += 1

        else:
            print("  Unknown key -- use X / C / V / B / Q")

    prefetcher.stop()
    cv2.destroyAllWindows()

    # Only delete JPEGs once the site is FULLY classified
    remaining_after = [t for t in all_tifs if str(t) not in done_filepaths(df)]
    if not remaining_after:
        print(f"\n  {site_name} fully classified -- deleting its temp JPEGs...")
        for tif in all_tifs:
            jpeg_cache_path(tif).unlink(missing_ok=True)
    else:
        print(f"\n  {len(remaining_after)} images still unclassified -- "
              f"keeping temp JPEGs for {site_name}.")

    print(f"  {len(df)} total classified for {site_name}.  CSV: {csv_path}\n")
    if quit_requested:
        return "quit"


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 66)
    print("  Planet Cloud Cover Classifier")
    print(f"  Base dir : {BASE_DIR}")
    print(f"  JPEG temp: {TEMP_JPEG_DIR}")
    print("=" * 66)

    # Discover all TIFs across all sites upfront
    all_tifs_global = []
    site_dirs = {}
    for site_name in STUDY_SITES:
        sd = BASE_DIR / f"{site_name} 50x50 km"
        if not sd.exists():
            print(f"  WARNING: {sd} not found -- skipping {site_name}.")
            continue
        site_dirs[site_name] = sd
        tifs = find_freezeup_tifs(sd)
        all_tifs_global.extend(tifs)
        print(f"  {site_name}: {len(tifs)} Freezeup TIFs found.")

    if not all_tifs_global:
        print("  No TIF files found. Check BASE_DIR. Exiting.")
        return

    already_done = sum(1 for t in all_tifs_global if jpeg_cache_path(t).exists())
    print(f"\n  Total TIFs : {len(all_tifs_global)}")
    print(f"  JPEGs ready: {already_done}")
    print(f"  To convert : {len(all_tifs_global) - already_done}")

    # --- Start global converter immediately on launch ---
    converter = GlobalConverter(all_tifs_global)
    threading.Thread(
        target=progress_monitor, args=(converter,), daemon=True, name="progress-monitor"
    ).start()

    print("\n  Background JPEG conversion started.")
    print("  Leave this running to pre-build all JPEGs, or press ENTER to")
    print("  start classifying now (images load on the fly if not yet ready).\n")

    input("  >>> Press ENTER to begin classification: ")
    print()

    for site_name, site_dir in site_dirs.items():
        result = classify_site(site_name, site_dir)
        if result == "quit":
            print("  Stopped early. Re-run to resume from where you left off.")
            break

    converter.stop()

    try:
        if TEMP_JPEG_DIR.exists() and not any(TEMP_JPEG_DIR.iterdir()):
            TEMP_JPEG_DIR.rmdir()
            print("  Temp folder removed (empty).")
    except Exception:
        pass

    print("\nAll done.")


if __name__ == "__main__":
    main()