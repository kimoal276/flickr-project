from __future__ import annotations
from src.trainer.geo_utils import haversine_km
import argparse
import csv
import os
import sys
from pathlib import Path
from simple_image_cache import SimpleImageCache
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
_TRAINER_DIR = PROJECT_ROOT / "src" / "trainer"
if _TRAINER_DIR.exists():
    sys.path.insert(0, str(_TRAINER_DIR))

load_dotenv()

find_matches = None
_import_err = None
for _modpath in ("src.trainer.matching", "matching", "geo_mapillary"):
    try:
        _mod = __import__(_modpath, fromlist=["find_matches"])
        find_matches = getattr(_mod, "find_matches")
        break
    except Exception as exc:  
        _import_err = exc
if find_matches is None:
    raise ImportError(
        "Could not import find_matches from matching.py. Last error:\n"
        f"  {type(_import_err).__name__}: {_import_err}\n"
        "Adjust the import paths above to point at your matching module."
    )

# --------------------------------------------------------------------------
# Output schema. find_matches writes these columns into the row dataframe:
#   mapillary_candidates, mapillary_id, mapillary_lon, mapillary_lat,
#   p_match, mapillary_compass_angle, mapillary_captured_at, mapillary_pic_url
# (only mapillary_candidates is guaranteed; the rest appear when a match exists)
# --------------------------------------------------------------------------
OUT_FIELDS = [
    "id", "title", "source_dataset",
    "flickr_lat", "flickr_lon", "p_building",
    "mapillary_candidates",
    "mapillary_id", "mapillary_lat", "mapillary_lon",
    "p_match", "mapillary_compass_angle", "mapillary_captured_at",
    "mapillary_pic_url",
    "distance_km", "status",
]


def _result_record(src_row: pd.Series, matched_row: pd.Series, status: str) -> dict:
    """Flatten one matched row into the flat OUT_FIELDS record."""
    rec = {k: None for k in OUT_FIELDS}
    rec["id"] = src_row.get("id")
    rec["title"] = src_row.get("title")
    rec["source_dataset"] = src_row.get("source_dataset")
    rec["flickr_lat"] = src_row.get("latitude")
    rec["flickr_lon"] = src_row.get("longitude")
    rec["p_building"] = src_row.get("p_building")
    rec["status"] = status

    if matched_row is not None:
        rec["mapillary_candidates"] = matched_row.get("mapillary_candidates")
        m_id = matched_row.get("mapillary_id")
        if m_id is not None and not pd.isna(m_id):
            rec["mapillary_id"] = m_id
            rec["mapillary_lat"] = matched_row.get("mapillary_lat")
            rec["mapillary_lon"] = matched_row.get("mapillary_lon")
            rec["p_match"] = matched_row.get("p_match")
            rec["mapillary_compass_angle"] = matched_row.get("mapillary_compass_angle")
            rec["mapillary_captured_at"] = matched_row.get("mapillary_captured_at")
            rec["mapillary_pic_url"] = matched_row.get("mapillary_pic_url")
            try:
                rec["distance_km"] = haversine_km(
                    float(src_row["latitude"]), float(src_row["longitude"]),
                    float(matched_row["mapillary_lat"]), float(matched_row["mapillary_lon"]),
                )
            except Exception:  
                rec["distance_km"] = None
    return rec


def load_done_ids(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    try:
        # on_bad_lines='skip' tolerates a half-written final row from a hard
        # power cut, so resume still recognises every fully-written id.
        prev = pd.read_csv(out_path, usecols=["id"], on_bad_lines="skip")
        return set(prev["id"].dropna().astype(str))
    except Exception:  
        return set()


def main() -> None:
    ap = argparse.ArgumentParser(description="Match Flickr CSV rows to Mapillary imagery.")
    ap.add_argument("--input", default="flickr_clusters.csv")
    ap.add_argument("--output", default="mapillary_matches.csv")
    ap.add_argument("--cache-dir", default="image_cache")
    ap.add_argument("--limit", type=int, default=None, help="process at most N rows (debug)")
    ap.add_argument("--precache", action="store_true",
                    help="download all Flickr query images to disk first, then match")
    args = ap.parse_args()

    if not os.getenv("MAPILLARY_ACCESS_TOKEN"):
        sys.exit("ERROR: MAPILLARY_ACCESS_TOKEN is not set. Put it in a .env file "
                 "at the project root (MAPILLARY_ACCESS_TOKEN=MLY|...).")

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        sys.exit(f"ERROR: input CSV not found: {in_path.resolve()}")

    # load + adapt the CSV to what find_matches expects
    df = pd.read_csv(in_path)
    df = df.rename(columns={"image_url": "url_o"})        # the column find_matches reads
    df = df.dropna(subset=["url_o", "latitude", "longitude"])
    df = df[(df["latitude"] != 0) & (df["longitude"] != 0)]
    # mirror fast_mapillary's ordering (p_building_given_descr -> p_building)
    sort_key = "p_building" if "p_building" in df.columns else None
    if sort_key:
        df = df.sort_values(sort_key, ascending=False)
    df = df.reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)

    cache = SimpleImageCache(args.cache_dir)

    if args.precache:
        print(f"Pre-caching {df['url_o'].nunique()} Flickr images to disk ...")
        n_ok = cache.precache(df["url_o"].tolist(), disk_save=True)
        print(f"  cached {n_ok} images on disk")

    done_ids = load_done_ids(out_path)
    if done_ids:
        print(f"Resuming: {len(done_ids)} ids already in {out_path.name}, skipping them.")

    write_header = not out_path.exists()
    fh = out_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=OUT_FIELDS)
    if write_header:
        writer.writeheader()

    n_matched = n_no_match = n_error = 0
    try:
        for i in tqdm(range(len(df)), desc="Matching images"):
            src_row = df.iloc[i]
            pid = str(src_row["id"])
            if pid in done_ids:
                continue

            one = df.iloc[[i]].copy()        # single-row df
            try:
                matched = find_matches(one, cache)
                matched_row = matched.iloc[0]
                has_match = (
                    "mapillary_id" in matched.columns
                    and not pd.isna(matched_row.get("mapillary_id"))
                )
                status = "ok" if has_match else "no_match"
                rec = _result_record(src_row, matched_row, status)
                if has_match:
                    n_matched += 1
                else:
                    n_no_match += 1
            except Exception as exc:          
                rec = _result_record(src_row, None, f"error:{type(exc).__name__}:{exc}")
                n_error += 1

            writer.writerow(rec)
            fh.flush()                        # push Python buffer to the OS
            os.fsync(fh.fileno())             # force OS to write to physical disk (survives power loss)
            done_ids.add(pid)
    finally:
        fh.close()

    print("\n=============================================")
    print(f"matched : {n_matched}")
    print(f"no_match: {n_no_match}")
    print(f"errors  : {n_error}")
    print(f"results -> {out_path.resolve()}")


if __name__ == "__main__":
    main()