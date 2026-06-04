"""
visualize_matches.py
--------------------
Build side-by-side Flickr-vs-Mapillary comparison JPEGs for your matched rows,
reusing your existing geo_utils.save_comparison().

Why this wrapper is needed:
  * save_comparison(photo, top, out_dir) expects:
        photo = {image_url, source_dataset, date_taken, title, photo_id}
        top   = {thumb_url, pred_lat, pred_lon}
  * mapillary_matches.csv carries the match side, but NOT the Flickr image_url
    or date_taken -> so we join back to flickr_clusters.csv on `id`.
  * Mapillary thumbnail URLs (fbcdn, with an `oe=` expiry) go stale within
    hours/days. By default this re-fetches a FRESH thumb_1024_url for each
    match via the Graph API using MAPILLARY_ACCESS_TOKEN, so it works anytime.
    Use --no-refresh to use the stored URLs (only good right after a run).

Run from the project root:
    python visualize_matches.py                      # confident matches (p_match >= 0.05)
    python visualize_matches.py --min-pmatch 0.0     # include weak ones too
    python visualize_matches.py --out comparisons --no-refresh
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# make src.trainer / top-level modules importable from the project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
_TRAINER = PROJECT_ROOT / "src" / "trainer"
if _TRAINER.exists():
    sys.path.insert(0, str(_TRAINER))

load_dotenv()

# import save_comparison from your geo_utils (root or src/trainer)
save_comparison = None
for _modpath in ("src.trainer.geo_utils", "geo_utils"):
    try:
        _mod = __import__(_modpath, fromlist=["save_comparison"])
        save_comparison = getattr(_mod, "save_comparison")
        break
    except Exception:  
        pass
if save_comparison is None:
    sys.exit("Could not import save_comparison from geo_utils.py — check its location.")

GRAPH = "https://graph.mapillary.com"


def fresh_thumb_url(image_id: int, token: str) -> str | None:
    """Ask the Mapillary Graph API for a current (non-expired) thumb URL."""
    try:
        r = requests.get(
            f"{GRAPH}/{image_id}",
            params={"access_token": token, "fields": "thumb_1024_url"},
            timeout=20,
        )
        r.raise_for_status()
        return r.json().get("thumb_1024_url")
    except Exception:  
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Render Flickr vs Mapillary comparison images.")
    ap.add_argument("--matches", default="mapillary_matches.csv")
    ap.add_argument("--source", default="flickr_clusters.csv")
    ap.add_argument("--out", default="comparisons")
    ap.add_argument("--min-pmatch", type=float, default=0.05,
                    help="only render matches with p_match >= this (0.05 = your friend's keep threshold)")
    ap.add_argument("--no-refresh", action="store_true",
                    help="use stored thumb URLs instead of re-fetching fresh ones from the API")
    args = ap.parse_args()

    matches = pd.read_csv(args.matches)
    src = pd.read_csv(args.source)[["id", "image_url", "date_taken"]]
    df = matches.merge(src, on="id", how="left")

    df = df[
        (df["status"] == "ok")
        & (df["mapillary_id"].notna())
        & (df["p_match"] >= args.min_pmatch)
    ].sort_values("p_match", ascending=False)

    if df.empty:
        sys.exit(f"No matches with p_match >= {args.min_pmatch}. "
                 f"Try a lower --min-pmatch to include weaker matches.")

    token = os.getenv("MAPILLARY_ACCESS_TOKEN")
    if not args.no_refresh and not token:
        print("WARNING: no MAPILLARY_ACCESS_TOKEN; using stored URLs (may be expired).")
        args.no_refresh = True

    out_dir = Path(args.out)
    n_ok = 0
    for _, row in df.iterrows():
        thumb = row.get("mapillary_pic_url")
        if not args.no_refresh:
            fresh = fresh_thumb_url(int(row["mapillary_id"]), token)
            if fresh:
                thumb = fresh

        photo = {
            "image_url":      row.get("image_url"),
            "source_dataset": row.get("source_dataset"),
            "date_taken":     "" if pd.isna(row.get("date_taken")) else str(row.get("date_taken")),
            "title":          "" if pd.isna(row.get("title")) else str(row.get("title")),
            "photo_id":       row["id"],
        }
        top = {
            "thumb_url": thumb,
            "pred_lat":  row.get("mapillary_lat"),
            "pred_lon":  row.get("mapillary_lon"),
        }

        if not photo["image_url"] or not top["thumb_url"] or pd.isna(top["thumb_url"]):
            print(f"  skip {row['id']}: missing Flickr or Mapillary URL")
            continue

        path = save_comparison(photo, top, out_dir)
        if path:
            n_ok += 1
            inliers = int(round(row["p_match"] * 1000 - 1))
            print(f"  saved {path.name}  (p_match={row['p_match']:.3f} ~{inliers} inliers, "
                  f"{row['distance_km']:.3f} km)")
        else:
            print(f"  FAILED {row['id']} (URL may have expired — drop --no-refresh)")

    print(f"\n{n_ok} comparison image(s) -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
