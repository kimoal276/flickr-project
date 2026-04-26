"""
diagnose_candidates.py
----------------------
Diagnostic for the flico geolocator pipeline.

Given a Flickr photo's GPS coords and a hand-verified ground-truth
Mapillary image ID, this script:

  1. Tiles the bbox at multiple search radii (0.25, 0.5, 1, 2 km).
  2. Exhaustively fetches every Mapillary image in each bbox
     (no early-stop bug — see fetch_all() below).
  3. Reports whether the ground-truth ID is in the candidate set.
  4. Saves the full candidate list to a JSON file for inspection.

If the truth ID is not in the set even at 2 km radius, the matcher is
not the bug — fetching is.

Setup
-----
  pip install requests python-dotenv
  export MAPILLARY_ACCESS_TOKEN=...   # or put it in a .env file

Usage
-----
  python diagnose_candidates.py \
      --lat 42.9222 --lon -78.8660 \
      --truth_id 145823109876543      # from mapillary.com share link

  # Optional: also report drift from real building location
  python diagnose_candidates.py \
      --lat 42.9222 --lon -78.8660 \
      --truth_lat 42.8956 --truth_lon -78.8769 \
      --truth_id 145823109876543
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


MAPILLARY_BASE = "https://graph.mapillary.com"
TILE_SIZE = 0.005          # max safe Mapillary bbox side, ~555 m at equator
PER_TILE_LIMIT = 2000        # ask for plenty per tile — Mapillary caps internally


# ── helpers ──────────────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bbox_from_center(lat: float, lon: float, r_km: float):
    dlat = r_km / 111.0
    dlon = r_km / (111.0 * max(math.cos(math.radians(lat)), 1e-8))
    return lat - dlat, lon - dlon, lat + dlat, lon + dlon


# ── exhaustive fetcher (no early stop) ───────────────────────────────────────

def fetch_all(min_lat, min_lon, max_lat, max_lon, token, verbose=False):
    """
    Tile the bbox and fetch ALL images from EVERY tile.

    Unlike the production fetch_candidates() this does NOT bail out as soon
    as some quota is reached — that's the whole point. We want to know
    what the *complete* candidate set looks like before any subsampling.
    """
    seen: set[str] = set()
    out: list[dict] = []
    n_tiles = 0
    n_tiles_with_data = 0
    n_errors = 0

    lat = min_lat
    while lat < max_lat:
        lon = min_lon
        while lon < max_lon:
            n_tiles += 1
            t_max_lon = min(lon + TILE_SIZE, max_lon)
            t_max_lat = min(lat + TILE_SIZE, max_lat)

            params = {
                "access_token": token,
                "fields":       "id,geometry,thumb_1024_url,captured_at",
                "bbox":         f"{lon},{lat},{t_max_lon},{t_max_lat}",
                "limit":        PER_TILE_LIMIT,
            }
            try:
                resp = requests.get(
                    f"{MAPILLARY_BASE}/images", params=params, timeout=30
                )
                resp.raise_for_status()
                data = resp.json().get("data", [])
                if data:
                    n_tiles_with_data += 1
                for item in data:
                    mid = item.get("id")
                    coords = item.get("geometry", {}).get("coordinates") or []
                    if mid and mid not in seen and len(coords) >= 2:
                        seen.add(mid)
                        out.append({
                            "id":          mid,
                            "lat":         coords[1],
                            "lon":         coords[0],
                            "thumb":       item.get("thumb_1024_url"),
                            "captured_at": item.get("captured_at"),
                        })
            except requests.HTTPError as e:
                n_errors += 1
                if verbose:
                    print(f"    [tile error] {lat:.4f},{lon:.4f}: {e}")
            except requests.RequestException as e:
                n_errors += 1
                if verbose:
                    print(f"    [net error] {lat:.4f},{lon:.4f}: {e}")
            lon += TILE_SIZE
        lat += TILE_SIZE

    return out, n_tiles, n_tiles_with_data, n_errors


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--lat", type=float, required=True,
                    help="Flickr GPS latitude")
    ap.add_argument("--lon", type=float, required=True,
                    help="Flickr GPS longitude")
    ap.add_argument("--truth_id", default=None,
                    help="Hand-verified Mapillary image ID (from share URL)")
    ap.add_argument("--truth_lat", type=float, default=None,
                    help="Optional: building's actual lat (for drift calc)")
    ap.add_argument("--truth_lon", type=float, default=None)
    ap.add_argument("--radii", type=float, nargs="+",
                    default=[0.25, 0.5, 1.0, 2.0],
                    help="Search radii in km to test (default: 0.25 0.5 1 2)")
    ap.add_argument("--save", default="candidates_dump.json",
                    help="Where to save the full candidate dump")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    token = os.getenv("MAPILLARY_ACCESS_TOKEN")
    if not token:
        sys.exit("ERROR: set MAPILLARY_ACCESS_TOKEN in your env or .env")

    print()
    print(f"Flickr GPS    : {args.lat:.5f}, {args.lon:.5f}")
    if args.truth_lat is not None and args.truth_lon is not None:
        drift = haversine_km(args.lat, args.lon, args.truth_lat, args.truth_lon)
        print(f"Real building : {args.truth_lat:.5f}, {args.truth_lon:.5f}")
        print(f"GPS drift     : {drift:.3f} km")
    if args.truth_id:
        print(f"Truth ID      : {args.truth_id}")
    print()

    summary: dict = {}
    for r in sorted(args.radii):
        min_lat, min_lon, max_lat, max_lon = bbox_from_center(
            args.lat, args.lon, r
        )
        print(f"── Radius {r} km " + "─" * 50)

        cands, n_tiles, n_with_data, n_err = fetch_all(
            min_lat, min_lon, max_lat, max_lon, token, verbose=args.verbose
        )

        print(f"  Tiles queried        : {n_tiles}")
        print(f"  Tiles with coverage  : {n_with_data}")
        print(f"  Tiles with errors    : {n_err}")
        print(f"  Unique candidates    : {len(cands)}")

        truth_hit = None
        if args.truth_id:
            truth_hit = next(
                (c for c in cands if str(c["id"]) == str(args.truth_id)),
                None,
            )
            if truth_hit:
                d_search = haversine_km(
                    args.lat, args.lon, truth_hit["lat"], truth_hit["lon"]
                )
                print(f"  ✓ TRUTH FOUND        : at {d_search:.3f} km from search center")
            else:
                print(f"  ✗ TRUTH NOT FOUND    : id {args.truth_id} not in set")

        summary[f"radius_{r}km"] = {
            "n_candidates":      len(cands),
            "n_tiles_queried":   n_tiles,
            "n_tiles_with_data": n_with_data,
            "truth_in_set":      bool(truth_hit) if args.truth_id else None,
            "candidate_ids":     [c["id"] for c in cands],
        }
        print()

    Path(args.save).write_text(json.dumps(summary, indent=2))
    print(f"Full dump written to {args.save}")


if __name__ == "__main__":
    main()