"""
mapillary_client.py

Fetches Mapillary street-level image candidates and ranks them against a
historical archive photo using visual feature matching.

Public API

sample_candidate
"""

from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image
from dataclasses import dataclass


from .encoder import (
    encode,
    similarity,
)
from .building_matcher import match_buildings
from .geo_utils import bbox_from_center, haversine_km
from typing import Optional

load_dotenv()

# Auth 
def _get_token() -> str:
    token = os.getenv("MAPILLARY_ACCESS_TOKEN")
    if not token:
        raise ValueError("MAPILLARY_ACCESS_TOKEN must be set in .env")
    return token
from dataclasses import dataclass
import requests

MAPILLARY_BASE = "https://graph.mapillary.com"

@dataclass
class MapillaryPicture:
    id: int
    lat: float
    lon: float
    pic_url: str

@dataclass
class MapillarySampler:
    lon: float
    lat: float
    candidates: list

    @classmethod
    def create(cls, longitude: float, latitude: float, st_km: float = 0.05)-> Optional[MapillarySampler]:
        """creates a sampler: """
        token = _get_token()
        params = {
            "access_token": token,
            "fields": "id,geometry,thumb_1024_url,captured_at",
            "bbox": f"{longitude - 5*st_km},{latitude - 5*st_km},{longitude + 5*st_km},{latitude + 5*st_km}",
            "limit": 1000,
        }
        try:
            resp = requests.get(f"{MAPILLARY_BASE}/images", params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            candidates = [
                MapillaryPicture(
                    id=item.get("id"),
                    lat=item.get("geometry", {}).get("coordinates", [None, None])[1],
                    lon=item.get("geometry", {}).get("coordinates", [None, None])[0],
                    pic_url=item.get("thumb_1024_url"),
                )
                for item in data
            ]
            return cls(longitude, latitude, candidates)
        except requests.RequestException:
            return None

        @classmethod
        def sample():
            """returns a MapillayPicture sampled from the object"""

# Fetching 
def fetch_candidates(min_lat, min_lon, max_lat, max_lon, limit=100):
    token = _get_token()
    TILE_SIZE = 0.005
    PER_TILE  = 25                       
    seen, candidates = set(), []

    lat = min_lat
    while lat < max_lat:
        lon = min_lon
        while lon < max_lon:
            tile = (lon, lat,
                    min(lon + TILE_SIZE, max_lon),
                    min(lat + TILE_SIZE, max_lat))
            params = {
                "access_token": token,
                "fields":       "id,geometry,thumb_1024_url,captured_at",
                "bbox":         f"{tile[0]},{tile[1]},{tile[2]},{tile[3]}",
                "limit":        1000,
            }
            try:
                resp = requests.get(f"{MAPILLARY_BASE}/images",
                                    params=params, timeout=60)
                resp.raise_for_status()
                for item in resp.json().get("data", []):
                    mid = item.get("id")
                    coords = item.get("geometry", {}).get("coordinates", [])
                    thumb  = item.get("thumb_1024_url")
                    if mid and mid not in seen and thumb and len(coords) >= 2:
                        seen.add(mid)
                        candidates.append({"mapillary_id": mid,
                                           "lon": coords[0], "lat": coords[1],
                                           "thumb_url": thumb})
            except requests.HTTPError:
                pass
            lon += TILE_SIZE
        lat += TILE_SIZE

    # Then subsample to `limit` *spatially* — keep candidates near the
    # bbox center first so we don't blow LoFTR budget on the periphery.
    cx = (min_lat + max_lat) / 2
    cy = (min_lon + max_lon) / 2
    candidates.sort(key=lambda c: (c["lat"]-cx)**2 + (c["lon"]-cy)**2)
    return candidates[:limit]

# LoFTR-based ranking (PROPER building identification) 
def rank_candidates_loftr(
    archive_image: Union[str, Image.Image],
    candidates: list[dict],
    min_inliers: int = 8,
    prefilter_top_k: int = 50,    # Can determine whether siglip is used (if k=mapillary_limit only loftr runs)
) -> list[dict]:
    """
    Two-stage ranking:
      1. SigLIP cosine similarity on all candidates  (fast, ~ms per image)
      2. LoFTR + RANSAC on the top-`prefilter_top_k`  (slow but accurate)
    Set prefilter_top_k=None to disable the filter and run LoFTR on everything.
    """
    from .building_matcher import load_matcher
    try:
        load_matcher()
    except Exception as exc:
        print(f"  [fatal] Could not load LoFTR model: {exc}")
        raise

    # Pre-load archive ONCE
    if isinstance(archive_image, str):
        from io import BytesIO
        import time as _t
        archive_pil = None
        for attempt in range(3):
            try:
                resp = requests.get(archive_image, timeout=30)
                resp.raise_for_status()
                archive_pil = Image.open(BytesIO(resp.content)).convert("RGB")
                break
            except Exception as exc:
                wait = 2 ** attempt
                print(f"  [archive load attempt {attempt+1}/3 failed: {exc}] "
                      f"retrying in {wait}s …")
                _t.sleep(wait)
        if archive_pil is None:
            print(f"  [fatal] Could not download archive image")
            return []
    else:
        archive_pil = archive_image

    # STAGE 1: SigLIP pre-filter     
    if prefilter_top_k is not None and len(candidates) > prefilter_top_k:
        print(f"  Stage 1: SigLIP pre-filter on {len(candidates)} candidates …")
        archive_vec = encode(archive_pil, preprocess_archive=True)

        prefiltered = []
        for idx, cand in enumerate(candidates):
            cand_vec = safe_encode(cand["thumb_url"],
                                    preprocess_archive=True)
            if cand_vec is None:
                continue
            sim = similarity(archive_vec, cand_vec)
            prefiltered.append({**cand, "siglip": sim})

        prefiltered.sort(key=lambda x: x["siglip"], reverse=True)
        candidates = prefiltered[:prefilter_top_k]
        print(f"  → keeping top {len(candidates)} for LoFTR  "
              f"(SigLIP scores: {candidates[0]['siglip']:.3f} … "
              f"{candidates[-1]['siglip']:.3f})")

    # STAGE 2: LoFTR + RANSAC on the survivors     
    scored = []
    for idx, cand in enumerate(candidates):
        try:
            result = match_buildings(archive_pil, cand["thumb_url"])
        except Exception as exc:
            print(f"  [match error] candidate {cand['mapillary_id']}: {exc}")
            continue

        n_in   = result["inliers"]
        total  = result["total"]
        ratio  = result["inlier_ratio"]
        is_match = n_in >= min_inliers

        scored.append({
            **cand,
            "similarity":   float(n_in),
            "inliers":      n_in,
            "match_total":  total,
            "inlier_ratio": ratio,
            "distance_km":  None,
            "final_score":  float(n_in),
            "is_match":     is_match,
        })

        marker = "✓" if is_match else "·"
        print(f"  [{idx+1:2d}/{len(candidates)}] {marker} "
              f"inliers={n_in:3d}/{total:<3d} (ratio={ratio:.2f})  "
              f"id={cand['mapillary_id']}")

    scored.sort(
        key=lambda x: (x["inliers"], x["inlier_ratio"], x["mapillary_id"]),
        reverse=True,
    )
    
    return scored

# Internals 

def safe_encode(
    url: str,
    preprocess_archive: bool,
) -> Optional[np.ndarray]:
    try:
        return encode(url, preprocess_archive=preprocess_archive)
    except Exception as exc:
        print(f"  [encode error] {url}: {exc}")
        return None
