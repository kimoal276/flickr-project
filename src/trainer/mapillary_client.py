"""
mapillary_client.py

Fetches Mapillary street-level image candidates and ranks them against a
historical archive photo using visual feature matching.

Ranking score :

Each candidate is scored by:

score = visual_similarity − distance_weight × distance_km

where visual_similarity is computed by the encoder module.  The default
encoder is DINOv2 (structural/geometric features), optionally fused with
SigLIP (semantic features) via a weighted average — see encoder.py for
the full rationale.

The GPS distance penalty keeps the search geographically anchored but is
intentionally small: we want the *visual match* to drive the ranking, with
geography as a soft constraint, not the other way round.

Public API

fetch_candidates(min_lat, min_lon, max_lat, max_lon, limit) → list[dict]
rank_candidates(archive_vec, candidates, ref_lat, ref_lon,
                distance_weight, candidate_model) → list[ScoredCandidate]
rank_candidates_dual(archive_img, candidates, ref_lat, ref_lon,
                     distance_weight, alpha) → list[ScoredCandidate]
rerank_by_cluster(candidates, cluster_radius_km) → list[ScoredCandidate]
predict_location(ranked, centroid_radius_km) → (pred_lat, pred_lon)
"""

from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image

from .encoder import (
    EncoderModel,
    dual_similarity,
    encode,
    similarity,
)
from .building_matcher import match_buildings
from .geo_utils import bbox_from_center, haversine_km

load_dotenv()

MAPILLARY_BASE = "https://graph.mapillary.com"


# Auth 
def _get_token() -> str:
    token = os.getenv("MAPILLARY_ACCESS_TOKEN")
    if not token:
        raise ValueError("MAPILLARY_ACCESS_TOKEN must be set in .env")
    return token


# Fetching 
def fetch_candidates(min_lat, min_lon, max_lat, max_lon, limit=100):
    token = _get_token()
    TILE_SIZE = 0.005
    PER_TILE  = 25                       # was effectively 10
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


# Single-model ranking 
def rank_candidates(
    archive_vec: np.ndarray,
    candidates: list[dict],
    ref_lat: Optional[float] = None,
    ref_lon: Optional[float] = None,
    distance_weight: float = 0.05,
    candidate_model: EncoderModel = EncoderModel.DINOV2,
) -> list[dict]:
    """
    Rank Mapillary candidates against a pre-computed archive embedding.

    Encodes each candidate with `candidate_model` and scores:
        score = cosine_similarity(archive_vec, candidate_vec)
                − distance_weight × distance_km

    Use this when the archive image was already encoded with encode()
    and you want a single-model comparison.

    Parameters
    
    archive_vec:      Pre-encoded archive photo embedding (768-d float32).
    candidates:       Candidates from fetch_candidates().
    ref_lat/ref_lon:  Archive GPS for the distance penalty. None to disable.
    distance_weight:  GPS distance penalty strength (km⁻¹).
    candidate_model:  Must match the model used to produce archive_vec.
    """
    scored = []
    for idx, cand in enumerate(candidates):
        cand_vec = _safe_encode(cand["thumb_url"], model=candidate_model,
                                preprocess_archive=False)
        if cand_vec is None:
            continue

        sim               = similarity(archive_vec, cand_vec)
        dist_km, score    = _apply_distance_penalty(
            sim, cand["lat"], cand["lon"], ref_lat, ref_lon, distance_weight
        )

        scored.append(
            {
                **cand,
                "embedding":    cand_vec,
                "similarity":   sim,
                "distance_km":  dist_km,
                "final_score":  score,
                "dino_score":   sim if candidate_model == EncoderModel.DINOV2 else None,
                "siglip_score": sim if candidate_model == EncoderModel.SIGLIP else None,
            }
        )
        _log(idx + 1, len(candidates), sim, dist_km, score, cand["mapillary_id"])

    scored.sort(key=lambda x: x["final_score"], reverse=True)
    return scored


# Dual-model ranking (recommended for archive-to-street matching) 

def rank_candidates_dual(
    archive_image: Union[str, Image.Image],
    candidates: list[dict],
    ref_lat: Optional[float] = None,
    ref_lon: Optional[float] = None,
    distance_weight: float = 0.05,
    alpha: float = 0.7,
    preprocess_archive: bool = True,
) -> list[dict]:
    """
    Rank candidates using a DINOv2 + SigLIP fused similarity score.

    This is the **recommended** function for cross-domain archive-to-street
    matching.  For each candidate the fused visual score is:

        fused = alpha × sim_dinov2 + (1 - alpha) × sim_siglip

    and the final ranked score is:

        score = fused − distance_weight × distance_km

    DINOv2 captures structural/geometric patterns (edge layout, facade
    geometry) that are stable across the archive-to-street domain gap.
    SigLIP adds semantic context (what kind of scene is this?).

    The archive image is preprocessed (grayscale + contrast normalisation)
    before DINOv2 encoding to reduce the style gap between historical photos
    and modern street imagery.  Mapillary thumbnails are processed the same
    way so both embeddings live in the same colour-neutral space.

    Parameters
    
    archive_image:      URL string or PIL Image of the historical photo.
    candidates:         Candidates from fetch_candidates().
    ref_lat/ref_lon:    Archive GPS for distance penalty (None to disable).
    distance_weight:    GPS distance penalty strength (km⁻¹).
    alpha:              DINOv2 weight in [0, 1].  0.7 recommended.
    preprocess_archive: Apply grayscale + contrast normalisation.
                        Set False only when both images are modern colour.
    """
    scored = []
    for idx, cand in enumerate(candidates):
        try:
            fused, dino_sc, siglip_sc = dual_similarity(
                archive_image,
                cand["thumb_url"],
                alpha=alpha,
                preprocess_archive=preprocess_archive,
            )
        except Exception as exc:
            print(f"  [encode error] candidate {cand['mapillary_id']}: {exc}")
            continue

        dist_km, score = _apply_distance_penalty(
            fused, cand["lat"], cand["lon"], ref_lat, ref_lon, distance_weight
        )

        scored.append(
            {
                **cand,
                "similarity":   fused,
                "dino_score":   dino_sc,
                "siglip_score": siglip_sc,
                "distance_km":  dist_km,
                "final_score":  score,
            }
        )
        _log_dual(idx + 1, len(candidates),
                  fused, dino_sc, siglip_sc, dist_km, score,
                  cand["mapillary_id"])

    scored.sort(key=lambda x: x["final_score"], reverse=True)
    return scored


# LoFTR-based ranking (PROPER building identification) 
def rank_candidates_loftr(
    archive_image: Union[str, Image.Image],
    candidates: list[dict],
    min_inliers: int = 14,
    prefilter_top_k: int = 30,
) -> list[dict]:
    """
    Two-stage ranking:
      1. SigLIP cosine similarity on all candidates  (fast, ~ms per image)
      2. LoFTR + RANSAC on the top-`prefilter_top_k`  (slow but accurate)
    Set prefilter_top_k=None to disable the filter and run LoFTR on everything.
    """
    from .building_matcher import _load_matcher
    try:
        _load_matcher()
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

    # STAGE 1: SigLIP pre-filter     if prefilter_top_k is not None and len(candidates) > prefilter_top_k:
        print(f"  Stage 1: SigLIP pre-filter on {len(candidates)} candidates …")
        archive_vec = encode(archive_pil, model=EncoderModel.SIGLIP,
                             preprocess_archive=True)

        prefiltered = []
        for idx, cand in enumerate(candidates):
            cand_vec = _safe_encode(cand["thumb_url"],
                                    model=EncoderModel.SIGLIP,
                                    preprocess_archive=True)
            if cand_vec is None:
                continue
            sim = similarity(archive_vec, cand_vec)
            prefiltered.append({**cand, "_siglip": sim})

        prefiltered.sort(key=lambda x: x["_siglip"], reverse=True)
        candidates = prefiltered[:prefilter_top_k]
        print(f"  → keeping top {len(candidates)} for LoFTR  "
              f"(SigLIP scores: {candidates[0]['_siglip']:.3f} … "
              f"{candidates[-1]['_siglip']:.3f})")

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
    print(f"  [debug] LoFTR survivors: {[c['mapillary_id'] for c in candidates]}")
    return scored

# Internals 

def _safe_encode(
    url: str,
    model: EncoderModel,
    preprocess_archive: bool,
) -> Optional[np.ndarray]:
    try:
        return encode(url, model=model, preprocess_archive=preprocess_archive)
    except Exception as exc:
        print(f"  [encode error] {url}: {exc}")
        return None


def _apply_distance_penalty(
    sim: float,
    cand_lat: float,
    cand_lon: float,
    ref_lat: Optional[float],
    ref_lon: Optional[float],
    weight: float,
) -> tuple[Optional[float], float]:
    if ref_lat is not None and ref_lon is not None:
        dist_km = haversine_km(ref_lat, ref_lon, cand_lat, cand_lon)
        return dist_km, sim - weight * dist_km
    return None, sim


def _log(
    idx: int, total: int,
    sim: float, dist_km: Optional[float],
    score: float, mid: str,
) -> None:
    dist_str = f"dist={dist_km:.2f}km" if dist_km is not None else "dist=N/A"
    print(f"  [{idx:3d}/{total}] sim={sim:.4f}  {dist_str}  score={score:.4f}  {mid}")


def _log_dual(
    idx: int, total: int,
    fused: float, dino: float, siglip: float,
    dist_km: Optional[float], score: float, mid: str,
) -> None:
    dist_str = f"dist={dist_km:.2f}km" if dist_km is not None else "dist=N/A"
    print(
        f"  [{idx:3d}/{total}] "
        f"fused={fused:.4f}  dino={dino:.4f}  siglip={siglip:.4f}  "
        f"{dist_str}  score={score:.4f}  {mid}"
    )