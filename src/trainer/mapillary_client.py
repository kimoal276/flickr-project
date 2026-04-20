"""
mapillary_client.py
-------------------
Fetches Mapillary street-level image candidates and ranks them against a
historical archive photo using visual feature matching.

Ranking score
-------------
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
----------
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


# ── Auth ──────────────────────────────────────────────────────────────────────

def _get_token() -> str:
    token = os.getenv("MAPILLARY_ACCESS_TOKEN")
    if not token:
        raise ValueError("MAPILLARY_ACCESS_TOKEN must be set in .env")
    return token


# ── Fetching ──────────────────────────────────────────────────────────────────

def fetch_candidates(
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    limit: int = 50,
) -> list[dict]:
    """
    Query the Mapillary Graph API for street-level images inside a bounding box.

    Returns a list of dicts, each with:
        mapillary_id, lat, lon, thumb_url

    Mapillary returns 500 when the bbox is larger than ~0.005 degrees.
    We tile the search area into 0.005° x 0.005° cells and query each
    one individually, collecting unique results until we reach `limit`.
    """
    token     = _get_token()
    TILE_SIZE = 0.005           # max safe bbox side in degrees
    seen_ids: set[str] = set()
    candidates: list[dict] = []

    # Generate grid of tiles covering the full bbox
    lat = min_lat
    while lat < max_lat and len(candidates) < limit:
        lon = min_lon
        while lon < max_lon and len(candidates) < limit:
            tile_min_lon = lon
            tile_min_lat = lat
            tile_max_lon = min(lon + TILE_SIZE, max_lon)
            tile_max_lat = min(lat + TILE_SIZE, max_lat)

            params = {
                "access_token": token,
                "fields":       "id,geometry,thumb_1024_url",
                "bbox":         f"{tile_min_lon},{tile_min_lat},{tile_max_lon},{tile_max_lat}",
                "limit":        min(10, limit - len(candidates)),
            }

            try:
                resp = requests.get(f"{MAPILLARY_BASE}/images", params=params, timeout=60)
                resp.raise_for_status()
                for item in resp.json().get("data", []):
                    mid    = item.get("id")
                    coords = item.get("geometry", {}).get("coordinates", [])
                    thumb  = item.get("thumb_1024_url")
                    if mid and mid not in seen_ids and thumb and len(coords) >= 2:
                        seen_ids.add(mid)
                        candidates.append({
                            "mapillary_id": mid,
                            "lon":          coords[0],
                            "lat":          coords[1],
                            "thumb_url":    thumb,
                        })
            except requests.HTTPError:
                pass  # skip tiles with no coverage

            lon += TILE_SIZE
        lat += TILE_SIZE

    return candidates


# ── Single-model ranking ──────────────────────────────────────────────────────

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
    ----------
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


# ── Dual-model ranking (recommended for archive-to-street matching) ───────────

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
    ----------
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


# ── LoFTR-based ranking (PROPER building identification) ──────────────────────

def rank_candidates_loftr(
    archive_image: Union[str, Image.Image],
    candidates: list[dict],
    min_inliers: int = 8,
) -> list[dict]:
    """
    Rank Mapillary candidates by geometric feature matching against an archive
    photo.  This is the CORRECT approach for building identification.

    For each candidate:
      1. LoFTR finds corresponding points between archive and candidate.
      2. RANSAC keeps only geometrically consistent matches.
      3. The inlier count is the ranking score.

    Unlike SigLIP cosine similarity, this tells you whether the two photos
    actually depict the same physical building (via point correspondences
    and geometric consistency).

    Parameters
    ----------
    archive_image:  URL or PIL image of the historical archive photo.
    candidates:     Candidates from fetch_candidates().
    min_inliers:    Below this inlier count, the candidate is still ranked
                    but treated as a non-match.

    Returns
    -------
    List of candidates augmented with:
        similarity     — inlier count (integer, repurposed for CSV compat)
        inliers        — RANSAC inlier count
        match_total    — LoFTR matches above confidence threshold
        inlier_ratio   — inliers / match_total
        final_score    — inlier count (same as similarity, for CSV compat)
        distance_km    — always None (no GPS penalty)
    """
    # Load the model once up-front so a download/SSL failure fails fast
    # rather than retrying for every candidate.
    from .building_matcher import _load_matcher
    try:
        _load_matcher()
    except Exception as exc:
        print(f"  [fatal] Could not load LoFTR model: {exc}")
        raise

    scored = []
    for idx, cand in enumerate(candidates):
        try:
            result = match_buildings(archive_image, cand["thumb_url"])
        except Exception as exc:
            print(f"  [match error] candidate {cand['mapillary_id']}: {exc}")
            continue

        n_in      = result["inliers"]
        total     = result["total"]
        ratio     = result["inlier_ratio"]
        is_match  = n_in >= min_inliers

        scored.append(
            {
                **cand,
                "similarity":   float(n_in),
                "inliers":      n_in,
                "match_total":  total,
                "inlier_ratio": ratio,
                "distance_km":  None,
                "final_score":  float(n_in),
                "is_match":     is_match,
            }
        )

        marker = "✓" if is_match else "·"
        print(
            f"  [{idx+1:3d}/{len(candidates)}] {marker} "
            f"inliers={n_in:3d}/{total:<3d} (ratio={ratio:.2f})  "
            f"id={cand['mapillary_id']}"
        )

    scored.sort(key=lambda x: x["inliers"], reverse=True)
    return scored


# ── Spatial cluster re-ranking ────────────────────────────────────────────────

def rerank_by_cluster(
    candidates: list[dict],
    cluster_radius_km: float = 0.2,
) -> list[dict]:
    """
    Boost candidates that are surrounded by other high-scoring neighbours.

    For each candidate c:
        cluster_score = final_score + 0.1 × mean_neighbour_similarity

    The neighbour count is capped at 5 so that a busy intersection with 40
    Mapillary images does not outrank a visually correct but quieter street.
    Only applied when similarity > MIN_SIM to prevent weak matches getting
    a free spatial lift.
    """
    MIN_SIM = 0.30
    reranked = []
    for i, cand in enumerate(candidates):
        neighbours = [
            other for j, other in enumerate(candidates)
            if j != i
            and haversine_km(cand["lat"], cand["lon"],
                             other["lat"], other["lon"]) <= cluster_radius_km
        ]
        # Cap at 5 so dense streets don't dominate over visual similarity
        top_neighbours = sorted(neighbours, key=lambda x: x["similarity"], reverse=True)[:5]
        n        = len(top_neighbours)
        mean_sim = sum(nb["similarity"] for nb in top_neighbours) / n if n else 0.0

        if cand["similarity"] >= MIN_SIM:
            clust_score = cand["final_score"] + 0.10 * mean_sim
        else:
            clust_score = cand["final_score"]

        reranked.append(
            {
                **cand,
                "neighbors":     len(neighbours),   # log real count for diagnostics
                "support_score": mean_sim,
                "cluster_score": clust_score,
            }
        )

    reranked.sort(key=lambda x: x["cluster_score"], reverse=True)
    return reranked


# ── Location prediction ───────────────────────────────────────────────────────

def predict_location(
    ranked: list[dict],
    centroid_radius_km: float = 0.15,
) -> tuple[float, float]:
    """
    Predict (lat, lon) as the centroid of the top match's tight cluster.

    Falls back to a 2-element average when no other candidates lie within
    centroid_radius_km of the top match.
    """
    top     = ranked[0]
    cluster = [
        r for r in ranked
        if haversine_km(top["lat"], top["lon"],
                        r["lat"],  r["lon"]) <= centroid_radius_km
    ]
    if len(cluster) < 2:
        cluster = ranked[: min(2, len(ranked))]

    return (
        sum(r["lat"] for r in cluster) / len(cluster),
        sum(r["lon"] for r in cluster) / len(cluster),
    )


# ── Internals ─────────────────────────────────────────────────────────────────

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