from typing import Optional
import requests
from PIL import Image
import numpy as np
from .encoder import (
    encode,
    similarity,
)
from .building_matcher import (
    match_buildings,load_matcher
)

# LoFTR-based ranking (PROPER building identification) 
def rank_candidates_loftr(
    archive_image: Image.Image,
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
        """print(f"  → keeping top {len(candidates)} for LoFTR  "
              f"(SigLIP scores: {candidates[0]['siglip']:.3f} … "
              f"{candidates[-1]['siglip']:.3f})")"""""

    # STAGE 2: LoFTR + RANSAC on the survivors
    prefiltered = []
    prefiltered.sort(key=lambda x: x["siglip"], reverse=True)
    candidates = prefiltered[:prefilter_top_k]     
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
