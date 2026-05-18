"""
building_matcher.py
-------------------
Computer-vision based identification of the same building across photos.

The central idea
----------------
Global image embeddings (SigLIP, CLIP) answer "what kind of scene is this?"
— they are trained on (image, caption) pairs and learn semantic similarity.
Two different churches in different cities will score high on SigLIP.

To decide whether two photos show *the same specific building* we need
LOCAL feature matching: find points in image A that correspond to specific
points in image B ("this window corner" → "that window corner"), then
verify the correspondences are GEOMETRICALLY consistent — i.e. they can
be related by a valid viewpoint change (fundamental matrix).

The count of geometrically consistent matches (RANSAC inliers) is a
robust signal for "same building", invariant to:
  - black-and-white vs colour
  - age of the photograph
  - viewpoint, zoom, weather, time of day
  - partial occlusion (cars, people, foliage)

Pipeline
--------
1. LoFTR — transformer-based dense matcher,
   pretrained on MegaDepth outdoor scenes.  Produces keypoint
   correspondences and per-match confidence scores.
2. Filter by LoFTR confidence (≥ 0.5).
3. RANSAC with fundamental-matrix model (USAC_MAGSAC) to keep only
   geometrically consistent matches.
4. Return the inlier count.

Typical inlier counts
---------------------
   >30   very strong match, almost certainly the same building
   15–30 probable match, worth checking
   5–15  visually similar but likely a different building
    <5   unrelated

Public API
----------
match_buildings(image_a, image_b, **kwargs) → dict
    Returns {"inliers": int, "total": int, "inlier_ratio": float}
"""

from __future__ import annotations

import ssl
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Union

import cv2
import kornia.feature as KF
import numpy as np
import requests
import torch
from PIL import Image
from .geo_utils import load_image

# Lazy model loader 

_matcher: KF.LoFTR | None = None
_device:  torch.device  | None = None
load_failed: bool = False                     # sticky flag — don't retry on failure

_LOFTR_URL  = "http://cmp.felk.cvut.cz/~mishkdmy/models/loftr_outdoor.ckpt"
_LOFTR_NAME = "loftr_outdoor.ckpt"


def ensure_checkpoint() -> Path:
    """
    Pre-download the LoFTR checkpoint into torch's hub cache, bypassing SSL
    verification.  Works around Windows Python + Czech university cert chain
    issues.  No-op if the file is already cached.
    """
    cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt = cache_dir / _LOFTR_NAME

    if ckpt.exists() and ckpt.stat().st_size > 1_000_000:
        return ckpt

    print(f"  Downloading LoFTR checkpoint → {ckpt}")
    # Bypass SSL verification for this one-time download
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(_LOFTR_URL, headers={"User-Agent": "flico/1.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=120) as resp, \
         open(ckpt, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0)) or None
        read  = 0
        while True:
            chunk = resp.read(64 * 1024)
            if not chunk:
                break
            out.write(chunk)
            read += len(chunk)
            if total:
                pct = 100 * read / total
                print(f"\r    {read / 1e6:6.1f} / {total / 1e6:.1f} MB "
                      f"({pct:5.1f}%)", end="", flush=True)
        print()
    print(f"  ✓ Checkpoint saved ({ckpt.stat().st_size / 1e6:.1f} MB)")
    return ckpt


def load_matcher() -> tuple[KF.LoFTR, torch.device]:
    """Lazy-load LoFTR outdoor weights.  Raises on persistent failure."""
    global _matcher, _device, load_failed

    if load_failed:
        raise RuntimeError(
            "LoFTR model failed to load earlier in this session — not retrying."
        )

    if _matcher is None:
        try:
            ensure_checkpoint()          # pre-download if missing
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"  Loading LoFTR (outdoor) on {_device} …")
            _matcher = KF.LoFTR(pretrained="outdoor").eval().to(_device)
        except Exception:
            load_failed = True           # don't retry on every candidate
            raise

    return _matcher, _device


# Image I/O 

def to_gray_tensor(img: Image.Image, max_size: int, device: torch.device) -> torch.Tensor:
    """Convert PIL image to LoFTR-compatible [1, 1, H, W] grayscale tensor."""
    gray = img.convert("L")
    w, h = gray.size
    scale = max_size / max(w, h)
    if scale < 1:
        gray = gray.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    # LoFTR prefers dimensions divisible by 8
    w2, h2 = gray.size
    w2 = (w2 // 8) * 8
    h2 = (h2 // 8) * 8
    gray = gray.resize((w2, h2), Image.BILINEAR)
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, None].to(device)


# Public API 
def match_buildings(
    image_a: Union[str, Image.Image],
    image_b: Union[str, Image.Image],
    max_size: int = 512,
    confidence_threshold: float = 0.5,
    ransac_threshold: float = 3.0,
) -> dict:
    """
    Count geometrically consistent matches between two building photos.

    Parameters
    ----------
    image_a, image_b:     URLs or PIL images (archive photo, candidate).
    max_size:             Longest side the images are resized to (default 640).
                          Larger = more detail, slower.
    confidence_threshold: LoFTR per-match confidence filter (default 0.5).
    ransac_threshold:     RANSAC reprojection threshold in pixels (default 3).

    Returns
    -------
    {
      "inliers":      int,     # RANSAC inlier count — primary ranking score
      "total":        int,     # LoFTR matches above confidence_threshold
      "inlier_ratio": float,   # inliers / total
    }
    """
    matcher, device = load_matcher()

    img_a = load_image(image_a)
    img_b = load_image(image_b)

    t_a = to_gray_tensor(img_a, max_size, device)
    t_b = to_gray_tensor(img_b, max_size, device)

    with torch.no_grad():
        out = matcher({"image0": t_a, "image1": t_b})

    kp0  = out["keypoints0"].cpu().numpy()
    kp1  = out["keypoints1"].cpu().numpy()
    conf = out["confidence"].cpu().numpy()

    mask  = conf >= confidence_threshold
    kp0   = kp0[mask]
    kp1   = kp1[mask]
    total = len(kp0)

    if total < 8:   # need minimum 8 points for fundamental matrix estimation
        return {"inliers": 0, "total": total, "inlier_ratio": 0.0}

    _, inliers = cv2.findHomography(
        kp0, kp1,
        cv2.USAC_MAGSAC,
        ransacReprojThreshold=ransac_threshold,
        confidence=0.99,
        maxIters=2_000,
    )
    n_inliers = int(inliers.sum()) if inliers is not None else 0

    return {
        "inliers":      n_inliers,
        "total":        total,
        "inlier_ratio": n_inliers / max(1, total),
    }