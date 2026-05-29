import ssl
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple
import cv2
import kornia.feature as KF
import numpy as np
import requests
import torch
from PIL import Image



_matcher: KF.LoFTR | None = None
_device:  torch.device  | None = None
load_failed: bool = False                     

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


def load_picture(url: str, timeout: int = 20) -> Optional[Image.Image]:

    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception as exc:
        print(f"  [load_picture] failed for {url}: {exc}")
        return None
    

def compute_loftr_matches(
    t1,
    t2,
    matcher,
    confidence_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:

    with torch.no_grad():
        out = matcher({"image0": t1, "image1": t2})

    kp0  = out["keypoints0"].cpu().numpy()
    kp1  = out["keypoints1"].cpu().numpy()
    conf = out["confidence"].cpu().numpy()

    mask = conf >= confidence_threshold
    return kp0[mask], kp1[mask]


def compute_ransac_inliers(
    kp0: np.ndarray,
    kp1: np.ndarray,
    ransac_threshold: float = 3.0,
    confidence: float = 0.99,
    max_iterations: int = 500,
) -> int:
    
    if len(kp0) < 8 or len(kp1) < 8:
        return 0

    F, inliers = cv2.findFundamentalMat(
    kp0, kp1,
    cv2.USAC_MAGSAC,
    ransacReprojThreshold=ransac_threshold,
    confidence=confidence,
    maxIters=max_iterations,
)
    return int(inliers.sum()) if inliers is not None else 0
