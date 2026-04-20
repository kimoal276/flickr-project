"""
encoder.py
----------
Image encoding for the geolocator pipeline.

Uses SigLIP (google/siglip-base-patch16-224) to encode images into 768-d
float32 embeddings for cosine-similarity ranking.

Public API
----------
encode(image_or_url, model, preprocess_archive) → np.ndarray
similarity(vec_a, vec_b) → float
dual_similarity(archive, candidate, alpha, preprocess_archive)
    → (fused_score, dino_score, siglip_score)
    Note: dual_similarity is a SigLIP-only wrapper kept for API compatibility.
          dino_score and siglip_score will both equal the SigLIP score.

encode_with_siglip(image_or_url) → np.ndarray   [legacy alias]
encode_pil_image(image) → np.ndarray             [legacy alias]
encode_image_from_url(url) → np.ndarray          [legacy alias]
"""

from __future__ import annotations

from enum import Enum
from io import BytesIO
from typing import Union

import numpy as np
import requests
import torch
from PIL import Image, ImageFilter
from transformers import AutoModel, AutoProcessor


# ── Model registry ────────────────────────────────────────────────────────────

class EncoderModel(str, Enum):
    SIGLIP = "siglip"
    DINOV2 = "dinov2"  # kept in enum so CLI flags still parse; routes to SigLIP


_SIGLIP_NAME = "google/siglip-base-patch16-224"

_siglip_processor = None
_siglip_model     = None


def _load_siglip():
    global _siglip_processor, _siglip_model
    if _siglip_model is None:
        print(f"  Loading SigLIP ({_SIGLIP_NAME}) …")
        _siglip_processor = AutoProcessor.from_pretrained(_SIGLIP_NAME)
        _siglip_model     = AutoModel.from_pretrained(_SIGLIP_NAME).eval()
    return _siglip_processor, _siglip_model


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_for_cross_domain(image: Image.Image) -> Image.Image:
    """
    Reduce the style gap between historical archive photos and modern
    street-level imagery before encoding.

    Steps: grayscale → histogram equalisation → mild Gaussian smoothing
    → back to RGB (required by ViT).
    """
    from PIL import ImageOps
    gray      = image.convert("L")
    equalized = ImageOps.equalize(gray)
    smoothed  = equalized.filter(ImageFilter.GaussianBlur(radius=0.8))
    return smoothed.convert("RGB")


# ── Core API ──────────────────────────────────────────────────────────────────

def encode(
    image_or_url: Union[str, Image.Image],
    model: EncoderModel = EncoderModel.SIGLIP,
    preprocess_archive: bool = False,
) -> np.ndarray:
    """
    Encode a single image to a float32 SigLIP embedding vector.

    Parameters
    ----------
    image_or_url:       URL string or PIL Image.
    model:              Ignored (SigLIP is always used). Kept for API compat.
    preprocess_archive: Apply grayscale + contrast normalisation before
                        encoding (set True for historical archive photos).

    Returns
    -------
    Float32 numpy array of shape (768,).
    """
    img = _load_image(image_or_url)
    if preprocess_archive:
        img = preprocess_for_cross_domain(img)
    return _encode_siglip(img)


def similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two embedding vectors."""
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-8
    return float(np.dot(vec_a, vec_b) / denom)


def dual_similarity(
    archive_url_or_img: Union[str, Image.Image],
    candidate_url_or_img: Union[str, Image.Image],
    alpha: float = 0.7,
    preprocess_archive: bool = True,
) -> tuple[float, float, float]:
    """
    Compatibility wrapper — returns (score, score, score) using SigLIP only.

    The alpha and dual-encoder arguments are accepted but ignored.
    Both dino_score and siglip_score are set to the SigLIP cosine similarity
    so that any downstream code reading those fields still gets a valid number.
    """
    archive_img   = _load_image(archive_url_or_img)
    candidate_img = _load_image(candidate_url_or_img)

    if preprocess_archive:
        archive_img   = preprocess_for_cross_domain(archive_img)
        candidate_img = preprocess_for_cross_domain(candidate_img)

    vec_a = _encode_siglip(archive_img)
    vec_b = _encode_siglip(candidate_img)
    score = similarity(vec_a, vec_b)
    return score, score, score   # fused, dino_score, siglip_score


# ── Legacy wrappers ───────────────────────────────────────────────────────────

def encode_with_siglip(image_or_url: Union[str, Image.Image]) -> np.ndarray:
    return encode(image_or_url, model=EncoderModel.SIGLIP, preprocess_archive=False)


def encode_pil_image(image: Image.Image) -> np.ndarray:
    return _encode_siglip(image)


def encode_image_from_url(url: str, timeout: int = 20) -> np.ndarray:
    return _encode_siglip(_load_image(url, timeout=timeout))


# ── Internals ─────────────────────────────────────────────────────────────────

def _load_image(image_or_url: Union[str, Image.Image], timeout: int = 20) -> Image.Image:
    if isinstance(image_or_url, str):
        resp = requests.get(image_or_url, timeout=timeout)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    if isinstance(image_or_url, Image.Image):
        return image_or_url.convert("RGB")
    raise TypeError(f"Expected URL string or PIL Image, got {type(image_or_url)}")


def _encode_siglip(image: Image.Image) -> np.ndarray:
    processor, model = _load_siglip()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        vec = model.get_image_features(**inputs).pooler_output[0]
    arr = vec.cpu().numpy().astype(np.float32)
    assert arr.shape == (768,), f"SigLIP: expected (768,), got {arr.shape}"
    return arr