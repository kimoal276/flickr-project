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
from .geo_utils import load_image

SIGLIP_NAME = "google/siglip-base-patch16-224"
siglip_processor = None
siglip_model     = None


def load_siglip():
    global siglip_processor, siglip_model
    if siglip_model is None:
        print(f"  Loading SigLIP ({SIGLIP_NAME}) …")
        siglip_processor = AutoProcessor.from_pretrained(SIGLIP_NAME)
        siglip_model     = AutoModel.from_pretrained(SIGLIP_NAME).eval()
    return siglip_processor, siglip_model


# Preprocessing 

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


# Core API 
def encode(
    image_or_url: Union[str, Image.Image],
    preprocess_archive: bool = False,
) -> np.ndarray:
    """
    Encode a single image to a float32 SigLIP embedding vector.

    Parameters
    ----------
    image_or_url:       URL string or PIL Image.
    preprocess_archive: Apply grayscale + contrast normalisation before
                        encoding (set True for historical archive photos).

    Returns
    -------
    Float32 numpy array of shape (768,).
    """
    img = load_image(image_or_url)
    if preprocess_archive:
        img = preprocess_for_cross_domain(img)
    return encode_siglip(img)


def similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two embedding vectors."""
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-8
    return float(np.dot(vec_a, vec_b) / denom)


def encode_siglip(image: Image.Image) -> np.ndarray:
    processor, model = load_siglip()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        vec = model.get_image_features(**inputs).pooler_output[0]
    arr = vec.cpu().numpy().astype(np.float32)
    assert arr.shape == (768,), f"SigLIP: expected (768,), got {arr.shape}"
    return arr