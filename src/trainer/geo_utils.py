"""
geo_utils.py
------------
Shared geographic utility functions used across the geolocator pipeline.

Functions
---------
haversine_km        Distance between two (lat, lon) points in kilometres.
bbox_from_center    Bounding box around a (lat, lon) centre with a given radius.
parse_float         Safe float parser that maps "0" / "" / None → None.
"""

import math
from typing import Optional
from io import BytesIO
from typing import Union
import requests
from PIL import Image

# Geography 
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in kilometres between two GPS points."""
    R = 6_371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bbox_from_center(
    lat: float,
    lon: float,
    radius_km: float,
) -> tuple[float, float, float, float]:
    """
    Return (min_lat, min_lon, max_lat, max_lon) for a square bounding box
    centred on (lat, lon) with the given radius in kilometres.
    """
    deg_lat = radius_km / 111.0
    cos_lat = math.cos(math.radians(lat))
    cos_lat = max(cos_lat, 1e-8)           # guard against poles
    deg_lon = radius_km / (111.0 * cos_lat)
    return lat - deg_lat, lon - deg_lon, lat + deg_lat, lon + deg_lon

# Parsing 

def parse_float(value) -> Optional[float]:
    """
    Parse a value to float.  Returns None for missing / zero / invalid inputs
    (Flickr encodes "no GPS" as 0.0).
    """
    if value in (None, "", "0", "0.0", 0):
        return None
    try:
        f = float(value)
        return None if f == 0.0 else f
    except Exception:
        return None
    
def load_image(image_or_url: Union[str, Image.Image], timeout: int = 20) -> Image.Image:
    if isinstance(image_or_url, str):
        resp = requests.get(image_or_url, timeout=timeout)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    if isinstance(image_or_url, Image.Image):
        return image_or_url.convert("RGB")
    raise TypeError(f"Expected URL string or PIL Image, got {type(image_or_url)}")