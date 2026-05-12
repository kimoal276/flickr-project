"""
geo_utils.py
------------
Shared geographic utility functions used across the geolocator pipeline.

Functions
---------
haversine_km        Distance between two (lat, lon) points in kilometres.
bbox_from_center    Bounding box around a (lat, lon) centre with a given radius.
cosine_similarity   Normalised dot-product between two embedding vectors.
parse_float         Safe float parser that maps "0" / "" / None → None.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


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