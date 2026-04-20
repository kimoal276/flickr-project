"""
text_geocoder.py
----------------
Attempts to refine a search centre using free-text metadata (title,
description) before falling back to the raw GPS coordinates from the CSV.

The pipeline is:
    1. Clean the text (strip HTML, dates, archive boilerplate).
    2. Send it to Nominatim (OpenStreetMap geocoder, no API key needed).
    3. Accept the result only when it lies within max_drift_km of the
       original GPS point — a sanity check against titles like
       "Library of Congress Collection, 1905".

The module deliberately sleeps ≥ 1.1 s between Nominatim calls to respect
the service's rate limit policy.

Public API
----------
refine_center(gps_lat, gps_lon, title, description, max_drift_km)
    → (lat, lon, source)   where source ∈ {"title", "description", "gps"}
"""

from __future__ import annotations

import re
import time
from typing import Optional

import requests

from .geo_utils import haversine_km

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_HEADERS = {"User-Agent": "flico-geolocator/1.0 (academic research)"}

# Regex patterns used to strip noise before geocoding
_ARCHIVE_NOISE = re.compile(
    r"\b(ca\.?|circa|c\.|photograph(s|ed)?|photo|image|view|scene(s)?|"
    r"collection|archive|commons|library|museum|university|institute|"
    r"negatives?|glass\s+plates?|lantern\s+slides?|album)\b",
    flags=re.IGNORECASE,
)
_DATES = re.compile(
    r"\b(ca\.?\s*)?\d{4}s?\b|\b\d+(st|nd|rd|th)\b",
    flags=re.IGNORECASE,
)


# ── Public API ────────────────────────────────────────────────────────────────

def refine_center(
    gps_lat: float,
    gps_lon: float,
    title: str,
    description: str = "",
    max_drift_km: float = 50.0,
) -> tuple[float, float, str]:
    """
    Try to improve the search centre using Nominatim geocoding.

    Tries the photo title first; if the result drifts more than max_drift_km
    from the GPS point it falls back to the description; if that also fails
    the original GPS coordinates are returned unchanged.

    Returns
    -------
    (lat, lon, source)
        source is one of "title", "description", or "gps".
    """
    for text, label in [(title, "title"), (description, "description")]:
        if not text or len(text.strip()) < 5:
            continue

        coords = _nominatim_geocode(text)
        if coords is None:
            print(f"  Text geocoding ({label}): no result")
            continue

        nom_lat, nom_lon = coords
        drift = haversine_km(gps_lat, gps_lon, nom_lat, nom_lon)

        if drift <= max_drift_km:
            print(
                f"  Text geocoding ({label}): accepted — drift {drift:.2f} km  "
                f"→ ({nom_lat:.5f}, {nom_lon:.5f})"
            )
            return nom_lat, nom_lon, label

        print(f"  Text geocoding ({label}): rejected — drift {drift:.2f} km > {max_drift_km} km")

    print("  Text geocoding: all sources failed — using GPS centre")
    return gps_lat, gps_lon, "gps"


# ── Internals ─────────────────────────────────────────────────────────────────

def _clean_for_geocoding(text: str) -> str:
    """Strip HTML, dates, and archive boilerplate from a free-text field."""
    text = re.sub(r"<[^>]+>", " ", text)    # remove HTML tags
    text = _DATES.sub("", text)
    text = _ARCHIVE_NOISE.sub("", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,;:-.")
    return text


def _nominatim_geocode(text: str, timeout: int = 6) -> Optional[tuple[float, float]]:
    """
    Geocode a single text string via Nominatim.

    Always sleeps ≥ 1.1 s after a request to honour Nominatim's
    1 request/second rate limit policy.

    Returns (lat, lon) on success, None otherwise.
    """
    cleaned = _clean_for_geocoding(text)
    if len(cleaned) < 5:
        return None

    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={"q": cleaned, "format": "json", "limit": 1},
            headers=_HEADERS,
            timeout=timeout,
        )
        resp.raise_for_status()
        results = resp.json()
        if results:
            return float(results[0]["lat"]), float(results[0]["lon"])
    except Exception:
        pass
    finally:
        time.sleep(1.1)

    return None