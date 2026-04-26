"""text_geocoder.py — Nominatim with visible errors and distance-based selection."""

from __future__ import annotations

import re
import time
from typing import Optional

import requests

from .geo_utils import haversine_km

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
# IMPORTANT: replace the email below with yours. Nominatim requires a real UA.
_HEADERS = {"User-Agent": "flico-thesis/1.0 (nathanael.ambert@epfl.ch)"}

# Conservative noise regex — DO NOT strip building-type words like
# "library/museum/university/institute"; those are often the actual subject.
_ARCHIVE_NOISE = re.compile(
    r"\b(ca\.?|circa|c\.|photograph(s|ed)?|photo|image|view|scene(s)?|"
    r"negatives?|glass\s+plates?|lantern\s+slides?|album|"
    r"reference\s+n[\w.\-/]*|creator|location|unidentified)\b",
    flags=re.IGNORECASE,
)
_DATES = re.compile(
    r"\b(ca\.?\s*)?\d{4}s?\b|\b\d+(st|nd|rd|th)\b|\bbetween\b",
    flags=re.IGNORECASE,
)


def refine_center(
    gps_lat: float,
    gps_lon: float,
    title: str,
    description: str = "",
    max_drift_km: float = 50.0,
) -> tuple[float, float, str]:
    """
    Geocode title/description with Nominatim, biased to the Flickr GPS.
    Picks the closest result within max_drift_km; otherwise falls back to GPS.
    Always returns a tuple — never None.
    """
    for text, label in [(title, "title"), (description, "description")]:
        if not text or len(text.strip()) < 5:
            continue

        cleaned = _clean_for_geocoding(text)
        if len(cleaned) < 5:
            continue

        # Build progressively simpler query variants.
        variants = [cleaned]
        if "," in cleaned:
            parts = [p.strip() for p in cleaned.split(",") if p.strip()]
            if len(parts) >= 2:
                variants.append(f"{parts[0]}, {parts[-1]}")
            if parts:
                variants.append(parts[0])
            if len(parts) >= 2:
                variants.append(parts[-1])
        # Deduplicate while preserving order.
        seen: set[str] = set()
        variants = [v for v in variants if not (v in seen or seen.add(v))]

        for q in variants:
            candidates = _nominatim_geocode(q, gps_lat=gps_lat, gps_lon=gps_lon)
            if not candidates:
                continue

            within = [
                (la, lo, haversine_km(gps_lat, gps_lon, la, lo))
                for la, lo in candidates
            ]
            within = [(la, lo, d) for la, lo, d in within if d <= max_drift_km]

            if within:
                nom_lat, nom_lon, drift = min(within, key=lambda x: x[2])
                print(f"  Text geocoding ({label}): '{q}' → "
                      f"({nom_lat:.5f}, {nom_lon:.5f})  drift={drift:.2f} km  "
                      f"(picked closest of {len(candidates)} hits) ✓")
                return nom_lat, nom_lon, label

            best_drift = min(haversine_km(gps_lat, gps_lon, la, lo)
                             for la, lo in candidates)
            print(f"  Text geocoding ({label}): '{q}' → "
                  f"{len(candidates)} hits, closest {best_drift:.1f} km > "
                  f"{max_drift_km} — rejected")

    print("  Text geocoding: all sources failed — using GPS centre")
    return gps_lat, gps_lon, "gps"


def _clean_for_geocoding(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)        # strip HTML
    text = _DATES.sub("", text)
    text = _ARCHIVE_NOISE.sub("", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,;:-.")
    return text


def _nominatim_geocode(
    text: str,
    gps_lat: Optional[float] = None,
    gps_lon: Optional[float] = None,
    timeout: int = 10,
) -> list[tuple[float, float]]:
    """Return up to 10 candidate (lat, lon) pairs. Errors are logged, not silenced."""
    if len(text) < 5:
        return []

    params: dict = {"q": text, "format": "json", "limit": 10}
    if gps_lat is not None and gps_lon is not None:
        params["viewbox"] = (
            f"{gps_lon - 0.5},{gps_lat + 0.5},"
            f"{gps_lon + 0.5},{gps_lat - 0.5}"
        )
        params["bounded"] = 0

    try:
        resp = requests.get(NOMINATIM_URL, params=params,
                            headers=_HEADERS, timeout=timeout)
        if resp.status_code != 200:
            print(f"  [nominatim {resp.status_code}] '{text}' → "
                  f"{resp.text[:120]}")
            return []
        results = resp.json() or []
        return [(float(r["lat"]), float(r["lon"])) for r in results]
    except requests.RequestException as e:
        print(f"  [nominatim network error] '{text}': {e}")
        return []
    except (ValueError, KeyError) as e:
        print(f"  [nominatim parse error] '{text}': {e}")
        return []
    finally:
        time.sleep(1.1)