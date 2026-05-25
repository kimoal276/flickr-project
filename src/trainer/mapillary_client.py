"""
mapillary_client.py

Fetches Mapillary street-level image candidates and ranks them against a
historical archive photo using visual feature matching.

Public API

sample_candidate
"""

import os

import numpy as np
import requests
from dotenv import load_dotenv
from dataclasses import dataclass
import math
import random
from typing import Optional
from collections import defaultdict
from .geo_utils import haversine_km

load_dotenv()

# Auth 
def _get_token() -> str:
    token = os.getenv("MAPILLARY_ACCESS_TOKEN")
    if not token:
        raise ValueError("MAPILLARY_ACCESS_TOKEN must be set in .env")
    return token
from dataclasses import dataclass
import requests

MAPILLARY_BASE = "https://graph.mapillary.com"

@dataclass
class MapillaryPicture:
    id: int
    lat: float
    lon: float
    compass_angle: int
    captured_at: int
    pic_url: str

@dataclass
class MapillarySampler:
    lon: float
    lat: float
    candidates: list
    st_km: float = 0.05

    def sample_candidates(self) -> Optional[MapillaryPicture]:
            """
            Return one candidate picture, drawn at random with a 2-D Gaussian
            weighting centred on (self.lat, self.lon).
            """
            if not self.candidates:
                return None

            # Weight each candidate by the unnormalised Gaussian PDF at its location.
            # random.choices normalises internally, so we don't need to divide.
            weights = [
                math.exp(-0.5 * (haversine_km(self.lat, self.lon, c.lat, c.lon) / self.st_km) ** 2)
                for c in self.candidates
            ]
            return random.choices(self.candidates, weights=weights, k=1)[0]

def create_sampler(longitude: float, latitude: float, st_km: float = 0.05)-> Optional[MapillarySampler]:
    """creates a Sampler"""
    token = _get_token()
    deg_lat = 5 * st_km / 111.0
    deg_lon = 5 * st_km / (111.0 * math.cos(math.radians(latitude)))
    params = {
        "access_token": token,
        "fields": "id,geometry,captured_at,compass_angle,thumb_1024_url",
        "bbox": (
            f"{longitude - deg_lon},{latitude - deg_lat},"
            f"{longitude + deg_lon},{latitude + deg_lat}"
        ),
        "limit": 1000,
    }
    try:
        resp = requests.get(f"{MAPILLARY_BASE}/images", params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        candidates = [
            MapillaryPicture(
                id=item.get("id"),
                lat=item.get("geometry", {}).get("coordinates", [None, None])[1],
                lon=item.get("geometry", {}).get("coordinates", [None, None])[0],
                captured_at=item.get("captured_at"),
                compass_angle=item.get("compass_angle"),
                pic_url=item.get("thumb_1024_url"),
            )
            for item in data
        ]
        if len(candidates) > 0:
            return MapillarySampler(longitude, latitude, candidates, st_km)
    except requests.RequestException:
        return None
    return None

def smart_angle_candidates(longitude: float, latitude:float, TILE_SIDE_KM=0.1, BIN_SIZE=15):
    """
    Returns candidates sampled spatially and by viewing angle.

    For each of the 9 surrounding tiles:
    - fetch Mapillary pictures
    - bin by compass angle (15° bins)
    - sample 1 random picture per non-empty bin
    """
    token = _get_token()

    deg_lat =  TILE_SIDE_KM / 111.0
    deg_lon = TILE_SIDE_KM / (111.0 * math.cos(math.radians(latitude)))

    tile_centers = [
        (
            longitude + deg_lon * lon_offset,
            latitude + deg_lat * lat_offset,
        )
        for lon_offset, lat_offset in [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),  (0, 0),  (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    ]
    candidates = []
    
    for center_lon, center_lat in tile_centers:
        params = {
            "access_token": token,
            "fields": "id,geometry,captured_at,compass_angle,thumb_1024_url",
            "bbox": (
                f"{center_lon - 0.5*deg_lon},{center_lat - 0.5*deg_lat},"
                f"{center_lon + 0.5*deg_lon},{center_lat + 0.5*deg_lat}"
            ),
            "limit": 1000,
        }
        try:
            resp = requests.get(f"{MAPILLARY_BASE}/images", params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            tile_candidates = [
                MapillaryPicture(
                    id=item.get("id"),
                    lat=item.get("geometry", {}).get("coordinates", [None, None])[1],
                    lon=item.get("geometry", {}).get("coordinates", [None, None])[0],
                    captured_at=item.get("captured_at"),
                    compass_angle=item.get("compass_angle"),
                    pic_url=item.get("thumb_1024_url"),
                )
                for item in data
            ]
            if not tile_candidates:
                continue
            angle_bins = defaultdict(list)
            for candidate in tile_candidates:
                angle = candidate.compass_angle % 360
                bin_idx = int(angle // BIN_SIZE)
                angle_bins[bin_idx].append(candidate)
            for bin_candidates in angle_bins.values():
                candidates.append(random.choice(bin_candidates))
        except requests.RequestException:
            continue
    
    return candidates
    

"""""
# Fetching 
def fetch_candidates(min_lat, min_lon, max_lat, max_lon, limit=100):
    token = _get_token()
    TILE_SIZE = 0.005
    PER_TILE  = 25                       
    seen, candidates = set(), []

    lat = min_lat
    while lat < max_lat:
        lon = min_lon
        while lon < max_lon:
            tile = (lon, lat,
                    min(lon + TILE_SIZE, max_lon),
                    min(lat + TILE_SIZE, max_lat))
            params = {
                "access_token": token,
                "fields":       "id,geometry,thumb_1024_url,captured_at",
                "bbox":         f"{tile[0]},{tile[1]},{tile[2]},{tile[3]}",
                "limit":        1000,
            }
            try:
                resp = requests.get(f"{MAPILLARY_BASE}/images",
                                    params=params, timeout=60)
                resp.raise_for_status()
                for item in resp.json().get("data", []):
                    mid = item.get("id")
                    coords = item.get("geometry", {}).get("coordinates", [])
                    thumb  = item.get("thumb_1024_url")
                    if mid and mid not in seen and thumb and len(coords) >= 2:
                        seen.add(mid)
                        candidates.append({"mapillary_id": mid,
                                           "lon": coords[0], "lat": coords[1],
                                           "thumb_url": thumb})
            except requests.HTTPError:
                pass
            lon += TILE_SIZE
        lat += TILE_SIZE

    # Then subsample to `limit` *spatially* — keep candidates near the
    # bbox center first so we don't blow LoFTR budget on the periphery.
    cx = (min_lat + max_lat) / 2
    cy = (min_lon + max_lon) / 2
    candidates.sort(key=lambda c: (c["lat"]-cx)**2 + (c["lon"]-cy)**2)
    return candidates[:limit]
"""""
