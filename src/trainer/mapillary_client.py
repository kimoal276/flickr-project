import os
from dotenv import load_dotenv
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import requests
import numpy as np
import math
import random
import heapq
import mercantile
import vt2geojson.tools
from mapbox_vector_tile import decode

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
VECTOR_TILE_URL = "https://tiles.mapillary.com/maps/vtp/mly1_public/2/{z}/{x}/{y}"


@dataclass
class MapillaryPicture:
    id: int
    lat: float
    lon: float
    compass_angle: int
    captured_at: int
    pic_url: str

@dataclass
class TileStats:
    count: int = 0
    heading_bins: set = None

    def __post_init__(self):
        if self.heading_bins is None:
            self.heading_bins = set()

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
    
def slow_stupid_monument_candidates(longitude: float, latitude:float):
    """
    Find candidates pictures around a point in a 10km radius:
    - Identify the 10 highest density 100m x 100m tiles that are likely 
      corresponding to famous monuments
    - for each monument tile, sample a picture pointing in every bin direction
      to cover all angles of the monument
    """
    SEARCH_TILE_SIDE_KM = 0.1 #size of a tile
    N_TILE_RADIUS = 50 #search radius as a number of tile
    N_BEST = 10 
    DIVERSITY_DEGREE = 15
    BIN_SIZE_DEGREE = 45

    deg_lat =  SEARCH_TILE_SIDE_KM / 111.0
    deg_lon = SEARCH_TILE_SIDE_KM / (111.0 * math.cos(math.radians(latitude)))

    params = {
        "access_token": _get_token(),
        "fields": "id,geometry,captured_at,compass_angle,thumb_1024_url",
        "bbox": (
            f"{longitude - (N_TILE_RADIUS+0.5)*deg_lon},{latitude - (N_TILE_RADIUS+0.5)*deg_lat},"
            f"{longitude + (N_TILE_RADIUS+0.5)*deg_lon},{latitude + (N_TILE_RADIUS+0.5)*deg_lat}"
        ),
        "limit": 10000,
    }
    try:
        resp = requests.get(f"{MAPILLARY_BASE}/images", params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        tile_candidates = [
            MapillaryPicture(
                id=item.get("id"),
                lon=item.get("geometry", {}).get("coordinates", [None, None])[0],
                lat=item.get("geometry", {}).get("coordinates", [None, None])[1],
                captured_at=item.get("captured_at"),
                compass_angle=item.get("compass_angle"),
                pic_url=item.get("thumb_1024_url"),
            )
            for item in data
        ]
    except requests.RequestException:
        return []
    if not tile_candidates:
        return []

    pics_per_tile = defaultdict(list)

    for candidate in tile_candidates:
        lon_id = int((candidate.lon - longitude) / deg_lon)
        lat_id = int((candidate.lat - latitude) / deg_lat)
        pics_per_tile[(lon_id, lat_id)].append(candidate)

    def tile_score(candidates):
        bins = {
            int((c.compass_angle % 360) // DIVERSITY_DEGREE)
            for c in candidates
            if c.compass_angle is not None
        }

        return len(candidates) * max(len(bins), 1)

    best_tiles = sorted(
        pics_per_tile.items(),
        key=lambda kv: tile_score(kv[1]),
        reverse=True,
    )[:N_BEST]

    final_candidates = []
    for id_tuple, candidates in best_tiles.items():
        angle_bins = defaultdict(list)
        for candidate in candidates:
            angle = candidate.compass_angle % 360
            bin_idx = int(angle // BIN_SIZE_DEGREE)
            angle_bins[bin_idx].append(candidate)
        for bin_candidates in angle_bins.values():
            final_candidates.append(random.choice(bin_candidates))

    return final_candidates

def _fetch_vector_tile(z: int, x: int, y: int):
    url = VECTOR_TILE_URL.format(z=z,x=x,y=y, token=_get_token())
    params = {"access_token": _get_token()}
    resp = requests.get(url, params=params, timeout=30)
    print(resp.status_code)
    print(resp.headers.get("Content-Type"))
    print(resp.headers.get("Content-Encoding"))
    print(resp.content[:100])
    resp.raise_for_status()
    return decode(resp.content)

def _bbox_around_point(longitude, latitude, radius_km):
    deg_lat = radius_km / 111.0
    cos_lat = max(0.01, abs(math.cos(math.radians(latitude))))
    deg_lon = radius_km / (111.0 * cos_lat)
    return (longitude - deg_lon, latitude - deg_lat, longitude + deg_lon, latitude + deg_lat)

def smart_monument_candidates(longitude: float, latitude: float):
    """
    Optimized monument-oriented candidate sampler.

    Strategy:
    1. Use Mapillary vector tiles to cheaply identify dense regions
    2. Only fetch image metadata inside the best regions
    3. Preserve directional diversity
    """

    SEARCH_TILE_SIDE_KM = 0.1
    N_TILE_RADIUS = 50
    N_BEST = 10
    DIVERSITY_DEGREE = 15
    BIN_SIZE_DEGREE = 45
    SEARCH_RADIUS_KM = SEARCH_TILE_SIDE_KM * N_TILE_RADIUS

    # ==========================================================
    # PHASE 1 — COARSE HOTSPOT DISCOVERY VIA VECTOR TILES
    # ==========================================================

    ZOOM = 14 # z14 ≈ neighborhood-level density estimation
    bbox = _bbox_around_point(longitude, latitude, SEARCH_RADIUS_KM,)
    tiles = list(mercantile.tiles(*bbox, zooms=[ZOOM]))
    print(tiles[0])
    coarse_stats = defaultdict(TileStats)
    for tile in tiles:
        try:
            decoded = _fetch_vector_tile(tile.z, tile.x,tile.y)
        except Exception as e:
            print(e)
            continue
        layer = (decoded.get("image") or decoded.get("images")or {})
        features = layer.get("features", [])
        print(len(features), end=' ')
        for feat in features:
            geom = feat.get("geometry")
            props = feat.get("properties", {})
            if not geom:
                continue
            coords = geom.get("coordinates")
            if not coords:
                continue
            lon, lat = coords
            dx = lon - longitude
            dy = lat - latitude

            # original 100m tile quantization preserved
            cos_lat = max(0.01, abs(math.cos(math.radians(latitude))))
            deg_lat = SEARCH_TILE_SIDE_KM / 111.0
            deg_lon = SEARCH_TILE_SIDE_KM / (111.0 * cos_lat)

            lon_id = int(dx / deg_lon)
            lat_id = int(dy / deg_lat)

            heading = props.get("compass_angle")

            stat = coarse_stats[(lon_id, lat_id)]

            stat.count += 1

            if heading is not None:
                stat.heading_bins.add(
                    int((heading % 360) // DIVERSITY_DEGREE)
                )

    if not coarse_stats:
        return []

    # ==========================================================
    # PHASE 2 — SCORE HOTSPOTS
    # ==========================================================

    def tile_score(stat: TileStats):
        diversity = max(len(stat.heading_bins), 1)

        # favors:
        # - dense regions
        # - many viewing directions
        return stat.count * diversity

    best_tiles = heapq.nlargest(
        N_BEST,
        coarse_stats.items(),
        key=lambda kv: tile_score(kv[1]),
    )

    # ==========================================================
    # PHASE 3 — FETCH DETAILED IMAGES ONLY FOR BEST TILES
    # ==========================================================

    final_candidates = []

    cos_lat = max(
        0.01,
        abs(math.cos(math.radians(latitude))),
    )

    deg_lat = SEARCH_TILE_SIDE_KM / 111.0
    deg_lon = SEARCH_TILE_SIDE_KM / (111.0 * cos_lat)

    for (lon_id, lat_id), _ in best_tiles:

        center_lon = longitude + lon_id * deg_lon
        center_lat = latitude + lat_id * deg_lat

        tile_bbox = (
            center_lon - 0.5 * deg_lon,
            center_lat - 0.5 * deg_lat,
            center_lon + 0.5 * deg_lon,
            center_lat + 0.5 * deg_lat,
        )

        params = {
            "access_token": _get_token(),
            "fields": (
                "id,"
                "geometry,"
                "compass_angle,"
                "captured_at,"
                "thumb_1024_url"
            ),
            "bbox": ",".join(map(str, tile_bbox)),
            "limit": 500,
        }

        try:
            resp = requests.get(
                f"{MAPILLARY_BASE}/images",
                params=params,
                timeout=30,
            )

            resp.raise_for_status()

            data = resp.json().get("data", [])

        except Exception:
            continue

        angle_bins = defaultdict(list)

        for item in data:

            geometry = item.get("geometry")

            if not geometry:
                continue

            coords = geometry.get("coordinates")

            if not coords:
                continue

            heading = item.get("compass_angle")

            if heading is None:
                continue

            candidate = MapillaryPicture(
                id=item.get("id"),
                lon=coords[0],
                lat=coords[1],
                captured_at=item.get("captured_at"),
                compass_angle=heading,
                pic_url=item.get("thumb_1024_url"),
            )

            bin_idx = int((heading % 360) // BIN_SIZE_DEGREE)

            angle_bins[bin_idx].append(candidate)

        # preserve directional diversity
        for bin_candidates in angle_bins.values():
            final_candidates.append(
                random.choice(bin_candidates)
            )

    return final_candidates