"""
geo_utils.py
------------
Shared utility functions used across the pipeline.

Functions
---------
haversine_km        Distance between two (lat, lon) points in kilometres.
bbox_from_center    Bounding box around a (lat, lon) centre with a given radius.
parse_float         Safe float parser that maps "0" / "" / None → None.
"""
import csv
from pathlib import Path
from typing import Optional, Union
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import requests
import traceback
import math


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

# visualise results

def save_comparison(photo: dict, top: dict, out_dir: Path) -> Optional[Path]:
    """Side-by-side comparison JPEG,  prints what it's doing at each step."""

    try:
        archive_resp = requests.get(photo["image_url"], timeout=30)
        archive_resp.raise_for_status()
        mapil_resp = requests.get(top["thumb_url"], timeout=30)
        mapil_resp.raise_for_status()

        archive = Image.open(BytesIO(archive_resp.content)).convert("RGB")
        mapil   = Image.open(BytesIO(mapil_resp.content)).convert("RGB")

        target_h = 600
        a_w = int(archive.width * target_h / archive.height)
        m_w = int(mapil.width   * target_h / mapil.height)
        archive = archive.resize((a_w, target_h), Image.LANCZOS)
        mapil   = mapil.resize((m_w, target_h),   Image.LANCZOS)

        gap, cap_h = 16, 70
        canvas = Image.new("RGB", (a_w + m_w + gap, target_h + cap_h), "white")
        canvas.paste(archive, (0, cap_h))
        canvas.paste(mapil,   (a_w + gap, cap_h))

        draw = ImageDraw.Draw(canvas)
        try:
            font_big   = ImageFont.truetype("arial.ttf", 16)
            font_small = ImageFont.truetype("arial.ttf", 13)
        except Exception:
            font_big = font_small = ImageFont.load_default()

        # Header
        institution = (photo.get('source_dataset') or '').strip()
        date_str    = (photo.get('date_taken')     or '').strip()
        title_disp  = (photo.get('title')          or '').strip()
        if len(title_disp) > 90:
            title_disp = title_disp[:87].rstrip() + "…"
        meta_bits = [b for b in (institution, date_str) if b]
        line1 = title_disp + (' — ' + ', '.join(meta_bits) if meta_bits else '')
        line2 = (f"Located via Mapillary at "
                 f"{top.get('pred_lat', 0):.5f}, {top.get('pred_lon', 0):.5f}")

        draw.text((10, 8),  line1, fill="black", font=font_big)
        draw.text((10, 32), line2, fill="#444",  font=font_small)
        draw.line([(10, 56), (a_w + m_w + gap - 10, 56)], fill="#bbb", width=1)

        out_dir.mkdir(parents=True, exist_ok=True)

        safe_title = (photo.get('title') or '').strip()
        for ch in '<>:"/\\|?*\n\r\t':
            safe_title = safe_title.replace(ch, '')
        safe_title = ' '.join(safe_title.split())
        if len(safe_title) > 80:
            safe_title = safe_title[:80].rstrip()
        if not safe_title:
            safe_title = 'untitled'

        out_path = out_dir / f"{safe_title}_{photo['photo_id']}.jpg"
        canvas.save(out_path, quality=85)
        return out_path

    except Exception as exc:
        print(f"  [comparison] FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return None

CSV_FIELDS = [
    "photo_id", "source_dataset", "title", "vision_label",
    "flickr_lat", "flickr_lon",
    "pred_lat", "pred_lon", "mapillary_id",
    "similarity","final_score", "distance_km", "status",
]

def to_row(photo: dict, result: Optional[dict]) -> dict:
    base = {
        "photo_id":       photo["photo_id"],
        "source_dataset": photo["source_dataset"],
        "title":          photo["title"],
        "vision_label":   photo["vision_label"],
        "flickr_lat":     photo["latitude"],
        "flickr_lon":     photo["longitude"],
        "status": "ok"
    }
    if result is None:
        return {**base, **{k: None for k in CSV_FIELDS if k not in base},
                "status": "no_result"}
    
    t = result["top_match"]
    return {
        **base,
        "pred_lat":      t["pred_lat"],
        "pred_lon":      t["pred_lon"],
        "mapillary_id":  t["mapillary_id"],
        "similarity":    t["similarity"],
        "distance_km":   t["distance_km"],
    }


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved → {path}")


def print_summary(rows: list[dict]) -> None:
    ok  = [r for r in rows if r["status"] == "ok" and r["distance_km"] is not None]
    sep = "=" * 62
    print(f"\n{sep}")
    if not ok:
        print("No successful geolocation results.")
        return
# print helpers 

def print_result(
    distance_km: float,
    top: dict,
) -> None:
    mapillary_img = f"https://www.mapillary.com/app/?image_key={top['mapillary_id']}"
    mapillary_map = (
        f"https://www.mapillary.com/app/"
        f"?lat={top['lat']:.6f}&lng={top['lon']:.6f}&z=17"
        f"&image_key={top['mapillary_id']}"
    )

    # LoFTR path: show inlier count
    if top.get("inliers") is not None:
        match_str = (
            f"inliers={top['inliers']}/{top.get('match_total', '?')} "
            f"(ratio={top.get('inlier_ratio', 0):.2f})"
        )
    confident = top["inliers"] >= 15   # ≥15 geometric inliers = strong match
    flag = "✓" if confident else "⚠ LOW CONFIDENCE"

    print(f"\n  {flag} Best match")
    print(f"    Mapillary ID : {top['mapillary_id']}")
    print(f"    Coordinates  : {top['lat']:.6f}, {top['lon']:.6f}")
    print(f"    Distance     : {distance_km:.3f} km from archive GPS")
    print(f"    Match        : {match_str}")
    print(f"  Open in Mapillary")
    print(f"    Street-level image : {mapillary_img}")
    print(f"    Location on map    : {mapillary_map}")
