from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .geo_utils import bbox_from_center, haversine_km, parse_float
from .mapillary_client import (
    fetch_candidates,
    rank_candidates_loftr,
)

load_dotenv()

SEARCH_RADIUS_KM = 0.5

# CSV loading 

def load_photos(
    csv_path: str = "flickr_clusters.csv",
    vision_filter: Optional[str] = None,
    institution_filter: Optional[str] = None,
    title_filter: Optional[str] = None,
    photo_id: Optional[str] = None,
    num_photos: int = 10,
    seed: Optional[int] = None,
) -> list[dict]:
    """
    Load and optionally filter photos from flickr_clusters.csv.

    Parameters
    ----------
    csv_path:           Path to the CSV file.
    vision_filter:      "YES" (buildings only), "NO" (non-buildings), None (all).
    institution_filter: Case-insensitive substring match on source_dataset.
    photo_id:           Pin to a single photo by its string ID.
    num_photos:         Maximum number of photos to return after shuffling.
    seed:               Random seed for reproducible photo selection.

    Returns
    -------
    List of photo dicts with keys:
        photo_id, title, description, date_taken, latitude, longitude,
        image_url, tags, source_dataset, vision_label, vision_reason
    """
    photos: list[dict] = []

    with open(csv_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            lat = parse_float(row.get("latitude"))
            lon = parse_float(row.get("longitude"))
            if lat is None or lon is None:
                continue

            image_url = (row.get("image_url") or "").strip()
            pid       = (row.get("id") or "").strip()
            if not pid or not image_url:
                continue

            vision_label = (row.get("vision_label") or "").strip().upper()
            source       = (row.get("source_dataset") or "").strip()

            if photo_id          and pid != str(photo_id):
                continue
            if vision_filter     and vision_label != vision_filter.upper():
                continue
            if institution_filter and institution_filter.lower() not in source.lower():
                continue
            if title_filter and title_filter.lower() not in (row.get("title") or "").lower():
                continue

            photos.append(
                {
                    "photo_id":       pid,
                    "title":          (row.get("title") or "").strip(),
                    "description":    (row.get("description") or "").strip(),
                    "date_taken":     (row.get("date_taken") or "").strip(),
                    "latitude":       lat,
                    "longitude":      lon,
                    "image_url":      image_url,
                    "tags":           (row.get("tags") or "").strip(),
                    "source_dataset": source,
                    "vision_label":   vision_label,
                    "vision_reason":  (row.get("vision_reason") or "").strip(),
                }
            )

    if photo_id:
        if not photos:
            raise ValueError(f"Photo ID {photo_id!r} not found in {csv_path}")
        print(f"Loaded 1 photo  (pinned id={photo_id})")
        return photos[:1]

    if seed is not None:
        random.seed(seed)
    random.shuffle(photos)
    selected = photos[:num_photos]

    dist: dict[str, int] = {}
    for p in selected:
        dist[p["vision_label"]] = dist.get(p["vision_label"], 0) + 1
    return selected


# Core geolocator 

def geolocate(
    photo: dict,
    radius_km: Optional[float] = None,
    mapillary_limit: int = 100,
    top_k: int = 5,
) -> Optional[dict]:
    """
    Geolocate a single archive photo against Mapillary street-level imagery.

    Parameters
    ----------
    photo:              Photo dict as returned by load_photos().
    radius_km:          Override the adaptive search radius (km).
    mapillary_limit:    Maximum number of Mapillary candidates to retrieve.
    top_k:              Number of ranked matches included in the output.
    single_encoder:     Backbone to use when use_dual_encoder=False.

    Returns
    -------
    Result dict or None on failure.

    Result schema
    -------------
    {
      "photo":     { photo_id, source_dataset, title, date_taken,
                     vision_label, vision_reason, original_lat, original_lon,
                     image_url, center_source },
      "top_match": { mapillary_id, pred_lat, pred_lon,
                     top_image_lat, top_image_lon,
                     similarity, dino_score, siglip_score,
                     distance_km, final_score, cluster_score,
                     neighbors, support_score, thumb_url },
      "all_ranked": [ { mapillary_id, lat, lon, similarity,
                         distance_km, final_score, cluster_score }, ... ]
    }
    """
    photo_id    = photo["photo_id"]
    title       = photo["title"]
    description = photo.get("description", "")
    vision      = photo["vision_label"]
    gps_lat     = photo["latitude"]
    gps_lon     = photo["longitude"]

    center_lat, center_lon, = gps_lat, gps_lon
    # step 1: fetch Mapillary candidates 
    min_lat, min_lon, max_lat, max_lon = bbox_from_center(
        center_lat, center_lon, SEARCH_RADIUS_KM
    )
    print(f"  Photo ID : {photo_id}")
    print(f"  Title    : {title}")
    print(f"  GPS      : {gps_lat:.5f}, {gps_lon:.5f}")
    print(f"  Bbox     : ({min_lat:.4f},{min_lon:.4f}) → ({max_lat:.4f},{max_lon:.4f})")

    candidates = fetch_candidates(min_lat, min_lon, max_lat, max_lon,
                                  limit=mapillary_limit)
    print(f"  {len(candidates)} Mapillary candidates fetched")

    if not candidates:
        retry_radius = SEARCH_RADIUS_KM * 2
        min_lat, min_lon, max_lat, max_lon = bbox_from_center(
            center_lat, center_lon, retry_radius
        )
        print(f"  No candidates — retrying with radius={retry_radius:.2f} km …")
        candidates = fetch_candidates(min_lat, min_lon, max_lat, max_lon,
                                      limit=mapillary_limit)
        print(f"  {len(candidates)} candidates after retry")
        if not candidates:
            print("No Mapillary coverage found — skipping photo.")
            return None

    # step 3: visual ranking 
    # Geometric feature matching — the correct CV approach for
    # identifying the same building across photos.
    print(f"  Matching against {len(candidates)} candidates with LoFTR …")
    ranked = rank_candidates_loftr(
    archive_image=photo["image_url"],
    candidates=candidates,
    )
    
    # step 4: pick top visual match 
    # Pure visual matching: the top-ranked similar image is the prediction.
    # No cluster re-ranking (which biases toward busy streets).
    # No centroid averaging (which pulls the prediction away from the actual
    # matched image).  The top Mapillary image's own coordinates are returned.
    if not ranked:
        return None

    top = ranked[0]
    # Reject if no real geometric match was found
    MIN_INLIERS = 6
    if top.get("inliers") is not None and top["inliers"] < MIN_INLIERS:
        print(f"  ✗ No confident match — best inlier count was {top['inliers']} "
          f"(Photo likely not in Mapillary coverage.")
        return None
    pred_lat, pred_lon = top["lat"], top["lon"]
    distance_km = haversine_km(gps_lat, gps_lon, pred_lat, pred_lon)
    

    print_result(distance_km, top)

    return {
        "photo": {
            "photo_id":       photo_id,
            "source_dataset": photo["source_dataset"],
            "title":          title,
            "date_taken":     photo["date_taken"],
            "vision_label":   vision,
            "vision_reason":  photo.get("vision_reason", ""),
            "original_lat":   gps_lat,
            "original_lon":   gps_lon,
            "image_url":      photo["image_url"],
        },
        "top_match": {
            "mapillary_id":  top["mapillary_id"],
            "pred_lat":      pred_lat,
            "pred_lon":      pred_lon,
            "top_image_lat": top["lat"],
            "top_image_lon": top["lon"],
            "similarity":    top["similarity"],
            "thumb_url":     top["thumb_url"],
            "distance_km":   distance_km,
            "final_score":   top["final_score"],
            "cluster_score": top.get("cluster_score", top["final_score"]),
        },
        "all_ranked": [
    {
        "mapillary_id":  r["mapillary_id"],
        "lat":           r["lat"],
        "lon":           r["lon"],
        "thumb_url":     r.get("thumb_url"),
        "similarity":    r["similarity"],
        "inliers":       r.get("inliers"),
        "match_total":   r.get("match_total"),
        "inlier_ratio":  r.get("inlier_ratio"),
        "distance_km":   r["distance_km"],
    }
    for r in ranked[:top_k]
],
    }


# Batch runner 

def batch_geolocate(
    csv_path: str = "flickr_clusters.csv",
    num_photos: int = 10,
    out_csv: str = "downloads/cluster_geolocator_results.csv",
    vision_filter: Optional[str] = None,
    institution_filter: Optional[str] = None,
    title_filter: Optional[str] = None,
    photo_id: Optional[str] = None,
    radius_km: Optional[float] = None,
    mapillary_limit: int = 50,
    top_k: int = 5,
    seed: Optional[int] = None,
) -> None:
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    photos = load_photos(
        csv_path=csv_path,
        vision_filter=vision_filter,
        institution_filter=institution_filter,
        title_filter=title_filter,
        photo_id=photo_id,
        num_photos=num_photos,
        seed=seed,
    )
    if not photos:
        print("No eligible photos found — check your filters.")
        return

    rows: list[dict] = []

    for i, photo in enumerate(photos, 1):
        print(f"\n[{i}/{len(photos)}]")
        try:
            result = geolocate(
                photo=photo,
                radius_km=radius_km,
                mapillary_limit=mapillary_limit,
                top_k=top_k,
            )
        except Exception as exc:
            print(f"  Unhandled error: {exc}")
            result = None

        # Save best-match comparison (now correctly inside the loop)
        if result and result.get("all_ranked"):
            cand = result["all_ranked"][0]
            if cand.get("thumb_url"):
                top_for_render = {
                    "mapillary_id": cand["mapillary_id"],
                    "thumb_url":    cand["thumb_url"],
                    "pred_lat":     cand["lat"],
                    "pred_lon":     cand["lon"],
                }
                comp = save_comparison(
                    photo=result["photo"],
                    top=top_for_render,
                    out_dir=Path(out_csv).parent / "comparisons",
                )
                if comp:
                    inl = cand.get("inliers", "?")
                    ratio = cand.get("inlier_ratio") or 0
                    print(f"  Comparison ({inl} inliers, ratio={ratio:.2f}) → {comp}")

        rows.append(to_row(photo, result))

    write_csv(rows, out_path)
    print_summary(rows)

def save_comparison(photo: dict, top: dict, out_dir: Path) -> Optional[Path]:
    """Side-by-side comparison JPEG,  prints what it's doing at each step."""
    from io import BytesIO
    from PIL import Image, ImageDraw, ImageFont
    import requests
    import traceback


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

    p = result["photo"]
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

    dists    = sorted(r["distance_km"] for r in ok)
    mean_d   = sum(dists) / len(dists)
    median_d = dists[len(dists) // 2]
    u500     = sum(1 for d in dists if d < 0.5)
    u1km     = sum(1 for d in dists if d < 1.0)

    print(f"Successful       : {len(ok)}/{len(rows)}")
    print(f"Mean distance    : {mean_d:.3f} km")
    print(f"Median distance  : {median_d:.3f} km")
    print(f"< 500 m          : {u500}/{len(ok)}  ({100 * u500 // len(ok)}%)")
    print(f"< 1 km           : {u1km}/{len(ok)}  ({100 * u1km // len(ok)}%)")

    bld = [r for r in ok if r["vision_label"] == "YES"]
    non = [r for r in ok if r["vision_label"] != "YES"]
    if bld:
        print(f"Building mean    : "
              f"{sum(r['distance_km'] for r in bld)/len(bld):.3f} km  (n={len(bld)})")
    if non:
        print(f"Non-bld mean     : "
              f"{sum(r['distance_km'] for r in non)/len(non):.3f} km  (n={len(non)})")


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


# CLI 

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Geolocate historical archive photos from flickr_clusters.csv "
            "against present-day Mapillary street-level imagery.\n\n"
            "Visual matching uses a DINOv2 + SigLIP dual-encoder by default. "
            "DINOv2 captures structural/geometric features robust to the "
            "archive-to-street domain gap; SigLIP adds semantic context."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv",            default="flickr_clusters.csv")
    p.add_argument("--num_photos",     type=int,   default=10)
    p.add_argument("--out_csv",        default="downloads/cluster_geolocator_results.csv")
    p.add_argument("--photo_id",       default=None, help="Pin to a single photo ID")
    p.add_argument("--only_buildings", action="store_true",
                   help="Shorthand for --vision_filter YES")
    p.add_argument("--vision_filter",  default=None, choices=["YES", "NO"])
    p.add_argument("--institution",    default=None,
                   help="Substring match on source_dataset column")
    p.add_argument("--title",          default=None,
                   help="Substring match on title column (case-insensitive)")
    p.add_argument("--radius_km",      type=float, default=None,
                   help="Override adaptive search radius (km)")
    p.add_argument("--mapillary_limit",type=int,   default=50)
    p.add_argument("--top_k",          type=int,   default=5)
    p.add_argument("--seed",           type=int,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    batch_geolocate(
        csv_path=args.csv,
        num_photos=args.num_photos,
        out_csv=args.out_csv,
        vision_filter="YES" if args.only_buildings else args.vision_filter,
        institution_filter=args.institution,
        title_filter=args.title,
        photo_id=args.photo_id,
        radius_km=args.radius_km,
        mapillary_limit=args.mapillary_limit,
        top_k=args.top_k,
        seed=args.seed,
    )