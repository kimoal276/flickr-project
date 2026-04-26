from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .encoder import EncoderModel, encode
from .geo_utils import bbox_from_center, haversine_km, parse_float
from .mapillary_client import (
    fetch_candidates,
    rank_candidates,
    rank_candidates_dual,
    rank_candidates_loftr,
)
from .text_geocoder import refine_center

load_dotenv()


# ── Configuration constants ───────────────────────────────────────────────────
# Adaptive parameters split by vision_label (YES = building, else = generic)

RADIUS_BUILDING: float = 0.5           # search radius (km)
RADIUS_DEFAULT:  float = 1.0

# GPS distance penalty weight (km⁻¹) — set to 0: pure visual matching.
# The archive GPS is often cluster-level (one decimal place), so using
# distance to rerank actively degrades quality.
DIST_WEIGHT_BUILDING: float = 0.0
DIST_WEIGHT_DEFAULT:  float = 0.0

# Minimum visual similarity for a confident match.  Below this, the top
# Mapillary candidate is reported but flagged as "low_confidence" — the
# building likely isn't in Mapillary's coverage for this location.
# Typical SigLIP values: 0.3–0.5 = unrelated, 0.5–0.7 = weak, 0.7+ = match.
MIN_CONFIDENCE: float = 0.65

# Dual-encoder fusion weight for DINOv2 (1-alpha goes to SigLIP)
DEFAULT_ALPHA: float = 0.7


# ── CSV loading ────────────────────────────────────────────────────────────────

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
    print(f"Loaded {len(selected)} photos  (label distribution: {dist})")
    return selected


# ── Core geolocator ───────────────────────────────────────────────────────────

def geolocate(
    photo: dict,
    radius_km: Optional[float] = None,
    mapillary_limit: int = 200,
    top_k: int = 5,
    use_text_geocoding: bool = True,
    use_dual_encoder: bool = False,
    alpha: float = DEFAULT_ALPHA,
    single_encoder: EncoderModel = EncoderModel.SIGLIP,
    matcher: str = "loftr",
) -> Optional[dict]:
    """
    Geolocate a single archive photo against Mapillary street-level imagery.

    Parameters
    ----------
    photo:              Photo dict as returned by load_photos().
    radius_km:          Override the adaptive search radius (km).
    mapillary_limit:    Maximum number of Mapillary candidates to retrieve.
    top_k:              Number of ranked matches included in the output.
    use_text_geocoding: Attempt Nominatim geocoding from title/description.
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
    is_building = vision == "YES"

    # ── per-type hyper-parameters ─────────────────────────────────────────────
    eff_radius  = radius_km if radius_km is not None else (
                      RADIUS_BUILDING if is_building else RADIUS_DEFAULT)
    dist_weight = DIST_WEIGHT_BUILDING if is_building else DIST_WEIGHT_DEFAULT

    _print_header(photo, eff_radius, dist_weight, use_dual_encoder, alpha)

    # ── step 1: text-geocode the search centre ────────────────────────────────
    center_lat, center_lon, center_src = gps_lat, gps_lon, "gps"
    if use_text_geocoding:
        center_lat, center_lon, center_src = refine_center(
            gps_lat, gps_lon, title=title, description=description
        )

    # ── step 2: fetch Mapillary candidates ────────────────────────────────────
    min_lat, min_lon, max_lat, max_lon = bbox_from_center(
        center_lat, center_lon, eff_radius
    )
    print(
        f"  Bbox ({center_src}): "
        f"({min_lat:.4f},{min_lon:.4f}) → ({max_lat:.4f},{max_lon:.4f})"
    )

    candidates = fetch_candidates(min_lat, min_lon, max_lat, max_lon,
                                  limit=mapillary_limit)
    print(f"  {len(candidates)} Mapillary candidates fetched")

    if not candidates:
        retry_radius = eff_radius * 2
        min_lat, min_lon, max_lat, max_lon = bbox_from_center(
            center_lat, center_lon, retry_radius
        )
        print(f"  No candidates — retrying with radius={retry_radius:.2f} km …")
        candidates = fetch_candidates(min_lat, min_lon, max_lat, max_lon,
                                      limit=mapillary_limit)
        print(f"  {len(candidates)} candidates after retry")
        if not candidates:
            print("  No Mapillary coverage found — skipping photo.")
            return None

    # ── step 3: visual ranking ────────────────────────────────────────────────
    if matcher == "loftr":
        # Geometric feature matching — the correct CV approach for
        # identifying the same building across photos.
        print(f"  Matching against {len(candidates)} candidates with LoFTR …")
        ranked = rank_candidates_loftr(
            archive_image=photo["image_url"],
            candidates=candidates,
        )
    elif use_dual_encoder:
        # Pass the raw image URL — dual encoder handles loading + preprocessing
        ranked = rank_candidates_dual(
            archive_image=photo["image_url"],
            candidates=candidates,
            ref_lat=gps_lat,
            ref_lon=gps_lon,
            distance_weight=dist_weight,
            alpha=alpha,
            preprocess_archive=True,    # grayscale + contrast norm for archive
        )
    else:
        # Pre-encode archive image, then rank candidates with the same model
        print(f"  Encoding archive image with {single_encoder.value} …")
        try:
            archive_vec = encode(
                photo["image_url"],
                model=single_encoder,
                preprocess_archive=True,
            )
            print(f"  Encoded  (dim={archive_vec.shape[0]})")
        except Exception as exc:
            print(f"  Encoding failed: {exc}")
            return None

        ranked = rank_candidates(
            archive_vec=archive_vec,
            candidates=candidates,
            ref_lat=gps_lat,
            ref_lon=gps_lon,
            distance_weight=dist_weight,
            candidate_model=single_encoder,
        )

    # ── step 4: pick top visual match ─────────────────────────────────────────
    # Pure visual matching: the top-ranked similar image IS the prediction.
    # No cluster re-ranking (which biases toward busy streets).
    # No centroid averaging (which pulls the prediction away from the actual
    # matched image).  The top Mapillary image's own coordinates are returned.
    if not ranked:
        return None

    top = ranked[0]
    pred_lat, pred_lon = top["lat"], top["lon"]
    distance_km = haversine_km(gps_lat, gps_lon, pred_lat, pred_lon)

    # Confidence check: is the visual match actually strong enough?
    confident = top["similarity"] >= MIN_CONFIDENCE

    # CSV-schema compatibility (these columns still exist in the output)
    top.setdefault("cluster_score", top["final_score"])
    top.setdefault("neighbors",     0)
    top.setdefault("support_score", 0.0)

    _print_result(pred_lat, pred_lon, distance_km, top, center_src,
                  archive_lat=gps_lat, archive_lon=gps_lon)

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
            "center_source":  center_src,
        },
        "top_match": {
            "mapillary_id":  top["mapillary_id"],
            "pred_lat":      pred_lat,
            "pred_lon":      pred_lon,
            "top_image_lat": top["lat"],
            "top_image_lon": top["lon"],
            "similarity":    top["similarity"],
            "dino_score":    top.get("dino_score"),
            "siglip_score":  top.get("siglip_score"),
            "thumb_url":     top["thumb_url"],
            "distance_km":   distance_km,
            "final_score":   top["final_score"],
            "cluster_score": top.get("cluster_score", top["final_score"]),
            "neighbors":     top.get("neighbors", 0),
            "support_score": top.get("support_score", 0.0),
        },
        "all_ranked": [
            {
                "mapillary_id": r["mapillary_id"],
                "lat":          r["lat"],
                "lon":          r["lon"],
                "similarity":   r["similarity"],
                "dino_score":   r.get("dino_score"),
                "siglip_score": r.get("siglip_score"),
                "distance_km":  r["distance_km"],
                "final_score":  r["final_score"],
                "cluster_score": r.get("cluster_score", r["final_score"]),
            }
            for r in ranked[:top_k]
        ],
    }


# ── Batch runner ──────────────────────────────────────────────────────────────

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
    use_text_geocoding: bool = True,
    use_dual_encoder: bool = False,
    alpha: float = DEFAULT_ALPHA,
    single_encoder: EncoderModel = EncoderModel.SIGLIP,
    matcher: str = "loftr",
    seed: Optional[int] = None,
) -> None:
    """
    Run geolocate() on a batch of photos from flickr_clusters.csv
    and write results to a CSV file.

    Parameters
    ----------
    csv_path:           Source CSV (default: flickr_clusters.csv).
    num_photos:         How many photos to process.
    out_csv:            Destination CSV path for results.
    vision_filter:      "YES" / "NO" / None — filter by vision_label.
    institution_filter: Case-insensitive substring match on source_dataset.
    photo_id:           Run on a single specific photo ID.
    radius_km:          Override adaptive radius for all photos (km).
    mapillary_limit:    Max Mapillary candidates fetched per photo.
    top_k:              Number of top matches saved per photo in the CSV.
    use_text_geocoding: Enable Nominatim text geocoding for search refinement.
    use_dual_encoder:   Use DINOv2 + SigLIP fusion (recommended).
    alpha:              DINOv2 weight in the fused score.
    single_encoder:     Backbone to use when use_dual_encoder=False.
    seed:               Random seed for reproducible photo selection.
    """
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
                use_text_geocoding=use_text_geocoding,
                use_dual_encoder=use_dual_encoder,
                alpha=alpha,
                single_encoder=single_encoder,
                matcher=matcher,
            )
        except Exception as exc:
            print(f"  Unhandled error: {exc}")
            result = None

        if result and result.get("top_match", {}).get("thumb_url"):
            comp = _save_comparison(
                photo=result["photo"],
                top=result["top_match"],
                out_dir=Path(out_csv).parent / "comparisons",
            )
            if comp:
                print(f"  Comparison → {comp}")

        rows.append(_to_row(photo, result))

    _write_csv(rows, out_path)
    _print_summary(rows)


# ── CSV output helpers ────────────────────────────────────────────────────────

_CSV_FIELDS = [
    "photo_id", "source_dataset", "title", "vision_label",
    "flickr_lat", "flickr_lon",
    "pred_lat", "pred_lon", "mapillary_id",
    "similarity", "dino_score", "siglip_score",
    "final_score", "cluster_score", "neighbors",
    "distance_km", "center_source", "status",
]

def _save_comparison(photo: dict, top: dict, out_dir: Path) -> Optional[Path]:
    """Side-by-side comparison JPEG. Verbose: prints what it's doing at each step."""
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

def _to_row(photo: dict, result: Optional[dict]) -> dict:
    base = {
        "photo_id":       photo["photo_id"],
        "source_dataset": photo["source_dataset"],
        "title":          photo["title"],
        "vision_label":   photo["vision_label"],
        "flickr_lat":     photo["latitude"],
        "flickr_lon":     photo["longitude"],
    }
    if result is None:
        return {**base, **{k: None for k in _CSV_FIELDS if k not in base},
                "status": "no_result"}

    p = result["photo"]
    t = result["top_match"]
    return {
        **base,
        "pred_lat":      t["pred_lat"],
        "pred_lon":      t["pred_lon"],
        "mapillary_id":  t["mapillary_id"],
        "similarity":    t["similarity"],
        "dino_score":    t.get("dino_score"),
        "siglip_score":  t.get("siglip_score"),
        "final_score":   t["final_score"],
        "cluster_score": t["cluster_score"],
        "neighbors":     t["neighbors"],
        "distance_km":   t["distance_km"],
        "center_source": p["center_source"],
        "status":        "ok",
    }


def _write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved → {path}")


def _print_summary(rows: list[dict]) -> None:
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
    refined  = sum(1 for r in rows
                   if r.get("center_source") in ("title", "description"))

    print(f"Successful       : {len(ok)}/{len(rows)}")
    print(f"Mean distance    : {mean_d:.3f} km")
    print(f"Median distance  : {median_d:.3f} km")
    print(f"< 500 m          : {u500}/{len(ok)}  ({100 * u500 // len(ok)}%)")
    print(f"< 1 km           : {u1km}/{len(ok)}  ({100 * u1km // len(ok)}%)")
    print(f"Text-geocoded    : {refined}/{len(rows)}")

    bld = [r for r in ok if r["vision_label"] == "YES"]
    non = [r for r in ok if r["vision_label"] != "YES"]
    if bld:
        print(f"Building mean    : "
              f"{sum(r['distance_km'] for r in bld)/len(bld):.3f} km  (n={len(bld)})")
    if non:
        print(f"Non-bld mean     : "
              f"{sum(r['distance_km'] for r in non)/len(non):.3f} km  (n={len(non)})")


# ── Pretty-print helpers ──────────────────────────────────────────────────────

def _print_header(
    photo: dict,
    radius: float,
    dist_weight: float,
    dual: bool,
    alpha: float,
) -> None:
    encoder_str = (
        f"DINOv2+SigLIP  (α={alpha})"
        if dual else
        f"single encoder"
    )
    sep = "─" * 62
    print(f"\n{sep}")
    print(f"  Photo    : {photo['photo_id']}")
    print(f"  Dataset  : {photo['source_dataset']}")
    print(f"  Title    : {photo['title'][:72]}")
    if photo.get("description"):
        print(f"  Desc     : {photo['description'][:72]}")
    print(f"  GPS      : {photo['latitude']:.5f}, {photo['longitude']:.5f}")
    print(f"  Vision   : {photo['vision_label']}  ({photo.get('vision_reason', '')})")
    print(f"  Radius   : {radius} km  |  dist_weight={dist_weight}")
    print(f"  Encoder  : {encoder_str}")


def _print_result(
    pred_lat: float,
    pred_lon: float,
    distance_km: float,
    top: dict,
    center_src: str,
    archive_lat: float,
    archive_lon: float,
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
    else:
        match_str = f"similarity={top['similarity']:.4f}"
        confident = top["similarity"] >= MIN_CONFIDENCE

    flag = "✓" if confident else "⚠ LOW CONFIDENCE —"

    print(f"\n  {flag} Best match")
    print(f"    Mapillary ID : {top['mapillary_id']}")
    print(f"    Coordinates  : {top['lat']:.6f}, {top['lon']:.6f}")
    print(f"    Distance     : {distance_km:.3f} km from archive GPS")
    print(f"    Match        : {match_str}")
    print(f"    Center src   : {center_src}")
    print(f"")
    print(f"  ── Open in Mapillary ──────────────────────────────────")
    print(f"    Street-level image : {mapillary_img}")
    print(f"    Location on map    : {mapillary_map}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
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
    p.add_argument("--no_text_geocoding", action="store_true",
                   help="Skip Nominatim text geocoding (faster, GPS only)")
    p.add_argument("--matcher",        default="loftr",
                   choices=["loftr", "siglip", "dual"],
                   help=(
                       "Visual matching method. "
                       "loftr  = LoFTR feature matching + RANSAC (default, correct for building ID). "
                       "siglip = SigLIP global embedding cosine similarity (fast, less accurate). "
                       "dual   = DINOv2 + SigLIP fusion (legacy)."
                   ))
    p.add_argument("--alpha",          type=float, default=DEFAULT_ALPHA,
                   help="DINOv2 weight in the fused similarity (only for --matcher dual)")
    p.add_argument("--seed",           type=int,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    use_dual   = args.matcher == "dual"
    single_enc = (
        EncoderModel.SIGLIP if args.matcher == "siglip"
        else EncoderModel.SIGLIP
    )

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
        use_text_geocoding=not args.no_text_geocoding,
        use_dual_encoder=use_dual,
        alpha=args.alpha,
        single_encoder=single_enc,
        matcher=args.matcher,
        seed=args.seed,
    )