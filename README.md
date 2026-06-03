# flickr-matching

DataFrame-oriented Flickr → Mapillary geolocation matcher, designed to slot into
Nathanael's DB-backed `flickr-commons-metadata` project. Consumes building-positive
photographs from [`flickr-filtering`](../flickr-filtering) and re-anchors each one to a
modern Mapillary street-level panorama, emitting a refined coordinate and a confidence
score.

## Structure

```
flickr-matching/
├── run_csv_mapillary.py     ← local CSV runner (development entry point)
├── visualize_matches.py     ← side-by-side Flickr vs Mapillary comparison JPEGs
├── simple_image_cache.py    ← SimpleImageCache (.get / .get_images)
├── requirements.txt
└── src/trainer/
    ├── matching.py          ← public API: find_matches(df, cache) → df  (+ tuning constants)
    ├── mapillary_client.py  ← smart_angle_candidates, MapillarySampler, create_sampler
    ├── building_matcher.py  ← load_matcher, compute_loftr_matches, compute_ransac_inliers, to_gray_tensor
    └── geo_utils.py         ← haversine_km, write_csv, save_comparison
```

```python
from src.trainer.matching import find_matches
```

`find_matches(df, cache)` is the deliverable. The same function is consumed unchanged by
the local CSV runner (development) and by Nathanael's production trainer via
`pipeline.fast_mapillary`. The bridge between the two is the output schema, not a
translation layer — a row written by either path is indistinguishable.

## Pipeline order

This matcher runs **after** `flickr-filtering`, on confirmed buildings only:

1. **Filter** (upstream, `flickr-filtering`) — embed → pre-filter → `label_buildings()` → `_cluster()`, producing rows with `is_building = True` and a Flickr coordinate.
2. **Match** — `find_matches()` adds the eight `mapillary_*` columns (slow, resumable).

Match **building-positive rows only.** Each query costs a LoFTR pass against every
candidate, so running the matcher on non-building photos just burns the (dominant) LoFTR
budget. Internally `find_matches` runs four stages per query: Mapillary candidate
sampling (`smart_angle_candidates`, a 3×3 tile grid with 15° compass-angle bins) →
LoFTR detector-free matching (Kornia `KF.LoFTR`, `outdoor` checkpoint) → MAGSAC++ RANSAC
verification (`cv2.USAC_MAGSAC`, τ = 3.0 px) → rank by inlier count.

## Rate limits and resume

Candidate sampling hits the Mapillary Graph API: `smart_angle_candidates` issues **9
`/images` requests per query** (the dominant network cost). Image fetches — both the
Flickr query image and the Mapillary thumbnails — go through plain HTTP with 20–30 s
timeouts. A failed fetch fails that one query but **never interrupts the batch**.

- **No Mapillary coverage** in the query bbox → `status = no_result`, all match columns `NULL` (so `p_match IS NULL` cleanly separates an absent match from a weak one).

**Resume model differs by path:**

- **Production (DB):** pass only rows `WHERE p_match IS NULL`; `update_ml_photo` commits each row as it finishes, so the next run picks up where the last left off.
- **Local CSV:** each completed `id` is flushed and `fsync`-ed to disk per row and skipped on re-run; a half-written final row from a hard cut is tolerated on reload.

There is no sleeping or blocking inside the library — progress across rate limits is
**manual reruns**.

## Usage example (DataFrames)

You must provide a **cache** object exposing `.get(url)` and `.get_images(urls, ...)`
that return PIL RGB images or `None` (`SimpleImageCache` does this; Nathanael's project
supplies an equivalent for proxy / download handling).

```python
from src.trainer.matching import find_matches
from simple_image_cache import SimpleImageCache

cache = SimpleImageCache("image_cache")

# df: building-positive rows with id, url_o (Flickr image URL), latitude, longitude
matched = find_matches(df, cache)

# matched has the eight mapillary_* columns added.
# status == "ok"        → a candidate was matched and geometrically verified
# status == "no_result" → no Mapillary coverage; match columns are NULL
```

Local CSV runner (resumable, appends to the output):

```bash
python run_csv_mapillary.py                       # input flickr_clusters.csv → mapillary_matches.csv
python run_csv_mapillary.py --limit 50 --precache # debug a small batch, pre-download query images
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `flickr_clusters.csv` | Building-positive rows (`id`, `image_url`, `latitude`, `longitude`) |
| `--output` | `mapillary_matches.csv` | Output CSV (matches appended) |
| `--cache-dir` | `image_cache` | Persistent image cache directory |
| `--limit N` | none | Process at most N rows |
| `--precache` | off | Download all query images to disk first, then match |

Comparison images for manual review:

```bash
python visualize_matches.py                  # confident matches (p_match >= 0.05)
python visualize_matches.py --min-pmatch 0.0 # include weak matches too
```

Mapillary thumbnail URLs expire within hours, so `visualize_matches.py` re-fetches a
fresh `thumb_1024_url` per match by default; use `--no-refresh` only right after a run.

## Match output columns

`find_matches` augments the DataFrame with eight columns (mirroring `machine_learning_photo`):

| Column | Type | Meaning |
|--------|------|---------|
| `mapillary_id` | `int` or `None` | Matched Mapillary image id; `None` when no match |
| `p_match` | `float` or `None` | Confidence, `min((inliers + 1) / 1000, 0.999999)`; `None` = not scored / no result |
| `mapillary_lat` | `float` or `None` | Refined latitude (from the matched panorama) |
| `mapillary_lon` | `float` or `None` | Refined longitude |
| `mapillary_compass_angle` | `int` or `None` | Heading of the matched panorama |
| `mapillary_captured_at` | `int` or `None` | Mapillary capture timestamp (epoch ms) |
| `mapillary_pic_url` | `str` or `None` | Thumbnail URL of the matched panorama (expires) |
| `mapillary_candidates` | `int` | Candidate count considered before ranking (always set) |

A `status` field carries `ok` / `no_result`. The local CSV runner additionally writes
`flickr_lat`, `flickr_lon`, `p_building`, and `distance_km`.

Two operational thresholds: **review** at 15 inliers (`p_match >= 0.016`, permissive,
for manual triage) and **production retention** at `p_match > 0.05` (applied by the DB
cleanup routine).

## Required DB schema additions

```sql
ALTER TABLE machine_learning_photo
    ADD COLUMN mapillary_id            BIGINT,           -- matched Mapillary image id
    ADD COLUMN p_match                 DOUBLE PRECISION, -- confidence in [0, 1); NULL = retry / no result
    ADD COLUMN mapillary_lat           DOUBLE PRECISION, -- refined latitude
    ADD COLUMN mapillary_lon           DOUBLE PRECISION, -- refined longitude
    ADD COLUMN mapillary_compass_angle INTEGER,          -- panorama heading
    ADD COLUMN mapillary_captured_at   BIGINT,           -- capture timestamp (epoch ms)
    ADD COLUMN mapillary_pic_url       TEXT,             -- panorama thumbnail URL (expires)
    ADD COLUMN mapillary_candidates    INTEGER;          -- candidate count per query
```

The production path writes directly into these columns; the CSV path uses the same names
so the DataFrame schema is identical and only the destination differs.

## Configuration via environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MAPILLARY_ACCESS_TOKEN` | — | **Required.** Read from a `.env` file at the project root (`MAPILLARY_ACCESS_TOKEN=MLY\|...`). The runner exits if unset. |

The LoFTR `outdoor` checkpoint is fixed in `building_matcher.py` and auto-downloads into
torch's hub cache on first use; no env var, no manual download. CUDA is strongly
recommended — a (query, candidate) pair is a couple of seconds on GPU versus minutes on
CPU, and LoFTR is the dominant cost.

## Tuning constants (`src/trainer/matching.py`)

| Constant | Value | Meaning |
|----------|-------|---------|
| `MAX_TENSOR_SIZE` | 512 | Longest image side before LoFTR (then cropped to multiples of 8) |
| `TILE_SIDE_KM` | 0.1 | Tile side for the 3×3 sampling grid (≈ 300 m search box) |
| `BIN_SIZE` | 15 | Compass-angle bin width (degrees) |
| `MIN_LOFTR_INLIER_COUNT` | 10 | Minimum LoFTR correspondences before attempting RANSAC |
| `LOFTR_CONFIDENCE` | 0.5 | Minimum per-correspondence LoFTR confidence |
| `RANSAC_THRESHOLD` | 3.0 | MAGSAC++ reprojection threshold (pixels) |
