from src.trainer.mapillary_client import fetch_candidates
from src.trainer.geo_utils import bbox_from_center

bbox = bbox_from_center(43.70499, 7.28067, radius_km=0.5)
candidates = fetch_candidates(*bbox, limit=50)
print(f'{len(candidates)} candidates')
for i, c in enumerate(candidates):
    mid = c['mapillary_id']
    lat = c['lat']
    lon = c['lon']
    print(f'[{i+1}] https://www.mapillary.com/app/?image_key={mid}  ({lat:.5f}, {lon:.5f})')
