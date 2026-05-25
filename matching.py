import pandas as pd
from typing import Optional, Tuple
from .src.trainer.mapillary_client import MapillaryPicture, create_sampler
from src.trainer.building_matcher import (
    load_picture,
    compute_loftr_matches,
    compute_ransac_inliers,
)

SAMPLING_COUNT = 100
SAMPLING_RADIUS = 0.05
MIN_LOFTR_INLIER_COUNT = 20
RANSAC_THRESHOLD = 3.0
RANSAC_CONFIDENCE = 0.99
RANSAC_MAX_ITERATIONS = 500
  
def _best_matching_pair_with_confidence(cache, img_url: str, longitude: float, latitude: float)-> Tuple[Optional[MapillaryPicture], float]:

    flickr_pic = cache.get(img_url)
    if not flickr_pic:
        return None, 0
    sampler = create_sampler(longitude=longitude, latitude=latitude, st_km=SAMPLING_RADIUS)

    best_ransac_inlier_count = 0
    best_candidate = None
    for i in range(SAMPLING_COUNT):
        candidate = sampler.sample_candidates()
        if not candidate:
            return None, 0
        mapillary_pic = cache.get(candidate.pic_url)
        if not mapillary_pic:
            continue
        kp0, kp1 = compute_loftr_matches(flickr_pic, mapillary_pic)
        loft_inlier_count = len(kp0)
        if loft_inlier_count > MIN_LOFTR_INLIER_COUNT:
            ransac_inlier_count = compute_ransac_inliers(kp0, kp1, RANSAC_THRESHOLD, RANSAC_CONFIDENCE, RANSAC_MAX_ITERATIONS)
        if ransac_inlier_count > best_ransac_inlier_count:
            best_ransac_inlier_count = ransac_inlier_count
            best_candidate = candidate

    return best_candidate, max(best_ransac_inlier_count / 1000, 0.999999)



def find_matches(df: pd.DataFrame, cache)-> pd.DataFrame:

    for row in df:
        mapillary_pic, confidence = _best_matching_pair_with_confidence(cache, row['url_o'], row['longitude'], row['latitude'])
    
    df[["mappilary_id", "mapillary_lon", "mapillary_lat" ,"p_match", "mapillary_compass_angle",
     "mapillary_captured_at", "mapillary_pic_url"]] = 