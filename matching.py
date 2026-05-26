import pandas as pd
from typing import Optional, Tuple
from tqdm import tqdm
from .src.trainer.mapillary_client import MapillaryPicture, create_sampler, smart_angle_candidates
from .src.trainer.building_matcher import (
    load_picture,
    load_matcher,
    compute_loftr_matches,
    compute_ransac_inliers,
    to_gray_tensor,
)

MAX_TENSOR_SIZE = 512
SAMPLING_COUNT = 40
SAMPLING_RADIUS = 0.1 #100m
TILE_SIDE_KM=0.1
BIN_SIZE=15
MIN_LOFTR_INLIER_COUNT = 10
LOFTR_CONFIDENCE = 0.5
RANSAC_THRESHOLD = 3.0
RANSAC_CONFIDENCE = 0.99
RANSAC_MAX_ITERATIONS = 100
  
def _best_matching_pair_with_confidence(cache, img_url: str, longitude: float, latitude: float)-> Tuple[Optional[MapillaryPicture], float]:

    matcher, device = load_matcher()
    flickr_tensor = to_gray_tensor(cache.get(img_url), MAX_TENSOR_SIZE, device)

    # old sampler
    # sampler = create_sampler(longitude=longitude, latitude=latitude, st_km=SAMPLING_RADIUS)
    # candidates = [sampler.sample_candidates() for i in range(SAMPLING_COUNT)]
    #new sampler:
    candidates = smart_angle_candidates(longitude=longitude, latitude=latitude, TILE_SIDE_KM=TILE_SIDE_KM, BIN_SIZE=BIN_SIZE)

    mapillary_pictures = cache.get_images([c.pic_url for c in candidates], download_missing=True, fast_cache=False, disk_save=False)
    valid_candidates = [ (to_gray_tensor(mapillary_pic, MAX_TENSOR_SIZE, device), candidate)
        for mapillary_pic, candidate in zip(mapillary_pictures, candidates)
        if mapillary_pic is not None
    ]

    best_ransac_inlier_count = 0
    best_candidate = None
    for mapillary_tensor, candidate in tqdm(valid_candidates, total=SAMPLING_COUNT, desc='find best candidate'):
        kp0, kp1 = compute_loftr_matches(flickr_tensor, mapillary_tensor, matcher, confidence_threshold=LOFTR_CONFIDENCE)
        loft_inlier_count = len(kp0)
        if loft_inlier_count > MIN_LOFTR_INLIER_COUNT:
            ransac_inlier_count = compute_ransac_inliers(kp0, kp1, RANSAC_THRESHOLD, RANSAC_CONFIDENCE, RANSAC_MAX_ITERATIONS)
            if ransac_inlier_count > best_ransac_inlier_count:
                best_ransac_inlier_count = ransac_inlier_count
                best_candidate = candidate

    # return best_candidate, loft_inlier_count
    return best_candidate, min((best_ransac_inlier_count+1) / 1000, 0.999999)



def find_matches(df: pd.DataFrame, cache) -> pd.DataFrame:
    """
    For each row in the dataframe, find the best matching Mapillary picture
    and store the match information in new dataframe columns.
    """

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Matching images"):
        mapillary_pic, confidence = _best_matching_pair_with_confidence(
                cache,
                row["url_o"],
                row["longitude"],
                row["latitude"],
            )

        if mapillary_pic:
            df.loc[idx, [
                "mapillary_id",
                "mapillary_lon",
                "mapillary_lat",
                "p_match",
                "mapillary_compass_angle",
                "mapillary_captured_at",
                "mapillary_pic_url",
            ]] = [
                mapillary_pic.id,
                mapillary_pic.lon,
                mapillary_pic.lat,
                confidence,
                mapillary_pic.compass_angle,
                mapillary_pic.captured_at,
                mapillary_pic.pic_url,
            ]

    return df
 