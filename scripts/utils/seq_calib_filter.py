"""Helper for filtering sequences' camera views by calibration availability.

The HOI-M3 dataset has 3 dates with one bad camera each in the refined calibration
(/simurgh2/datasets/HOI-M3/calib_ground_refined/<date>/calibration.json):
    20230916 view 40, 20230924 view 32, 20230927 view 40.

Sequences captured on those dates have masks/images for the bad view, but the
camera calibration (RT) is unusable, so downstream multi-view code (triangulation,
fitting, 3D back-projection) should skip the bad view per sequence.

Usage:
    from scripts.utils.seq_calib_filter import unavail_views, available_views

    bad = unavail_views('livingroom_data30')         # -> [32]
    good = available_views('livingroom_data30', 42)  # -> [0,1,...,31,33,...,41]

The map is loaded once from /simurgh2/datasets/HOI-M3/seq_unavail_views.json.
"""
import json
import os
from functools import lru_cache

_MAP_PATH = '/simurgh2/datasets/HOI-M3/seq_unavail_views.json'


@lru_cache(maxsize=1)
def _load_map() -> dict:
    if not os.path.isfile(_MAP_PATH):
        return {}
    with open(_MAP_PATH) as f:
        data = json.load(f)
    return data.get('seq_unavail_views', {})


def unavail_views(seq: str) -> list[int]:
    """Returns sorted list of view indices whose calibration is unavailable for this seq."""
    return sorted(_load_map().get(seq, []))


def available_views(seq: str, num_views: int = 42) -> list[int]:
    """Returns sorted list of view indices to USE (excludes calib-unavailable views)."""
    bad = set(unavail_views(seq))
    return [v for v in range(num_views) if v not in bad]


def filter_views(seq: str, views) -> list[int]:
    """Filter a given views iterable, removing calib-unavailable ones."""
    bad = set(unavail_views(seq))
    return [int(v) for v in views if int(v) not in bad]
