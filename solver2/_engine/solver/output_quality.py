"""Minimum geometric rectangle quality, measured before execution gaps."""

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union


MINIMUM_OUTPUT_FILL_RATIO = 0.95


def rectangle_fill_ratio(vertices) -> float:
    arrays = [np.asarray(points, dtype=float) for points in vertices]
    if not arrays or any(not np.isfinite(points).all() for points in arrays):
        return 0.0
    polygons = [Polygon(points) for points in arrays]
    if any(not p.is_valid or p.is_empty or p.area <= 0 for p in polygons):
        return 0.0
    union = unary_union(polygons)
    area = float(union.minimum_rotated_rectangle.area)
    return float(np.clip(union.area / area, 0.0, 1.0)) if area > 0 else 0.0
