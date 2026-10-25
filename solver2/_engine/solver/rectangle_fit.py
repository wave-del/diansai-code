"""矩形放置工具。"""

from __future__ import annotations

import numpy as np
from ..config import A4_HEIGHT_MM, A4_WIDTH_MM, DIVIDER_Y_MM, TARGET_CENTER_A4_MM


def target_offset(bounds: tuple[float, float, float, float]) -> np.ndarray:
    minx, miny, maxx, maxy = bounds
    width, height = maxx - minx, maxy - miny
    center_x, center_y = TARGET_CENTER_A4_MM
    center_x = min(max(center_x, 10.0 + width / 2), A4_WIDTH_MM - 10.0 - width / 2)
    center_y = min(max(center_y, DIVIDER_Y_MM + 10.0 + height / 2), A4_HEIGHT_MM - 10.0 - height / 2)
    return np.asarray([center_x - (minx + maxx) / 2, center_y - (miny + maxy) / 2])

