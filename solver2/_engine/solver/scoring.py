"""候选置信度和统一评分字段。"""

from __future__ import annotations

import math
from ..models import Score


def confidence_from_candidates(scores: list[float], absolute_error: float, fill_ratio: float) -> tuple[float, bool]:
    if not scores:
        return 0.0, True
    gap = scores[1] - scores[0] if len(scores) > 1 else max(1.0, scores[0])
    relative_gap = gap / max(abs(scores[0]), 1.0)
    confidence = 0.25 + 0.35 * min(1.0, relative_gap) + 0.4 * max(0.0, min(1.0, fill_ratio))
    confidence *= math.exp(-max(0.0, absolute_error) / 20.0)
    ambiguous = len(scores) > 1 and relative_gap < 0.03
    if ambiguous:
        confidence = min(confidence, 0.65)
    return float(max(0.01, min(0.99, confidence))), ambiguous


def fixed_score(rms: float, fill_ratio: float, overlap: float, uncovered: float, width: float, height: float) -> Score:
    return Score(
        total=rms + overlap + uncovered / 100.0,
        matched_edge_rms_mm=rms,
        matched_vertex_rms_mm=rms,
        outer_edge_rms_mm=rms,
        overlap_area_mm2=overlap,
        uncovered_area_mm2=uncovered,
        rectangle_fill_ratio=fill_ratio,
        width_mm=width,
        height_mm=height,
    )

