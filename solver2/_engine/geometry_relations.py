"""具有数值容差的共线线段重叠与邻接关系。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SegmentOverlap:
    length_mm: float
    first_parameters: tuple[float, float]
    second_parameters: tuple[float, float]
    start_point: tuple[float, float]
    end_point: tuple[float, float]
    support_line_distance_mm: float
    parallel_error: float


def robust_segment_overlap(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
    *,
    line_tolerance_mm: float = 1e-6,
    parallel_tolerance: float = 1e-8,
) -> SegmentOverlap | None:
    a0 = np.asarray(first_start, dtype=float)
    a1 = np.asarray(first_end, dtype=float)
    b0 = np.asarray(second_start, dtype=float)
    b1 = np.asarray(second_end, dtype=float)
    da = a1 - a0
    db = b1 - b0
    la = float(np.linalg.norm(da))
    lb = float(np.linalg.norm(db))
    if la <= 1e-12 or lb <= 1e-12:
        return None
    ua = da / la
    ub = db / lb
    parallel_error = float(abs(ua[0] * ub[1] - ua[1] * ub[0]))
    if parallel_error > parallel_tolerance:
        return None
    normal = np.asarray([-ua[1], ua[0]])
    distances = np.abs(
        np.asarray([np.dot(b0 - a0, normal), np.dot(b1 - a0, normal)])
    )
    line_distance = float(np.max(distances))
    if line_distance > line_tolerance_mm:
        return None
    b_projection = sorted(
        (float(np.dot(b0 - a0, ua)), float(np.dot(b1 - a0, ua)))
    )
    overlap_start = max(0.0, b_projection[0])
    overlap_end = min(la, b_projection[1])
    if overlap_end <= overlap_start:
        return None
    world_start = a0 + overlap_start * ua
    world_end = a0 + overlap_end * ua
    b_direction = b1 - b0
    b_denominator = float(np.dot(b_direction, b_direction))
    b_parameters = sorted(
        (
            float(np.dot(world_start - b0, b_direction) / b_denominator),
            float(np.dot(world_end - b0, b_direction) / b_denominator),
        )
    )
    return SegmentOverlap(
        length_mm=float(overlap_end - overlap_start),
        first_parameters=(overlap_start / la, overlap_end / la),
        second_parameters=(b_parameters[0], b_parameters[1]),
        start_point=(float(world_start[0]), float(world_start[1])),
        end_point=(float(world_end[0]), float(world_end[1])),
        support_line_distance_mm=line_distance,
        parallel_error=parallel_error,
    )
