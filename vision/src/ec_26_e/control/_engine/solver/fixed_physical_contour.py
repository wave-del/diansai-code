"""Physical-contour extraction for fixed templates.

Raw contour vertices may contain arbitrary samples along a straight physical
edge.  Those samples are retained for line fitting but are not structural
corners and never receive assembly-anchor identities.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from shapely.geometry import Polygon


@dataclass(frozen=True)
class ObservedPhysicalEdge:
    edge_index: int
    start_corner_index: int
    end_corner_index: int
    sample_indices: tuple[int, ...]
    support_point: tuple[float, float]
    support_direction: tuple[float, float]
    fit_rms_mm: float
    length_mm: float


@dataclass
class PhysicalContourObservation:
    raw_vertices: np.ndarray
    structural_corners: np.ndarray
    physical_edges: tuple[ObservedPhysicalEdge, ...]
    edge_sample_groups: tuple[tuple[int, ...], ...]
    raw_to_edge_assignment: tuple[int, ...]
    polygon_orientation: int
    valid: bool
    simplification_diagnostics: dict[str, Any]


def _point_segment_distance(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> tuple[float, float]:
    delta = end - start
    norm2 = float(np.dot(delta, delta))
    if norm2 <= 1e-12:
        return float(np.linalg.norm(point - start)), 0.0
    parameter = float(np.dot(point - start, delta) / norm2)
    projected = start + np.clip(parameter, 0.0, 1.0) * delta
    return float(np.linalg.norm(point - projected)), parameter


def _straight_deviation_deg(
    previous: np.ndarray, point: np.ndarray, following: np.ndarray
) -> float:
    first, second = previous - point, following - point
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return 180.0
    angle = math.degrees(math.acos(float(np.clip(
        np.dot(first, second) / denominator, -1.0, 1.0
    ))))
    return abs(180.0 - angle)


def _cyclic_indices(start: int, end: int, count: int) -> tuple[int, ...]:
    values = [start]
    current = start
    while current != end:
        current = (current + 1) % count
        values.append(current)
        if len(values) > count + 1:
            raise RuntimeError("cyclic contour traversal did not terminate")
    return tuple(values)


def _robust_line_fit(
    points: np.ndarray, preferred_direction: np.ndarray, scale_mm: float
) -> tuple[np.ndarray, np.ndarray, float]:
    weights = np.ones(len(points), dtype=float)
    center = np.mean(points, axis=0)
    direction = preferred_direction / max(
        float(np.linalg.norm(preferred_direction)), 1e-12
    )
    for _ in range(8):
        total = max(float(weights.sum()), 1e-12)
        center = np.sum(points * weights[:, None], axis=0) / total
        centered = points - center
        covariance = (centered * weights[:, None]).T @ centered / total
        values, vectors = np.linalg.eigh(covariance)
        direction = vectors[:, int(np.argmax(values))]
        if float(np.dot(direction, preferred_direction)) < 0.0:
            direction = -direction
        normal = np.asarray([-direction[1], direction[0]], dtype=float)
        residuals = np.abs(centered @ normal)
        huber = max(float(scale_mm), 1e-3)
        updated = np.where(
            residuals <= huber,
            1.0,
            huber / np.maximum(residuals, 1e-12),
        )
        if np.max(np.abs(updated - weights)) <= 1e-6:
            weights = updated
            break
        weights = updated
    normal = np.asarray([-direction[1], direction[0]], dtype=float)
    residuals = (points - center) @ normal
    rms = float(np.sqrt(np.average(residuals**2, weights=weights)))
    return center, direction, rms


def _line_intersection(
    first_point: np.ndarray,
    first_direction: np.ndarray,
    second_point: np.ndarray,
    second_direction: np.ndarray,
    fallback: np.ndarray,
) -> np.ndarray:
    matrix = np.column_stack((first_direction, -second_direction))
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) <= 1e-4:
        return fallback.copy()
    return first_point + first_direction * np.linalg.solve(
        matrix, second_point - first_point
    )[0]


def extract_physical_contour_features(
    ordered_vertices: np.ndarray | list,
    uncertainty_mm: float = 0.0,
    *,
    collinear_distance_tolerance_mm: float | None = None,
    collinear_angle_tolerance_deg: float = 18.0,
) -> PhysicalContourObservation:
    """Extract structural corners without discarding raw edge samples."""

    raw = np.asarray(ordered_vertices, dtype=float)
    if (
        raw.ndim != 2
        or raw.shape[1:] != (2,)
        or len(raw) < 3
        or not np.all(np.isfinite(raw))
    ):
        raise ValueError("fixed physical contour must be finite Nx2")
    tolerance = float(
        collinear_distance_tolerance_mm
        if collinear_distance_tolerance_mm is not None
        else max(0.25, min(2.0, 0.5 + 1.5 * float(uncertainty_mm)))
    )
    angle_tolerance = float(max(
        collinear_angle_tolerance_deg,
        min(25.0, 6.0 + 3.0 * float(uncertainty_mm)),
    ))
    active = list(range(len(raw)))
    removed: list[dict[str, float | int]] = []
    while len(active) > 3:
        candidates = []
        for position, raw_index in enumerate(active):
            previous = active[position - 1]
            following = active[(position + 1) % len(active)]
            distance, parameter = _point_segment_distance(
                raw[raw_index], raw[previous], raw[following]
            )
            deviation = _straight_deviation_deg(
                raw[previous], raw[raw_index], raw[following]
            )
            if (
                -0.05 <= parameter <= 1.05
                and distance <= tolerance
                and deviation <= angle_tolerance
            ):
                candidates.append((
                    distance / max(tolerance, 1e-9)
                    + deviation / max(angle_tolerance, 1e-9),
                    position,
                    raw_index,
                    distance,
                    deviation,
                ))
        if not candidates:
            break
        _, position, raw_index, distance, deviation = min(candidates)
        removed.append({
            "raw_index": int(raw_index),
            "line_distance_mm": float(distance),
            "straight_deviation_deg": float(deviation),
        })
        active.pop(position)

    groups = tuple(
        _cyclic_indices(
            active[index],
            active[(index + 1) % len(active)],
            len(raw),
        )
        for index in range(len(active))
    )
    line_rows = []
    for index, group in enumerate(groups):
        points = raw[list(group)]
        preferred = (
            raw[active[(index + 1) % len(active)]] - raw[active[index]]
        )
        center, direction, rms = _robust_line_fit(
            points, preferred, max(tolerance, 0.25)
        )
        line_rows.append((center, direction, rms))

    corners = []
    for index, raw_index in enumerate(active):
        previous = line_rows[index - 1]
        following = line_rows[index]
        corners.append(_line_intersection(
            previous[0],
            previous[1],
            following[0],
            following[1],
            raw[raw_index],
        ))
    structural = np.asarray(corners, dtype=float)
    assignments = [-1] * len(raw)
    edges = []
    for index, group in enumerate(groups):
        for raw_index in group[:-1]:
            assignments[raw_index] = index
        start, end = structural[index], structural[(index + 1) % len(structural)]
        center, direction, rms = line_rows[index]
        edges.append(ObservedPhysicalEdge(
            edge_index=index,
            start_corner_index=index,
            end_corner_index=(index + 1) % len(structural),
            sample_indices=group,
            support_point=(float(center[0]), float(center[1])),
            support_direction=(float(direction[0]), float(direction[1])),
            fit_rms_mm=float(rms),
            length_mm=float(np.linalg.norm(end - start)),
        ))
    for index, value in enumerate(assignments):
        if value < 0:
            assignments[index] = next(
                edge.edge_index for edge in edges
                if index in edge.sample_indices
            )
    polygon = Polygon(structural)
    signed_twice_area = float(np.sum(
        structural[:, 0] * np.roll(structural[:, 1], -1)
        - np.roll(structural[:, 0], -1) * structural[:, 1]
    ))
    valid = bool(
        polygon.is_valid
        and polygon.area > 1e-6
        and all(edge.length_mm > 1e-6 for edge in edges)
    )
    return PhysicalContourObservation(
        raw_vertices=raw.copy(),
        structural_corners=structural,
        physical_edges=tuple(edges),
        edge_sample_groups=groups,
        raw_to_edge_assignment=tuple(int(value) for value in assignments),
        polygon_orientation=1 if signed_twice_area > 0.0 else -1,
        valid=valid,
        simplification_diagnostics={
            "raw_vertex_count": len(raw),
            "structural_corner_count": len(structural),
            "removed_collinear_sample_count": len(removed),
            "removed_samples": removed,
            "distance_tolerance_mm": tolerance,
            "angle_tolerance_deg": angle_tolerance,
            "maximum_support_line_rms_mm": max(
                (edge.fit_rms_mm for edge in edges), default=0.0
            ),
        },
    )
