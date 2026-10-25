"""Scale-aware shape and complete-candidate similarity in normalized [0, 1]."""

from __future__ import annotations

from itertools import permutations
from math import atan2, degrees, exp, pi
from typing import Any

import numpy as np
from shapely.geometry import Polygon

from .models import (
    CandidateGeometryDescriptor,
    CandidateSimilarityResult,
    ShapeSimilarityResult,
)


def _vertices(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        value = value.get("vertices", value.get("vertices_mm"))
    elif hasattr(value, "vertices"):
        value = value.vertices
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("piece shape requires an N x 2 polygon")
    return points


def _edge_vectors(points: np.ndarray) -> np.ndarray:
    return np.roll(points, -1, axis=0) - points


def _lengths(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(_edge_vectors(points), axis=1)


def _turns(points: np.ndarray) -> np.ndarray:
    edges = _edge_vectors(points)
    rows = []
    for previous, current in zip(np.roll(edges, 1, axis=0), edges):
        cross = previous[0] * current[1] - previous[1] * current[0]
        dot = float(np.dot(previous, current))
        rows.append(atan2(cross, dot) / pi)
    return np.asarray(rows, dtype=float)


def _internal_angles(points: np.ndarray) -> np.ndarray:
    return 1.0 - np.abs(_turns(points))


def _principal_aspect(points: np.ndarray) -> float:
    centered = points - np.mean(points, axis=0)
    values = np.linalg.eigvalsh(centered.T @ centered / max(len(points), 1))
    return float(np.sqrt(max(values[-1], 1e-12) / max(values[0], 1e-12)))


def _ratio_score(first: float, second: float) -> float:
    if first <= 1e-12 or second <= 1e-12:
        return float(first <= 1e-12 and second <= 1e-12)
    return float(exp(-abs(np.log(first / second))))


def _sequence_score(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) != len(second):
        return 0.0
    return float(np.clip(1.0 - np.mean(np.abs(first - second)), 0.0, 1.0))


def _rotation_fit(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    a = first - np.mean(first, axis=0)
    b = second - np.mean(second, axis=0)
    scale_a = float(np.sqrt(np.mean(np.sum(a * a, axis=1))))
    scale_b = float(np.sqrt(np.mean(np.sum(b * b, axis=1))))
    if min(scale_a, scale_b) <= 1e-12:
        return 0.0, 0.0
    a /= scale_a
    b /= scale_b
    u, _s, vt = np.linalg.svd(a.T @ b)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    aligned = a @ rotation.T
    rms = float(np.sqrt(np.mean(np.sum((aligned - b) ** 2, axis=1))))
    angle = degrees(atan2(rotation[1, 0], rotation[0, 0]))
    return float(exp(-3.0 * rms)), angle


def _best_orientation(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[float, int, float, dict[str, float]]:
    if len(first) != len(second):
        return 0.0, 0, 0.0, {
            "length": 0.0, "angle": 0.0, "turn": 0.0, "fit": 0.0,
        }
    first_lengths = _lengths(first) / max(float(np.sum(_lengths(first))), 1e-12)
    first_angles = _internal_angles(first)
    first_turns = _turns(first)
    best = (-1.0, 0, 0.0, {})
    for shift in range(len(second)):
        shifted = np.roll(second, -shift, axis=0)
        second_lengths = _lengths(shifted)
        second_lengths /= max(float(np.sum(second_lengths)), 1e-12)
        components = {
            "length": _sequence_score(first_lengths, second_lengths),
            "angle": _sequence_score(first_angles, _internal_angles(shifted)),
            "turn": _sequence_score(first_turns, _turns(shifted)),
        }
        components["fit"], rotation = _rotation_fit(first, shifted)
        score = (
            0.32 * components["length"]
            + 0.24 * components["angle"]
            + 0.20 * components["turn"]
            + 0.24 * components["fit"]
        )
        if score > best[0]:
            best = (float(score), shift, rotation, components)
    return best


def compare_piece_shapes(piece_a: Any, piece_b: Any) -> ShapeSimilarityResult:
    """Compare public/estimated polygons without truth or reflection matching."""
    first, second = _vertices(piece_a), _vertices(piece_b)
    count_score = float(exp(-1.5 * abs(len(first) - len(second))))
    legal, shift, rotation, components = _best_orientation(first, second)
    mirrored_score, _mirror_shift, _mirror_rotation, _mirror_components = (
        _best_orientation(first, second[::-1])
    )
    first_polygon, second_polygon = Polygon(first), Polygon(second)
    perimeter_a, perimeter_b = first_polygon.length, second_polygon.length
    compactness_a = (
        4.0 * pi * first_polygon.area / max(perimeter_a * perimeter_a, 1e-12)
    )
    compactness_b = (
        4.0 * pi * second_polygon.area / max(perimeter_b * perimeter_b, 1e-12)
    )
    scale_score = _ratio_score(perimeter_a, perimeter_b)
    compactness_score = _ratio_score(compactness_a, compactness_b)
    aspect_score = _ratio_score(
        _principal_aspect(first), _principal_aspect(second)
    )
    similarity = float(np.clip(
        0.12 * count_score
        + 0.25 * components.get("length", 0.0)
        + 0.18 * components.get("angle", 0.0)
        + 0.12 * components.get("turn", 0.0)
        + 0.10 * scale_score
        + 0.08 * compactness_score
        + 0.07 * aspect_score
        + 0.08 * components.get("fit", 0.0),
        0.0,
        1.0,
    ))
    mirrored = bool(mirrored_score > legal + 0.03)
    return ShapeSimilarityResult(
        similarity=similarity,
        edge_count_score=count_score,
        normalized_length_score=components.get("length", 0.0),
        angle_score=components.get("angle", 0.0),
        direction_change_score=components.get("turn", 0.0),
        scale_score=scale_score,
        compactness_score=compactness_score,
        aspect_ratio_score=aspect_score,
        best_cyclic_shift=int(shift),
        best_rotation_deg=float(rotation),
        mirrored=mirrored,
        mirrored_similarity=float(np.clip(mirrored_score, 0.0, 1.0)),
    )


def _jaccard(first: set[Any], second: set[Any]) -> float:
    union = first | second
    return 1.0 if not union else len(first & second) / len(union)


def _mapped_adjacency(
    pairs: set[frozenset[str]], mapping: dict[str, str]
) -> set[frozenset[str]]:
    return {
        frozenset(mapping[piece_id] for piece_id in pair)
        for pair in pairs
        if all(piece_id in mapping for piece_id in pair)
    }


def _layout_score(
    first: CandidateGeometryDescriptor,
    second: CandidateGeometryDescriptor,
    mapping: dict[str, str],
) -> tuple[float, bool]:
    ids = sorted(mapping)
    centroids_a = np.asarray([
        np.mean(np.asarray(first.target_vertices[item], dtype=float), axis=0)
        for item in ids
    ])
    centroids_b = np.asarray([
        np.mean(
            np.asarray(second.target_vertices[mapping[item]], dtype=float),
            axis=0,
        )
        for item in ids
    ])
    a = centroids_a - np.mean(centroids_a, axis=0)
    b = centroids_b - np.mean(centroids_b, axis=0)
    diagonal = max(
        np.hypot(first.width_mm, first.height_mm),
        np.hypot(second.width_mm, second.height_mm),
        1e-9,
    )
    if len(ids) > 1:
        u, _s, vt = np.linalg.svd(a.T @ b)
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1] *= -1
            rotation = vt.T @ u.T
        a = a @ rotation.T
    position_rms = float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))
    rotation_errors = []
    base_a = first.rotations_deg[ids[0]]
    base_b = second.rotations_deg[mapping[ids[0]]]
    for item in ids:
        relative_a = (first.rotations_deg[item] - base_a) % 360.0
        relative_b = (
            second.rotations_deg[mapping[item]] - base_b
        ) % 360.0
        difference = abs((relative_a - relative_b + 180.0) % 360.0 - 180.0)
        rotation_errors.append(difference)
    rotation_rms = float(np.sqrt(np.mean(np.square(rotation_errors))))
    score = float(np.clip(
        0.7 * exp(-6.0 * position_rms / diagonal)
        + 0.3 * exp(-rotation_rms / 20.0),
        0.0,
        1.0,
    ))
    identical = bool(position_rms <= 1e-6 and rotation_rms <= 1e-6)
    return score, identical


def compare_safe_candidates(
    candidate_a: CandidateGeometryDescriptor,
    candidate_b: CandidateGeometryDescriptor,
) -> CandidateSimilarityResult:
    """Compare complete candidates that already passed geometry safety."""
    if not candidate_a.safe or not candidate_b.safe:
        raise ValueError("candidate similarity requires safe candidates")
    scale = max(
        abs(candidate_a.geometry_score),
        abs(candidate_b.geometry_score),
        1.0,
    )
    geometry_score = float(exp(
        -abs(candidate_a.geometry_score - candidate_b.geometry_score) / scale
    ))
    ids_a = sorted(candidate_a.piece_vertices)
    ids_b = sorted(candidate_b.piece_vertices)
    if len(ids_a) != len(ids_b):
        raise ValueError("candidate piece counts differ")
    best: tuple[
        float, tuple[str, ...], dict[str, str], float, float, float, bool
    ] | None = None
    for order in permutations(ids_b):
        mapping = dict(zip(ids_a, order))
        shape_score = float(np.mean([
            compare_piece_shapes(
                candidate_a.piece_vertices[source],
                candidate_b.piece_vertices[target],
            ).similarity
            for source, target in mapping.items()
        ]))
        adjacency_score = _jaccard(
            _mapped_adjacency(candidate_a.adjacency_pairs, mapping),
            candidate_b.adjacency_pairs,
        )
        layout_score, identical = _layout_score(
            candidate_a, candidate_b, mapping
        )
        direct_dimensions = 0.5 * (
            _ratio_score(candidate_a.width_mm, candidate_b.width_mm)
            + _ratio_score(candidate_a.height_mm, candidate_b.height_mm)
        )
        quarter_turn_dimensions = 0.5 * (
            _ratio_score(candidate_a.width_mm, candidate_b.height_mm)
            + _ratio_score(candidate_a.height_mm, candidate_b.width_mm)
        )
        dimension_score = max(
            direct_dimensions, quarter_turn_dimensions
        )
        layout_score = float(
            0.75 * layout_score + 0.25 * dimension_score
        )
        similarity = float(np.clip(
            0.40 * shape_score
            + 0.25 * adjacency_score
            + 0.20 * layout_score
            + 0.15 * geometry_score,
            0.0,
            1.0,
        ))
        row = (
            similarity,
            tuple(order),
            mapping,
            shape_score,
            float(adjacency_score),
            layout_score,
            identical,
        )
        if best is None or row[0] > best[0] + 1e-12 or (
            abs(row[0] - best[0]) <= 1e-12 and row[1] < best[1]
        ):
            best = row
    assert best is not None
    (
        similarity,
        _order,
        mapping,
        shape_score,
        adjacency_score,
        layout_score,
        identical,
    ) = best
    identity_mapping = all(mapping.get(item) == item for item in mapping)
    rotation_deltas = [
        (
            candidate_b.rotations_deg[mapping[item]]
            - candidate_a.rotations_deg[item]
            + 180.0
        ) % 360.0 - 180.0
        for item in mapping
    ]
    rotation_differs = any(abs(value) > 1e-6 for value in rotation_deltas)
    common_rotation = bool(
        rotation_differs
        and max(rotation_deltas) - min(rotation_deltas) <= 1e-6
    )
    if not identity_mapping:
        ambiguity_type = "similar_piece_swap"
    elif common_rotation:
        ambiguity_type = "global_rotation_ambiguity"
    elif rotation_differs:
        ambiguity_type = "piece_rotation_ambiguity"
    elif candidate_a.interface_ids != candidate_b.interface_ids:
        ambiguity_type = "alternative_interface_assignment"
    elif not identical:
        ambiguity_type = "multiple_equivalent_layouts"
    else:
        ambiguity_type = None
    same_physical_targets = bool(
        set(candidate_a.target_vertices) == set(candidate_b.target_vertices)
        and all(
            Polygon(candidate_a.target_vertices[piece_id]).hausdorff_distance(
                Polygon(candidate_b.target_vertices[piece_id])
            ) <= 1e-6
            for piece_id in candidate_a.target_vertices
        )
    )
    return CandidateSimilarityResult(
        similarity=similarity,
        exchanged_piece_shape_similarity=shape_score,
        adjacency_similarity=float(adjacency_score),
        target_layout_similarity=layout_score,
        geometry_score_similarity=geometry_score,
        piece_mapping=mapping,
        identical_physical_placement=(
            same_physical_targets
            and identity_mapping
            and not rotation_differs
        ),
        identity_or_pose_differs=bool(
            not identity_mapping
            or rotation_differs
            or candidate_a.interface_ids != candidate_b.interface_ids
            or not identical
        ),
        ambiguity_type=ambiguity_type,
    )
