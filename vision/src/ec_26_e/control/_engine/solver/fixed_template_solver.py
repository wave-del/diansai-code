"""固定模板的全排列、循环起点与刚体匹配。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import json
import logging
from time import perf_counter

import numpy as np

from ..generator.fixed_generator import fixed_physical_contours
from ..transforms import transform_points
from .fixed_physical_contour import (
    PhysicalContourObservation,
    extract_physical_contour_features,
)
from .preprocess import ProcessedPiece
from .robust_pose import compare_pose_estimators, automatic_vertex_weights
from ...acceleration import RigidFitBackend, RigidFitRequest


@dataclass
class PoseCandidate:
    piece_id: str
    template_id: str
    angle: float
    translation: np.ndarray
    rms: float
    target_vertices_local: np.ndarray
    template_vertices_local: np.ndarray
    estimator: str
    estimator_rms: dict[str, float]
    shape_score: float
    rotation_std_deg: float
    translation_std_mm: float
    predicted_max_vertex_error_mm: float
    physical_observation: PhysicalContourObservation
    structural_target_vertices_local: np.ndarray
    support_line_rms_mm: float
    physical_edge_length_score: float
    physical_area_score: float


@dataclass
class FixedCandidate:
    poses: list[PoseCandidate]
    total: float
    piece_assignment_score: float
    vertex_fit_score: float
    edge_direction_score: float
    edge_length_ratio_score: float
    joint_pose_score: float
    predicted_execution_uncertainty: float


def _point_line_rms(
    physical: PhysicalContourObservation,
    transformed_raw: np.ndarray,
    shifted_template: np.ndarray,
) -> float:
    residuals: list[float] = []
    for edge_index, group in enumerate(physical.edge_sample_groups):
        start = shifted_template[edge_index]
        end = shifted_template[(edge_index + 1) % len(shifted_template)]
        direction = end - start
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        normal = np.asarray([-direction[1], direction[0]], dtype=float)
        residuals.extend(
            ((transformed_raw[list(group)] - start) @ normal).tolist()
        )
    return float(np.sqrt(np.mean(np.square(residuals)))) if residuals else 0.0


def _best_pose(
    observed: ProcessedPiece,
    template_id: str,
    template: np.ndarray,
    uncertainty_mm: float = 0.0,
    *, initial_fits=None, full_diagnostics: bool = True, physical=None, fit_stats=None,
) -> PoseCandidate | None:
    if physical is None:
        physical = extract_physical_contour_features(observed.vertices, uncertainty_mm)
    if (
        not physical.valid
        or len(physical.structural_corners) != len(template)
        or physical.polygon_orientation < 0
    ):
        return None
    best: PoseCandidate | None = None
    shift_scores: list[float] = []
    for shift in range(len(template)):
        shifted = np.roll(template, -shift, axis=0)
        angle, translation, rms, estimator, diagnostics = compare_pose_estimators(
            physical.structural_corners,
            shifted,
            np.ones(len(physical.structural_corners), dtype=float),
            full_diagnostics=full_diagnostics,
            **({"initial_fits": initial_fits[shift]} if initial_fits is not None else {}),
        )
        observed_lengths = np.linalg.norm(
            np.roll(physical.structural_corners, -1, axis=0)
            - physical.structural_corners,
            axis=1,
        )
        template_lengths = np.linalg.norm(np.roll(shifted, -1, axis=0) - shifted, axis=1)
        length_score = float(np.sqrt(np.mean(
            (observed_lengths / observed_lengths.sum() - template_lengths / template_lengths.sum()) ** 2
        )))
        fitted_structural = transform_points(
            physical.structural_corners, angle, translation
        )
        fitted_raw = transform_points(
            physical.raw_vertices, angle, translation
        )
        edge_a = np.roll(fitted_structural, -1, axis=0) - fitted_structural
        edge_b = np.roll(shifted, -1, axis=0) - shifted
        direction_score = float(np.mean(np.abs(
            edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]
        ) / np.maximum(np.linalg.norm(edge_a, axis=1) * np.linalg.norm(edge_b, axis=1), 1e-9)))
        support_rms = _point_line_rms(
            physical, fitted_raw, shifted
        )
        first_delta = (
            physical.structural_corners[1]
            - physical.structural_corners[0]
        )
        second_delta = (
            physical.structural_corners[2]
            - physical.structural_corners[0]
        )
        observed_area = abs(float(
            first_delta[0] * second_delta[1]
            - first_delta[1] * second_delta[0]
        )) / 2.0 if len(template) == 3 else float(
            abs(np.sum(
                physical.structural_corners[:, 0]
                * np.roll(physical.structural_corners[:, 1], -1)
                - np.roll(physical.structural_corners[:, 0], -1)
                * physical.structural_corners[:, 1]
            )) / 2.0
        )
        template_area = float(abs(np.sum(
            shifted[:, 0] * np.roll(shifted[:, 1], -1)
            - np.roll(shifted[:, 0], -1) * shifted[:, 1]
        )) / 2.0)
        area_score = abs(
            observed_area / max(template_area, 1e-12) - 1.0
        )
        shape_score = (
            rms
            + support_rms
            + 25.0 * length_score
            + 5.0 * direction_score
            + 5.0 * area_score
        )
        radius = max(float(np.linalg.norm(
            physical.structural_corners
            - np.mean(physical.structural_corners, axis=0),
            axis=1,
        ).max()), 1.0)
        combined_rms = float(np.hypot(rms, support_rms))
        translation_std = combined_rms / max(
            np.sqrt(len(physical.structural_corners)), 1.0
        )
        rotation_std = np.degrees(
            combined_rms / radius
            / max(np.sqrt(len(physical.structural_corners)), 1.0)
        )
        predicted_error = float(
            translation_std
            + np.deg2rad(rotation_std) * radius
            + 2.0 * combined_rms
        )
        pose = PoseCandidate(
            observed.piece_id,
            template_id,
            angle,
            translation,
            rms,
            fitted_raw,
            shifted,
            estimator,
            diagnostics,
            shape_score,
            float(rotation_std),
            float(translation_std),
            predicted_error,
            physical,
            fitted_structural,
            support_rms,
            length_score,
            area_score,
        )
        shift_scores.append(pose.shape_score)
        if best is None or pose.shape_score < best.shape_score:
            best = pose
    if initial_fits is not None and len(shift_scores) > 1:
        ordered = sorted(shift_scores)
        if ordered[1] - ordered[0] < 0.02:
            # A near-tied cyclic origin must retain the original CPU tie order.
            if fit_stats is not None:
                fit_stats['cpu_tie_fallbacks'] += 1
            return _best_pose(observed, template_id, template, uncertainty_mm,
                              full_diagnostics=full_diagnostics, physical=physical)
    return best


def solve_fixed(
    pieces: list[ProcessedPiece],
    uncertainty_mm: float = 0.0,
    *, rigid_backend: RigidFitBackend | None = None, full_diagnostics: bool = True,
) -> list[FixedCandidate]:
    started = perf_counter()
    fit_stats = dict(cpu_tie_fallbacks=0, fit_requests=0, fit_seconds=0.0)
    templates = fixed_physical_contours()
    if len(pieces) != 4:
        raise ValueError("fixed 模式必须恰好有 4 片")
    candidates: list[FixedCandidate] = []
    # 同一观测片与模板的位姿不依赖全排列中的其他片。缓存仅属于本次
    # 求解，同时缓存不相容的配对，避免在 24 个排列中重复鲁棒拟合。
    pose_cache: dict[tuple[int, str], PoseCandidate | None] = {}
    fit_cache = {}
    physical_cache = [extract_physical_contour_features(p.vertices, uncertainty_mm) for p in pieces]
    fit_stats['feature_seconds'] = perf_counter() - started
    if rigid_backend is not None:
        # Submit all compatible piece/template/cyclic-origin fits as one logical batch.
        # Iterative robust estimators and geometry/execution checks remain on PS.
        requests, keys = [], []
        for index, observed in enumerate(pieces):
            physical = physical_cache[index]
            for template_piece in templates:
                template = np.asarray(template_piece.corner_vertices, dtype=float)
                if (not physical.valid or physical.polygon_orientation < 0
                        or len(physical.structural_corners) != len(template)):
                    continue
                for shift in range(len(template)):
                    source = physical.structural_corners
                    target = np.roll(template, -shift, axis=0)
                    weights = automatic_vertex_weights(source, target, np.ones(len(source)))
                    requests.extend((RigidFitRequest(source, target), RigidFitRequest(source, target, weights)))
                    keys.append((index, template_piece.piece_template_id, shift))
        fit_started = perf_counter()
        results = rigid_backend.fit_batch(requests)
        fit_stats.update(fit_requests=len(requests), fit_seconds=perf_counter()-fit_started)
        if len(results) != len(requests):
            raise RuntimeError("刚体后端返回的批次长度不匹配")
        for result in results:
            if (np.shape(result.translation) != (2,)
                    or not np.isfinite([result.angle, *result.translation, result.rms]).all()
                    or result.rms < 0):
                raise RuntimeError("刚体后端返回无效结果")
        for i, (index, template_id, shift) in enumerate(keys):
            fit_cache.setdefault((index, template_id), {})[shift] = results[2*i:2*i+2]
    for perm in permutations(templates):
        poses: list[PoseCandidate] = []
        valid = True
        for index, (observed, template_piece) in enumerate(zip(pieces, perm)):
            key = (index, template_piece.piece_template_id)
            if key not in pose_cache:
                pose_cache[key] = _best_pose(
                    observed,
                    template_piece.piece_template_id,
                    np.asarray(template_piece.corner_vertices, dtype=float),
                    uncertainty_mm,
                    physical=physical_cache[index],
                    **({"full_diagnostics": False} if not full_diagnostics else {}),
                    **({"initial_fits": fit_cache[key], "fit_stats": fit_stats} if key in fit_cache else {}),
                )
            pose = pose_cache[key]
            if pose is None:
                valid = False
                break
            poses.append(pose)
        if valid:
            vertex = float(np.sqrt(np.mean([p.rms**2 for p in poses])))
            assignment = float(np.mean([p.shape_score for p in poses]))
            edge_length = float(np.mean([max(p.shape_score - p.rms, 0.0) for p in poses]))
            total = assignment
            candidates.append(FixedCandidate(
                poses, total, assignment, vertex, edge_length / 5.0,
                edge_length / 25.0, vertex,
                max(p.predicted_max_vertex_error_mm for p in poses),
            ))
    fit_stats['total_seconds'] = perf_counter() - started
    logging.getLogger(__name__).info('FIXED_FIT_PERFORMANCE %s', json.dumps(fit_stats))
    return sorted(candidates, key=lambda candidate: candidate.total)[:5]
