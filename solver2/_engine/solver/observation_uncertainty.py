"""Observation-derived uncertainty for the production random solver.

Generator noise labels are deliberately excluded.  The hard radius is a
physical safety envelope; typical sigma is only a ranking hint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import hypot
from typing import Any, Iterable

import numpy as np


NUMERICAL_EPSILON = 1e-9
DEFAULT_HARD_ERROR_RADIUS_MM = 3.0


@dataclass(frozen=True)
class ObservationUncertaintyProfile:
    hard_error_radius_mm: float
    typical_sigma_mm: float | None
    calibration_rms_mm: float | None
    edge_fit_rms_mm: float | None
    mean_vertex_confidence: float | None
    source: str
    quality_level: str
    absolute_measurement_sigma_known: bool = False

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VertexDistanceMetrics:
    nominal_max_mm: float
    estimated_risk_max_mm: float
    worst_case_robust_max_mm: float
    competition_limit_mm: float
    estimated_budget_absorbed_by_gap_mm: float

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


def vertex_distance_metrics(
    nominal_max_mm: float,
    profile: ObservationUncertaintyProfile,
    *,
    target_gap_mm: float,
    machine_position_margin_mm: float,
    pose_residual_mm: float = 0.0,
    competition_limit_mm: float = 20.0,
) -> VertexDistanceMetrics:
    """Separate nominal, likely risk and hard-bound worst case.

    The typical budget already represented by the commanded seam gap is not
    added a second time.  The hard bound remains an independent diagnostic.
    """
    nominal = float(nominal_max_mm)
    typical_pair_budget = float(
        2.0 * (profile.typical_sigma_mm or 0.0)
        + machine_position_margin_mm
        + max(0.0, pose_residual_mm)
    )
    absorbed = min(float(target_gap_mm), typical_pair_budget)
    estimated = nominal + max(0.0, typical_pair_budget - absorbed)
    worst = (
        nominal
        + 2.0 * profile.hard_error_radius_mm
        + machine_position_margin_mm
        + max(0.0, pose_residual_mm)
    )
    return VertexDistanceMetrics(
        nominal,
        float(estimated),
        float(worst),
        float(competition_limit_mm),
        float(absorbed),
    )


def _finite_optional(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) and number >= 0.0 else None


def _mean(values: Iterable[float]) -> float | None:
    rows = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(rows)) if rows else None


def build_observation_uncertainty(
    observations: dict[str, Any],
    pieces: Iterable[Any],
    *,
    default_hard_error_radius_mm: float = DEFAULT_HARD_ERROR_RADIUS_MM,
) -> ObservationUncertaintyProfile:
    """Build uncertainty only from public observation-quality evidence."""
    quality = observations.get("vision_quality")
    quality = quality if isinstance(quality, dict) else {}
    calibration = _finite_optional(
        quality.get("calibration_reprojection_rms_mm")
    )
    edge_fit = _finite_optional(quality.get("edge_fit_rms_mm"))

    explicit_confidence = _finite_optional(
        quality.get("mean_vertex_confidence")
    )
    piece_confidences = [
        float(value)
        for piece in pieces
        for value in np.asarray(
            getattr(piece, "vertex_confidence", []), dtype=float
        )
        if np.isfinite(value)
    ]
    mean_confidence = (
        explicit_confidence
        if explicit_confidence is not None
        else _mean(piece_confidences)
    )

    explicit_sigmas: list[float] = []
    for piece in pieces:
        values = getattr(piece, "vertex_estimated_sigma_mm", None)
        if values is not None:
            explicit_sigmas.extend(
                float(value) for value in np.asarray(values, dtype=float)
                if np.isfinite(value) and value >= 0.0
            )
    for item in observations.get("pieces", []):
        for vertex in item.get("vertices", []):
            if isinstance(vertex, dict):
                sigma = _finite_optional(vertex.get("estimated_sigma_mm"))
                if sigma is not None:
                    explicit_sigmas.append(sigma)
        for sigma in item.get("vertex_estimated_sigma_mm", []):
            value = _finite_optional(sigma)
            if value is not None:
                explicit_sigmas.append(value)

    if explicit_sigmas:
        typical_sigma = _mean(explicit_sigmas)
        source = "vision"
        absolute_known = True
    elif calibration is not None or edge_fit is not None:
        typical_sigma = hypot(calibration or 0.0, edge_fit or 0.0)
        source = "vision"
        absolute_known = True
    else:
        typical_sigma = None
        source = "fallback_from_candidate_statistics"
        absolute_known = False

    rms_hint = max(calibration or 0.0, edge_fit or 0.0)
    if (
        mean_confidence is not None
        and mean_confidence >= 0.90
        and (not absolute_known or rms_hint <= 0.75)
    ):
        quality_level = "high"
    elif (
        mean_confidence is None
        or mean_confidence >= 0.65
    ) and (not absolute_known or rms_hint <= 1.5):
        quality_level = "medium"
    else:
        quality_level = "low"

    return ObservationUncertaintyProfile(
        hard_error_radius_mm=float(default_hard_error_radius_mm),
        typical_sigma_mm=typical_sigma,
        calibration_rms_mm=calibration,
        edge_fit_rms_mm=edge_fit,
        mean_vertex_confidence=mean_confidence,
        source=source,
        quality_level=quality_level,
        absolute_measurement_sigma_known=absolute_known,
    )
