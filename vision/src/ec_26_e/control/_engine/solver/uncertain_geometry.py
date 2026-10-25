"""可选观测协方差与边拟合不确定度的统一表示。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from .preprocess import ProcessedPiece


@dataclass(frozen=True)
class UncertainPieceGeometry:
    observed_vertices: np.ndarray
    observed_edges: tuple[dict[str, Any], ...]
    vertex_covariance: np.ndarray
    edge_direction_variance: np.ndarray
    edge_offset_variance: np.ndarray
    edge_length_variance: np.ndarray
    area_uncertainty: float
    angle_uncertainty: float
    edge_direction_sigma_deg: np.ndarray
    vertex_angle_sigma_deg: np.ndarray
    confidence: float

    @classmethod
    def from_piece(
        cls,
        piece: ProcessedPiece,
        noise_bound_mm: float,
    ) -> "UncertainPieceGeometry":
        count = len(piece.vertices)
        confidence = np.clip(piece.vertex_confidence, 1e-3, 1.0)
        base_variance = max(float(noise_bound_mm), 1e-6) ** 2 / 3.0
        covariance = (
            np.asarray(piece.vertex_covariance_mm2, dtype=np.float64)
            if piece.vertex_covariance_mm2 is not None
            else np.asarray(
                [
                    np.eye(2) * base_variance / value
                    for value in confidence
                ],
                dtype=np.float64,
            )
        )
        length_scale = np.maximum(piece.edge_lengths, 1.0)
        direction_variance = base_variance / np.square(length_scale)
        offset_variance = np.full(count, base_variance)
        length_variance = np.full(count, 2.0 * base_variance)
        for index, edge in enumerate(piece.edges):
            edge_index = int(edge.get("start_vertex_index", index)) % count
            if edge.get("direction_std_deg") is not None:
                direction_variance[edge_index] = np.deg2rad(
                    float(edge["direction_std_deg"])
                ) ** 2
            if edge.get("offset_std_mm") is not None:
                offset_variance[edge_index] = float(
                    edge["offset_std_mm"]
                ) ** 2
            if edge.get("length_std_mm") is not None:
                length_variance[edge_index] = float(
                    edge["length_std_mm"]
                ) ** 2
        direction_sigma_deg = np.rad2deg(np.sqrt(direction_variance))
        vertex_angle_sigma_deg = np.sqrt(
            np.square(np.roll(direction_sigma_deg, 1))
            + np.square(direction_sigma_deg)
            + 1.0
        )
        return cls(
            observed_vertices=piece.vertices,
            observed_edges=tuple(piece.edges),
            vertex_covariance=covariance,
            edge_direction_variance=direction_variance,
            edge_offset_variance=offset_variance,
            edge_length_variance=length_variance,
            area_uncertainty=float(
                np.sqrt(base_variance) * piece.edge_lengths.sum()
            ),
            angle_uncertainty=float(np.deg2rad(
                vertex_angle_sigma_deg.mean()
            )),
            edge_direction_sigma_deg=direction_sigma_deg,
            vertex_angle_sigma_deg=vertex_angle_sigma_deg,
            confidence=float(piece.piece_confidence * confidence.mean()),
        )


def covariance_weighted_kabsch(
    source: np.ndarray,
    target: np.ndarray,
    covariance: np.ndarray,
) -> tuple[float, np.ndarray, float]:
    from .robust_pose import weighted_kabsch

    cov = np.asarray(covariance, dtype=np.float64)
    weights = 1.0 / np.maximum(np.trace(cov, axis1=1, axis2=2), 1e-12)
    return weighted_kabsch(source, target, weights)
