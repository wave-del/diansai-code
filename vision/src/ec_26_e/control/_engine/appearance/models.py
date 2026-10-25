"""Data contracts shared by geometry ambiguity and future vision modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ShapeSimilarityResult:
    similarity: float
    edge_count_score: float
    normalized_length_score: float
    angle_score: float
    direction_change_score: float
    scale_score: float
    compactness_score: float
    aspect_ratio_score: float
    best_cyclic_shift: int
    best_rotation_deg: float
    mirrored: bool
    mirrored_similarity: float


@dataclass(frozen=True)
class CandidateGeometryDescriptor:
    candidate_id: str
    piece_vertices: dict[str, Any]
    target_vertices: dict[str, Any]
    rotations_deg: dict[str, float]
    adjacency_pairs: set[frozenset[str]]
    interface_ids: tuple[str, ...]
    width_mm: float
    height_mm: float
    geometry_score: float
    safe: bool = True
    mirrored: bool = False
    variant_id: str = "raw"


@dataclass(frozen=True)
class CandidateSimilarityResult:
    similarity: float
    exchanged_piece_shape_similarity: float
    adjacency_similarity: float
    target_layout_similarity: float
    geometry_score_similarity: float
    piece_mapping: dict[str, str]
    identical_physical_placement: bool
    identity_or_pose_differs: bool
    ambiguity_type: str | None


@dataclass(frozen=True)
class GeometryAmbiguityResult:
    detected: bool
    threshold: float
    candidate_groups: list[list[str]]
    maximum_similarity: float
    ambiguity_type: str | None
    requires_appearance_resolution: bool
    pairwise_diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def as_json(self) -> dict[str, Any]:
        return {
            "detected": self.detected,
            "threshold": self.threshold,
            "candidate_groups": self.candidate_groups,
            "candidate_group_size": max(
                (len(group) for group in self.candidate_groups), default=0
            ),
            "maximum_similarity": self.maximum_similarity,
            "ambiguity_type": self.ambiguity_type,
            "requires_appearance_resolution": (
                self.requires_appearance_resolution
            ),
            "pairwise_diagnostics": self.pairwise_diagnostics,
        }


@dataclass(frozen=True)
class AppearanceResolutionResult:
    required: bool
    completed: bool
    resolved: bool
    winner_candidate_id: str | None
    candidate_scores: dict[str, float]
    minimum_interface_scores: dict[str, float]
    winner_margin: float
    winner_margin_threshold: float
    block_reason: str | None

    def as_json(self) -> dict[str, Any]:
        rows = [
            {
                "candidate_id": candidate_id,
                "score": score,
                "minimum_interface_score": (
                    self.minimum_interface_scores.get(candidate_id)
                ),
            }
            for candidate_id, score in sorted(
                self.candidate_scores.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]
        return {
            "required": self.required,
            "completed": self.completed,
            "resolved": self.resolved,
            "winner_candidate_id": self.winner_candidate_id,
            "winner_margin": self.winner_margin,
            "winner_margin_threshold": self.winner_margin_threshold,
            "candidate_scores": rows,
            "block_reason": self.block_reason,
        }
