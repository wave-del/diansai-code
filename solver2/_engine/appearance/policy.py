"""Small surface policy layered after the common geometry-safe pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .ambiguity_detector import detect_geometry_ambiguity
from .candidate_resolver import resolve_candidates_by_appearance
from .variant_pool import distinct_safe_variants
from .models import (
    AppearanceResolutionResult,
    CandidateGeometryDescriptor,
    GeometryAmbiguityResult,
)


@dataclass(frozen=True)
class SurfaceSelection:
    selected_variant: Any | None
    geometry_ambiguity: GeometryAmbiguityResult
    appearance_resolution: AppearanceResolutionResult


def default_geometry_ambiguity(
    threshold: float,
) -> GeometryAmbiguityResult:
    return GeometryAmbiguityResult(
        detected=False,
        threshold=float(threshold),
        candidate_groups=[],
        maximum_similarity=0.0,
        ambiguity_type=None,
        requires_appearance_resolution=False,
        pairwise_diagnostics=[],
    )


def default_appearance_resolution(
    margin: float,
) -> AppearanceResolutionResult:
    return AppearanceResolutionResult(
        required=False,
        completed=False,
        resolved=True,
        winner_candidate_id=None,
        candidate_scores={},
        minimum_interface_scores={},
        winner_margin=0.0,
        winner_margin_threshold=float(margin),
        block_reason=None,
    )


def normalize_appearance_observation(
    data: dict[str, Any],
) -> dict[str, Any] | None:
    """Normalize public external-vision fields without reading truth."""
    supplied = data.get("appearance_observation")
    normalized = dict(supplied) if isinstance(supplied, dict) else {}
    if "candidate_scores" not in normalized:
        scores = data.get("appearance_candidate_scores")
        if scores is not None:
            normalized["candidate_scores"] = scores
    if "interface_scores" not in normalized:
        scores = data.get("appearance_interface_scores")
        if scores is not None:
            normalized["interface_scores"] = scores
    return normalized or None


def _variant_appearance_observation(
    data: dict[str, Any],
    variants: list[Any],
) -> dict[str, Any] | None:
    explicit = normalize_appearance_observation(data)
    if explicit is not None:
        return explicit
    if not data.get("appearance_edge_matches"):
        return None
    return {
        "candidate_scores": {
            variant.candidate_id: float(
                variant.candidate.appearance_score
            )
            for variant in variants
        }
    }


def _descriptor(variant: Any) -> CandidateGeometryDescriptor:
    candidate = variant.candidate
    return CandidateGeometryDescriptor(
        candidate_id=variant.candidate_id,
        piece_vertices={
            record["piece_id"]: np.asarray(
                record["observed"], dtype=float
            )
            for record in variant.records
        },
        target_vertices={
            record["piece_id"]: np.asarray(
                record["target"], dtype=float
            )
            for record in variant.records
        },
        rotations_deg={
            record["piece_id"]: float(record["angle"])
            for record in variant.records
        },
        adjacency_pairs={
            frozenset((
                interface.first_piece,
                interface.second_piece,
            ))
            for interface in candidate.interfaces
        },
        interface_ids=tuple(sorted(
            interface.interface_id
            for interface in candidate.interfaces
        )),
        width_mm=float(variant.width),
        height_mm=float(variant.height),
        geometry_score=float(candidate.geometry_score),
        safe=bool(variant.safe),
        mirrored=False,
        variant_id=variant.variant_id,
    )


def resolve_patterned_variants(
    data: dict[str, Any],
    safe_variants: list[Any],
    fallback_variant: Any | None,
    config: Any,
    *,
    appearance_scorer: Any | None = None,
) -> SurfaceSelection:
    """Retain one best variant per discrete candidate, then resolve surface."""
    safe_by_candidate: dict[str, Any] = {}
    for variant in safe_variants:
        if not variant.safe or not variant.machine_executable:
            continue
        previous = safe_by_candidate.get(variant.candidate_id)
        if (
            previous is None
            or variant.quality_key < previous.quality_key
        ):
            safe_by_candidate[variant.candidate_id] = variant
    safe_pool = sorted(
        safe_by_candidate.values(),
        key=lambda item: item.quality_key,
    )[:config.max_appearance_candidates]
    safe_pool = distinct_safe_variants(safe_pool)
    geometry = detect_geometry_ambiguity(
        [_descriptor(variant) for variant in safe_pool],
        threshold=config.geometry_ambiguity_threshold,
    )
    if appearance_scorer is not None:
        appearance_pool = safe_pool
    elif geometry.detected:
        ambiguous_ids = {
            candidate_id
            for group in geometry.candidate_groups
            for candidate_id in group
        }
        appearance_pool = [
            variant for variant in safe_pool
            if variant.candidate_id in ambiguous_ids
        ]
    else:
        appearance_pool = [safe_pool[0]] if safe_pool else []
    observation = (
        appearance_scorer(appearance_pool)
        if appearance_scorer is not None and appearance_pool
        else _variant_appearance_observation(data, appearance_pool)
    )
    appearance = resolve_candidates_by_appearance(
        appearance_pool,
        observation,
        config,
    )
    selected = fallback_variant
    if appearance.resolved:
        selected = next(
            (
                variant for variant in appearance_pool
                if variant.candidate_id
                == appearance.winner_candidate_id
            ),
            fallback_variant,
        )
    return SurfaceSelection(selected, geometry, appearance)
