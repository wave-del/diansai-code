"""Unified progressive-search contracts for all random observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class ProgressiveSearchStage:
    name: str
    max_matches_per_edge: int
    beam_width: int
    candidate_limit: int
    node_limit: int
    optimize_candidate_limit: int
    allow_generic_partial: bool
    allow_anchor_switch: bool

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchAmbiguityMetrics:
    interface_candidate_count: int
    mean_candidates_per_edge: float
    maximum_candidates_per_edge: int
    best_second_margin_mean: float
    interface_entropy: float
    candidate_score_margin: float | None
    closure_residual_mm: float | None
    near_safe_candidate_count: int

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


def stages_from_config(config: Any) -> tuple[ProgressiveSearchStage, ...]:
    return (
        ProgressiveSearchStage(
            "fast",
            config.fast_max_matches_per_edge,
            config.fast_beam_width,
            config.fast_candidate_limit,
            config.fast_node_limit,
            config.fast_optimize_candidate_limit,
            False,
            False,
        ),
        ProgressiveSearchStage(
            "expanded",
            config.expanded_max_matches_per_edge,
            config.expanded_beam_width,
            config.expanded_candidate_limit,
            config.expanded_node_limit,
            config.expanded_optimize_candidate_limit,
            True,
            False,
        ),
        ProgressiveSearchStage(
            "recovery",
            config.recovery_max_matches_per_edge,
            config.recovery_beam_width,
            config.recovery_candidate_limit,
            config.max_search_nodes,
            config.recovery_candidate_limit,
            True,
            True,
        ),
    )


def interface_ambiguity_metrics(
    placements: Iterable[Any],
    *,
    candidate_scores: Iterable[float] = (),
    closure_residual_mm: float | None = None,
    near_safe_candidate_count: int = 0,
) -> SearchAmbiguityMetrics:
    groups: dict[tuple[str, int], list[float]] = {}
    rows = list(placements)
    for item in rows:
        groups.setdefault(
            (item.first_piece, item.first_segment.edge_index), []
        ).append(float(item.score))
        groups.setdefault(
            (item.second_piece, item.second_segment.edge_index), []
        ).append(float(item.score))
    counts = [len(values) for values in groups.values()]
    margins: list[float] = []
    entropies: list[float] = []
    for values in groups.values():
        ordered = sorted(values)
        if len(ordered) >= 2:
            margins.append(max(0.0, ordered[1] - ordered[0]))
        weights = np.exp(-np.asarray(ordered, dtype=float))
        probabilities = weights / max(float(weights.sum()), 1e-12)
        entropies.append(float(
            -sum(p * log(max(float(p), 1e-12)) for p in probabilities)
        ))
    scores = sorted(float(value) for value in candidate_scores)
    return SearchAmbiguityMetrics(
        interface_candidate_count=len(rows),
        mean_candidates_per_edge=float(np.mean(counts)) if counts else 0.0,
        maximum_candidates_per_edge=max(counts, default=0),
        best_second_margin_mean=(
            float(np.mean(margins)) if margins else float("inf")
        ),
        interface_entropy=(
            float(np.mean(entropies)) if entropies else 0.0
        ),
        candidate_score_margin=(
            scores[1] - scores[0] if len(scores) >= 2 else None
        ),
        closure_residual_mm=closure_residual_mm,
        near_safe_candidate_count=int(near_safe_candidate_count),
    )
