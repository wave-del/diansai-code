"""Appearance-score adapter; no image model and no truth access."""

from __future__ import annotations

from typing import Any

from .config import (
    APPEARANCE_WINNER_MARGIN,
    MINIMUM_INTERFACE_APPEARANCE_SCORE,
)
from .models import AppearanceResolutionResult


def _candidate_id(candidate: Any) -> str:
    if isinstance(candidate, str):
        return candidate
    if isinstance(candidate, dict):
        return str(candidate["candidate_id"])
    return str(candidate.candidate_id)


def resolve_candidates_by_appearance(
    candidates: list[Any],
    appearance_observation: dict[str, Any] | None,
    config: Any | None = None,
) -> AppearanceResolutionResult:
    """Resolve prevalidated candidates from external appearance scores only."""
    candidate_ids = [_candidate_id(candidate) for candidate in candidates]
    required = len(candidate_ids) > 1
    margin_threshold = float(
        getattr(config, "appearance_winner_margin", APPEARANCE_WINNER_MARGIN)
    )
    minimum_threshold = getattr(
        config,
        "minimum_interface_appearance_score",
        MINIMUM_INTERFACE_APPEARANCE_SCORE,
    )
    if not appearance_observation:
        return AppearanceResolutionResult(
            required=required,
            completed=False,
            resolved=False,
            winner_candidate_id=None,
            candidate_scores={},
            minimum_interface_scores={},
            winner_margin=0.0,
            winner_margin_threshold=margin_threshold,
            block_reason="appearance_data_missing",
        )
    raw_scores = appearance_observation.get(
        "candidate_scores",
        appearance_observation.get("appearance_candidate_scores", {}),
    )
    if isinstance(raw_scores, list):
        raw_scores = {
            str(item["candidate_id"]): float(
                item.get("appearance_score", item.get("score", 0.0))
            )
            for item in raw_scores
        }
    scores = {
        candidate_id: float(raw_scores[candidate_id])
        for candidate_id in candidate_ids
        if candidate_id in raw_scores
    }
    raw_interfaces = appearance_observation.get(
        "minimum_interface_scores", {}
    )
    if not raw_interfaces:
        interface_scores = appearance_observation.get(
            "interface_scores", {}
        )
        raw_interfaces = {
            candidate_id: min(
                (
                    float(value)
                    for value in interface_scores.get(candidate_id, {}).values()
                ),
                default=1.0,
            )
            for candidate_id in candidate_ids
        }
    minimum_interfaces = {
        candidate_id: float(raw_interfaces.get(candidate_id, 1.0))
        for candidate_id in scores
    }
    if len(scores) != len(candidate_ids):
        return AppearanceResolutionResult(
            required=required,
            completed=False,
            resolved=False,
            winner_candidate_id=None,
            candidate_scores=scores,
            minimum_interface_scores=minimum_interfaces,
            winner_margin=0.0,
            winner_margin_threshold=margin_threshold,
            block_reason="appearance_scores_incomplete",
        )
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    winner_id, winner_score = ordered[0]
    runner_up = ordered[1][1] if len(ordered) > 1 else 0.0
    margin = float(winner_score - runner_up)
    interface_ok = bool(
        minimum_threshold is None
        or minimum_interfaces[winner_id] >= float(minimum_threshold)
    )
    resolved = bool(
        (not required or margin >= margin_threshold) and interface_ok
        and winner_score >= float(appearance_observation.get("minimum_candidate_score",0.0))
    )
    block_reason = None
    if not interface_ok:
        block_reason = "appearance_interface_score_too_low"
    elif winner_score < float(appearance_observation.get("minimum_candidate_score",0.0)):
        block_reason = "appearance_continuity_too_low"
    elif required and margin < margin_threshold:
        block_reason = "appearance_winner_margin_too_small"
    return AppearanceResolutionResult(
        required=required,
        completed=True,
        resolved=resolved,
        winner_candidate_id=winner_id if resolved else None,
        candidate_scores=scores,
        minimum_interface_scores=minimum_interfaces,
        winner_margin=margin,
        winner_margin_threshold=margin_threshold,
        block_reason=block_reason,
    )
