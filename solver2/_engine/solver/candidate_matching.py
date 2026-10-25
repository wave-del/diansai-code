"""公共边候选筛选。"""

from __future__ import annotations

from dataclasses import dataclass

from .edge_features import EdgeDescriptor


@dataclass
class EdgeMatch:
    first: EdgeDescriptor
    second: EdgeDescriptor
    length_error_mm: float
    score: float


def candidate_edge_matches(edges: list[EdgeDescriptor]) -> list[EdgeMatch]:
    matches: list[EdgeMatch] = []
    for i, first in enumerate(edges):
        for second in edges[i + 1 :]:
            if first.piece_id == second.piece_id:
                continue
            error = abs(first.length - second.length)
            threshold = max(12.0, 0.25 * (first.length + second.length) / 2.0)
            if error <= threshold:
                angle_error = abs(first.start_internal_angle_deg - second.end_internal_angle_deg)
                matches.append(EdgeMatch(first, second, error, error + 0.05 * angle_error))
    return sorted(matches, key=lambda item: item.score)

