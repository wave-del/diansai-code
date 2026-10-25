"""Group complete safe candidates that geometry alone cannot distinguish."""

from __future__ import annotations

from .config import GEOMETRY_AMBIGUITY_THRESHOLD
from .models import CandidateGeometryDescriptor, GeometryAmbiguityResult
from .shape_similarity import compare_safe_candidates


def detect_geometry_ambiguity(
    safe_candidates: list[CandidateGeometryDescriptor],
    threshold: float = GEOMETRY_AMBIGUITY_THRESHOLD,
) -> GeometryAmbiguityResult:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("geometry ambiguity threshold must be in [0, 1]")
    candidates = [
        candidate for candidate in safe_candidates
        if candidate.safe and not candidate.mirrored
    ]
    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        root_a, root_b = find(first), find(second)
        if root_a != root_b:
            parent[root_b] = root_a

    maximum = 0.0
    diagnostics = []
    types: list[str] = []
    for first in range(len(candidates)):
        for second in range(first + 1, len(candidates)):
            result = compare_safe_candidates(
                candidates[first], candidates[second]
            )
            maximum = max(maximum, result.similarity)
            qualifies = bool(
                result.similarity >= threshold
                and not result.identical_physical_placement
                and result.identity_or_pose_differs
            )
            diagnostics.append({
                "candidate_a": candidates[first].candidate_id,
                "candidate_b": candidates[second].candidate_id,
                "similarity": result.similarity,
                "qualifies": qualifies,
                "ambiguity_type": result.ambiguity_type,
                "components": {
                    "piece_shape": (
                        result.exchanged_piece_shape_similarity
                    ),
                    "adjacency": result.adjacency_similarity,
                    "target_layout": result.target_layout_similarity,
                    "geometry_score": result.geometry_score_similarity,
                },
            })
            if qualifies:
                union(first, second)
                if result.ambiguity_type:
                    types.append(result.ambiguity_type)
    grouped: dict[int, list[str]] = {}
    for index, candidate in enumerate(candidates):
        grouped.setdefault(find(index), []).append(candidate.candidate_id)
    groups = sorted(
        (
            sorted(group) for group in grouped.values()
            if len(group) >= 2
        ),
        key=lambda group: (group[0], len(group)),
    )
    ambiguity_type = types[0] if types and len(set(types)) == 1 else (
        "multiple_equivalent_layouts" if types else None
    )
    return GeometryAmbiguityResult(
        detected=bool(groups),
        threshold=float(threshold),
        candidate_groups=groups,
        maximum_similarity=float(maximum),
        ambiguity_type=ambiguity_type,
        requires_appearance_resolution=bool(groups),
        pairwise_diagnostics=diagnostics,
    )
