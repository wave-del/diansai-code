"""Geometry ambiguity and appearance-resolution interfaces."""

from .ambiguity_detector import detect_geometry_ambiguity
from .candidate_resolver import resolve_candidates_by_appearance
from .shape_similarity import compare_piece_shapes, compare_safe_candidates

__all__ = [
    "compare_piece_shapes",
    "compare_safe_candidates",
    "detect_geometry_ambiguity",
    "resolve_candidates_by_appearance",
]
