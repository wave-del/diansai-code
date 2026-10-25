#!/usr/bin/env python3
"""Small repeatable regression set for the standalone geometry solver."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import PuzzleDocument
from core.geometry import matrix_pose_error
from core.solver import RectanglePuzzleSolver


def has_truth(document: PuzzleDocument, result) -> bool:
    for candidate in result.candidates:
        root = min(candidate.poses)
        global_pose = document.true_pose(root) @ np.linalg.inv(candidate.poses[root])
        errors = [matrix_pose_error(global_pose @ candidate.poses[piece_id], document.true_pose(piece_id)) for piece_id in candidate.poses]
        if max(error[0] for error in errors) < 0.5 and max(error[1] for error in errors) < math.radians(0.5):
            return True
    return False


def has_trusted_rectangle(result) -> bool:
    """An artificial cut can admit a second legal rectangle.

    The contest guarantees uniqueness, but this compact regression T-cut does
    not: it can also form a 50 x 120 mm rectangle.  Once early convergence is
    enabled, either the original pose or such a fully validated rectangle is a
    correct solver outcome.
    """
    return any(
        candidate.metrics["coverage"] >= 0.99
        and candidate.metrics["perimeter_ratio"] >= 0.99
        and candidate.metrics["overlap_mm2"] <= 0.80
        and candidate.metrics["frame_corners"] >= 4.0
        and candidate.metrics["outer_piece_count"] >= len(candidate.poses)
        for candidate in result.candidates
    )


def run_case(name: str, cuts: list[tuple[int, tuple[float, float], tuple[float, float]]]) -> None:
    document = PuzzleDocument(100.0, 60.0)
    for cut in cuts:
        ok, message = document.cut(*cut)
        assert ok, message
    document.scatter(20260802)
    result = RectanglePuzzleSolver(beam_width=120, top_k=120).solve(document.solver_pieces(), timeout_s=20.0)
    assert result.candidates, f"{name}: no rectangle candidate"
    assert has_truth(document, result) or has_trusted_rectangle(result), f"{name}: no valid rectangle candidate"
    print(f"PASS {name}: {len(result.candidates)} candidates, {result.elapsed_s:.2f}s")


def run_polyline_case() -> None:
    document = PuzzleDocument(100.0, 60.0)
    # Exterior start -> interior bend -> exterior end.  Each output is a valid
    # five-sided V-shaped piece rather than a result of an infinite straight cut.
    ok, message = document.cut_polyline(0, [(-10.0, 30.0), (50.0, 15.0), (110.0, 30.0)])
    assert ok, message
    document.scatter(20260802)
    result = RectanglePuzzleSolver(beam_width=120, top_k=120).solve(document.solver_pieces(), timeout_s=20.0)
    assert result.candidates, "polyline: no rectangle candidate"
    assert has_truth(document, result) or has_trusted_rectangle(result), "polyline: no valid rectangle candidate"
    print(f"PASS polyline: {len(result.candidates)} candidates, {result.elapsed_s:.2f}s")


def run_saved_case(filename: str) -> None:
    """回归真实手工切割案例，覆盖折线和分段长边的数值闭环。"""

    document = PuzzleDocument()
    document.load(str(Path(__file__).resolve().parents[1] / "cases" / filename))
    result = RectanglePuzzleSolver().solve(
        document.solver_pieces(), timeout_s=28.0
    )
    assert result.candidates, f"{filename}: no rectangle candidate"
    assert has_truth(document, result) or has_trusted_rectangle(result), (
        f"{filename}: no valid rectangle candidate"
    )
    print(
        f"PASS {filename}: {len(result.candidates)} candidates, "
        f"{result.elapsed_s:.2f}s"
    )


def main() -> None:
    run_case("full-edge", [(0, (43, -10), (43, 70))])
    # A 60-mm edge is jointly matched by two 30-mm edges: a T-junction case.
    run_case(
        "t-junction",
        [
            (0, (50, -10), (50, 70)),
            (0, (-10, 30), (55, 30)),
        ],
    )
    run_polyline_case()
    run_saved_case("case4p-fali-7.json")


if __name__ == "__main__":
    main()
