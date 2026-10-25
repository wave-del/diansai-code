"""调用唯一维护的 solver2 几何引擎生成随机拼图和完整取放计划。"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) in sys.path:
    sys.path.remove(str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR))

from solver2 import PlanningError, PlanningOptions, PuzzlePiece  # noqa: E402
from solver2.planner import _validate_public_input  # noqa: E402
from solver2._engine.solver.solve import solve_observations  # noqa: E402


FINALIZATION_RESERVE_SECONDS = 2.5


def _solve_reference(
    pieces: Sequence[PuzzlePiece],
    output_dir: Path,
    *,
    timeout_s: float | None,
    noise_bound_mm: float,
    search_top_n: int,
    minimum_execution_confidence: float,
    surface_mode: str = "white",
    appearance_scorer: Callable | None = None,
    candidate_target: int | None = None,
    minimum_piece_edge_mm: float = 20.0,
) -> dict[str, Any]:
    """白色和花纹共用边链搜索、联合优化、留缝及最终执行检查。"""
    if timeout_s is not None and (not math.isfinite(timeout_s) or timeout_s <= FINALIZATION_RESERVE_SECONDS + 1):
        raise ValueError("timeout_s 必须大于 3.5 秒且有限")
    if candidate_target is not None and (type(candidate_target) is not int or candidate_target < 1):
        raise ValueError("candidate_target 必须是正整数")
    if not math.isfinite(minimum_piece_edge_mm) or minimum_piece_edge_mm <= 0:
        raise ValueError("minimum_piece_edge_mm 必须是有限正数")
    options = PlanningOptions(
        mode="random", noise_bound_mm=noise_bound_mm, search_top_n=search_top_n,
        minimum_execution_confidence=minimum_execution_confidence,
        competition_rules=True,
    )
    typed_pieces = tuple(pieces)
    if any(not isinstance(piece, PuzzlePiece) for piece in typed_pieces):
        raise TypeError("pieces 必须包含 PuzzlePiece")
    _validate_public_input(typed_pieces)
    for piece in typed_pieces:
        vertices = np.asarray(piece.vertices_mm, dtype=float)
        if not 3 <= len(vertices) <= 5:
            raise PlanningError(f"{piece.piece_id}: 赛题要求每片 3～5 条边")
        lengths = np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1)
        if lengths.min() + 2 * options.noise_bound_mm < minimum_piece_edge_mm - 1e-6:
            raise PlanningError(f"{piece.piece_id}: 最短边不满足 {minimum_piece_edge_mm:g} mm（含端点观测误差）")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0", "mode": "random", "surface_mode": surface_mode,
        "noise_bound_mm": options.noise_bound_mm, "search_top_n": options.search_top_n,
        "pieces": [
            {"id": piece.piece_id, "vertices_mm": [list(p) for p in piece.vertices_mm]}
            for piece in typed_pieces
        ],
    }
    input_path = output_dir / "observations.json"
    input_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    budget = timeout_s - FINALIZATION_RESERVE_SECONDS if timeout_s is not None else 30.0
    try:
        result = solve_observations(
            input_path, output_dir, print_plan=False,
            appearance_scorer=appearance_scorer,
            runtime_config={
                "execution_profile": "production", "execution_policy": "strict",
                "top_n": options.search_top_n,
                "minimum_execution_confidence": options.minimum_execution_confidence,
                "random_competition_rules": True,
                "minimum_piece_edge_mm": minimum_piece_edge_mm,
                "unlimited_search": timeout_s is None,
                "minimum_safe_candidates": candidate_target if candidate_target is not None else (4 if surface_mode == "patterned" else 2),
                "rectangle_corner_hypotheses": True,
                "max_search_seconds": budget, "wall_time_guard_seconds": budget,
                "recovery_total_deadline_seconds": budget,
                "recovery_finalization_reserve_seconds": 1.2,
                "recovery_initial_stage_seconds": min(12.0, budget - 1.2),
                "appearance_resolution_reserve_seconds": 3.0 if surface_mode == "patterned" else 0.0,
                "minimum_interface_appearance_score": 0.4 if surface_mode == "patterned" else None,
            },
        )
    finally:
        logger = logging.getLogger("solver2")
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
    result["solver_backend"] = "directed_edge_chain_reference"
    result["solver_package"] = "solver2"
    result["solver_package_path"] = str(Path(sys.modules["solver2"].__file__).resolve().parent)
    result["minimum_piece_edge_mm"] = minimum_piece_edge_mm
    (output_dir / "solution.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def solve_random_white(
    pieces: Sequence[PuzzlePiece], output_dir: Path, *, timeout_s: float | None = None,
    noise_bound_mm: float = 2.0, search_top_n: int = 100,
    minimum_execution_confidence: float = 0.6,
    candidate_target: int = 2,
    minimum_piece_edge_mm: float = 20.0,
) -> dict[str, Any]:
    """求解几何；候选数量、最短边门槛可由调用方显式配置。"""
    return _solve_reference(
        pieces, output_dir, timeout_s=timeout_s, noise_bound_mm=noise_bound_mm,
        search_top_n=search_top_n, minimum_execution_confidence=minimum_execution_confidence,
        candidate_target=candidate_target,
        minimum_piece_edge_mm=minimum_piece_edge_mm,
    )
