"""边链几何引擎的安全候选，使用现场图像拼缝连续性消歧。"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any, Sequence

import numpy as np
from shapely.geometry import Polygon

from random_white_solver_adapter import _solve_reference
from solver2 import PuzzlePiece
from solver2._engine.transforms import rotation_matrix
from solver.core.models import Candidate, PuzzlePiece as TexturePiece
from solver.core.texture import SceneTextureRanker


TOTAL_BUDGET_SECONDS = 60.0


def _texture_candidate(variant: Any, indices: dict[str, int]) -> Candidate:
    """将紧密拼合位姿转换给图像评分；角度保持引擎的顺时针语义。"""
    poses = {}
    for record in variant.records:
        matrix = np.eye(3)
        matrix[:2, :2] = rotation_matrix(float(record["angle"]))
        matrix[:2, 2] = record["translation"]
        poses[indices[str(record["piece_id"])]] = matrix
    return Candidate(poses=poses, score=0.0, metrics={})


def solve_random_patterned(
    pieces: Sequence[PuzzlePiece], output_dir: Path, *, scene_path: Path | None,
    timeout_s: float | None = None, noise_bound_mm: float = 2.0,
    search_top_n: int = 100, minimum_execution_confidence: float = 0.6,
) -> dict[str, Any]:
    started = perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = {str(piece.piece_id): i for i, piece in enumerate(pieces)}
    texture_pieces = [TexturePiece(i, Polygon(piece.vertices_mm)) for i, piece in enumerate(pieces)]
    summary: dict[str, Any] = {"used": False, "reason": "未提供现场花纹图"}
    ranker = None
    captured_scene = None
    scored: list[tuple[str, Candidate]] = []
    if scene_path is not None and Path(scene_path).is_file():
        try:
            captured_scene = output_dir / "captured_patterned_scene.png"
            if Path(scene_path).resolve() != captured_scene.resolve():
                shutil.copy2(scene_path, captured_scene)
            ranker = SceneTextureRanker(
                captured_scene, texture_pieces, paper_width_mm=207.0, paper_height_mm=292.0,
            )
        except Exception as error:
            summary["reason"] = str(error)

    def score_safe_variants(variants: list[Any]) -> dict[str, Any] | None:
        if ranker is None:
            return None
        scores, minimum_scores = {}, {}
        for variant in variants:
            if timeout_s is not None and perf_counter() >= started + timeout_s - 2.5:
                summary["reason"] = "花纹评分预算耗尽"
                return None
            candidate = _texture_candidate(variant, indices)
            try:
                ranker.score_candidate(candidate)
            except Exception as error:
                summary["reason"] = f"花纹评分失败：{error}"
                return None
            metrics = candidate.metrics
            if (not all(np.isfinite(float(value)) for value in metrics.values())
                    or metrics.get("texture_information", 0.0) < 0.03
                    or metrics.get("texture_seam_count", 0.0) < 1.0
                    or metrics.get("texture_graph_connected",1.0) < .5):
                summary["reason"] = "现场拼缝花纹信息不足"
                return None
            key = str(variant.candidate_id)
            scores[key] = float(metrics["texture_reference_continuity"])
            minimum_scores[key] = float(metrics.get("texture_min_seam",metrics["texture_min_segment"]))
            scored.append((key, candidate))
        summary.update(used=True, reason=None, mode="contour_seams",
                       scored_candidates=len(scored), candidate_scores=scores)
        return {"candidate_scores": scores, "minimum_interface_scores": minimum_scores,
                "minimum_candidate_score": .45}

    result = _solve_reference(
        pieces, output_dir, timeout_s=timeout_s, noise_bound_mm=noise_bound_mm,
        search_top_n=search_top_n, minimum_execution_confidence=minimum_execution_confidence,
        surface_mode="patterned", appearance_scorer=score_safe_variants,
    )
    appearance = result.get("appearance_resolution", {})
    summary["resolved"] = bool(appearance.get("resolved", False))
    summary["block_reason"] = appearance.get("block_reason")
    if ranker is not None and scored:
        previews = []
        ordered = sorted(scored, key=lambda item: item[1].metrics["texture_reference_continuity"], reverse=True)
        for index, (candidate_id, candidate) in enumerate(ordered[:3], 1):
            try:
                preview = ranker.render_reconstruction(
                    candidate, output_dir / "texture_top_candidates" / f"candidate_{index:02d}.png",
                )
                previews.append({"rank": index, "candidate_id": candidate_id, "preview": str(preview),
                                 "continuity": candidate.metrics["texture_reference_continuity"]})
            except Exception as error:
                summary["candidate_preview_error"] = str(error)
                break
        summary["top_candidates"] = previews
    reconstruction = None
    if ranker is not None and result.get("moves"):
        try:
            # 使用最终留缝后的坐标生成预览，与发往执行器的 MovePlan 一致。
            records = [
                {"observed": move["observed_vertices_a4_mm"],
                 "target": move["target_vertices_a4_mm"]}
                for move in result["moves"]
            ]
            reconstruction = ranker.render_planned_reconstruction(
                records, output_dir / "pattern_reconstruction.png",
                status_text="EXECUTION PREVIEW" if result.get("executable") else "REVIEW REQUIRED",
            )
        except Exception as error:
            summary["preview_error"] = str(error)
    result["pattern_texture"] = summary
    result["pattern_scene_path"] = str(captured_scene) if captured_scene else None
    result["pattern_reconstruction_path"] = str(reconstruction) if reconstruction else None
    result["elapsed_seconds"] = perf_counter() - started
    (output_dir / "solution.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
