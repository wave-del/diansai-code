"""提供冻结伪真值的毫米几何误差和分阶段性能统计。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ec_26_e.vision.contracts import Detect1Result


def percentile_summary(values: list[float]) -> dict[str, float | None]:
    """返回 mean/P50/P95/max；空样本使用 None。"""

    if not values:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(np.mean(array)), 4),
        "p50": round(float(np.percentile(array, 50)), 4),
        "p95": round(float(np.percentile(array, 95)), 4),
        "max": round(float(np.max(array)), 4),
    }


def _align(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray | None:
    """寻找顶点数相同时不翻转方向的最小循环对齐。"""

    if len(reference) != len(candidate):
        return None
    options = [np.roll(candidate, shift, axis=0) for shift in range(len(candidate))]
    return min(options, key=lambda item: float(np.mean(np.linalg.norm(item - reference, axis=1))))


def _edge_errors(reference: np.ndarray, candidate: np.ndarray, mm_x: float, mm_y: float) -> tuple[list[float], list[float]]:
    """计算候选边相对参考边的法向距离毫米值和方向误差。"""

    distances: list[float] = []
    angles: list[float] = []
    scale = np.asarray([mm_x, mm_y])
    reference_mm = reference * scale
    candidate_mm = candidate * scale
    for ref_start, ref_end, cand_start, cand_end in zip(
        reference_mm,
        np.roll(reference_mm, -1, axis=0),
        candidate_mm,
        np.roll(candidate_mm, -1, axis=0),
    ):
        ref_direction = ref_end - ref_start
        ref_direction /= max(float(np.linalg.norm(ref_direction)), 1e-9)
        normal = np.array([ref_direction[1], -ref_direction[0]])
        distances.extend([abs(float((cand_start - ref_start) @ normal)), abs(float((cand_end - ref_start) @ normal))])
        cand_direction = cand_end - cand_start
        cand_direction /= max(float(np.linalg.norm(cand_direction)), 1e-9)
        angles.append(float(np.degrees(np.arccos(np.clip(abs(ref_direction @ cand_direction), 0.0, 1.0)))))
    return distances, angles


@dataclass(slots=True)
class GeometryAccumulator:
    """累计参考多边形与算法多边形的毫米级误差。"""

    vertex_mm: list[float]
    edge_normal_mm: list[float]
    direction_deg: list[float]
    unmatched_references: int = 0
    unmatched_candidates: int = 0

    @classmethod
    def create(cls) -> GeometryAccumulator:
        """创建空累计器。"""

        return cls([], [], [])

    def add(self, reference_pieces: list[list[list[float]]], result: Detect1Result) -> None:
        """按中心最近邻对齐一帧参考与检测结果并累计。"""

        width, height = result.image_size
        mm_x = 292.0 / max(width - 1, 1)
        mm_y = 207.0 / max(height - 1, 1)
        references = [np.asarray(piece, dtype=np.float64) for piece in reference_pieces]
        candidates = [np.asarray(piece.corners_px, dtype=np.float64) for piece in result.pieces]
        remaining = set(range(len(candidates)))
        for reference in references:
            compatible = [index for index in remaining if len(candidates[index]) == len(reference)]
            if not compatible:
                self.unmatched_references += 1
                continue
            index = min(
                compatible,
                key=lambda slot: float(np.linalg.norm(np.mean(candidates[slot], axis=0) - np.mean(reference, axis=0))),
            )
            remaining.remove(index)
            aligned = _align(reference, candidates[index])
            if aligned is None:
                self.unmatched_references += 1
                continue
            errors = (aligned - reference) * np.asarray([mm_x, mm_y])
            self.vertex_mm.extend(np.linalg.norm(errors, axis=1).tolist())
            distances, angles = _edge_errors(reference, aligned, mm_x, mm_y)
            self.edge_normal_mm.extend(distances)
            self.direction_deg.extend(angles)
        self.unmatched_candidates += len(remaining)

    def to_dict(self) -> dict[str, object]:
        """输出顶点、边和匹配统计。"""

        return {
            "vertex_error_mm": percentile_summary(self.vertex_mm),
            "edge_normal_error_mm": percentile_summary(self.edge_normal_mm),
            "direction_error_deg": percentile_summary(self.direction_deg),
            "unmatched_references": self.unmatched_references,
            "unmatched_candidates": self.unmatched_candidates,
        }
