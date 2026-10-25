"""可选 PL 候选距离批处理；只提供粗筛下界，完整几何判定留在 PS。"""
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import combinations
from typing import Protocol, Sequence

import numpy as np


class CandidateScanBackend(Protocol):
    def pair_distances(self, vertices: Sequence[np.ndarray], deltas: np.ndarray) -> np.ndarray | None:
        """返回候选×碎片对距离（mm），按 combinations(range(n), 2) 顺序。

        最大误差 0.02 mm；不支持的几何返回 None，协议/超时必须抛错。
        """
        ...


_backend: ContextVar[CandidateScanBackend | None] = ContextVar('candidate_scan_backend', default=None)
DISTANCE_GUARD_MM = 0.02


@contextmanager
def candidate_acceleration(backend):
    token = _backend.set(backend)
    try:
        yield
    finally:
        _backend.reset(token)


def scan_candidate_distances(vertices, deltas):
    backend = _backend.get()
    if backend is None:
        return None
    distances = backend.pair_distances(vertices, np.asarray(deltas, dtype=float))
    if distances is None:
        return None
    distances = np.asarray(distances, dtype=float)
    if (distances.shape != (len(deltas), len(tuple(combinations(vertices, 2))))
            or not np.isfinite(distances).all() or np.any(distances < 0)):
        raise RuntimeError('候选 PL 返回的距离形状或数值无效')
    return distances


def cannot_improve(distances, adjacency_indices, best_score, minimum_gap, maximum_gap, target_gap):
    """仅在已有无重叠、端点合格候选时，用乐观下界排除更差的候选。

    PL 不报告重叠面积，不接受布局。距离落在误差保护带内时保留原 PS
    检查；顺序和相同分数的先到先得规则不变。
    """
    if best_score is None or best_score[0] != 0 or best_score[1] != 0:
        return False
    gaps = distances[adjacency_indices]
    low, high = np.maximum(0, gaps-DISTANCE_GUARD_MM), gaps+DISTANCE_GUARD_MM
    violation = float(np.sum(np.maximum(0, minimum_gap-high)**2 + np.maximum(0, low-maximum_gap)**2))
    if violation > best_score[2] + 1e-10:
        return True
    # Only compare the next lexicographic term when the previous term is
    # provably no better (both zero); otherwise an improved gap could win.
    if best_score[2] != 0:
        return False
    target_error = float(np.sum(np.maximum(0, low-target_gap)**2 + np.maximum(0, target_gap-high)**2))
    return target_error > best_score[3] + 1e-10
