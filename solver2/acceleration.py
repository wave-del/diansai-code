"""可选批量刚体后端；算法库不依赖 PYNQ，默认求解路径不变。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
import json
import logging
from time import perf_counter
import numpy as np


@dataclass(frozen=True)
class RigidFitRequest:
    source: np.ndarray
    target: np.ndarray
    weights: np.ndarray | None = None


@dataclass(frozen=True)
class RigidFitResult:
    angle: float
    translation: np.ndarray
    rms: float

    def pose(self) -> tuple[float, np.ndarray, float]:
        return self.angle, self.translation, self.rms


class RigidFitBackend(Protocol):
    def fit_batch(self, requests: Sequence[RigidFitRequest]) -> list[RigidFitResult]: ...


class ClearanceScanBackend(Protocol):
    """四个凸多边形在 401 个质心扩张比例下的最小间距（mm）。

    不支持的几何返回 None，传输/协议错误必须抛出异常。结果只用于
    带 0.02 mm 数值保护带的粗筛，最终候选仍在 CPU 完整验证。
    """
    def minimum_distances(self, vertices: Sequence[np.ndarray], centroids: np.ndarray) -> np.ndarray | None: ...


def validate_request(request: RigidFitRequest) -> None:
    src, dst = np.asarray(request.source), np.asarray(request.target)
    if src.ndim != 2 or src.shape[1] != 2 or src.shape != dst.shape or len(src) < 2:
        raise ValueError("刚体配准输入必须为相同形状的 Nx2，N >= 2")
    if not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise ValueError("刚体配准坐标必须有限")
    if request.weights is not None:
        w = np.asarray(request.weights)
        if w.shape != (len(src),) or not np.isfinite(w).all() or np.any(w <= 0):
            raise ValueError("权重必须为每点一个有限正数")


class CpuRigidFitBackend:
    """独立 SVD 参考实现，保留原 estimate_rigid/weighted_kabsch 语义。"""

    def fit_batch(self, requests: Sequence[RigidFitRequest]) -> list[RigidFitResult]:
        from ._engine.transforms import estimate_rigid
        from ._engine.solver.robust_pose import weighted_kabsch
        results = []
        for request in requests:
            validate_request(request)
            pose = (estimate_rigid(request.source, request.target) if request.weights is None
                    else weighted_kabsch(request.source, request.target, request.weights))
            results.append(RigidFitResult(*pose))
        return results


_search_backend: ContextVar[RigidFitBackend | None] = ContextVar('search_rigid_backend', default=None)
_search_stats: ContextVar[dict | None] = ContextVar('search_rigid_stats', default=None)


@contextmanager
def search_acceleration(backend: RigidFitBackend | None):
    """一次同步候选枚举独占后端；异常、嵌套调用均恢复原上下文。"""
    token = _search_backend.set(backend)
    stats = dict(batch_calls=0, fit_requests=0, fit_seconds=0.0)
    stats_token = _search_stats.set(stats)
    try:
        yield
    finally:
        _search_backend.reset(token)
        _search_stats.reset(stats_token)
        logging.getLogger(__name__).info('RANDOM_FIT_PERFORMANCE %s', json.dumps(stats))


def fit_search_batch(requests: Sequence[RigidFitRequest]) -> list[RigidFitResult]:
    """随机边链批量配准；CPU 参考与硬件共用严格的结果契约。"""
    backend = _search_backend.get() or CpuRigidFitBackend()
    started = perf_counter()
    try:
        results = backend.fit_batch(requests)
    finally:
        stats = _search_stats.get()
        if stats is not None:
            stats['batch_calls'] += 1
            stats['fit_requests'] += len(requests)
            stats['fit_seconds'] += perf_counter() - started
    if len(results) != len(requests):
        raise RuntimeError('刚体后端返回的批次长度不匹配')
    for result in results:
        if (np.shape(result.translation) != (2,)
                or not np.isfinite([result.angle, *result.translation, result.rms]).all()
                or result.rms < 0):
            raise RuntimeError('刚体后端返回无效结果')
    return results
