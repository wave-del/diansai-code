"""带几何来源的物理边参数区间与区间占用工具。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class IntervalEndpointSource:
    """说明一个参数端点为什么存在，而不是只为凑长度人为生成。"""

    source_type: str
    source_id: str
    confidence: float = 1.0


@dataclass(frozen=True)
class AtomicInterval:
    piece_id: str
    edge_index: int
    t_start: float
    t_end: float
    start_point_local: tuple[float, float]
    end_point_local: tuple[float, float]
    length_mm: float
    support_line_local: tuple[tuple[float, float], tuple[float, float]]
    start_sources: tuple[IntervalEndpointSource, ...]
    end_sources: tuple[IntervalEndpointSource, ...]
    confidence: float = 1.0

    @property
    def start_point(self) -> tuple[float, float]:
        return self.start_point_local

    @property
    def end_point(self) -> tuple[float, float]:
        return self.end_point_local

    @property
    def source_types(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    source.source_type
                    for source in self.start_sources + self.end_sources
                }
            )
        )

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    source.source_id
                    for source in self.start_sources + self.end_sources
                }
            )
        )

    def as_json(self) -> dict:
        result = asdict(self)
        result["source_types"] = list(self.source_types)
        result["source_ids"] = list(self.source_ids)
        return result


# 兼容原有导入名；新代码统一按 AtomicInterval 语义使用。
AtomicSegment = AtomicInterval


def point_on_edge(
    vertices: np.ndarray, edge_index: int, parameter: float
) -> np.ndarray:
    start = np.asarray(vertices[edge_index], dtype=float)
    end = np.asarray(vertices[(edge_index + 1) % len(vertices)], dtype=float)
    return start + float(parameter) * (end - start)


def make_atomic_segment(
    piece_id: str,
    vertices: np.ndarray,
    edge_index: int,
    t_start: float = 0.0,
    t_end: float = 1.0,
    *,
    start_sources: Sequence[IntervalEndpointSource] | None = None,
    end_sources: Sequence[IntervalEndpointSource] | None = None,
    confidence: float = 1.0,
    require_sources: bool = False,
) -> AtomicInterval:
    lo, hi = sorted((float(t_start), float(t_end)))
    if lo < -1e-12 or hi > 1.0 + 1e-12 or hi - lo <= 1e-12:
        raise ValueError("原子区间参数必须满足 0 <= t_start < t_end <= 1")
    lo = max(0.0, lo)
    hi = min(1.0, hi)
    edge_id = f"{piece_id}:edge_{int(edge_index)}"
    if start_sources is None and abs(lo) <= 1e-12:
        start_sources = (
            IntervalEndpointSource(
                "physical_vertex", f"{piece_id}:vertex_{int(edge_index)}"
            ),
        )
    if end_sources is None and abs(hi - 1.0) <= 1e-12:
        end_sources = (
            IntervalEndpointSource(
                "physical_vertex",
                f"{piece_id}:vertex_{(int(edge_index) + 1) % len(vertices)}",
            ),
        )
    start_sources = tuple(start_sources or ())
    end_sources = tuple(end_sources or ())
    if require_sources and (not start_sources or not end_sources):
        raise ValueError(
            f"{edge_id}[{lo:.6f},{hi:.6f}] 的两个端点都必须有几何来源"
        )
    start = point_on_edge(vertices, edge_index, lo)
    end = point_on_edge(vertices, edge_index, hi)
    support_start = np.asarray(vertices[edge_index], dtype=float)
    support_end = np.asarray(
        vertices[(edge_index + 1) % len(vertices)], dtype=float
    )
    return AtomicInterval(
        piece_id=piece_id,
        edge_index=int(edge_index),
        t_start=lo,
        t_end=hi,
        start_point_local=(float(start[0]), float(start[1])),
        end_point_local=(float(end[0]), float(end[1])),
        length_mm=float(np.linalg.norm(end - start)),
        support_line_local=(
            (float(support_start[0]), float(support_start[1])),
            (float(support_end[0]), float(support_end[1])),
        ),
        start_sources=start_sources,
        end_sources=end_sources,
        confidence=float(confidence),
    )


def intervals_overlap(
    first: tuple[float, float],
    second: tuple[float, float],
    tolerance: float = 1e-9,
) -> bool:
    """端点相接不算重复占用，只有正长度交叠才冲突。"""
    a0, a1 = sorted(first)
    b0, b1 = sorted(second)
    return min(a1, b1) - max(a0, b0) > tolerance


def interval_is_available(
    occupied: Iterable[tuple[float, float]],
    candidate: tuple[float, float],
    tolerance: float = 1e-9,
) -> bool:
    return not any(
        intervals_overlap(existing, candidate, tolerance)
        for existing in occupied
    )


def merge_intervals(
    intervals: Iterable[tuple[float, float]], tolerance: float = 1e-9
) -> list[tuple[float, float]]:
    ordered = sorted((min(a, b), max(a, b)) for a, b in intervals)
    merged: list[list[float]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1] + tolerance:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(float(start), float(end)) for start, end in merged]


def interval_coverage(
    intervals: Iterable[tuple[float, float]]
) -> tuple[float, float]:
    """返回 [0,1] 上的覆盖率和内部空缺长度（参数单位）。"""
    merged = merge_intervals(intervals)
    coverage = sum(max(0.0, end - start) for start, end in merged)
    gap = 0.0
    cursor = 0.0
    for start, end in merged:
        gap += max(0.0, start - cursor)
        cursor = max(cursor, end)
    gap += max(0.0, 1.0 - cursor)
    return float(min(1.0, coverage)), float(gap)
