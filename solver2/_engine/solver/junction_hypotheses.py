"""由可解释边事件建立全局 T 节点及多路节点假设。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from .atomic_segments import AtomicInterval


@dataclass(frozen=True)
class JunctionHypothesis:
    junction_id: str
    junction_type: str
    participating_pieces: tuple[str, ...]
    participating_edges: tuple[tuple[str, int], ...]
    local_parameters: tuple[tuple[str, int, float], ...]
    candidate_world_point: tuple[float, float] | None
    residuals: tuple[float, ...]
    confidence: float

    def as_json(self) -> dict:
        return asdict(self)


def hypotheses_from_intervals(
    intervals: Iterable[AtomicInterval],
) -> list[JunctionHypothesis]:
    """按投影事件来源聚合同一个候选节点，不读取任何 truth 信息。"""
    grouped: dict[str, list[tuple[AtomicInterval, float]]] = {}
    for interval in intervals:
        for sources, parameter in (
            (interval.start_sources, interval.t_start),
            (interval.end_sources, interval.t_end),
        ):
            for source in sources:
                if source.source_type == "physical_vertex":
                    continue
                grouped.setdefault(source.source_id, []).append(
                    (interval, parameter)
                )
    result: list[JunctionHypothesis] = []
    for index, (source_id, entries) in enumerate(sorted(grouped.items()), 1):
        pieces = tuple(
            sorted({interval.piece_id for interval, _parameter in entries})
        )
        edges = tuple(
            sorted(
                {
                    (interval.piece_id, interval.edge_index)
                    for interval, _parameter in entries
                }
            )
        )
        parameters = tuple(
            sorted(
                {
                    (interval.piece_id, interval.edge_index, float(parameter))
                    for interval, parameter in entries
                }
            )
        )
        if len(edges) >= 3:
            kind = "MULTI_T_JUNCTION"
        else:
            kind = "T_JUNCTION"
        confidence = float(
            min(interval.confidence for interval, _parameter in entries)
        )
        result.append(
            JunctionHypothesis(
                junction_id=f"junction_{index:05d}",
                junction_type=kind,
                participating_pieces=pieces,
                participating_edges=edges,
                local_parameters=parameters,
                candidate_world_point=None,
                residuals=(0.0,),
                confidence=confidence,
            )
        )
    return result


def point_contact_hypothesis(
    junction_id: str,
    participating_pieces: Iterable[str],
    world_point: np.ndarray,
    residual_mm: float,
    confidence: float,
) -> JunctionHypothesis:
    return JunctionHypothesis(
        junction_id=junction_id,
        junction_type="POINT_CONTACT",
        participating_pieces=tuple(sorted(set(participating_pieces))),
        participating_edges=(),
        local_parameters=(),
        candidate_world_point=(
            float(world_point[0]),
            float(world_point[1]),
        ),
        residuals=(float(residual_mm),),
        confidence=float(confidence),
    )
