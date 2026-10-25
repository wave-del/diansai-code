"""由独立几何约束生成相对 SE(2) 位姿假设。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from ..transforms import estimate_rigid
from .interface_matching import PairwiseInterfacePlacement


@dataclass(frozen=True)
class PoseHypothesis:
    theta: float | None
    tx: float | None
    ty: float | None
    source_constraints: tuple[str, ...]
    independent_constraint_rank: int
    residual: float
    covariance_estimate: tuple[tuple[float, float, float], ...] | None
    deterministic: bool

    def as_json(self) -> dict:
        return asdict(self)


def pose_from_interface(
    placement: PairwiseInterfacePlacement,
    target_segment: np.ndarray,
) -> PoseHypothesis:
    """有限区间的两个有来源端点提供秩2约束，可确定刚体位姿。"""
    source = np.asarray(
        [
            placement.second_segment.start_point,
            placement.second_segment.end_point,
        ],
        dtype=float,
    )
    if not (
        placement.second_segment.start_sources
        and placement.second_segment.end_sources
    ):
        return PoseHypothesis(
            None,
            None,
            None,
            (placement.interface_id,),
            1,
            float("inf"),
            None,
            False,
        )
    theta, translation, residual = estimate_rigid(
        source, np.asarray(target_segment, dtype=float)
    )
    variance = max(float(residual) ** 2, 1e-12)
    return PoseHypothesis(
        float(theta),
        float(translation[0]),
        float(translation[1]),
        (placement.interface_id,),
        2,
        float(residual),
        (
            (variance, 0.0, 0.0),
            (0.0, variance, 0.0),
            (0.0, 0.0, variance),
        ),
        True,
    )


def combine_pose_constraints(
    hypotheses: Iterable[PoseHypothesis],
    tolerance: float,
) -> PoseHypothesis:
    deterministic = [item for item in hypotheses if item.deterministic]
    if not deterministic:
        return PoseHypothesis(None, None, None, (), 0, float("inf"), None, False)
    reference = deterministic[0]
    values = np.asarray(
        [[item.theta, item.tx, item.ty] for item in deterministic], dtype=float
    )
    residual = float(np.max(np.linalg.norm(values - values[0], axis=1)))
    if residual > tolerance:
        return PoseHypothesis(
            None,
            None,
            None,
            tuple(
                constraint
                for item in deterministic
                for constraint in item.source_constraints
            ),
            sum(item.independent_constraint_rank for item in deterministic),
            residual,
            None,
            False,
        )
    mean = values.mean(axis=0)
    return PoseHypothesis(
        float(mean[0]),
        float(mean[1]),
        float(mean[2]),
        tuple(
            constraint
            for item in deterministic
            for constraint in item.source_constraints
        ),
        sum(item.independent_constraint_rank for item in deterministic),
        residual,
        reference.covariance_estimate,
        True,
    )
