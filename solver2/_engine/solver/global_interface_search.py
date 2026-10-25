"""全局接口搜索使用的可行域、优先级与规范化状态工具。"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .interface_matching import PairwiseInterfacePlacement


def feasible_width_interval(total_area_mm2: float) -> tuple[float, float]:
    """由 W*H=S、W∈[90,120]、H∈[50,90] 得到可行 W 区间。"""
    return (
        float(max(90.0, total_area_mm2 / 90.0)),
        float(min(120.0, total_area_mm2 / 50.0)),
    )


def placement_priority(
    placement: PairwiseInterfacePlacement,
) -> tuple[float, float, float, float, float, float, float, float, str]:
    """先按有向几何合法层级排序，再比较局部长度残差。"""
    side = placement.opposite_side_result
    return (
        0.0 if side is None or side.valid or side.ambiguous else 1.0,
        0.0 if placement.directed_endpoint_context_valid else 1.0,
        float(placement.score),
        float(placement.directed_endpoint_cost),
        float(-placement.quality_score),
        float(placement.outer_edge_risk),
        float(placement.remaining_dof + placement.ambiguity_penalty),
        float(-placement.geometric_explanation_score),
        placement.interface_id,
    )


def can_embed_in_legal_rectangle(
    points: np.ndarray,
    *,
    tolerance_mm: float = 1e-6,
) -> bool:
    """检查是否存在一个凸包边方向可把当前状态装入 120×90 外框。

    对凸多边形的定向包围盒，临界方向来自凸包边。这里检查所有边，
    而不是错误地只检查最小面积包围盒选中的单一方向。
    """
    vertices = np.asarray(points, dtype=float)
    if len(vertices) <= 1:
        return True
    diagonal_limit = np.hypot(120.0, 90.0) + tolerance_mm
    if any(
        np.linalg.norm(first - second) > diagonal_limit
        for index, first in enumerate(vertices)
        for second in vertices[index + 1 :]
    ):
        return False
    for index, start in enumerate(vertices):
        edge = vertices[(index + 1) % len(vertices)] - start
        norm = float(np.linalg.norm(edge))
        if norm <= 1e-12:
            continue
        axis = edge / norm
        normal = np.asarray([-axis[1], axis[0]])
        width = float(np.ptp(vertices @ axis))
        height = float(np.ptp(vertices @ normal))
        if (
            width <= 120.0 + tolerance_mm
            and height <= 90.0 + tolerance_mm
        ) or (
            width <= 90.0 + tolerance_mm
            and height <= 120.0 + tolerance_mm
        ):
            return True
    return False


def canonical_pose_state(
    piece_poses: Iterable[tuple[str, float, float, float]],
    *,
    exact: bool | None = None,
) -> tuple:
    """消除全局90度等价方向后生成 transposition-table 键。"""
    rows = list(piece_poses)
    if not rows:
        return ()
    # `exact` is a deprecated compatibility argument.  A stable precision is
    # used for every observation so generator labels cannot alter deduplication.
    precision = 5
    encodings = []
    for global_rotation in (0.0, 90.0, 180.0, 270.0):
        radians = np.deg2rad(global_rotation)
        rotation = np.asarray(
            [
                [np.cos(radians), np.sin(radians)],
                [-np.sin(radians), np.cos(radians)],
            ]
        )
        transformed = []
        translations = []
        for piece_id, theta, tx, ty in rows:
            translation = rotation @ np.asarray([tx, ty], dtype=float)
            translations.append(translation)
            transformed.append(
                [
                    piece_id,
                    round((theta + global_rotation) % 360.0, precision),
                    translation,
                ]
            )
        origin = np.min(np.vstack(translations), axis=0)
        encodings.append(
            tuple(
                sorted(
                    (
                        piece_id,
                        theta,
                        round(float(translation[0] - origin[0]), precision),
                        round(float(translation[1] - origin[1]), precision),
                    )
                    for piece_id, theta, translation in transformed
                )
            )
        )
    return min(encodings)
