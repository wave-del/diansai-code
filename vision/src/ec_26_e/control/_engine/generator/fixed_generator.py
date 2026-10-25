"""题目规定的固定四片模板。"""

from __future__ import annotations

from functools import lru_cache
from dataclasses import dataclass

from ..models import Piece
from .validator import validate_partition

FIXED_WIDTH_MM = 100.0
FIXED_HEIGHT_MM = 60.0
# 固定拼图的纸片必须保持非重叠；该目标缝为视觉与机械落位提供余量。
FIXED_TARGET_SEAM_GAP_MM = 10.0

# 固定模式的唯一最终落位。坐标来自
# outputs/analysis/fixed_static_target_center_cw245deg/final_target_positions.svg。
# 识别阶段只负责把实物片映射到 template_id；不得再按单帧观测重排此布局。
FIXED_FINAL_TARGET_TRANSFORMS: dict[
    str, tuple[float, tuple[float, float]]
] = {
    "fixed_piece_0": (0.0, (50.0, 180.55)),
    "fixed_piece_1": (0.0, (53.0, 189.75)),
    "fixed_piece_2": (0.0, (53.0, 193.95)),
    # 片 3 绕其模板质心顺时针旋转 2.45° 后的刚体落位。
    "fixed_piece_3": (2.45, (61.922464, 182.433444)),
}


def fixed_final_target_transform(
    template_id: str,
) -> tuple[float, tuple[float, float]]:
    """返回固定模板到 A4 最终落位的刚体变换。"""
    try:
        return FIXED_FINAL_TARGET_TRANSFORMS[template_id]
    except KeyError as error:
        raise ValueError(f"未知固定模板: {template_id}") from error


@dataclass(frozen=True)
class PhysicalTemplateEdge:
    edge_index: int
    start_corner_index: int
    end_corner_index: int
    start: tuple[float, float]
    end: tuple[float, float]


@dataclass(frozen=True)
class PhysicalContourTemplate:
    piece_template_id: str
    corner_vertices: tuple[tuple[float, float], ...]
    physical_edges: tuple[PhysicalTemplateEdge, ...]


@dataclass(frozen=True)
class AssemblyAnchor:
    anchor_id: str
    host_piece_template_id: str
    host_physical_edge_index: int
    edge_parameter_t: float
    incident_piece_template_ids: tuple[str, ...]
    semantic_role: str


def _contour(
    piece_id: str,
    corners: tuple[tuple[float, float], ...],
    *,
    edge_overrides: dict[
        int, tuple[int, int, tuple[float, float], tuple[float, float]]
    ] | None = None,
) -> PhysicalContourTemplate:
    overrides = edge_overrides or {}
    edges = []
    for index in range(len(corners)):
        start_index, end_index = index, (index + 1) % len(corners)
        start, end = corners[start_index], corners[end_index]
        if index in overrides:
            start_index, end_index, start, end = overrides[index]
        edges.append(PhysicalTemplateEdge(
            index, start_index, end_index, start, end
        ))
    return PhysicalContourTemplate(piece_id, corners, tuple(edges))


@lru_cache(maxsize=1)
def fixed_physical_contours() -> tuple[PhysicalContourTemplate, ...]:
    """Authoritative observable contours; collinear assembly nodes are absent."""

    return (
        _contour("fixed_piece_0", (
            (0.0, 0.0), (20.0, 0.0), (36.0, 12.0), (0.0, 20.0),
        )),
        _contour("fixed_piece_1", (
            (0.0, 20.0), (36.0, 12.0), (76.0, 42.0), (0.0, 30.0),
        )),
        _contour("fixed_piece_2", (
            (0.0, 30.0), (76.0, 42.0), (100.0, 60.0), (0.0, 60.0),
        )),
        # Physical polygon order is 20,0 -> 100,0 -> 100,60.  Edge 2 is
        # parameterized in the opposite direction so historical assembly
        # anchors retain the documented t=0.20 and t=0.70 semantics.
        _contour(
            "fixed_piece_3",
            ((20.0, 0.0), (100.0, 0.0), (100.0, 60.0)),
            edge_overrides={
                2: (0, 2, (20.0, 0.0), (100.0, 60.0)),
            },
        ),
    )


def fixed_assembly_anchors() -> tuple[AssemblyAnchor, ...]:
    return (
        AssemblyAnchor(
            "fixed_anchor_020",
            "fixed_piece_3",
            2,
            0.20,
            ("fixed_piece_0", "fixed_piece_1", "fixed_piece_3"),
            "virtual_interface_junction",
        ),
        AssemblyAnchor(
            "fixed_anchor_070",
            "fixed_piece_3",
            2,
            0.70,
            ("fixed_piece_1", "fixed_piece_2", "fixed_piece_3"),
            "virtual_interface_junction",
        ),
    )


def fixed_anchor_local_point(anchor: AssemblyAnchor) -> tuple[float, float]:
    contour = next(
        item for item in fixed_physical_contours()
        if item.piece_template_id == anchor.host_piece_template_id
    )
    edge = contour.physical_edges[anchor.host_physical_edge_index]
    start, end = edge.start, edge.end
    t = float(anchor.edge_parameter_t)
    return (
        float(start[0] + t * (end[0] - start[0])),
        float(start[1] + t * (end[1] - start[1])),
    )


def fixed_pieces() -> list[Piece]:
    pieces = [
        Piece(item.piece_template_id, list(item.corner_vertices))
        for item in fixed_physical_contours()
    ]
    validate_partition(pieces, FIXED_WIDTH_MM, FIXED_HEIGHT_MM, min_edge_mm=1.0, min_area_mm2=1.0)
    return pieces
