"""跨模块使用的轻量数据模型。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

Point = tuple[float, float]
Vertices = list[Point]


@dataclass
class Piece:
    piece_id: str
    vertices: Vertices
    source_id: str | None = None


@dataclass
class RigidTransform:
    rotation_deg_clockwise: float
    translation_a4_mm: Point


@dataclass
class Score:
    total: float
    matched_edge_rms_mm: float = 0.0
    matched_vertex_rms_mm: float = 0.0
    outer_edge_rms_mm: float = 0.0
    overlap_area_mm2: float = 0.0
    uncovered_area_mm2: float = 0.0
    outside_area_mm2: float = 0.0
    rectangle_fill_ratio: float = 0.0
    union_connectivity: int = 1
    width_mm: float = 0.0
    height_mm: float = 0.0
    dimension_penalty: float = 0.0
    unmatched_internal_edge_penalty: float = 0.0
    excessive_gap_penalty: float = 0.0
    reflection_penalty: float = 0.0
    optimization_success: bool = True


@dataclass
class Move:
    order: int
    piece_id: str
    pick_xy_a4_mm: Point
    pick_clearance_mm: float
    rotation_deg_clockwise: float
    place_xy_a4_mm: Point
    translation_a4_mm: Point
    observed_vertices_a4_mm: Vertices
    target_vertices_a4_mm: Vertices
    confidence: float
    template_id: str | None = None
    machine_coordinates: dict[str, Any] | None = None
    actions: list[str] = field(default_factory=list)

    def as_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["rigid_transform"] = {
            "rotation_deg_clockwise": data["rotation_deg_clockwise"],
            "translation_a4_mm": data.pop("translation_a4_mm"),
        }
        data.pop("confidence")
        if data["template_id"] is None:
            data.pop("template_id")
        return data

