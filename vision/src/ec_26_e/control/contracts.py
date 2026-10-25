"""定义拼图控制算法的稳定输入、配置、输出和错误契约。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal


A4_WIDTH_MM = 207.0
A4_HEIGHT_MM = 292.0
Point = tuple[float, float]
Vertices = tuple[Point, ...]
PuzzleMode = Literal["fixed", "random"]


def _validate_point(name: str, point: Point) -> None:
    """校验一个强类型二维毫米坐标。"""

    if not isinstance(point, tuple) or len(point) != 2:
        raise TypeError(f"{name} 必须是包含两个实数的元组")
    for value in point:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} 必须只包含实数")
        if not math.isfinite(value):
            raise ValueError(f"{name} 必须只包含有限数")


@dataclass(frozen=True, slots=True)
class PuzzlePiece:
    """描述一片已校准到竖向 A4 毫米坐标的纸片。"""

    piece_id: str
    vertices_mm: Vertices

    def __post_init__(self) -> None:
        """在进入数值算法前校验强类型结构和基础坐标范围。"""

        if not isinstance(self.piece_id, str):
            raise TypeError("piece_id 必须是字符串")
        if not self.piece_id.strip():
            raise ValueError("piece_id 不能为空")
        if not isinstance(self.vertices_mm, tuple):
            raise TypeError("vertices_mm 必须是顶点元组")
        if len(self.vertices_mm) < 3:
            raise ValueError(f"{self.piece_id}: 至少需要 3 个顶点")
        for index, point in enumerate(self.vertices_mm):
            _validate_point(f"{self.piece_id}.vertices_mm[{index}]", point)
            x, y = point
            if not 0.0 <= x <= A4_WIDTH_MM:
                raise ValueError(
                    f"{self.piece_id}: 顶点 x 必须位于 [0, 207] mm"
                )
            if not 0.0 <= y <= A4_HEIGHT_MM:
                raise ValueError(
                    f"{self.piece_id}: 顶点 y 必须位于 [0, 292] mm"
                )


@dataclass(frozen=True, slots=True)
class PlanningOptions:
    """描述一次同步拼图求解的模式、误差和排序参数。"""

    mode: PuzzleMode = "random"
    noise_bound_mm: float = 0.0
    search_top_n: int = 100
    start_xy_a4_mm: Point = (10.0, 10.0)
    minimum_execution_confidence: float = 0.6
    competition_rules: bool = False

    def __post_init__(self) -> None:
        """校验模式、搜索规模、噪声上界和拾取头起点。"""

        if type(self.competition_rules) is not bool:
            raise TypeError('competition_rules 必须为布尔值')

        if self.mode not in {"fixed", "random"}:
            raise ValueError("mode 必须为 fixed 或 random")
        if isinstance(self.noise_bound_mm, bool) or not isinstance(
            self.noise_bound_mm,
            (int, float),
        ):
            raise TypeError("noise_bound_mm 必须是实数")
        if (
            not math.isfinite(self.noise_bound_mm)
            or self.noise_bound_mm < 0
        ):
            raise ValueError("noise_bound_mm 必须是非负有限数")
        if isinstance(self.search_top_n, bool) or not isinstance(
            self.search_top_n,
            int,
        ):
            raise TypeError("search_top_n 必须是整数")
        if self.search_top_n <= 0:
            raise ValueError("search_top_n 必须是正整数")
        if isinstance(self.minimum_execution_confidence, bool) or not isinstance(
            self.minimum_execution_confidence,
            (int, float),
        ):
            raise TypeError("minimum_execution_confidence 必须是实数")
        if not 0.0 <= float(self.minimum_execution_confidence) <= 1.0:
            raise ValueError(
                "minimum_execution_confidence 必须位于 [0, 1]"
            )
        _validate_point("start_xy_a4_mm", self.start_xy_a4_mm)
        x, y = self.start_xy_a4_mm
        if not 0.0 <= x <= A4_WIDTH_MM or not 0.0 <= y <= A4_HEIGHT_MM:
            raise ValueError("start_xy_a4_mm 必须位于竖向 A4 范围内")


@dataclass(frozen=True, slots=True)
class PlannedMove:
    """描述一片纸从观测位置到目标位置的完整平面刚体移动。"""

    order: int
    piece_id: str
    pick_xy_a4_mm: Point
    pick_clearance_mm: float
    rotation_deg_clockwise: float
    place_xy_a4_mm: Point
    translation_a4_mm: Point
    observed_vertices_a4_mm: Vertices
    target_vertices_a4_mm: Vertices
    template_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        """转换为稳定、可直接 JSON 序列化的移动字典。"""

        result: dict[str, object] = {
            "order": self.order,
            "piece_id": self.piece_id,
            "pick_xy_a4_mm": list(self.pick_xy_a4_mm),
            "pick_clearance_mm": self.pick_clearance_mm,
            "rotation_deg_clockwise": self.rotation_deg_clockwise,
            "place_xy_a4_mm": list(self.place_xy_a4_mm),
            "observed_vertices_a4_mm": [
                list(point) for point in self.observed_vertices_a4_mm
            ],
            "target_vertices_a4_mm": [
                list(point) for point in self.target_vertices_a4_mm
            ],
            "rigid_transform": {
                "rotation_deg_clockwise": self.rotation_deg_clockwise,
                "translation_a4_mm": list(self.translation_a4_mm),
            },
        }
        if self.template_id is not None:
            result["template_id"] = self.template_id
        return result


@dataclass(frozen=True, slots=True)
class MovePlan:
    """描述求解成功后的目标矩形和全部有序移动。"""

    mode: PuzzleMode
    confidence: float
    ambiguous: bool
    target_width_mm: float
    target_height_mm: float
    target_center_a4_mm: Point
    moves: tuple[PlannedMove, ...]

    def to_dict(self) -> dict[str, object]:
        """转换为稳定、可直接 JSON 序列化的完整方案字典。"""

        return {
            "mode": self.mode,
            "confidence": self.confidence,
            "ambiguous": self.ambiguous,
            "target_rectangle": {
                "width_mm": self.target_width_mm,
                "height_mm": self.target_height_mm,
                "center_a4_mm": list(self.target_center_a4_mm),
            },
            "moves": [move.to_dict() for move in self.moves],
        }


class PlanningError(RuntimeError):
    """表示输入结构合法但无法构造完整矩形拼图方案。"""
