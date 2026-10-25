"""全局物理尺寸与默认配置。"""

from dataclasses import dataclass

A4_WIDTH_MM = 207.0
A4_HEIGHT_MM = 292.0
DIVIDER_Y_MM = 146.0
TARGET_CENTER_A4_MM = (103.5, 219.0)
GEOMETRY_TOLERANCE_MM = 1e-6
BOUNDARY_LENGTH_TOLERANCE_MM = 1e-6
SHARED_EDGE_LENGTH_TOLERANCE_MM = 1e-5
OVERLAP_NUMERICAL_EPSILON_MM2 = 1e-5
VERTEX_DISTANCE_LIMIT_MM = 20.0
TARGET_WIDTH_MIN_MM = 90.0
TARGET_WIDTH_MAX_MM = 120.0
TARGET_HEIGHT_MIN_MM = 50.0
TARGET_HEIGHT_MAX_MM = 90.0
COORDINATE_SYSTEM = {
    "unit": "mm",
    "origin": "a4_top_left",
    "x_direction": "right",
    "y_direction": "down",
    "positive_rotation": "clockwise",
    "angle_range": "[-180, 180)",
}


@dataclass(frozen=True)
class PlanningConfig:
    """运动顺序配置。"""

    machine_start_a4_mm: tuple[float, float] = (10.0, 10.0)
    safe_z_required: bool = True
