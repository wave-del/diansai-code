"""把控制算法 MovePlan 纯计算装配为安全的机器移动周期。"""

from __future__ import annotations

import math

from ec_26_e.actuation.contracts import (
    ActuationConfig,
    ActuationValidationError,
    TransferCycle,
)
from solver2 import MovePlan


def _quantize_coordinate(
    name: str,
    value_mm: float,
    minimum_mm: float,
    maximum_mm: float,
) -> int:
    """校验并把非负机器毫米坐标四舍五入为 0.01 mm。"""

    if isinstance(value_mm, bool) or not isinstance(value_mm, (int, float)):
        raise ActuationValidationError(f"{name} 必须是实数")
    value = float(value_mm)
    if not math.isfinite(value):
        raise ActuationValidationError(f"{name} 必须是有限数")
    if not minimum_mm <= value <= maximum_mm:
        raise ActuationValidationError(
            f"{name}={value:.3f} mm 超出软件限位 "
            f"[{minimum_mm:.3f}, {maximum_mm:.3f}]"
        )
    units = math.floor(value * 100.0 + 0.5)
    quantized = units / 100.0
    if not minimum_mm <= quantized <= maximum_mm:
        raise ActuationValidationError(f"{name} 量化后超出软件限位")
    return units


def _validate_rotation(value_degrees: float) -> float:
    """校验 MovePlan 的非负顺时针定向旋转角。"""

    if isinstance(value_degrees, bool) or not isinstance(
        value_degrees,
        (int, float),
    ):
        raise ActuationValidationError("旋转目标必须是实数")
    value = float(value_degrees)
    if not math.isfinite(value):
        raise ActuationValidationError("旋转目标必须是有限数")
    if not 0.0 <= value <= 360.0:
        raise ActuationValidationError("旋转目标必须位于 [0, 360] 度")
    return value


def compile_move_plan(
    plan: MovePlan,
    config: ActuationConfig,
) -> tuple[TransferCycle, ...]:
    """在硬件访问前把完整 MovePlan 转换为冻结移动周期。"""

    if not isinstance(plan, MovePlan):
        raise TypeError("plan 必须是 MovePlan")
    if not isinstance(config, ActuationConfig):
        raise TypeError("config 必须是 ActuationConfig")
    if not plan.moves:
        raise ActuationValidationError("MovePlan 不包含任何移动")

    cycles: list[TransferCycle] = []
    seen_ids: set[str] = set()
    for expected_order, move in enumerate(plan.moves, start=1):
        if move.order != expected_order:
            raise ActuationValidationError("MovePlan 移动顺序必须从 1 连续递增")
        if move.piece_id in seen_ids:
            raise ActuationValidationError(
                f"MovePlan 包含重复纸片 ID：{move.piece_id}"
            )
        seen_ids.add(move.piece_id)
        pick_x = config.a4_origin_x_mm + move.pick_xy_a4_mm[0]
        pick_y = config.a4_origin_y_mm + move.pick_xy_a4_mm[1]
        place_x = config.a4_origin_x_mm + move.place_xy_a4_mm[0]
        place_y = config.a4_origin_y_mm + move.place_xy_a4_mm[1]
        cycles.append(
            TransferCycle(
                order=move.order,
                piece_id=move.piece_id,
                pick_x_0p01mm=_quantize_coordinate(
                    "拾取 X",
                    pick_x,
                    config.x_min_mm,
                    config.x_max_mm,
                ),
                pick_y_0p01mm=_quantize_coordinate(
                    "拾取 Y",
                    pick_y,
                    config.y_min_mm,
                    config.y_max_mm,
                ),
                place_x_0p01mm=_quantize_coordinate(
                    "放置 X",
                    place_x,
                    config.x_min_mm,
                    config.x_max_mm,
                ),
                place_y_0p01mm=_quantize_coordinate(
                    "放置 Y",
                    place_y,
                    config.y_min_mm,
                    config.y_max_mm,
                ),
                rotation_degrees_clockwise=_validate_rotation(
                    move.rotation_deg_clockwise
                ),
            )
        )
    return tuple(cycles)
