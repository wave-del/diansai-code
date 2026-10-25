#!/usr/bin/env python3
"""不连接硬件的交付包自检。"""

from __future__ import annotations

import math

try:
    from .grbl_controller import (
        GrblStatus,
        apply_transform,
        build_absolute_move_command,
        build_pen_z_command,
    )
    from .writer_controller import WriterController
except ImportError:
    from grbl_controller import (
        GrblStatus,
        apply_transform,
        build_absolute_move_command,
        build_pen_z_command,
    )
    from writer_controller import WriterController


def main() -> int:
    status = GrblStatus(
        "<Idle|MPos:10.000,20.000,2.000,0.000|"
        "WCO:1.000,2.000,0.000,0.000>"
    )
    assert status.state == "Idle"
    assert status.work_position == (9.0, 18.0)
    assert apply_transform(10, 20, True, True, True) == (-20, -10)
    assert (
        build_absolute_move_command(-20, -10, 2000)
        == "G21 G90 G1 X-20.000 Y-10.000 F2000"
    )
    assert build_pen_z_command(2, 3000) == "G21 G90 G1 Z2.000 F3000"

    controller = WriterController()
    controller.configure_pen(up_z=0.0, down_z=1.5, speed=800)
    assert controller.pen_up_height == 0.0
    assert controller.pen_down_height == 1.5
    assert controller.pen_speed == 800.0
    try:
        controller.configure_pen(speed=math.nan)
    except ValueError:
        pass
    else:
        raise AssertionError("无效笔轴参数没有被拒绝")

    print("✅ 独立交付包自检通过（未连接硬件）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
