#!/usr/bin/env python3
"""视觉程序接入 WriterController 的最小示例。"""

from __future__ import annotations

try:
    from .writer_controller import WriterController
except ImportError:
    from writer_controller import WriterController


PORT = "/dev/ttyUSB0"


def main() -> None:
    controller = WriterController()
    try:
        controller.connect(port=PORT)

        # 每次上电：人工把 XY 放到约定最右下角、笔放到抬笔基准高度，
        # 然后调用一次 home()，把当前位置定义为 X0/Y0/Z0。
        input("人工归位完成后按 Enter...")
        controller.home()
        # home() 同时设置 $1=255；后续运动完成后电机继续使能保持位置。

        # up_z/down_z 单位 mm；speed 单位 mm/min。
        controller.configure_pen(up_z=0.0, down_z=2.0, speed=3000)
        controller.pen_up()

        # x/y 是用户绝对坐标，单位 mm；feedrate 单位 mm/min。
        # 下面表示以 1500mm/min 移动到 X=20mm、Y=10mm。
        controller.move_to(x=20.0, y=10.0, feedrate=1500)

        # 连续路径：内部以 ok 做流控，整条路径 Idle 后才返回。
        controller.move_path(
            points=[(30.0, 10.0), (30.0, 20.0), (20.0, 20.0)],
            feedrate=2000,
        )

        controller.pen_down()
        # 相对当前笔高向下移动 0.1mm，速度是 200mm/min。
        controller.pen_jog(dz=0.1, feedrate=200)
        # 再向上移动 0.1mm，回到微调前高度。
        controller.pen_jog(dz=-0.1, feedrate=200)
        controller.pen_up()
        controller.move_to(x=0.0, y=0.0)
    finally:
        if controller.is_connected():
            controller.disconnect()


if __name__ == "__main__":
    main()
