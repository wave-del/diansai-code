#!/usr/bin/env python3
"""F32C Python API 最小接入示例。"""

from __future__ import annotations

from f32c_controller import F32CMotor


def main() -> None:
    motor = F32CMotor(port="/dev/ttyAS3", address=2)
    try:
        # connect()只打开并独占串口，不使能、不查询、不运动。
        motor.connect()

        # enable()低速使能并先保持当前位置，清除驱动板残留目标，
        # 随后选择15 RPM；不会主动改变机械位置。
        motor.enable(rpm=15)
        print("使能后状态:", motor.get_status())

        # 相对于驱动板内持久机械零点移动到90°，沿最短路径，到位后返回。
        motor.move_to(angle_deg=90.0, rpm=15, wait=True)

        # 沿最短路径回到已经保存在驱动板中的机械零点。
        motor.return_to_mechanical_zero(wait=True)

        # 若确实需要从当前位置完整旋转一圈，使用专用接口：
        # motor.rotate_one_turn(direction=1, rpm=15, wait=True)
    finally:
        # 默认先失能，再释放 UART3。
        motor.close(disable=True)


if __name__ == "__main__":
    main()
