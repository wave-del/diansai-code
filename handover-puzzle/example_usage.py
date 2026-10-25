#!/usr/bin/env python3
"""在业务代码中调用拼图联动接口的最小示例。"""

from puzzle_controller import (
    PuzzleConfig,
    PuzzleController,
    TransferRequest,
)


def show_status(stage: str, detail: str) -> None:
    print(f"[{stage}] {detail}", flush=True)


config = PuzzleConfig(
    safe_z=-2.0,      # mm；标定完成前的临时保守命令
    pickup_z=0.0,     # mm，自然下垂的取件位置
    place_z=0.0,      # mm，自然下垂的放件位置
    rest_z=0.0,       # mm，回到 XY 原点后的最终落下位置
    xy_feed=2500.0,   # mm/min
    z_feed=1500.0,    # mm/min；实测低速不动，1500时1～3mm线性
    rotation_rpm=15,  # RPM
)

request = TransferRequest(
    pickup_x=50.0,    # mm，即 5 cm
    pickup_y=50.0,    # mm，即 5 cm
    place_x=100.0,    # mm，即 10 cm
    place_y=100.0,    # mm，即 10 cm
    # rotation_angle_deg=90.0,  # 可选；省略时保持当前角度
)

controller = PuzzleController(config=config, status_callback=show_status)
with controller:
    controller.connect(
        writer_port="/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0",
        motor_port="/dev/ttyAS3",
        motor_address=2,
    )

    input(
        "确认机构已到右下角、Z 已自然下垂到下限后，按 Enter 建立零点..."
    )
    controller.establish_origin()
    result = controller.transfer_piece(request)

print(
    f"完成：X={result.final_x:.3f} mm, "
    f"Y={result.final_y:.3f} mm, Z={result.final_z:.3f} mm"
)
