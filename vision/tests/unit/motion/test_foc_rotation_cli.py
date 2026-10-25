"""验证 F32C 外部旋转轴脚本的参数、互斥操作和安全关闭。

该文件只使用内存轴替身，不访问 UART 或 GPIO；可通过
``python -m unittest tests.unit.motion.test_foc_rotation_cli`` 独立运行。
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from typer.testing import CliRunner

from ec_26_e.motion import FocRotationConfig
from tests.external.motion import control_foc_rotation


class FakeAxis:
    """记录外部诊断脚本调用的旋转轴上下文替身。"""

    def __init__(self, config: FocRotationConfig) -> None:
        """保存传入配置并初始化调用记录。"""

        self.config = config
        self.calls: list[object] = []

    def __enter__(self) -> FakeAxis:
        """记录打开和安全使能完成后的上下文进入。"""

        self.calls.append("enter")
        return self

    def __exit__(self, *args: object) -> None:
        """记录上下文退出时的安全关闭。"""

        self.calls.append("exit")

    def move_clockwise_from_zero(self, angle_deg: float) -> float:
        """记录顺时针定向定位并返回模拟多圈目标。"""

        self.calls.append(("clockwise", angle_deg))
        return -90.0

    def return_counterclockwise_to_zero(
        self,
        angle_deg: float,
    ) -> float:
        """记录从目标逆时针回零并返回模拟多圈目标。"""

        self.calls.append(("counterclockwise", angle_deg))
        return 0.0

    def current_angle_deg(self) -> float:
        """返回模拟机械反馈。"""

        self.calls.append("get_angle")
        return 90.0


class FocRotationCliTest(unittest.TestCase):
    """验证帮助无副作用、角度/回零分支和配置传递。"""

    def setUp(self) -> None:
        """创建隔离的 Typer 调用器。"""

        self.runner = CliRunner()

    def test_help_does_not_run_motion(self) -> None:
        """查看帮助不得构造或打开旋转轴。"""

        result = self.runner.invoke(control_foc_rotation.app, ["--help"])

        self.assertEqual(result.exit_code, 0, result.output)

    def test_angle_command_passes_cli_configuration(self) -> None:
        """角度命令应把命令行配置和目标交给可测试核心函数。"""

        observed: list[tuple[FocRotationConfig, dict[str, object]]] = []

        def fake_run(
            config: FocRotationConfig,
            **kwargs: object,
        ) -> dict[str, object]:
            """记录 CLI 参数并返回最小 JSON 摘要。"""

            observed.append((config, kwargs))
            return {"operation": "move_clockwise_from_zero"}

        with patch.object(
            control_foc_rotation,
            "run_rotation",
            side_effect=fake_run,
        ):
            result = self.runner.invoke(
                control_foc_rotation.app,
                [
                    "--angle",
                    "90",
                    "--port",
                    "/dev/ttyUSB8",
                    "--rpm",
                    "20",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            json.loads(result.stdout),
            {"operation": "move_clockwise_from_zero"},
        )
        self.assertEqual(observed[0][0].port, "/dev/ttyUSB8")
        self.assertEqual(observed[0][0].rpm, 20)
        self.assertEqual(
            observed[0][1],
            {"angle_deg": 90.0, "return_to_zero": False},
        )

    def test_core_zero_and_angle_are_exclusive_and_close_axis(self) -> None:
        """回零应调用对应入口，互斥输入应在打开设备前失败。"""

        created: list[FakeAxis] = []

        def create_axis(config: FocRotationConfig) -> FakeAxis:
            """创建可在断言后检查的记录型轴。"""

            axis = FakeAxis(config)
            created.append(axis)
            return axis

        summary = control_foc_rotation.run_rotation(
            FocRotationConfig(),
            angle_deg=90.0,
            return_to_zero=True,
            axis_factory=create_axis,
        )

        self.assertEqual(
            summary["operation"],
            "return_counterclockwise_to_zero",
        )
        self.assertEqual(
            created[0].calls,
            [
                "enter",
                ("counterclockwise", 90.0),
                "get_angle",
                "exit",
            ],
        )
        with self.assertRaisesRegex(ValueError, "必须指定"):
            control_foc_rotation.run_rotation(
                FocRotationConfig(),
                return_to_zero=True,
                axis_factory=create_axis,
            )


if __name__ == "__main__":
    unittest.main()
