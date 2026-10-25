"""验证 F32C 旋转轴封装的环境配置和生命周期委托。

该文件只使用内存电机替身，不访问 UART 或 GPIO；可通过
``python -m unittest tests.unit.motion.test_foc_rotation`` 独立运行。
"""

from __future__ import annotations

import unittest

from ec_26_e.motion.f32c_controller import (
    F32CMotor,
    F32CProtocolError,
    FUNC_MULTI_TURN_ANGLE,
)
from ec_26_e.motion.foc_rotation import FocRotationAxis, FocRotationConfig


class FakeF32CMotor:
    """记录旋转轴包装调用的已验收 F32C 接口替身。"""

    def __init__(self) -> None:
        """初始化关闭、失能状态及调用记录。"""

        self.is_connected = False
        self.is_enabled = False
        self.calls: list[object] = []

    def connect(self, verify: bool = False) -> None:
        """记录安全连接请求。"""

        self.calls.append(("connect", verify))
        self.is_connected = True

    def enable(self, rpm: int = 15) -> None:
        """记录参考实现的安全使能请求。"""

        self.calls.append(("enable", rpm))
        self.is_enabled = True

    def disable(self) -> None:
        """记录失能请求。"""

        self.calls.append("disable")
        self.is_enabled = False

    def close(self, disable: bool = True) -> None:
        """记录关闭请求并模拟参考实现的默认失能。"""

        self.calls.append(("close", disable))
        if disable:
            self.is_enabled = False
        self.is_connected = False

    def get_angle(self) -> float:
        """返回固定的机械角度反馈。"""

        self.calls.append("get_angle")
        return 12.5

    def move_clockwise_from_zero(
        self,
        target_degrees: float,
        rpm: int | None = None,
        wait: bool = True,
        timeout: float | None = None,
    ) -> float:
        """记录顺时针定向目标并返回模拟内部多圈目标。"""

        self.calls.append(
            ("clockwise", target_degrees, rpm, wait, timeout)
        )
        return -42.0

    def return_counterclockwise_to_zero(
        self,
        target_degrees: float,
        rpm: int | None = None,
        wait: bool = True,
        timeout: float | None = None,
    ) -> float:
        """记录从目标逆时针回零请求。"""

        self.calls.append(
            ("counterclockwise", target_degrees, rpm, wait, timeout)
        )
        return 0.0

    def rotate_one_turn(self, direction: int = 1, rpm: int | None = None, wait: bool = True, timeout: float | None = None) -> float:
        """记录完整一圈请求。"""

        self.calls.append(("turn", direction, rpm, wait, timeout))
        return 360.0


class _OpenFakeUart:
    """让控制器认为离线替身处于已连接状态。"""

    is_open = True


class RecordingF32CMotor(F32CMotor):
    """绕过 UART 查询，仅记录多圈 T 型目标报文。"""

    def __init__(
        self,
        *,
        total_angle: float,
        mechanical_angle: float,
        clockwise_sign: int = 1,
    ) -> None:
        """注入当前反馈并直接置为离线使能状态。"""

        super().__init__(clockwise_mechanical_sign=clockwise_sign)
        self._uart = _OpenFakeUart()  # type: ignore[assignment]
        self._enabled = True
        self.total_angle = total_angle
        self.mechanical_angle = mechanical_angle
        self.sent: list[bytes] = []

    def get_total_angle(self) -> float:
        """返回固定多圈反馈。"""

        return self.total_angle

    def get_angle(self) -> float:
        """返回固定机械反馈。"""

        return self.mechanical_angle

    def _send(self, frame: bytes) -> None:
        """记录报文且不执行串口写入。"""

        self.sent.append(frame)

    @property
    def last_total_target(self) -> float:
        """解码最后一条多圈位置目标。"""

        frames = [
            frame
            for frame in self.sent
            if frame[2] == FUNC_MULTI_TURN_ANGLE
        ]
        value = int.from_bytes(frames[-1][3:7], "big", signed=True)
        return value / 10.0


class F32CDirectedMotionTest(unittest.TestCase):
    """验证多圈目标严格遵循顺时针去、逆时针回。"""

    def test_zero_and_small_target_keep_directed_semantics(self) -> None:
        """0° 不运动，90° 则按配置的顺时针方向往返。"""

        zero = RecordingF32CMotor(
            total_angle=10.0,
            mechanical_angle=0.0,
        )
        self.assertEqual(
            zero.move_clockwise_from_zero(0.0, wait=False),
            10.0,
        )
        self.assertFalse(
            any(frame[2] == FUNC_MULTI_TURN_ANGLE for frame in zero.sent)
        )

        outbound = RecordingF32CMotor(
            total_angle=10.0,
            mechanical_angle=0.0,
        )
        outbound.move_clockwise_from_zero(90.0, wait=False)
        self.assertEqual(outbound.last_total_target, -80.0)

        inbound = RecordingF32CMotor(
            total_angle=-80.0,
            mechanical_angle=90.0,
        )
        inbound.return_counterclockwise_to_zero(90.0, wait=False)
        self.assertEqual(inbound.last_total_target, 10.0)

    def test_clockwise_installation_sign_reverses_protocol_target(self) -> None:
        """反向安装只改变协议目标符号，不改变业务顺逆时针语义。"""

        motor = RecordingF32CMotor(
            total_angle=10.0,
            mechanical_angle=0.0,
            clockwise_sign=-1,
        )
        motor.move_clockwise_from_zero(90.0, wait=False)

        self.assertEqual(motor.last_total_target, 100.0)

    def test_large_target_does_not_choose_shortest_path(self) -> None:
        """270° 应完整顺时针执行，而不是反向走 90°。"""

        outbound = RecordingF32CMotor(
            total_angle=10.0,
            mechanical_angle=0.0,
        )
        returned = outbound.move_clockwise_from_zero(270.0, wait=False)

        self.assertEqual(returned, -260.0)
        self.assertEqual(outbound.last_total_target, -260.0)

        inbound = RecordingF32CMotor(
            total_angle=-260.0,
            mechanical_angle=270.0,
        )
        returned = inbound.return_counterclockwise_to_zero(
            270.0,
            wait=False,
        )

        self.assertEqual(returned, 10.0)
        self.assertEqual(inbound.last_total_target, 10.0)

    def test_360_is_a_real_turn_in_each_direction(self) -> None:
        """360° 去目标和回零都必须产生完整一圈多圈位移。"""

        outbound = RecordingF32CMotor(
            total_angle=10.0,
            mechanical_angle=0.0,
        )
        outbound.move_clockwise_from_zero(360.0, wait=False)
        self.assertEqual(outbound.last_total_target, -350.0)

        inbound = RecordingF32CMotor(
            total_angle=-350.0,
            mechanical_angle=0.0,
        )
        inbound.return_counterclockwise_to_zero(360.0, wait=False)
        self.assertEqual(inbound.last_total_target, 10.0)

    def test_invalid_target_and_wrong_start_position_are_rejected(
        self,
    ) -> None:
        """非法角度或错误起始位置不得发送位置目标。"""

        for target in (-0.1, 360.1, float("inf")):
            with self.subTest(target=target):
                motor = RecordingF32CMotor(
                    total_angle=0.0,
                    mechanical_angle=0.0,
                )
                with self.assertRaises(ValueError):
                    motor.move_clockwise_from_zero(target, wait=False)

        motor = RecordingF32CMotor(
            total_angle=0.0,
            mechanical_angle=10.0,
        )
        with self.assertRaises(F32CProtocolError):
            motor.move_clockwise_from_zero(90.0, wait=False)

        motor = RecordingF32CMotor(
            total_angle=-90.0,
            mechanical_angle=80.0,
        )
        with self.assertRaises(F32CProtocolError):
            motor.return_counterclockwise_to_zero(90.0, wait=False)


class FocRotationConfigTest(unittest.TestCase):
    """验证 UART3 默认值、.env 覆盖和非法配置。"""

    def test_defaults_target_verified_uart3(self) -> None:
        """默认配置应对应 Cubie A7Z 的 UART3 和参考速度。"""

        config = FocRotationConfig.from_environ({})

        self.assertEqual(config.port, "/dev/ttyAS3")
        self.assertEqual(config.address, 2)
        self.assertEqual(config.rpm, 15)

    def test_environment_overrides_are_typed(self) -> None:
        """已加载 .env 的 FOC 字段应覆盖默认值。"""

        config = FocRotationConfig.from_environ(
            {
                "FOC_SERIAL_PORT": "/dev/ttyUSB9",
                "FOC_ADDRESS": "7",
                "FOC_RPM": "20",
                "FOC_MAX_RPM": "25",
                "FOC_FEEDBACK_TIMEOUT_SECONDS": "0.6",
                "FOC_FEEDBACK_ATTEMPTS": "4",
                "MOTION_ROTATION_CLOCKWISE_SIGN": "-1",
            }
        )

        self.assertEqual(config.port, "/dev/ttyUSB9")
        self.assertEqual(config.address, 7)
        self.assertEqual(config.rpm, 20)
        self.assertEqual(config.max_rpm, 25)
        self.assertEqual(config.feedback_timeout_seconds, 0.6)
        self.assertEqual(config.feedback_attempts, 4)
        self.assertEqual(config.clockwise_mechanical_sign, -1)

    def test_invalid_environment_is_rejected_before_hardware(self) -> None:
        """空串口、非整数和超过上限的速度不能进入控制层。"""

        for environ in (
            {"FOC_SERIAL_PORT": ""},
            {"FOC_ADDRESS": "2.5"},
            {"FOC_RPM": "31", "FOC_MAX_RPM": "30"},
            {"MOTION_ROTATION_CLOCKWISE_SIGN": "0"},
        ):
            with self.subTest(environ=environ):
                with self.assertRaises((TypeError, ValueError)):
                    FocRotationConfig.from_environ(environ)


class FocRotationAxisTest(unittest.TestCase):
    """验证封装只做生命周期编排，不重写 F32C 控制逻辑。"""

    def test_open_motion_and_close_delegate_to_verified_motor(self) -> None:
        """打开需安全使能，业务方法原样委托，关闭必须要求失能。"""

        motor = FakeF32CMotor()
        axis = FocRotationAxis(FocRotationConfig(rpm=20), motor=motor)

        with self.assertRaisesRegex(RuntimeError, "open"):
            axis.move_clockwise_from_zero(90.0)
        axis.open()
        self.assertEqual(axis.current_angle_deg(), 12.5)
        self.assertEqual(
            axis.move_clockwise_from_zero(
                90.0,
                rpm=18,
                timeout=4.0,
            ),
            -42.0,
        )
        self.assertEqual(
            axis.return_counterclockwise_to_zero(90.0),
            0.0,
        )
        self.assertEqual(axis.rotate_one_turn(direction=-1), 360.0)
        axis.close()

        self.assertEqual(
            motor.calls,
            [
                ("connect", False),
                ("enable", 20),
                "get_angle",
                ("clockwise", 90.0, 18, True, 4.0),
                ("counterclockwise", 90.0, None, True, None),
                ("turn", -1, None, True, None),
                ("close", True),
            ],
        )

    def test_open_failure_closes_with_disable(self) -> None:
        """安全使能失败时必须要求参考控制器关闭并失能。"""

        class FailingMotor(FakeF32CMotor):
            """在使能阶段模拟参考实现抛出的通信异常。"""

            def enable(self, rpm: int = 15) -> None:
                """记录调用后抛出模拟失败。"""

                self.calls.append(("enable", rpm))
                raise RuntimeError("enable failed")

        motor = FailingMotor()
        axis = FocRotationAxis(motor=motor)

        with self.assertRaisesRegex(RuntimeError, "enable failed"):
            axis.open()
        self.assertEqual(
            motor.calls,
            [("connect", False), ("enable", 15), ("close", True)],
        )
