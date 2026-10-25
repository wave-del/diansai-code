"""验证 MovePlan 无回包执行、时序、FINISH 和安全清理。

该文件只使用假串口、假 GPIO 和假时钟，不访问任何真实设备。可通过
``python -m unittest tests.unit.tasks.test_task1_actuation`` 独立运行。
"""

from __future__ import annotations

from dataclasses import replace
import unittest

from ec_26_e.actuation import (
    ActuationConfig,
    ActuationExecutionError,
    ActuationShutdownError,
    ActuationValidationError,
    MagnetConfig,
    MotionLifecycleState,
    compile_move_plan,
)
from solver2 import MovePlan, PlannedMove
from ec_26_e.motion import (
    FocRotationConfig,
    MksXYController,
    MotionSerialConfig,
    XYState,
)
from ec_26_e.tasks.task1.actuation import MovePlanActuator


class FakeClock:
    """提供由 sleep 精确推进的单调时钟。"""

    def __init__(self) -> None:
        """从零秒开始。"""

        self.now = 0.0

    def monotonic(self) -> float:
        """返回当前假时间。"""

        return self.now

    def sleep(self, seconds: float) -> None:
        """直接推进假时间而不阻塞测试。"""

        self.now += seconds


class FakeTransport:
    """记录无回包串口生命周期和全部报文。"""

    def __init__(self, *, fail_disable: bool = False) -> None:
        """配置可选失能失败并创建空事件。"""

        self.fail_disable = fail_disable
        self.frames: list[bytes] = []
        self.frame_times: list[float] = []
        self.events: list[str] = []
        self.monotonic = lambda: 0.0
        self.opened = False
        self.closed = False

    def open(self) -> None:
        """记录打开。"""

        self.opened = True
        self.events.append("open")

    def send_frame(self, frame: bytes) -> None:
        """记录报文，并可在失能报文处模拟写入失败。"""

        if (
            self.fail_disable
            and len(frame) >= 3
            and frame[1] == 0xF3
            and frame[2] == 0
        ):
            raise RuntimeError("模拟失能写入失败")
        self.frames.append(frame)
        self.frame_times.append(self.monotonic())

    def discard_replies(self) -> None:
        """记录丢弃但不提供任何回包。"""

        self.events.append("discard")

    def take_reply(self) -> None:
        """无回包测试传输始终立即返回 None。"""

        return None

    def close(self) -> None:
        """记录幂等关闭。"""

        self.closed = True
        self.events.append("close")


class FakeMagnet:
    """记录磁铁电平并支持模拟上磁失败。"""

    def __init__(self, *, fail_on_enable: bool = False) -> None:
        """保存失败开关并初始化事件。"""

        self.fail_on_enable = fail_on_enable
        self.values: list[bool] = []
        self.value_times: list[float] = []
        self.monotonic = lambda: 0.0
        self.opened = False
        self.closed = False

    def open(self) -> None:
        """记录 GPIO 申请。"""

        self.opened = True

    def set_enabled(self, enabled: bool) -> None:
        """记录电平，并可在高电平处模拟故障。"""

        if enabled and self.fail_on_enable:
            raise RuntimeError("模拟上磁失败")
        self.values.append(enabled)
        self.value_times.append(self.monotonic())

    def close(self) -> None:
        """记录释放。"""

        self.closed = True


class FakeCompletionLed:
    """记录完成提示灯电平和生命周期。"""

    def __init__(self) -> None:
        """创建空事件记录。"""

        self.values: list[bool] = []
        self.value_times: list[float] = []
        self.monotonic = lambda: 0.0
        self.opened = False
        self.closed = False

    def open(self) -> None:
        """记录 pin33 已申请。"""

        self.opened = True

    def set_enabled(self, enabled: bool) -> None:
        """记录 pin33 目标电平。"""

        self.values.append(enabled)
        self.value_times.append(self.monotonic())

    def close(self) -> None:
        """记录 pin33 已释放。"""

        self.closed = True


class FakeRotation:
    """记录定向旋转、反馈和安全关闭，不访问真实 UART。"""

    def __init__(
        self,
        *,
        fail_move: bool = False,
        transport: FakeTransport | None = None,
    ) -> None:
        """从机械零点开始，并配置可选运动失败。"""

        self.fail_move = fail_move
        self.transport = transport
        self.angle = 0.0
        self.calls: list[object] = []
        self.frames_before_clockwise: tuple[bytes, ...] = ()
        self.frames_before_counterclockwise: list[tuple[bytes, ...]] = []
        self.opened = False
        self.closed = False

    def open(self) -> None:
        """记录旋转轴使能。"""

        self.opened = True
        self.calls.append("open")

    def current_angle_deg(self) -> float:
        """返回当前模拟机械角。"""

        self.calls.append("angle")
        return self.angle

    def move_clockwise_from_zero(
        self,
        target_degrees: float,
        *,
        wait: bool = True,
    ) -> float:
        """记录顺时针目标并更新模拟机械角。"""

        self.calls.append(("clockwise", target_degrees, wait))
        if self.transport is not None:
            self.frames_before_clockwise = tuple(self.transport.frames)
        if self.fail_move:
            raise RuntimeError("模拟旋转失败")
        self.angle = target_degrees % 360.0
        return -target_degrees

    def return_counterclockwise_to_zero(
        self,
        target_degrees: float,
        *,
        wait: bool = True,
    ) -> float:
        """记录逆时针原路回零并提交零位反馈。"""

        self.calls.append(("counterclockwise", target_degrees, wait))
        if self.transport is not None:
            self.frames_before_counterclockwise.append(
                tuple(self.transport.frames)
            )
        self.angle = 0.0
        return 0.0

    def close(self) -> None:
        """记录失能和关闭。"""

        self.calls.append("close")
        self.closed = True


def make_config(**overrides: object) -> ActuationConfig:
    """构造快速、合法且不关联真实设备的运动配置。"""

    values: dict[str, object] = {
        "serial": MotionSerialConfig(),
        "magnet": MagnetConfig(line_offset=12),
        "rotation": FocRotationConfig(),
        "xy_speed_mm_per_second": 1000.0,
        "xy_fixed_wait_seconds": 0.0,
        "z_boot_degrees": 0.0,
        "z_up_degrees": 180.0,
        "z_down_degrees": 0.0,
        "z_enable_wait_seconds": 0.5,
        "z_move_wait_seconds": 0.001,
        "magnet_attach_wait_seconds": 0.001,
        "magnet_release_wait_seconds": 0.001,
        "rotation_zero_wait_seconds": 0.001,
    }
    values.update(overrides)
    return ActuationConfig(**values)  # type: ignore[arg-type]


def make_plan(
    *,
    pick: tuple[float, float] = (10.0, 20.0),
    place: tuple[float, float] = (30.0, 40.0),
    rotation: float = 123.0,
) -> MovePlan:
    """构造一个包含非零旋转但可平移执行的单周期计划。"""

    vertices = ((1.0, 1.0), (2.0, 1.0), (1.0, 2.0))
    move = PlannedMove(
        order=1,
        piece_id="piece-1",
        pick_xy_a4_mm=pick,
        pick_clearance_mm=1.0,
        rotation_deg_clockwise=rotation,
        place_xy_a4_mm=place,
        translation_a4_mm=(20.0, 20.0),
        observed_vertices_a4_mm=vertices,
        target_vertices_a4_mm=vertices,
    )
    return MovePlan(
        mode="fixed",
        confidence=1.0,
        ambiguous=False,
        target_width_mm=1.0,
        target_height_mm=1.0,
        target_center_a4_mm=(30.0, 40.0),
        moves=(move,),
    )


def make_two_piece_plan() -> MovePlan:
    """构造两个位置不同的周期以验证跨周期并行预定位。"""

    plan = make_plan()
    second = replace(
        plan.moves[0],
        order=2,
        piece_id="piece-2",
        pick_xy_a4_mm=(50.0, 60.0),
        place_xy_a4_mm=(70.0, 80.0),
        rotation_deg_clockwise=45.0,
    )
    return replace(plan, moves=(*plan.moves, second))


class MovePlanActuatorTest(unittest.TestCase):
    """验证执行器正常、拒绝、故障和 FINISH 后退出路径。"""

    def _actuator(
        self,
        transport: FakeTransport,
        magnet: FakeMagnet,
        config: ActuationConfig | None = None,
        rotation: FakeRotation | None = None,
    ) -> tuple[MovePlanActuator, FakeClock, FakeCompletionLed]:
        """构造注入全部假外部依赖的执行器。"""

        clock = FakeClock()
        transport.monotonic = clock.monotonic
        magnet.monotonic = clock.monotonic
        led = FakeCompletionLed()
        led.monotonic = clock.monotonic
        actuator = MovePlanActuator(
            config or make_config(),
            transport_factory=lambda unused: transport,
            magnet_factory=lambda unused: magnet,
            completion_indicator_factory=lambda: led,
            rotation_factory=lambda unused: rotation or FakeRotation(),
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
        return actuator, clock, led

    def test_default_calibration_converts_a4_coordinates_to_motor_space(
        self,
    ) -> None:
        """默认标定应把 A4 原点映射为电机的 14.5、6.5 mm 位置。"""

        cycles = compile_move_plan(
            make_plan(pick=(0.0, 0.0), place=(10.125, 20.125)),
            make_config(),
        )

        self.assertEqual(
            (
                cycles[0].pick_x_0p01mm,
                cycles[0].pick_y_0p01mm,
                cycles[0].place_x_0p01mm,
                cycles[0].place_y_0p01mm,
            ),
            (1450, 650, 2463, 2663),
        )

    def test_calibration_override_is_applied_before_limit_validation(
        self,
    ) -> None:
        """环境装配后的标定覆盖应参与坐标换算和越界预检。"""

        config = make_config(a4_origin_x_mm=230.0, a4_origin_y_mm=6.5)
        with self.assertRaisesRegex(ActuationValidationError, "拾取 X"):
            compile_move_plan(make_plan(pick=(0.01, 0.0)), config)

    def test_full_turn_rotation_is_preserved_without_normalization(
        self,
    ) -> None:
        """360° 必须作为真实整圈目标保留，不能归一化为零。"""

        cycle = compile_move_plan(
            make_plan(rotation=360.0),
            make_config(),
        )[0]

        self.assertEqual(cycle.rotation_degrees_clockwise, 360.0)

    def test_normal_path_finishes_before_disabling_and_exits(self) -> None:
        """完整周期应先 FINISH，再失能四个电机并 EXITED。"""

        transport = FakeTransport()
        magnet = FakeMagnet()
        rotation = FakeRotation(transport=transport)
        actuator, _, _ = self._actuator(
            transport,
            magnet,
            rotation=rotation,
        )

        result = actuator.execute(make_plan())

        self.assertEqual(result.state, MotionLifecycleState.EXITED)
        self.assertEqual(result.completed_cycles, 1)
        finish_index = result.state_history.index(
            MotionLifecycleState.FINISH
        )
        shutdown_index = result.state_history.index(
            MotionLifecycleState.SHUTTING_DOWN
        )
        self.assertLess(finish_index, shutdown_index)
        self.assertEqual(result.state_history[-1], MotionLifecycleState.EXITED)
        self.assertEqual(
            [(frame[0], frame[1], frame[2]) for frame in transport.frames[-4:]],
            [
                (0xE0, 0xF3, 0),
                (0xE1, 0xF3, 0),
                (0xE2, 0xF3, 0),
                (0xE3, 0xF3, 0),
            ],
        )
        self.assertFalse(magnet.values[-1])
        self.assertEqual(
            magnet.values,
            [False, True, False, False, False],
        )
        self.assertTrue(magnet.closed)
        self.assertTrue(transport.closed)
        self.assertTrue(rotation.closed)
        self.assertIn(("clockwise", 123.0, True), rotation.calls)
        self.assertEqual(
            [frame[1] for frame in rotation.frames_before_clockwise[-3:]],
            [0xFD, 0xFD, 0xFD],
        )
        self.assertEqual(
            [
                call
                for call in rotation.calls
                if isinstance(call, tuple)
            ],
            [
                ("counterclockwise", 0.0, True),
                ("clockwise", 123.0, True),
                ("counterclockwise", 123.0, True),
            ],
        )

    def test_large_z_transition_is_split_without_reply_waits(self) -> None:
        """180°状态差应拆成 90°报文且不要求 transport 提供回包方法。"""

        transport = FakeTransport()
        actuator, _, _ = self._actuator(transport, FakeMagnet())

        actuator.execute(make_plan())

        z_moves = [
            frame
            for frame in transport.frames
            if frame[0] == 0xE3 and frame[1] == 0xFD
        ]
        self.assertGreaterEqual(len(z_moves), 6)
        self.assertTrue(
            all(int.from_bytes(frame[3:7], "big") <= 3200 for frame in z_moves)
        )

    def test_place_descent_disconnects_magnet_without_extra_wait(self) -> None:
        """放置下降只等待 Z 完成，随后应在同一时刻断开电磁铁。"""

        config = make_config(
            z_up_degrees=90.0,
            z_move_wait_seconds=0.25,
        )
        transport = FakeTransport()
        magnet = FakeMagnet()
        actuator, _, _ = self._actuator(transport, magnet, config)

        actuator.execute(make_plan())

        z_move_times = [
            timestamp
            for frame, timestamp in zip(
                transport.frames,
                transport.frame_times,
                strict=True,
            )
            if frame[0] == 0xE3 and frame[1] == 0xFD
        ]
        self.assertEqual(len(z_move_times), 5)
        placement_descent_started = z_move_times[-2]
        placement_magnet_disabled = magnet.value_times[2]
        self.assertAlmostEqual(
            placement_magnet_disabled - placement_descent_started,
            config.z_move_wait_seconds,
            places=9,
        )

    def test_completion_signal_keeps_z_raised_and_lights_pin33_one_second(
        self,
    ) -> None:
        """末片放下后应回 Z 抬升位，并只点亮 pin33 恰好一秒。"""

        config = make_config(
            z_up_degrees=90.0,
            z_move_wait_seconds=0.25,
            rotation_zero_wait_seconds=0.001,
        )
        transport = FakeTransport()
        magnet = FakeMagnet()
        actuator, _, led = self._actuator(transport, magnet, config)

        actuator.execute(make_plan())

        z_move_times = [
            timestamp
            for frame, timestamp in zip(
                transport.frames,
                transport.frame_times,
                strict=True,
            )
            if frame[0] == 0xE3 and frame[1] == 0xFD
        ]
        self.assertEqual(len(z_move_times), 5)
        completion_enabled = led.value_times[0]
        completion_disabled = led.value_times[1]
        self.assertEqual(led.values, [True, False])
        self.assertTrue(led.opened)
        self.assertTrue(led.closed)
        self.assertEqual(magnet.values, [False, True, False, False, False])
        self.assertAlmostEqual(
            completion_enabled - z_move_times[-1],
            config.z_move_wait_seconds + config.rotation_zero_wait_seconds,
            places=9,
        )
        self.assertAlmostEqual(
            completion_disabled - completion_enabled,
            1.0,
            places=9,
        )

    def test_next_pick_xy_starts_before_rotation_returns_to_zero(self) -> None:
        """Z 抬起后应先下发下一拾取 XY，再阻塞等待旋转轴归零。"""

        transport = FakeTransport()
        magnet = FakeMagnet()
        rotation = FakeRotation(transport=transport)
        actuator, _, _ = self._actuator(
            transport,
            magnet,
            rotation=rotation,
        )

        result = actuator.execute(make_two_piece_plan())

        self.assertEqual(result.completed_cycles, 2)
        self.assertEqual(len(rotation.frames_before_counterclockwise), 3)
        frames_at_first_cycle_return = (
            rotation.frames_before_counterclockwise[1]
        )
        self.assertEqual(
            [frame[0] for frame in frames_at_first_cycle_return[-5:]],
            [0xE3, 0xE3, 0xE0, 0xE1, 0xE2],
        )
        self.assertTrue(
            all(
                frame[1] == 0xFD
                for frame in frames_at_first_cycle_return[-5:]
            )
        )

    def test_xy_wait_uses_longest_axis_plus_fixed_value(self) -> None:
        """XY 外层截止时间应使用两轴距离最大值而不是相加。"""

        config = make_config(
            xy_speed_mm_per_second=10.0,
            xy_fixed_wait_seconds=0.5,
        )
        transport = FakeTransport()
        actuator, clock, _ = self._actuator(
            transport,
            FakeMagnet(),
            config,
        )
        xy = MksXYController(
            transport,
            monotonic=clock.monotonic,
        )
        xy.initialize()
        clock.sleep(0.5)
        xy.process()
        self.assertEqual(xy.state, XYState.READY)
        started = clock.now

        actuator._move_xy_to(1000, 2000, xy)

        self.assertAlmostEqual(clock.now - started, 2.5, places=6)

    def test_preflight_failure_never_constructs_hardware(self) -> None:
        """坐标越界时应在工厂调用之前失败。"""

        factory_calls: list[str] = []
        clock = FakeClock()
        actuator = MovePlanActuator(
            make_config(),
            transport_factory=lambda unused: factory_calls.append("serial"),
            magnet_factory=lambda unused: factory_calls.append("gpio"),
            rotation_factory=lambda unused: factory_calls.append("rotation"),
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

        with self.assertRaises(ActuationValidationError):
            actuator.execute(make_plan(pick=(231.0, 20.0)))

        self.assertEqual(factory_calls, [])
        self.assertEqual(actuator.state, MotionLifecycleState.FAILED)

    def test_execution_failure_demagnetizes_stops_and_disables(self) -> None:
        """FINISH 前异常应先消磁，再尽力停止和失能。"""

        transport = FakeTransport()
        magnet = FakeMagnet(fail_on_enable=True)
        actuator, _, _ = self._actuator(transport, magnet)

        with self.assertRaisesRegex(
            ActuationExecutionError,
            "电机运动失败.*模拟上磁失败",
        ):
            actuator.execute(make_plan())

        self.assertEqual(actuator.state, MotionLifecycleState.FAILED)
        self.assertFalse(magnet.values[-1])
        self.assertTrue(magnet.closed)
        functions = [frame[1] for frame in transport.frames]
        self.assertIn(0xF7, functions)
        self.assertTrue(
            any(frame[1] == 0xF3 and frame[2] == 0 for frame in transport.frames)
        )

    def test_invalid_rotation_is_rejected_before_hardware(self) -> None:
        """负角和超过 360° 的目标必须在构造硬件前失败。"""

        for target in (-0.1, 360.1, float("inf")):
            with self.subTest(target=target):
                factory_calls: list[str] = []
                actuator = MovePlanActuator(
                    make_config(),
                    transport_factory=lambda unused: factory_calls.append(
                        "serial"
                    ),
                    magnet_factory=lambda unused: factory_calls.append("gpio"),
                    rotation_factory=lambda unused: factory_calls.append(
                        "rotation"
                    ),
                )
                with self.assertRaises(ActuationValidationError):
                    actuator.execute(make_plan(rotation=target))
                self.assertEqual(factory_calls, [])

    def test_rotation_failure_uses_emergency_cleanup(self) -> None:
        """旋转异常时不得进入 FINISH，且旋转轴必须失能关闭。"""

        transport = FakeTransport()
        magnet = FakeMagnet()
        rotation = FakeRotation(fail_move=True)
        actuator, _, _ = self._actuator(
            transport,
            magnet,
            rotation=rotation,
        )

        with self.assertRaisesRegex(
            ActuationExecutionError,
            "电机运动失败.*模拟旋转失败",
        ):
            actuator.execute(make_plan(rotation=270.0))

        self.assertEqual(actuator.state, MotionLifecycleState.FAILED)
        self.assertTrue(rotation.closed)
        self.assertNotIn(
            MotionLifecycleState.FINISH,
            actuator.state_history,
        )

    def test_shutdown_failure_preserves_finish_state(self) -> None:
        """FINISH 后失能失败应保留 FINISH 并报告中文安全退出错误。"""

        transport = FakeTransport(fail_disable=True)
        magnet = FakeMagnet()
        actuator, _, _ = self._actuator(transport, magnet)

        with self.assertRaisesRegex(
            ActuationShutdownError,
            "MovePlan 已执行完成，但安全退出失败",
        ):
            actuator.execute(make_plan())

        self.assertEqual(actuator.state, MotionLifecycleState.FINISH)
        self.assertIn(
            MotionLifecycleState.FINISH,
            actuator.state_history,
        )
        self.assertTrue(magnet.closed)


if __name__ == "__main__":
    unittest.main()
