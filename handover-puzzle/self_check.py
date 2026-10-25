#!/usr/bin/env python3
"""不接硬件的联动顺序、错误恢复和参数校验。"""

from __future__ import annotations

import unittest

try:
    from .puzzle_controller import (
        PuzzleConfig,
        PuzzleController,
        PuzzleTransferError,
        TransferRequest,
    )
    from . import test_z_calibration as z_calibration
except ImportError:
    from puzzle_controller import (
        PuzzleConfig,
        PuzzleController,
        PuzzleTransferError,
        TransferRequest,
    )
    import test_z_calibration as z_calibration


class FakeWriter:
    def __init__(self, events: list[tuple], fail_at_xy: tuple | None = None):
        self.events = events
        self.fail_at_xy = fail_at_xy
        self._position = (0.0, 0.0)
        self._z = 0.0
        self._up_z = 0.0

    def connect(self, port: str, baud: int) -> None:
        self.events.append(("writer.connect", port, baud))

    def disconnect(self) -> None:
        self.events.append(("writer.disconnect",))

    def set_motor_hold(self, enable: bool = True) -> None:
        self.events.append(("writer.set_motor_hold", enable))

    def home(self) -> None:
        self.events.append(("writer.home",))
        self._position = (0.0, 0.0)
        self._z = 0.0

    def reanchor_at_a4_position(self, x: float, y: float, z: float) -> None:
        self.events.append(("writer.reanchor_at_a4_position", x, y, z))
        self._position = (x, y)
        self._z = z

    def configure_pen(self, up_z: float, down_z: float, speed: float) -> None:
        self.events.append(("writer.configure_pen", up_z, down_z, speed))
        self._up_z = up_z

    def pen_up(self) -> None:
        self.events.append(("writer.pen_up",))
        self._z = self._up_z

    def pen_move_to(self, z: float, feedrate: float) -> None:
        self.events.append(("writer.pen_move_to", z, feedrate))
        self._z = z

    def move_to(self, x: float, y: float, feedrate: float) -> None:
        self.events.append(("writer.move_to", x, y, feedrate))
        if self.fail_at_xy == (x, y):
            raise RuntimeError("模拟 XY 故障")
        self._position = (x, y)

    def return_to_machine_origin(self, feedrate: float) -> None:
        self.events.append(("writer.return_to_machine_origin", feedrate))
        self._position = (0.0, 0.0)

    def send_gcode(self, command: str) -> None:
        self.events.append(("writer.send_gcode", command))

    @property
    def position(self) -> tuple[float, float]:
        return self._position

    @property
    def pen_position(self) -> float:
        return self._z


class FakeMotor:
    def __init__(self, events: list[tuple]):
        self.events = events

    def connect(self, verify: bool = False) -> None:
        self.events.append(("motor.connect", verify))

    def enable(self, rpm: int) -> None:
        self.events.append(("motor.enable", rpm))

    def move_to(
        self,
        angle: float,
        rpm: int,
        wait: bool,
        tolerance_deg: float = 1.0,
        settle_retries: int = 0,
        allow_inexact_arrival: bool = False,
    ) -> None:
        self.events.append(
            (
                "motor.move_to",
                angle,
                rpm,
                wait,
                tolerance_deg,
                settle_retries,
                allow_inexact_arrival,
            )
        )

    def return_to_mechanical_zero(
        self,
        rpm: int,
        wait: bool,
        tolerance_deg: float = 1.0,
        settle_retries: int = 0,
        allow_inexact_arrival: bool = False,
    ) -> None:
        self.events.append(
            (
                "motor.zero",
                rpm,
                wait,
                tolerance_deg,
                settle_retries,
                allow_inexact_arrival,
            )
        )

    def close(self, disable: bool = True) -> None:
        self.events.append(("motor.close", disable))


class FakeMagnet:
    def __init__(self, events: list[tuple]):
        self.events = events
        self._engaged = False

    def open(self) -> None:
        self.events.append(("magnet.open",))

    def engage(self) -> None:
        self.events.append(("magnet.engage",))
        self._engaged = True

    def release(self) -> None:
        self.events.append(("magnet.release",))
        self._engaged = False

    @property
    def is_engaged(self) -> bool:
        self.events.append(("magnet.is_engaged",))
        return self._engaged

    def close(self, release: bool = True) -> None:
        self.events.append(("magnet.close", release))
        self._engaged = False


def make_controller(
    events: list[tuple],
    *,
    fail_at_xy: tuple | None = None,
    continuous_transfers: bool = False,
) -> PuzzleController:
    return PuzzleController(
        config=PuzzleConfig(
            magnet_pickup_dwell=0.0,
            magnet_release_dwell=0.0,
            continuous_transfers=continuous_transfers,
        ),
        writer=FakeWriter(events, fail_at_xy=fail_at_xy),
        motor=FakeMotor(events),
        magnet=FakeMagnet(events),
    )


class PuzzleControllerTests(unittest.TestCase):
    def test_complete_sequence_with_rotation(self) -> None:
        events: list[tuple] = []
        controller = make_controller(events)
        controller.connect("/dev/fake-writer", motor_port="/dev/fake-foc")
        controller.establish_origin()
        result = controller.transfer_piece(
            TransferRequest(
                pickup_x=50.0,
                pickup_y=50.0,
                place_x=100.0,
                place_y=100.0,
                rotation_angle_deg=90.0,
                return_rotation_to_zero=True,
            )
        )

        actions = [event[0] for event in events]
        expected_subsequence = [
            "writer.pen_up",
            "writer.move_to",
            "writer.pen_move_to",
            "magnet.engage",
            "writer.pen_up",
            "motor.move_to",
            "writer.move_to",
            "writer.pen_move_to",
            "magnet.release",
            "writer.pen_up",
            "writer.move_to",
            "motor.zero",
            "writer.pen_move_to",
            "writer.send_gcode",
            "magnet.is_engaged",
        ]
        origin_up = actions.index("writer.pen_up", actions.index("writer.home"))
        start = actions.index("writer.pen_up", origin_up + 1)
        self.assertEqual(actions[start:], expected_subsequence)
        self.assertIn(
            ("writer.configure_pen", -10.0, -2.0, 1500.0),
            events,
        )
        self.assertIn(
            ("motor.move_to", 90.0, 15, True, 1.5, 2, True), events
        )
        self.assertEqual(
            [
                event
                for event in events
                if event[0] == "writer.pen_move_to"
            ],
            [
                ("writer.pen_move_to", -2.0, 1500.0),
                ("writer.pen_move_to", -2.0, 1500.0),
                ("writer.pen_move_to", -10.0, 1500.0),
            ],
        )
        self.assertEqual((result.final_x, result.final_y), (0.0, -1.0))
        self.assertEqual(result.final_z, -10.0)
        self.assertFalse(result.magnet_engaged)
        self.assertIn(("writer.set_motor_hold", True), events)
        self.assertIn(("writer.send_gcode", "$1=255"), events)

    def test_no_rotation_command_when_angle_is_omitted(self) -> None:
        events: list[tuple] = []
        controller = make_controller(events)
        controller.connect("/dev/fake-writer")
        controller.establish_origin()
        controller.transfer_piece(
            TransferRequest(50.0, 50.0, 100.0, 100.0)
        )
        self.assertNotIn("motor.move_to", [event[0] for event in events])

    def test_reconnect_reanchors_known_safe_parking_point(self) -> None:
        events: list[tuple] = []
        controller = make_controller(events)
        controller.connect("/dev/fake-writer")
        controller.prepare_existing_origin()
        self.assertIn(
            ("writer.reanchor_at_a4_position", 0.0, -1.0, -10.0),
            events,
        )
        self.assertTrue(controller.is_ready)

    def test_continuous_transfer_stays_at_place_with_safe_z(self) -> None:
        events: list[tuple] = []
        controller = make_controller(events, continuous_transfers=True)
        controller.connect("/dev/fake-writer")
        controller.establish_origin()
        result = controller.transfer_piece(
            TransferRequest(50.0, 50.0, 100.0, 100.0)
        )
        self.assertEqual((result.final_x, result.final_y), (100.0, 100.0))
        self.assertEqual(result.final_z, -10.0)
        self.assertNotIn(
            ("writer.move_to", 0.0, -1.0, 3000.0), events
        )

    def test_rotation_barrier_runs_after_xy_before_pickup_descent(self) -> None:
        events: list[tuple] = []
        controller = make_controller(events)
        controller.connect("/dev/fake-writer")
        controller.establish_origin()
        barrier = lambda: events.append(("f32c.home.confirmed",))
        controller.transfer_piece(
            TransferRequest(
                50.0, 50.0, 100.0, 100.0,
                rotation_home_barrier=barrier,
            )
        )
        actions = [event[0] for event in events]
        pickup_move = actions.index("writer.move_to")
        barrier_index = actions.index("f32c.home.confirmed")
        pickup_down = actions.index("writer.pen_move_to")
        self.assertLess(pickup_move, barrier_index)
        self.assertLess(barrier_index, pickup_down)

    def test_failure_releases_and_returns_home(self) -> None:
        events: list[tuple] = []
        controller = make_controller(events, fail_at_xy=(100.0, 100.0))
        controller.connect("/dev/fake-writer")
        controller.establish_origin()

        with self.assertRaises(PuzzleTransferError) as caught:
            controller.transfer_piece(
                TransferRequest(50.0, 50.0, 100.0, 100.0)
            )

        self.assertEqual(caught.exception.stage, "移动到放件点")
        self.assertEqual(
            events[-3:],
            [
                ("writer.pen_up",),
                ("magnet.release",),
                ("writer.move_to", 0.0, -1.0, 3000.0),
            ],
        )

    def test_rejects_negative_coordinate_before_motion(self) -> None:
        events: list[tuple] = []
        controller = make_controller(events)
        controller.connect("/dev/fake-writer")
        controller.establish_origin()
        before = list(events)
        with self.assertRaises(ValueError):
            controller.transfer_piece(
                TransferRequest(-1.0, 50.0, 100.0, 100.0)
            )
        self.assertEqual(events, before)

    def test_calibration_cancel_restores_motor_hold(self) -> None:
        events: list[tuple] = []

        class HoldController:
            def set_motor_hold(self, enable: bool = True) -> None:
                events.append(("hold", enable))

            def send_gcode(self, command: str) -> None:
                events.append(("gcode", command))

        original_prompt = z_calibration._prompt
        original_sleep = z_calibration.time.sleep
        try:
            z_calibration._prompt = lambda _text: "cancel"
            z_calibration.time.sleep = lambda _seconds: None
            with self.assertRaises(z_calibration.CalibrationAbort):
                z_calibration._release_and_rezero(
                    HoldController(),
                    "取消恢复测试",
                )
        finally:
            z_calibration._prompt = original_prompt
            z_calibration.time.sleep = original_sleep

        self.assertEqual(events, [("hold", False), ("hold", True)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
