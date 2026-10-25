"""不访问相机或硬件，验证参考 MovePlan 到本机计划的适配。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
REFERENCE_SRC = HERE.parent / "vision" / "src"
for path in (HERE, REFERENCE_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ec_26_e.control import MovePlan, PlannedMove
from task1_adapter import (
    F32CRotationCalibration,
    TangramAdapterConfig,
    TangramHardwarePorts,
    TangramPlanExecutor,
    Task1AdaptationError,
    adapt_move_plan,
    load_rotation_calibration,
    save_rotation_calibration,
)


def _reference_plan(
    *,
    pick: tuple[float, float] = (50.0, 100.0),
    place: tuple[float, float] = (105.0, 222.75),
    rotation: float = 30.0,
) -> MovePlan:
    vertices = ((1.0, 1.0), (2.0, 1.0), (1.0, 2.0))
    move = PlannedMove(
        order=1,
        piece_id="piece-1",
        pick_xy_a4_mm=pick,
        pick_clearance_mm=1.0,
        rotation_deg_clockwise=rotation,
        place_xy_a4_mm=place,
        translation_a4_mm=(0.0, 0.0),
        observed_vertices_a4_mm=vertices,
        target_vertices_a4_mm=vertices,
    )
    return MovePlan(
        mode="fixed",
        confidence=0.9,
        ambiguous=False,
        target_width_mm=100.0,
        target_height_mm=60.0,
        target_center_a4_mm=(105.0, 222.75),
        moves=(move,),
    )


class _FakeMotor:
    def __init__(self) -> None:
        self.angle = 12.0
        self.home_calls: list[dict[str, object]] = []

    def get_angle(self) -> float:
        return self.angle

    def return_to_mechanical_zero(self, **kwargs: object) -> float:
        self.home_calls.append(dict(kwargs))
        self.angle = 0.0
        return 0.0


class _FakeController:
    def __init__(self) -> None:
        self.motor = _FakeMotor()
        self.events: list[object] = []

    def connect(self, *args: object, **kwargs: object) -> None:
        self.events.append("connect")

    def establish_origin(self) -> None:
        self.events.append("origin")

    def prepare_existing_origin(self) -> None:
        self.events.append("reuse-origin")

    def transfer_piece(self, request: object) -> object:
        if isinstance(request, dict):
            barrier = request.get("rotation_home_barrier")
            if barrier is not None:
                barrier()
            released = request.get("piece_released_callback")
            if released is not None:
                released()
        self.events.append(request)
        return request

    def park_safely(self) -> None:
        self.events.append("park")

    def close(self) -> None:
        self.events.append("close")


class Task1AdapterTest(unittest.TestCase):
    def test_default_z_profile_matches_validated_positive_lift_api(self) -> None:
        config = TangramAdapterConfig()
        self.assertEqual(config.safe_z_mm, -10.0)
        self.assertEqual(config.pickup_z_mm, -2.0)
        self.assertEqual(config.place_z_mm, -2.0)
        self.assertEqual(config.rest_z_mm, -10.0)

    def test_scales_reference_a4_to_real_blue_sheet(self) -> None:
        plan = adapt_move_plan(
            _reference_plan(rotation=0.0),
            TangramAdapterConfig(),
        )
        move = plan.moves[0]
        self.assertEqual(move.pickup_xy_mm, (206.5 * 50.0 / 210.0, 292.0 * 100.0 / 297.0))
        self.assertEqual(move.place_xy_mm, (103.25, 219.0))
        plan.require_executable()

    def test_random_profile_uses_207_by_292_reference_without_legacy_scaling(self) -> None:
        plan = adapt_move_plan(
            _reference_plan(pick=(103.5, 146.0), place=(155.25, 219.0), rotation=0.0),
            TangramAdapterConfig(
                reference_width_mm=207.0,
                reference_height_mm=292.0,
            ),
        )
        move = plan.moves[0]
        self.assertEqual(move.pickup_xy_mm, (103.25, 146.0))
        self.assertEqual(move.place_xy_mm, (154.875, 219.0))

    def test_clockwise_vision_path_reverses_both_reference_axes(self) -> None:
        plan = adapt_move_plan(
            _reference_plan(rotation=0.0),
            TangramAdapterConfig(reverse_reference_axes=True),
        )
        move = plan.moves[0]
        self.assertAlmostEqual(
            move.pickup_xy_mm[0], 206.5 * (1.0 - 50.0 / 210.0)
        )
        self.assertAlmostEqual(
            move.pickup_xy_mm[1], 292.0 * (1.0 - 100.0 / 297.0)
        )
        self.assertEqual(move.place_xy_mm, (103.25, 73.0))

    def test_uncalibrated_rotation_can_preview_but_not_execute(self) -> None:
        plan = adapt_move_plan(_reference_plan(), TangramAdapterConfig())
        self.assertIsNone(plan.moves[0].motor_rotation_delta_deg)
        with self.assertRaisesRegex(Task1AdaptationError, "未标定"):
            plan.require_executable()

    def test_clamps_coordinate_that_writer_would_clamp(self) -> None:
        plan = adapt_move_plan(
            _reference_plan(pick=(1.0, 100.0), rotation=0.0),
            TangramAdapterConfig(),
        )
        self.assertEqual(plan.moves[0].pickup_xy_mm[0], 23.5)
        self.assertAlmostEqual(
            plan.moves[0].pickup_xy_mm[1], 292.0 * 100.0 / 297.0
        )

    def test_executor_uses_current_motor_angle_plus_calibrated_delta(self) -> None:
        plan = adapt_move_plan(
            _reference_plan(rotation=30.0),
            TangramAdapterConfig(vision_clockwise_to_motor_sign=-1),
        )
        controller = _FakeController()
        executor = TangramPlanExecutor(
            TangramAdapterConfig(vision_clockwise_to_motor_sign=-1),
            TangramHardwarePorts(writer_port="writer", motor_port="motor"),
            controller_factory=lambda unused: controller,
            request_factory=lambda **kwargs: kwargs,
        )
        result = executor.execute(plan)
        self.assertEqual(len(result), 1)
        request = result[0]
        self.assertEqual(request["rotation_angle_deg"], 342.0)
        self.assertEqual(controller.events[0:2], ["connect", "origin"])
        self.assertEqual(controller.events[-1], "close")

    def test_continuous_execution_returns_rotation_to_zero_per_piece(self) -> None:
        plan = adapt_move_plan(
            _reference_plan(rotation=30.0),
            TangramAdapterConfig(vision_clockwise_to_motor_sign=1),
        )
        plan = replace(plan, moves=plan.moves * 2)
        controller = _FakeController()
        executor = TangramPlanExecutor(
            TangramAdapterConfig(
                vision_clockwise_to_motor_sign=1,
                continuous_transfers=True,
            ),
            TangramHardwarePorts(writer_port="writer", motor_port="motor"),
            controller_factory=lambda unused: controller,
            request_factory=lambda **kwargs: kwargs,
        )
        requests = executor.execute(plan)
        self.assertFalse(requests[0]["return_rotation_to_zero"])
        self.assertIsNone(requests[0]["rotation_home_barrier"])
        self.assertIsNotNone(requests[1]["rotation_home_barrier"])
        self.assertTrue(
            all(call["tolerance_deg"] == 1.5 for call in controller.motor.home_calls)
        )
        self.assertTrue(
            all(call["settle_retries"] == 2 for call in controller.motor.home_calls)
        )
        self.assertTrue(
            all(
                call["allow_inexact_arrival"] is True
                for call in controller.motor.home_calls
            )
        )

    def test_completion_callback_is_only_attached_to_last_piece(self) -> None:
        plan = adapt_move_plan(
            _reference_plan(rotation=0.0),
            TangramAdapterConfig(vision_clockwise_to_motor_sign=1),
        )
        plan = replace(plan, moves=plan.moves * 2)
        controller = _FakeController()
        executor = TangramPlanExecutor(
            TangramAdapterConfig(vision_clockwise_to_motor_sign=1),
            TangramHardwarePorts(writer_port="writer", motor_port="motor"),
            controller_factory=lambda unused: controller,
            request_factory=lambda **kwargs: kwargs,
        )
        notifications: list[str] = []
        requests = executor.execute(
            plan,
            completion_callback=lambda: notifications.append("released"),
        )
        self.assertIsNone(requests[0]["piece_released_callback"])
        self.assertIsNotNone(requests[1]["piece_released_callback"])
        self.assertEqual(notifications, ["released"])

    def test_executor_reuses_existing_origin_without_sending_g92(self) -> None:
        plan = adapt_move_plan(
            _reference_plan(rotation=0.0),
            TangramAdapterConfig(vision_clockwise_to_motor_sign=1),
        )
        controller = _FakeController()
        executor = TangramPlanExecutor(
            TangramAdapterConfig(vision_clockwise_to_motor_sign=1),
            TangramHardwarePorts(writer_port="writer", motor_port="motor"),
            controller_factory=lambda unused: controller,
            request_factory=lambda **kwargs: kwargs,
        )
        origin_modes: list[bool] = []
        executor.execute(
            plan,
            reuse_machine_origin=True,
            origin_ready_callback=origin_modes.append,
        )
        self.assertEqual(controller.events[0:2], ["connect", "reuse-origin"])
        self.assertNotIn("origin", controller.events)
        self.assertEqual(origin_modes, [True])

    def test_park_safely_reuses_existing_origin(self) -> None:
        controller = _FakeController()
        executor = TangramPlanExecutor(
            TangramAdapterConfig(),
            TangramHardwarePorts(writer_port="writer", motor_port="motor"),
            controller_factory=lambda unused: controller,
        )
        executor.park_safely()
        self.assertEqual(controller.events, ["connect", "reuse-origin", "park", "close"])

    def test_initialize_and_park_establishes_origin_before_moving(self) -> None:
        controller = _FakeController()
        executor = TangramPlanExecutor(
            TangramAdapterConfig(),
            TangramHardwarePorts(writer_port="writer", motor_port="motor"),
            controller_factory=lambda unused: controller,
        )
        origin_modes: list[bool] = []
        executor.initialize_and_park(origin_ready_callback=origin_modes.append)
        self.assertEqual(controller.events, ["connect", "origin", "park", "close"])
        self.assertEqual(origin_modes, [False])

    def test_rotation_calibration_round_trip(self) -> None:
        path = HERE / ".rotation-calibration-test.json"
        try:
            expected = F32CRotationCalibration(-1, 30.0)
            save_rotation_calibration(path, expected)
            self.assertEqual(load_rotation_calibration(path), expected)
        finally:
            path.unlink(missing_ok=True)

    def test_accepts_full_validated_fifteen_mm_lift_range(self) -> None:
        config = TangramAdapterConfig(
            safe_z_mm=-15.0,
            pickup_z_mm=0.0,
            place_z_mm=0.0,
            rest_z_mm=-15.0,
        )
        self.assertEqual(config.safe_z_mm, -15.0)

    def test_rejects_lift_beyond_validated_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "已验证"):
            TangramAdapterConfig(
                safe_z_mm=-16.0,
                pickup_z_mm=0.0,
                place_z_mm=0.0,
                rest_z_mm=-16.0,
            )


if __name__ == "__main__":
    unittest.main()
