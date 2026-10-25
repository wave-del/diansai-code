#!/usr/bin/env python3
"""鲁棒视觉 Task1 接线层的纯离线测试。"""

from __future__ import annotations

from pathlib import Path
import unittest

from run_task1_tangram import (
    _execution_motion_settings,
    _execution_xy_feed_mm_min,
    _parser,
)
from task1_yolo_flow import (
    _competition_override_allowed,
    _piece_count_is_plannable,
    _portrait_vertices,
    _solver_branch_name,
    load_yolo_task_config,
)


PACKAGE_DIR = Path(__file__).resolve().parent
# 视觉包（原独立仓库 ec_26_e）已并入本仓库的 vision/ 目录
VISION_DIR = PACKAGE_DIR.parent / "vision"


class CoordinateConversionTests(unittest.TestCase):
    def test_counterclockwise_landscape_to_portrait(self) -> None:
        result = _portrait_vertices(((0.0, 0.0), (1.0, 1.0)), "counterclockwise")
        self.assertEqual(result, ((0.0, 297.0), (210.0, 0.0)))

    def test_clockwise_landscape_to_portrait(self) -> None:
        result = _portrait_vertices(((0.0, 0.0), (1.0, 1.0)), "clockwise")
        self.assertEqual(result, ((210.0, 0.0), (0.0, 297.0)))

    def test_relative_coordinate_must_be_in_range(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            _portrait_vertices(((1.01, 0.5),), "counterclockwise")


class ConfigurationTests(unittest.TestCase):
    def test_random_mode_automatically_routes_two_to_four_pieces(self) -> None:
        self.assertFalse(_piece_count_is_plannable("fixed", 3))
        self.assertTrue(_piece_count_is_plannable("fixed", 4))
        self.assertFalse(_piece_count_is_plannable("random", 1))
        self.assertTrue(_piece_count_is_plannable("random", 2))
        self.assertTrue(_piece_count_is_plannable("random", 3))
        self.assertTrue(_piece_count_is_plannable("random", 4))
        self.assertFalse(_piece_count_is_plannable("random", 5))
        self.assertEqual(_solver_branch_name("random", 3), "随机三片拓扑")

    def test_loads_existing_camera_and_planning_config(self) -> None:
        config = load_yolo_task_config(
            PACKAGE_DIR / "task1.tangram.env",
            VISION_DIR / "configs" / "tasks" / "task1.json",
            params_name="new",
            process_fps=6.0,
            acquisition_timeout_seconds=None,
            plan_output_dir=PACKAGE_DIR / "outputs" / "task1_plans",
            draw=False,
            windowed=False,
        )
        self.assertEqual(config.width, 1280)
        self.assertEqual(config.height, 720)
        self.assertEqual(config.landscape_rotation, "counterclockwise")
        self.assertEqual(config.planning["mode"], "fixed")
        self.assertEqual(config.planning["minimum_execution_confidence"], 0.0)
        self.assertEqual(config.acquisition_timeout_seconds, 5.0)
        self.assertEqual(
            config.plan_output_dir,
            PACKAGE_DIR / "outputs" / "task1_plans",
        )

    def test_main_defaults_to_yolo_without_execution(self) -> None:
        args = _parser().parse_args([])
        self.assertEqual(args.vision, "yolo")
        self.assertFalse(args.execute)
        self.assertFalse(args.draw)
        self.assertEqual(args.landscape_rotation, "clockwise")
        self.assertFalse(args.allow_competition_compliant)
        self.assertIsNone(args.max_moves)
        self.assertEqual(args.travel_lift_mm, 10.0)
        self.assertEqual(args.pickup_lift_mm, 2.0)
        self.assertEqual(args.place_lift_mm, 2.0)
        self.assertEqual(args.rest_lift_mm, 10.0)
        self.assertFalse(args.continuous_transfers)

    def test_all_tasks_use_the_configured_fast_motion_profile(self) -> None:
        self.assertEqual(
            _execution_xy_feed_mm_min("fixed", 90.0), 4500.0
        )
        self.assertEqual(
            _execution_xy_feed_mm_min("random-white", 59.9), 4500.0
        )
        self.assertEqual(
            _execution_xy_feed_mm_min("random-white", 60.0), 4500.0
        )
        self.assertEqual(
            _execution_motion_settings("random-white", 60.0, 4500.0, 29),
            (4500.0, 29),
        )
        self.assertEqual(
            _execution_motion_settings("fixed", 90.0, 4500.0, 29),
            (4500.0, 29),
        )

    def test_local_rotation_override_is_used_for_our_left_to_right_layout(self) -> None:
        config = load_yolo_task_config(
            PACKAGE_DIR / "task1.tangram.env",
            VISION_DIR / "configs" / "tasks" / "task1.json",
            params_name="new",
            process_fps=6.0,
            acquisition_timeout_seconds=None,
            plan_output_dir=PACKAGE_DIR / "outputs" / "task1_plans",
            draw=False,
            windowed=False,
            landscape_rotation="clockwise",
        )
        self.assertEqual(config.landscape_rotation, "clockwise")

    def test_random_profile_uses_new_solver_coordinate_frame(self) -> None:
        config = load_yolo_task_config(
            PACKAGE_DIR / "task1.tangram.env",
            VISION_DIR / "configs" / "tasks" / "task2.json",
            params_name="new",
            process_fps=6.0,
            acquisition_timeout_seconds=None,
            plan_output_dir=PACKAGE_DIR / "outputs" / "task2_white_plans",
            draw=False,
            windowed=False,
            reference_width_mm=207.0,
            reference_height_mm=292.0,
            planning_start_xy_mm=(207.0, 292.0),
        )
        self.assertEqual(config.planning["mode"], "random")
        self.assertEqual(config.reference_width_mm, 207.0)
        self.assertEqual(config.reference_height_mm, 292.0)
        self.assertEqual(config.planning["start_xy_a4_mm"], [207.0, 292.0])

    def test_competition_override_requires_every_safety_condition(self) -> None:
        result = {
            "competition_gate_passed": True,
            "machine_executable": True,
            "search_candidate_valid": True,
            "ambiguous": False,
            "execution_quality": "competition_compliant",
        }
        self.assertTrue(
            _competition_override_allowed(result, enabled=True)
        )
        self.assertFalse(
            _competition_override_allowed(result, enabled=False)
        )
        for field in (
            "competition_gate_passed",
            "machine_executable",
            "search_candidate_valid",
        ):
            rejected = dict(result)
            rejected[field] = False
            self.assertFalse(
                _competition_override_allowed(rejected, enabled=True),
                field,
            )
        ambiguous = dict(result)
        ambiguous["ambiguous"] = True
        self.assertFalse(
            _competition_override_allowed(ambiguous, enabled=True)
        )


if __name__ == "__main__":
    unittest.main()
