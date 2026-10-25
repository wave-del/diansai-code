"""验证 Task1 与真实视觉链路、真实固定控制规划器的内存衔接。

该文件只使用合成图像和假相机，不依赖真实设备或文件。可通过
``python -m unittest tests.integration.tasks.test_task1_pipeline`` 运行。
"""

from __future__ import annotations

import unittest

from ec_26_e.control import PlanningOptions, plan_sheet_moves
from ec_26_e.tasks.task1 import Task1Options, run_task1
from tests.unit.control.test_planner import FIXED_PIECES
from tests.unit.tasks.task1_helpers import (
    ConstantDetector,
    FakeCamera,
    RecordingPlanner,
    SequenceResizer,
    make_camera_frame,
    make_detect_result,
    portrait_vertices_to_landscape_relative,
)


class Task1PipelineIntegrationTest(unittest.TestCase):
    """验证编排入口两侧均可替换且不需要字典或 JSON 中转。"""

    def test_fixed_task1_timeout_uses_last_reviewed_two_piece_scene(self) -> None:
        """稳定超时后应把最后一帧交给下一步，由规划器决定如何处理。"""

        detections = make_detect_result(
            tuple(
                (
                    index,
                    portrait_vertices_to_landscape_relative(
                        piece.vertices_mm
                    ),
                )
                for index, piece in enumerate(FIXED_PIECES[:2])
            )
        )
        frames = tuple(
            make_camera_frame(
                index,
                1_000_000_000 + index * 500_000_000,
            )
            for index in range(3)
        )
        camera = FakeCamera(frames)
        planner = RecordingPlanner(camera)

        run_task1(
            camera,
            SequenceResizer(frames[0]),
            ConstantDetector(detections),
            options=Task1Options(
                review_cycle_seconds=0.001,
                minimum_stable_frames=3,
                acquisition_timeout_seconds=0.003,
                frame_read_timeout_seconds=0.001,
                planning_options=PlanningOptions(mode="fixed"),
            ),
            planner=planner,
        )

        self.assertEqual(len(planner.calls), 1)
        self.assertEqual(len(planner.calls[0][0]), 2)

    def test_structured_detect1_results_run_real_fixed_planner(self) -> None:
        """四片结构化检测结果应直接进入真实 fixed 控制求解。"""

        detections = make_detect_result(
            tuple(
                (
                    index,
                    portrait_vertices_to_landscape_relative(
                        piece.vertices_mm
                    ),
                )
                for index, piece in enumerate(FIXED_PIECES)
            )
        )
        frames = (
            make_camera_frame(0, 1_000_000_000),
            make_camera_frame(1, 2_000_000_000),
        )
        camera = FakeCamera(frames)

        plan = run_task1(
            camera,
            SequenceResizer(frames[0]),
            ConstantDetector(detections),
            options=Task1Options(
                review_cycle_seconds=1.0,
                minimum_stable_frames=2,
                acquisition_timeout_seconds=5.0,
                planning_options=PlanningOptions(mode="fixed"),
            ),
            planner=plan_sheet_moves,
        )

        self.assertEqual(plan.mode, "fixed")
        self.assertEqual(len(plan.moves), 4)
        self.assertAlmostEqual(plan.target_width_mm, 100.0)
        self.assertAlmostEqual(plan.target_height_mm, 60.0)
        self.assertTrue(camera.closed)


if __name__ == "__main__":
    unittest.main()
