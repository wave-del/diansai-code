"""验证 Task1 公共契约、视觉审查和横竖 A4 坐标适配。

该文件只使用内存数据，不依赖相机、文件或外部环境。可通过
``python -m unittest tests.unit.tasks.test_task1_contracts`` 运行。
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import unittest

import ec_26_e.tasks as public_tasks
import ec_26_e.tasks.task1 as public_task1
from solver2 import PlanningOptions
from ec_26_e.tasks import TaskCancelledError, TaskContext
from ec_26_e.tasks.task1 import Task1Options
from ec_26_e.tasks.task1.review import review_visual_frame
from ec_26_e.vision import Detect1Result
from tests.unit.tasks.task1_helpers import (
    DEFAULT_RELATIVE_PIECES,
    make_camera_frame,
    make_detect_result,
    make_piece_detection,
    make_resize_result,
)


class Task1ContractTest(unittest.TestCase):
    """冻结 Task1 默认值、公共导出和配置校验语义。"""

    def test_public_exports_and_defaults(self) -> None:
        """Task1 子包和 tasks 根包应导出同一组稳定入口。"""

        task1_expected = {
            "Detect1Processor",
            "PlanActuator",
            "PuzzlePlanner",
            "ResizeProcessor",
            "Task1Error",
            "Task1FrameObserver",
            "Task1Options",
            "Task1TimeoutError",
            "run_task1",
        }
        self.assertEqual(set(public_task1.__all__), task1_expected)
        self.assertEqual(
            set(public_tasks.__all__),
            task1_expected | {"TaskCancelledError", "TaskContext"},
        )
        self.assertIs(public_tasks.Task1Options, Task1Options)
        self.assertIs(public_tasks.TaskContext, TaskContext)
        self.assertIs(public_tasks.TaskCancelledError, TaskCancelledError)

        options = Task1Options()
        self.assertEqual(
            tuple(field.name for field in fields(Task1Options)),
            (
                "minimum_stable_frames",
                "piece_vertex_tolerance_mm",
                "workspace_corner_x_tolerance_mm",
                "workspace_corner_y_tolerance_mm",
                "workspace_corner_distance_tolerance_mm",
                "review_cycle_seconds",
                "allowed_jump_frames",
                "acquisition_timeout_seconds",
                "frame_read_timeout_seconds",
                "landscape_rotation",
                "planning_options",
            ),
        )
        self.assertEqual(options.minimum_stable_frames, 5)
        self.assertEqual(options.piece_vertex_tolerance_mm, 2.0)
        self.assertEqual(options.workspace_corner_x_tolerance_mm, 2.0)
        self.assertEqual(options.workspace_corner_y_tolerance_mm, 2.0)
        self.assertEqual(
            options.workspace_corner_distance_tolerance_mm,
            2.5,
        )
        self.assertEqual(options.review_cycle_seconds, 2.0)
        self.assertEqual(options.allowed_jump_frames, 5)
        self.assertEqual(options.acquisition_timeout_seconds, 5.0)
        self.assertEqual(options.frame_read_timeout_seconds, 2.0)
        self.assertEqual(options.landscape_rotation, "counterclockwise")
        self.assertEqual(options.planning_options, PlanningOptions())
        self.assertFalse(hasattr(options, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            options.minimum_stable_frames = 3  # type: ignore[misc]

        context = TaskContext()
        self.assertEqual(
            tuple(field.name for field in fields(TaskContext)),
            ("draw_enabled",),
        )
        self.assertFalse(context.draw_enabled)
        self.assertFalse(hasattr(context, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            context.draw_enabled = True  # type: ignore[misc]
        with self.assertRaises(TypeError):
            TaskContext(draw_enabled=1)  # type: ignore[arg-type]

    def test_options_reject_invalid_and_impossible_values(self) -> None:
        """配置应在启动资源前拒绝非法值和不可能的超时组合。"""

        with self.assertRaises(TypeError):
            Task1Options(minimum_stable_frames=True)
        with self.assertRaises(ValueError):
            Task1Options(minimum_stable_frames=1)
        with self.assertRaises(ValueError):
            Task1Options(piece_vertex_tolerance_mm=0.0)
        with self.assertRaises(ValueError):
            Task1Options(workspace_corner_x_tolerance_mm=297.1)
        with self.assertRaises(ValueError):
            Task1Options(workspace_corner_y_tolerance_mm=210.1)
        with self.assertRaises(ValueError):
            Task1Options(
                review_cycle_seconds=2.0,
                acquisition_timeout_seconds=1.0,
            )
        with self.assertRaises(ValueError):
            Task1Options(allowed_jump_frames=-1)
        with self.assertRaises(ValueError):
            Task1Options(landscape_rotation="up")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            Task1Options(planning_options={})  # type: ignore[arg-type]


class Task1VisualReviewTest(unittest.TestCase):
    """验证单帧关联审查、坐标公式和视觉诊断字段隔离。"""

    def setUp(self) -> None:
        """构造一套尺寸一致的相机、Resize 和 Detect1 结果。"""

        self.frame = make_camera_frame(1, 1_000_000_000)
        self.resize_result = make_resize_result(self.frame)

    def test_clockwise_and_counterclockwise_coordinate_mapping(self) -> None:
        """两个可配置旋向必须严格按竖向 A4 公式换算。"""

        detection = make_detect_result(
            ((7, ((0.0, 0.0), (1.0, 0.0), (1.0, 0.5))),)
        )

        clockwise = review_visual_frame(
            self.frame,
            self.resize_result,
            detection,
            "clockwise",
        )
        counterclockwise = review_visual_frame(
            self.frame,
            self.resize_result,
            detection,
            "counterclockwise",
        )

        self.assertEqual(clockwise.pieces[0].piece_id, "7")
        self.assertEqual(
            clockwise.pieces[0].vertices_mm,
            ((207.0, 0.0), (207.0, 292.0), (103.5, 292.0)),
        )
        self.assertEqual(
            counterclockwise.pieces[0].vertices_mm,
            ((0.0, 292.0), (0.0, 0.0), (103.5, 0.0)),
        )

    def test_confidence_does_not_change_control_input(self) -> None:
        """视觉整体置信度变化不得进入或改变控制纸片。"""

        low = make_piece_detection(
            3,
            DEFAULT_RELATIVE_PIECES[0],
            confidence=0.01,
        )
        high = make_piece_detection(
            3,
            DEFAULT_RELATIVE_PIECES[0],
            confidence=0.99,
        )

        low_review = review_visual_frame(
            self.frame,
            self.resize_result,
            make_detect_result((low,)),
            "clockwise",
        )
        high_review = review_visual_frame(
            self.frame,
            self.resize_result,
            make_detect_result((high,)),
            "clockwise",
        )

        self.assertEqual(low_review.pieces, high_review.pieces)
        self.assertEqual(
            low_review.workspace_corners_source_px,
            self.resize_result.source_corners_px,
        )

    def test_rejects_mismatched_coordinate_system_and_duplicate_ids(
        self,
    ) -> None:
        """审查层应拒绝跨模块尺寸、坐标系和 ID 契约破坏。"""

        valid = make_detect_result()
        wrong_coordinate = Detect1Result(
            image_size=valid.image_size,
            pieces=valid.pieces,
            foreground_threshold=valid.foreground_threshold,
            workspace_area_px=valid.workspace_area_px,
            coordinate_system="other",
        )
        with self.assertRaisesRegex(ValueError, "坐标系"):
            review_visual_frame(
                self.frame,
                self.resize_result,
                wrong_coordinate,
                "clockwise",
            )

        duplicate = make_detect_result(
            (
                (1, DEFAULT_RELATIVE_PIECES[0]),
                (1, DEFAULT_RELATIVE_PIECES[1]),
            )
        )
        with self.assertRaisesRegex(ValueError, "group_id"):
            review_visual_frame(
                self.frame,
                self.resize_result,
                duplicate,
                "clockwise",
            )

    def test_rejects_unordered_repeated_and_self_intersecting_polygons(
        self,
    ) -> None:
        """审查层应在进入稳定跟踪前拒绝非法轮廓。"""

        invalid_shapes = (
            tuple(reversed(DEFAULT_RELATIVE_PIECES[0])),
            ((0.1, 0.1), (0.2, 0.1), (0.1, 0.1)),
            ((0.1, 0.1), (0.2, 0.2), (0.1, 0.2), (0.2, 0.1)),
        )
        for shape in invalid_shapes:
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    review_visual_frame(
                        self.frame,
                        self.resize_result,
                        make_detect_result(((0, shape),)),
                        "clockwise",
                    )


if __name__ == "__main__":
    unittest.main()
