"""验证 Task2 从内存视觉结果到随机 MovePlan 执行器的完整衔接。

测试使用 Fake 相机、检测器和执行器，不访问真实相机、串口或 GPIO。可通过
``python -m unittest tests.integration.tasks.test_task2_application`` 独立运行。
"""

from __future__ import annotations

import inspect
import unittest
from unittest.mock import Mock

from ec_26_e.config import load_task2_config
from ec_26_e.control import plan_sheet_moves
from ec_26_e.tasks import TaskContext
from ec_26_e.tasks.task2 import Task2Error
from ec_26_e.tasks.task2.application import execute_task2
from ec_26_e.vision import create_production_detector
from ec_26_e.tasks.task2 import Task2Options, run_task2
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


class RecordingActuator:
    """记录 Task2 传入的内存移动方案而不访问机器硬件。"""

    def __init__(self) -> None:
        """初始化空的执行记录。"""

        self.plans: list[object] = []

    def execute(self, plan: object) -> None:
        """保存执行请求供测试断言。"""

        self.plans.append(plan)


class Task2ApplicationTest(unittest.TestCase):
    """验证随机规划选项和执行器交接均保持在内存中。"""

    def test_default_detector_factory_is_unified_vision(self) -> None:
        """正式 Task2 装配也必须统一使用 Vision2。"""

        default = inspect.signature(execute_task2).parameters[
            "detector_factory"
        ].default
        self.assertIs(default, create_production_detector)

    def test_random_plan_reaches_actuator_after_camera_cleanup(self) -> None:
        """Task2 应关闭相机后把随机规划方案交给执行器。"""

        config = load_task2_config(
            environ={"MOTION_MAGNET_GPIO_LINE": "12"},
            env_path=None,
        )
        frames = tuple(
            make_camera_frame(
                index,
                1_000_000_000 + round(index * 2_000_000_000 / 9),
            )
            for index in range(10)
        )
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
        camera = FakeCamera(frames)
        planner = RecordingPlanner(camera)
        actuator = RecordingActuator()
        detector = ConstantDetector(detections)
        detector.close = Mock()  # type: ignore[attr-defined]

        execute_task2(
            config,
            TaskContext(),
            camera_factory=lambda unused: camera,
            resize_factory=lambda unused: SequenceResizer(frames[0]),
            detector_factory=lambda unused: detector,
            planner=planner,
            actuator_factory=lambda unused: actuator,
        )

        self.assertTrue(camera.closed)
        self.assertEqual(len(planner.calls), 1)
        self.assertEqual(planner.calls[0][1].mode, "random")
        self.assertEqual(len(actuator.plans), 1)
        detector.close.assert_called_once_with()  # type: ignore[attr-defined]

    def test_stable_scene_reaches_real_random_planner(self) -> None:
        """Task2 的稳定视觉结果应直接交给无模板随机求解器。"""

        frames = (
            make_camera_frame(0, 1_000_000_000),
            make_camera_frame(1, 2_000_000_000),
        )
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
        plan = run_task2(
            FakeCamera(frames),
            SequenceResizer(frames[0]),
            ConstantDetector(detections),
            options=Task2Options(
                review_cycle_seconds=1.0,
                minimum_stable_frames=2,
                acquisition_timeout_seconds=5.0,
            ),
            planner=plan_sheet_moves,
        )

        self.assertEqual(plan.mode, "random")
        self.assertTrue(
            all(move.template_id is None for move in plan.moves)
        )

    def test_inference_failure_closes_detector_and_skips_motion(self) -> None:
        """Task2 原生推理错误也必须映射为识别失败且不创建执行器。"""

        config = load_task2_config(
            environ={"MOTION_MAGNET_GPIO_LINE": "12"},
            env_path=None,
        )
        frame = make_camera_frame(0, 1_000_000_000)
        camera = FakeCamera((frame,))
        detector = ConstantDetector()
        detector.detect = Mock(side_effect=RuntimeError("模拟 NPU 推理失败"))  # type: ignore[method-assign]
        detector.close = Mock()  # type: ignore[attr-defined]
        actuator_factory = Mock()

        with self.assertRaisesRegex(Task2Error, "识别检测失败：模拟 NPU 推理失败"):
            execute_task2(
                config,
                TaskContext(),
                camera_factory=lambda unused: camera,
                resize_factory=lambda unused: SequenceResizer(frame),
                detector_factory=lambda unused: detector,
                planner=Mock(),
                actuator_factory=actuator_factory,
            )

        detector.close.assert_called_once_with()  # type: ignore[attr-defined]
        actuator_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
