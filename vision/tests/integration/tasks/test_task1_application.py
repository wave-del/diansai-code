"""验证 Task1 从内存规划到可替换运动执行器的完整衔接。

该文件只使用内存相机和假运动执行器，不访问真实串口、GPIO 或文件输出。
可通过 ``python -m unittest tests.integration.tasks.test_task1_application``
独立运行。
"""

from __future__ import annotations

from dataclasses import replace
import inspect
import threading
import unittest
from unittest.mock import Mock

from ec_26_e.cam import CameraCaptureError
from ec_26_e.config import load_task1_config
from ec_26_e.control import PlanningError
from ec_26_e.tasks import TaskCancelledError, TaskContext
from ec_26_e.tasks.task1 import Task1Error
from ec_26_e.tasks.task1.application import execute_task1
from ec_26_e.vision import create_production_detector
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


TEST_ENV = {"MOTION_MAGNET_GPIO_LINE": "12"}


def _stable_fixed_dependencies(
) -> tuple[FakeCamera, SequenceResizer, ConstantDetector]:
    """构造满足正式 fixed 稳定门槛的内存采集依赖。"""

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
    return (
        FakeCamera(frames),
        SequenceResizer(frames[0]),
        ConstantDetector(detections),
    )


class RecordingActuator:
    """记录应用层传入的内存 MovePlan。"""

    def __init__(self) -> None:
        """创建空调用记录。"""

        self.plans: list[object] = []

    def execute(self, plan: object) -> None:
        """记录 MovePlan 且不访问任何设备。"""

        self.plans.append(plan)


class Task1ApplicationIntegrationTest(unittest.TestCase):
    """验证完整任务应用装配顺序和中文失败边界。"""

    def setUp(self) -> None:
        """加载带必需 GPIO line 的纯配置。"""

        self.config = load_task1_config(
            environ=TEST_ENV,
            env_path=None,
        )

    def test_default_detector_factory_is_unified_vision(self) -> None:
        """正式 Task1 装配不得再默认创建纯传统视觉检测器。"""

        default = inspect.signature(execute_task1).parameters[
            "detector_factory"
        ].default
        self.assertIs(default, create_production_detector)

    def test_plan_stays_in_memory_and_task_returns_none(self) -> None:
        """相机关闭后应把计划交给执行器且不向调用方返回。"""

        camera, resizer, detector = _stable_fixed_dependencies()
        planner = RecordingPlanner(camera)
        actuator = RecordingActuator()
        detector.close = Mock()  # type: ignore[attr-defined]

        result = execute_task1(
            self.config,
            TaskContext(),
            camera_factory=lambda unused: camera,
            resize_factory=lambda unused: resizer,
            detector_factory=lambda unused: detector,
            planner=planner,
            actuator_factory=lambda unused: actuator,
        )

        self.assertIsNone(result)
        self.assertTrue(camera.closed)
        self.assertEqual(len(planner.calls), 1)
        self.assertEqual(len(actuator.plans), 1)
        detector.close.assert_called_once_with()  # type: ignore[attr-defined]

    def test_planning_failure_is_chinese_and_skips_actuator(self) -> None:
        """规划失败应关闭相机、打印边界中文并禁止运动。"""

        camera, resizer, detector = _stable_fixed_dependencies()
        planner = RecordingPlanner(
            camera,
            error=PlanningError("模拟无解"),
        )
        actuator_factory = Mock()
        detector.close = Mock()  # type: ignore[attr-defined]

        with self.assertRaisesRegex(Task1Error, "移动计划生成失败：模拟无解"):
            execute_task1(
                self.config,
                TaskContext(),
                camera_factory=lambda unused: camera,
                resize_factory=lambda unused: resizer,
                detector_factory=lambda unused: detector,
                planner=planner,
                actuator_factory=actuator_factory,
            )

        self.assertTrue(camera.closed)
        detector.close.assert_called_once_with()  # type: ignore[attr-defined]
        actuator_factory.assert_not_called()

    def test_camera_failure_is_chinese_and_skips_actuator(self) -> None:
        """识别设备失败应使用中文阶段信息并禁止运动。"""

        camera, resizer, detector = _stable_fixed_dependencies()

        def failed_start() -> None:
            """模拟相机初始化失败。"""

            raise CameraCaptureError("模拟相机不可用")

        camera.start = failed_start  # type: ignore[method-assign]
        actuator_factory = Mock()
        detector.close = Mock()  # type: ignore[attr-defined]
        with self.assertRaisesRegex(
            Task1Error,
            "识别检测失败：模拟相机不可用",
        ):
            execute_task1(
                self.config,
                TaskContext(),
                camera_factory=lambda unused: camera,
                resize_factory=lambda unused: resizer,
                detector_factory=lambda unused: detector,
                planner=RecordingPlanner(camera),
                actuator_factory=actuator_factory,
            )

        self.assertTrue(camera.closed)
        detector.close.assert_called_once_with()  # type: ignore[attr-defined]
        actuator_factory.assert_not_called()

    def test_disabled_context_never_constructs_observer(self) -> None:
        """关闭绘制时不应构造观察器，但仍执行内存计划。"""

        camera, resizer, detector = _stable_fixed_dependencies()
        observer_factory = Mock(
            side_effect=AssertionError("关闭时不得构造观察器")
        )
        actuator = RecordingActuator()

        execute_task1(
            self.config,
            TaskContext(draw_enabled=False),
            camera_factory=lambda unused: camera,
            resize_factory=lambda unused: resizer,
            detector_factory=lambda unused: detector,
            planner=RecordingPlanner(camera),
            observer_factory=observer_factory,
            actuator_factory=lambda unused: actuator,
        )

        observer_factory.assert_not_called()
        self.assertEqual(len(actuator.plans), 1)

    def test_preview_cancellation_skips_planner_and_actuator(self) -> None:
        """预览取消应释放视觉资源且不进入规划和运动。"""

        camera, resizer, detector = _stable_fixed_dependencies()
        planner = RecordingPlanner(camera)

        class CancellingObserver:
            """在第一张成功帧请求取消。"""

            def __init__(self) -> None:
                """初始化关闭标记。"""

                self.closed = False

            def start(self) -> None:
                """测试替身无需分配资源。"""

            def show_detection(
                self,
                frame: object,
                resize_result: object,
                detect_result: object,
            ) -> bool:
                """立即请求取消。"""

                del frame, resize_result, detect_result
                return True

            def show_failure(
                self,
                frame: object,
                *,
                stage: str,
                resize_result: object | None = None,
            ) -> bool:
                """失败帧也立即请求取消。"""

                del frame, stage, resize_result
                return True

            def close(self) -> None:
                """记录观察器释放。"""

                self.closed = True

        observer = CancellingObserver()
        actuator_factory = Mock()
        detector.close = Mock()  # type: ignore[attr-defined]
        with self.assertRaises(TaskCancelledError):
            execute_task1(
                self.config,
                TaskContext(draw_enabled=True),
                camera_factory=lambda unused: camera,
                resize_factory=lambda unused: resizer,
                detector_factory=lambda unused: detector,
                planner=planner,
                observer_factory=lambda: observer,
                actuator_factory=actuator_factory,
            )

        self.assertTrue(camera.closed)
        self.assertTrue(observer.closed)
        detector.close.assert_called_once_with()  # type: ignore[attr-defined]
        self.assertEqual(planner.calls, [])
        actuator_factory.assert_not_called()

    def test_npu_initialization_failure_skips_camera_planning_and_motion(self) -> None:
        """正式检测器初始化失败应统一映射为识别失败且不进入后续阶段。"""

        camera_factory = Mock()
        planner = Mock()
        actuator_factory = Mock()

        with self.assertRaisesRegex(Task1Error, "识别检测失败：模拟 NPU 初始化失败"):
            execute_task1(
                self.config,
                TaskContext(),
                camera_factory=camera_factory,
                detector_factory=Mock(side_effect=RuntimeError("模拟 NPU 初始化失败")),
                planner=planner,
                actuator_factory=actuator_factory,
            )

        camera_factory.assert_not_called()
        planner.assert_not_called()
        actuator_factory.assert_not_called()

    def test_inference_failure_closes_detector_and_skips_planning_and_motion(self) -> None:
        """原生推理错误应释放检测器并禁止规划和执行。"""

        camera, resizer, detector = _stable_fixed_dependencies()
        detector.detect = Mock(side_effect=RuntimeError("模拟原生推理失败"))  # type: ignore[method-assign]
        detector.close = Mock()  # type: ignore[attr-defined]
        planner = Mock()
        actuator_factory = Mock()

        with self.assertRaisesRegex(Task1Error, "识别检测失败：模拟原生推理失败"):
            execute_task1(
                self.config,
                TaskContext(),
                camera_factory=lambda unused: camera,
                resize_factory=lambda unused: resizer,
                detector_factory=lambda unused: detector,
                planner=planner,
                actuator_factory=actuator_factory,
            )

        self.assertTrue(camera.closed)
        detector.close.assert_called_once_with()  # type: ignore[attr-defined]
        planner.assert_not_called()
        actuator_factory.assert_not_called()

    def test_timeout_uses_last_reviewed_frame_and_closes_detector(self) -> None:
        """稳定超时应使用最后一帧继续下游，并确定性释放检测器。"""

        frame = make_camera_frame(0, 1_000_000_000)
        camera = FakeCamera((frame,))
        detector = ConstantDetector()
        detector.close = Mock()  # type: ignore[attr-defined]
        options = replace(
            self.config.options,
            review_cycle_seconds=0.001,
            acquisition_timeout_seconds=0.01,
            frame_read_timeout_seconds=0.001,
        )
        config = replace(self.config, options=options)

        planner = Mock()
        actuator = RecordingActuator()
        execute_task1(
            config,
            TaskContext(),
            camera_factory=lambda unused: camera,
            resize_factory=lambda unused: SequenceResizer(frame),
            detector_factory=lambda unused: detector,
            planner=planner,
            actuator_factory=lambda unused: actuator,
        )

        detector.close.assert_called_once_with()  # type: ignore[attr-defined]
        planner.assert_called_once()
        self.assertEqual(len(actuator.plans), 1)

    def test_detector_lifecycle_stays_on_application_thread(self) -> None:
        """NPU 的构造、推理和释放应保持在 execute_task1 调用线程。"""

        camera, resizer, template = _stable_fixed_dependencies()
        expected_thread = threading.get_ident()
        lifecycle_threads: list[int] = []

        def detector_factory(unused: object) -> ConstantDetector:
            """在当前应用线程构造并记录可关闭检测器。"""

            del unused
            lifecycle_threads.append(threading.get_ident())
            detector = ConstantDetector(template.result)
            original_detect = detector.detect

            def detect(image_bgr: object) -> object:
                """记录推理线程并转发到固定结果检测器。"""

                lifecycle_threads.append(threading.get_ident())
                return original_detect(image_bgr)  # type: ignore[arg-type]

            def close() -> None:
                """记录资源释放线程。"""

                lifecycle_threads.append(threading.get_ident())

            detector.detect = detect  # type: ignore[method-assign]
            detector.close = close  # type: ignore[attr-defined]
            return detector

        execute_task1(
            self.config,
            TaskContext(),
            camera_factory=lambda unused: camera,
            resize_factory=lambda unused: resizer,
            detector_factory=detector_factory,
            planner=RecordingPlanner(camera),
            actuator_factory=lambda unused: RecordingActuator(),
        )

        self.assertGreater(len(lifecycle_threads), 2)
        self.assertEqual(set(lifecycle_threads), {expected_thread})

    def test_detector_is_closed_before_actuator_construction(self) -> None:
        """进入任何执行器初始化逻辑前应已经释放 NPU。"""

        camera, resizer, detector = _stable_fixed_dependencies()
        detector.close = Mock()  # type: ignore[attr-defined]

        def failed_actuator_factory(unused: object) -> object:
            """确认关闭顺序后模拟执行前初始化失败。"""

            del unused
            detector.close.assert_called_once_with()  # type: ignore[attr-defined]
            raise RuntimeError("模拟执行器初始化失败")

        with self.assertRaisesRegex(RuntimeError, "模拟执行器初始化失败"):
            execute_task1(
                self.config,
                TaskContext(),
                camera_factory=lambda unused: camera,
                resize_factory=lambda unused: resizer,
                detector_factory=lambda unused: detector,
                planner=RecordingPlanner(camera),
                actuator_factory=failed_actuator_factory,
            )


if __name__ == "__main__":
    unittest.main()
