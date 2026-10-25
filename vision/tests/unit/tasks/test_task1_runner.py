"""验证 Task1 编排顺序、视觉重试、超时、清理和异常传播。

该文件只使用内存 Fake，不依赖真实相机或文件。可通过
``python -m unittest tests.unit.tasks.test_task1_runner`` 运行。
"""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ec_26_e.cam import CameraCaptureError
from solver2 import MovePlan, PlanningError
from ec_26_e.tasks import TaskCancelledError
from ec_26_e.tasks.task1 import (
    Task1Options,
    Task1TimeoutError,
    run_task1,
)
from tests.unit.tasks.task1_helpers import (
    ConstantDetector,
    FakeCamera,
    RecordingPlanner,
    SequenceResizer,
    make_camera_frame,
    make_detect_result,
)


def fast_options() -> Task1Options:
    """返回只需两帧、时间戳跨度一秒的快速测试选项。"""

    return Task1Options(
        review_cycle_seconds=1.0,
        minimum_stable_frames=2,
        acquisition_timeout_seconds=2.0,
        frame_read_timeout_seconds=0.1,
    )


class Task1RunnerTest(unittest.TestCase):
    """验证入口只在稳定后规划且始终释放相机。"""

    def test_success_closes_camera_before_single_planner_call(self) -> None:
        """成功路径应使用新帧、关闭相机并只规划一次。"""

        frames = (
            make_camera_frame(10, 1_000_000_000),
            make_camera_frame(14, 2_000_000_000),
        )
        camera = FakeCamera(frames)
        resizer = SequenceResizer(frames[0])
        planner = RecordingPlanner(camera)

        result = run_task1(
            camera,
            resizer,
            ConstantDetector(),
            options=fast_options(),
            planner=planner,
        )

        self.assertIsInstance(result, MovePlan)
        self.assertEqual(camera.events, ["start", "read", "read", "close"])
        self.assertEqual(len(planner.calls), 1)
        pieces, options = planner.calls[0]
        self.assertEqual(tuple(piece.piece_id for piece in pieces), ("0", "1"))
        self.assertEqual(options, fast_options().planning_options)

    def test_fixed_task_uses_single_frame_result_when_stable_result_is_empty(
        self,
    ) -> None:
        """稳定器暂未确认时，完整单帧结果仍可驱动 fixed 任务。"""

        frames = (
            make_camera_frame(10, 1_000_000_000),
            make_camera_frame(11, 2_000_000_000),
        )
        four_pieces = make_detect_result(
            tuple(
                (index, vertices)
                for index, vertices in enumerate(
                    (
                        ((0.10, 0.10), (0.20, 0.10), (0.20, 0.20)),
                        ((0.30, 0.10), (0.40, 0.10), (0.40, 0.20)),
                        ((0.10, 0.30), (0.20, 0.30), (0.20, 0.40)),
                        ((0.30, 0.30), (0.40, 0.30), (0.40, 0.40)),
                    )
                )
            )
        )

        class StabilizingDetector:
            """模拟有完整单帧结果但稳定器暂时输出空结果的真实检测器。"""

            def __init__(self) -> None:
                self.last_single_result = None

            def detect(self, image_bgr: object):
                del image_bgr
                self.last_single_result = four_pieces
                return make_detect_result(())

        options = Task1Options(
            review_cycle_seconds=1.0,
            minimum_stable_frames=2,
            acquisition_timeout_seconds=2.0,
            frame_read_timeout_seconds=0.1,
            planning_options=fast_options().planning_options.__class__(
                mode="fixed",
            ),
        )
        planner = RecordingPlanner()

        run_task1(
            FakeCamera(frames),
            SequenceResizer(frames[0]),
            StabilizingDetector(),
            options=options,
            planner=planner,
        )

        self.assertEqual(len(planner.calls), 1)
        self.assertEqual(len(planner.calls[0][0]), 4)

    def test_two_visual_failures_are_tolerated(self) -> None:
        """前两张 Resize 场景错误不得阻止后续稳定结果。"""

        frames = tuple(
            make_camera_frame(index, 1_000_000_000 + index * 500_000_000)
            for index in range(4)
        )
        camera = FakeCamera(frames)
        resizer = SequenceResizer(frames[0], failures=2)
        planner = RecordingPlanner(camera)

        run_task1(
            camera,
            resizer,
            ConstantDetector(),
            options=Task1Options(
                review_cycle_seconds=1.5,
                minimum_stable_frames=2,
                acquisition_timeout_seconds=2.0,
            ),
            planner=planner,
        )

        self.assertEqual(resizer.calls, 4)
        self.assertEqual(len(planner.calls), 1)
        self.assertTrue(camera.closed)

    def test_observer_receives_failures_and_detections_then_closes(
        self,
    ) -> None:
        """观察器应覆盖视觉失败和成功帧，并在规划前完成关闭。"""

        frames = tuple(
            make_camera_frame(index, 1_000_000_000 + index * 500_000_000)
            for index in range(3)
        )
        camera = FakeCamera(frames)
        events: list[object] = []

        class Observer:
            """记录 Task1 可视化生命周期和强类型帧事件。"""

            def start(self) -> None:
                """记录观察器启动。"""

                events.append("observer_start")

            def show_detection(
                self,
                frame: object,
                resize_result: object,
                detect_result: object,
            ) -> bool:
                """记录成功检测事件并继续任务。"""

                events.append(("detection", frame))
                return False

            def show_failure(
                self,
                frame: object,
                *,
                stage: str,
                resize_result: object | None = None,
            ) -> bool:
                """记录失败阶段和对应帧并继续任务。"""

                del resize_result
                events.append(("failure", stage, frame))
                return False

            def close(self) -> None:
                """记录观察器关闭。"""

                events.append("observer_close")

        class Planner:
            """断言规划只发生在观察器和相机均释放后。"""

            def __call__(self, pieces: object, options: object) -> MovePlan:
                """确认资源关闭后返回测试移动方案。"""

                del pieces, options
                self.assertions()
                return RecordingPlanner()((), fast_options().planning_options)

            @staticmethod
            def assertions() -> None:
                """断言相机和观察器均已释放。"""

                if not camera.closed or events[-1] != "observer_close":
                    raise AssertionError("资源必须在规划前关闭")

        run_task1(
            camera,
            SequenceResizer(frames[0], failures=1),
            ConstantDetector(),
            options=Task1Options(
                review_cycle_seconds=1.0,
                minimum_stable_frames=2,
                acquisition_timeout_seconds=2.0,
            ),
            planner=Planner(),
            observer=Observer(),
        )

        self.assertEqual(events[0], "observer_start")
        self.assertEqual(events[1][0:2], ("failure", "workspace"))
        self.assertEqual(
            [event[0] for event in events if isinstance(event, tuple)],
            ["failure", "detection", "detection"],
        )
        self.assertEqual(events[-1], "observer_close")

    def test_observer_cancellation_skips_planner_and_closes_resources(
        self,
    ) -> None:
        """观察器请求停止后应关闭资源并保持规划器零调用。"""

        frame = make_camera_frame(0, 1_000_000_000)
        camera = FakeCamera((frame,))
        planner = RecordingPlanner(camera)

        class CancellingObserver:
            """在第一张成功检测帧上模拟 Esc。"""

            def __init__(self) -> None:
                """初始化观察器生命周期状态。"""

                self.started = False
                self.closed = False

            def start(self) -> None:
                """记录观察器启动。"""

                self.started = True

            def show_detection(
                self,
                frame: object,
                resize_result: object,
                detect_result: object,
            ) -> bool:
                """在第一张成功帧上请求取消。"""

                del frame, resize_result, detect_result
                return True

            def show_failure(
                self,
                frame: object,
                *,
                stage: str,
                resize_result: object | None = None,
            ) -> bool:
                """失败帧不会触发本测试的取消。"""

                del frame, stage, resize_result
                return False

            def close(self) -> None:
                """记录观察器已释放。"""

                self.closed = True

        observer = CancellingObserver()
        with self.assertRaisesRegex(TaskCancelledError, "预览窗口"):
            run_task1(
                camera,
                SequenceResizer(frame),
                ConstantDetector(),
                options=fast_options(),
                planner=planner,
                observer=observer,
            )

        self.assertTrue(observer.started)
        self.assertTrue(observer.closed)
        self.assertTrue(camera.closed)
        self.assertEqual(planner.calls, [])

    def test_invalid_observer_is_rejected_before_camera_start(self) -> None:
        """观察器协议缺失应在创建任何运行资源前失败。"""

        frame = make_camera_frame(0, 1_000_000_000)
        camera = FakeCamera((frame,))
        with self.assertRaisesRegex(TypeError, "observer"):
            run_task1(
                camera,
                SequenceResizer(frame),
                ConstantDetector(),
                options=fast_options(),
                observer=object(),  # type: ignore[arg-type]
            )

        self.assertEqual(camera.events, [])

    def test_fatal_camera_error_propagates_and_closes(self) -> None:
        """非读取超时的相机故障应立即向上传播并释放资源。"""

        camera = FakeCamera((CameraCaptureError("设备断开"),))
        template = make_camera_frame(0, 1_000_000_000)

        with self.assertRaisesRegex(CameraCaptureError, "设备断开"):
            run_task1(
                camera,
                SequenceResizer(template),
                ConstantDetector(),
                options=fast_options(),
            )

        self.assertTrue(camera.closed)

    def test_planning_error_propagates_after_camera_close(self) -> None:
        """稳定后的控制无解错误应保持原类型且相机已经关闭。"""

        frames = (
            make_camera_frame(0, 1_000_000_000),
            make_camera_frame(1, 2_000_000_000),
        )
        camera = FakeCamera(frames)
        planner = RecordingPlanner(
            camera,
            error=PlanningError("没有方案"),
        )

        with self.assertRaisesRegex(PlanningError, "没有方案"):
            run_task1(
                camera,
                SequenceResizer(frames[0]),
                ConstantDetector(),
                options=fast_options(),
                planner=planner,
            )

        self.assertTrue(camera.closed)
        self.assertEqual(len(planner.calls), 1)

    def test_total_timeout_contains_review_diagnostics_and_closes(self) -> None:
        """没有新帧时应按总期限失败并给出审查计数。"""

        camera = FakeCamera(())
        template = make_camera_frame(0, 1_000_000_000)
        options = Task1Options(
            review_cycle_seconds=0.001,
            minimum_stable_frames=2,
            acquisition_timeout_seconds=0.003,
            frame_read_timeout_seconds=0.001,
        )

        with self.assertRaises(Task1TimeoutError) as context:
            run_task1(
                camera,
                SequenceResizer(template),
                ConstantDetector(),
                options=options,
            )

        message = str(context.exception)
        self.assertIn("已处理帧=0", message)
        self.assertIn("异常事件=", message)
        self.assertIn("最后原因=", message)
        self.assertTrue(camera.closed)

    def test_timeout_uses_last_reviewed_frame_for_planning(self) -> None:
        """稳定周期未通过时应把最后一张基础审查通过的帧交给规划。"""

        frame = make_camera_frame(0, 1_000_000_000)
        camera = FakeCamera((frame,))
        planner = RecordingPlanner(camera)

        run_task1(
            camera,
            SequenceResizer(frame),
            ConstantDetector(),
            options=Task1Options(
                review_cycle_seconds=0.001,
                minimum_stable_frames=2,
                acquisition_timeout_seconds=0.003,
                frame_read_timeout_seconds=0.001,
            ),
            planner=planner,
        )

        self.assertTrue(camera.closed)
        self.assertEqual(len(planner.calls), 1)
        pieces, _options = planner.calls[0]
        self.assertEqual(tuple(piece.piece_id for piece in pieces), ("0", "1"))

    def test_task1_has_no_file_side_effect(self) -> None:
        """纯编排入口不得在当前工作目录生成任何文件。"""

        frames = (
            make_camera_frame(0, 1_000_000_000),
            make_camera_frame(1, 2_000_000_000),
        )
        camera = FakeCamera(frames)
        original_directory = Path.cwd()
        with TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                run_task1(
                    camera,
                    SequenceResizer(frames[0]),
                    ConstantDetector(),
                    options=fast_options(),
                    planner=RecordingPlanner(camera),
                )
                self.assertEqual(list(Path(directory).iterdir()), [])
            finally:
                os.chdir(original_directory)


if __name__ == "__main__":
    unittest.main()
