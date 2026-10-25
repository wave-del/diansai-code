"""验证通用视觉绘制和屏幕预览适配器。

该文件只使用合成图像及 OpenCV Mock，不访问真实显示器。可通过
``python -m unittest tests.unit.test_draw`` 独立运行。
"""

from __future__ import annotations

import os
import subprocess
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from ec_26_e.draw import (
    DrawError,
    OpenCVPreview,
    Task1DrawObserver,
    draw_detection_failure,
    draw_live_detection,
)
from tests.unit.tasks.task1_helpers import (
    make_camera_frame,
    make_detect_result,
    make_resize_result,
)


class DrawingTest(unittest.TestCase):
    """验证绘制只修改副本并覆盖主要调试信息。"""

    def test_live_result_draws_workspace_pieces_and_labels_on_copy(
        self,
    ) -> None:
        """成功帧应保持源图不变并返回独立的非空标注副本。"""

        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame = make_camera_frame(7, 1_000_000_000, image=image)
        resize_result = make_resize_result(frame)
        source_snapshot = frame.image_bgr.copy()

        annotated = draw_live_detection(
            frame.image_bgr,
            resize_result,
            make_detect_result(),
            sequence_id=frame.sequence_id,
        )

        self.assertIsNot(annotated, frame.image_bgr)
        np.testing.assert_array_equal(frame.image_bgr, source_snapshot)
        self.assertGreater(np.count_nonzero(annotated), 0)
        self.assertTrue(
            np.any(annotated[75:90, 95:115] != 0),
            "A4 左上角应存在边框、角点或标签",
        )

    def test_failure_draws_stage_without_modifying_source(self) -> None:
        """失败帧应绘制阶段状态并可保留已有工作区边界。"""

        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame = make_camera_frame(3, 1_000_000_000, image=image)
        resize_result = make_resize_result(frame)
        source_snapshot = frame.image_bgr.copy()

        annotated = draw_detection_failure(
            frame.image_bgr,
            sequence_id=3,
            stage="detect1",
            resize_result=resize_result,
        )

        np.testing.assert_array_equal(frame.image_bgr, source_snapshot)
        self.assertGreater(np.count_nonzero(annotated), 0)

    def test_style_scales_with_resolution_with_bounded_values(self) -> None:
        """不同分辨率应使用有上下限的线宽和角点半径。"""

        observed: list[tuple[int, int]] = []
        for height, width in ((120, 160), (2160, 3840)):
            image = np.zeros((height, width, 3), dtype=np.uint8)
            frame = make_camera_frame(
                1,
                1_000_000_000,
                image=image,
            )
            with (
                patch(
                    "ec_26_e.draw.cv2.polylines",
                    wraps=cv2.polylines,
                ) as polylines,
                patch(
                    "ec_26_e.draw.cv2.circle",
                    wraps=cv2.circle,
                ) as circle,
            ):
                draw_detection_failure(
                    frame.image_bgr,
                    sequence_id=1,
                    stage="workspace",
                    resize_result=make_resize_result(frame),
                )
            observed.append(
                (
                    polylines.call_args.args[4],
                    circle.call_args.args[2],
                )
            )

        self.assertEqual(observed[0], (1, 2))
        self.assertEqual(observed[1], (2, 4))


class OpenCVPreviewTest(unittest.TestCase):
    """验证普通 OpenCV 窗口生命周期、停止信号和错误映射。"""

    def test_default_uses_board_display_and_escape_or_close_stops(
        self,
    ) -> None:
        """默认连接板端 :0，也允许显式选择继承环境显示会话。"""

        image = np.zeros((20, 30, 3), dtype=np.uint8)
        with (
            patch.dict(os.environ, {"DISPLAY": ":existing"}, clear=False),
            patch(
                "ec_26_e.draw._validate_display_session",
                side_effect=[":0", ":existing"],
            ),
            patch("ec_26_e.draw.cv2.namedWindow") as named_window,
            patch("ec_26_e.draw.cv2.setMouseCallback"),
            patch("ec_26_e.draw.cv2.imshow"),
            patch(
                "ec_26_e.draw.cv2.waitKey",
                side_effect=[27, -1, -1, -1],
            ),
            patch(
                "ec_26_e.draw.cv2.getWindowProperty",
                side_effect=[1.0, 0.0],
            ),
            patch("ec_26_e.draw.cv2.destroyWindow") as destroy,
        ):
            first = OpenCVPreview()
            first.start()
            self.assertTrue(first.show_and_should_stop(image))
            first.close()
            first.close()

            os.environ["DISPLAY"] = ":existing"
            second = OpenCVPreview(
                window_name="second",
                display_name=None,
            )
            second.start()
            self.assertTrue(second.show_and_should_stop(image))
            second.close()

            self.assertEqual(os.environ["DISPLAY"], ":existing")

        self.assertEqual(named_window.call_count, 2)
        self.assertEqual(destroy.call_count, 2)

    def test_missing_or_unreachable_display_fails_before_qt(self) -> None:
        """无显示会话或连接失败时不得调用可能终止进程的 Qt 入口。"""

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("ec_26_e.draw.cv2.namedWindow") as named_window,
        ):
            with self.assertRaisesRegex(DrawError, "DISPLAY"):
                OpenCVPreview(display_name=None).start()
        named_window.assert_not_called()

        failed_check = subprocess.CompletedProcess(
            args=("xdpyinfo",),
            returncode=1,
            stderr="unable to open display",
        )
        with (
            patch("ec_26_e.draw.shutil.which", return_value="/usr/bin/xdpyinfo"),
            patch("ec_26_e.draw.subprocess.run", return_value=failed_check),
            patch("ec_26_e.draw.cv2.namedWindow") as named_window,
        ):
            with self.assertRaisesRegex(DrawError, "无法连接显示会话"):
                OpenCVPreview(display_name=":99").start()
        named_window.assert_not_called()

    def test_gui_errors_become_draw_error(self) -> None:
        """窗口初始化及逐帧显示错误应转为可预期 DrawError。"""

        with (
            patch(
                "ec_26_e.draw._validate_display_session",
                return_value=":0",
            ),
            patch(
                "ec_26_e.draw.cv2.namedWindow",
                side_effect=cv2.error("no gui"),
            ),
        ):
            with self.assertRaisesRegex(DrawError, "无法创建"):
                OpenCVPreview().start()

        preview = OpenCVPreview()
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        with (
            patch(
                "ec_26_e.draw._validate_display_session",
                return_value=":0",
            ),
            patch("ec_26_e.draw.cv2.namedWindow"),
            patch("ec_26_e.draw.cv2.setMouseCallback"),
            patch(
                "ec_26_e.draw.cv2.imshow",
                side_effect=cv2.error("display failed"),
            ),
            patch("ec_26_e.draw.cv2.destroyWindow"),
            patch("ec_26_e.draw.cv2.waitKey", return_value=-1),
        ):
            preview.start()
            with self.assertRaisesRegex(DrawError, "显示失败"):
                preview.show_and_should_stop(image)
            preview.close()


class Task1DrawObserverTest(unittest.TestCase):
    """验证 Task1 观察器只承担绘制和显示适配。"""

    def test_observer_forwards_annotated_copy_and_stop_signal(self) -> None:
        """观察器应转发独立标注帧并原样返回预览停止信号。"""

        class Preview:
            """记录观察器生命周期及收到的帧。"""

            def __init__(self) -> None:
                """初始化生命周期和帧记录。"""

                self.started = False
                self.closed = False
                self.images: list[np.ndarray] = []

            def start(self) -> None:
                """记录预览启动。"""

                self.started = True

            def show_and_should_stop(
                self,
                image_bgr: np.ndarray,
            ) -> bool:
                """保存标注帧并模拟用户停止。"""

                self.images.append(image_bgr)
                return True

            def close(self) -> None:
                """记录预览关闭。"""

                self.closed = True

        frame = make_camera_frame(9, 1_000_000_000)
        preview = Preview()
        observer = Task1DrawObserver(preview)
        observer.start()
        should_stop = observer.show_detection(
            frame,
            make_resize_result(frame),
            make_detect_result(),
        )
        observer.close()

        self.assertTrue(preview.started)
        self.assertTrue(preview.closed)
        self.assertTrue(should_stop)
        self.assertEqual(len(preview.images), 1)
        self.assertIsNot(preview.images[0], frame.image_bgr)
        self.assertGreater(np.count_nonzero(preview.images[0]), 0)


if __name__ == "__main__":
    unittest.main()
