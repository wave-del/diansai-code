"""验证实时检测录像脚本的绘制、停止信号与安全编码关闭。

该文件使用合成帧和测试替身，不访问相机、显示器或 FFmpeg。可通过
``python -m unittest tests.unit.cam.test_live_detection`` 独立运行。
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from ec_26_e.cam import CameraConfig, CameraFrame
from ec_26_e.vision import (
    Detect1Config,
    Detect1Result,
    PieceDetection,
    ResizeConfig,
    ResizeResult,
)
from tests.external.cam.helpers import (
    CameraDeviceInfo,
    parse_camera_info,
    query_camera_info,
)
from tests.external.cam.record_detected_video import (
    AsyncPairRecorder,
    OpenCVPreview,
    PREVIEW_WINDOW_NAME,
    draw_detection_failure,
    draw_live_detection,
    record_detection_video,
    resolve_output_paths,
)


CAMERA_INFO_TEXT = """\
Driver Info:
\tDriver name      : uvcvideo
\tCard type        : USB Camera2: USB Camera2
\tBus info         : usb-xhci-hcd.41.auto-1.2
Format Video Capture:
\tWidth/Height      : 1280/720
\tPixel Format      : 'MJPG'
Streaming Parameters Video Capture:
\tFrames per second: 30.000 (30/1)
"""


def create_camera_info() -> CameraDeviceInfo:
    """创建实时录像编排测试使用的摄像头信息。"""

    return CameraDeviceInfo(
        device_path=Path("/dev/video0"),
        driver_name="fake",
        card_type="fake camera",
        bus_info="fake bus",
        width=160,
        height=120,
        pixel_format="MJPG",
        fps=30.0,
        raw_text="fake",
    )


def create_resize_result(image_bgr: np.ndarray) -> ResizeResult:
    """创建从 160×120 源图映射到 140×99 A4 图的结果。"""

    source = np.array(
        [[10, 10], [150, 10], [150, 109], [10, 109]],
        dtype=np.float32,
    )
    target = np.array(
        [[0, 0], [139, 0], [139, 98], [0, 98]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(source, target)
    rectified = cv2.warpPerspective(
        image_bgr,
        homography,
        (140, 99),
    )
    return ResizeResult(
        image_bgr=rectified,
        source_size=(160, 120),
        target_size=(140, 99),
        source_corners_px=tuple(
            (float(x), float(y)) for x, y in source
        ),
        homography=homography,
        workspace_threshold=0.0,
        workspace_ratio=0.75,
        line_residual=0.2,
    )


def create_detect1_result() -> Detect1Result:
    """创建含一个矩形纸片的 A4 检测结果。"""

    piece = PieceDetection(
        group_id=0,
        corners_px=((30.0, 20.0), (70.0, 20.0), (70.0, 50.0), (30.0, 50.0)),
        center_px=(50.0, 35.0),
        corners_relative=(
            (30.0 / 139.0, 20.0 / 98.0),
            (70.0 / 139.0, 20.0 / 98.0),
            (70.0 / 139.0, 50.0 / 98.0),
            (30.0 / 139.0, 50.0 / 98.0),
        ),
        center_relative=(50.0 / 139.0, 35.0 / 98.0),
        area_px=1200.0,
        area_relative=0.09,
        polygon_iou=1.0,
        line_residual=0.1,
        confidence=0.99,
    )
    return Detect1Result(
        image_size=(140, 99),
        pieces=(piece,),
        foreground_threshold=0.0,
        workspace_area_px=13000,
    )


class FakeCamera:
    """持续生成不同序号的小尺寸只读帧。"""

    def __init__(self, config: CameraConfig) -> None:
        """保存配置并初始化序号。"""

        self.config = config
        self.sequence_id = 0

    def __enter__(self) -> FakeCamera:
        """返回已启动语义的测试替身。"""

        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object | None,
    ) -> None:
        """测试替身无需释放外部资源。"""

    def read_latest(
        self,
        after_sequence: int | None = None,
        timeout: float | None = None,
    ) -> CameraFrame:
        """返回一张序号递增的帧。"""

        image = np.full(
            (self.config.height, self.config.width, 3),
            self.sequence_id,
            dtype=np.uint8,
        )
        frame = CameraFrame(
            sequence_id=self.sequence_id,
            received_at_ns=time.monotonic_ns(),
            image_bgr=image,
        )
        self.sequence_id += 1
        return frame


class InterruptingCamera(FakeCamera):
    """在第一帧之后用 KeyboardInterrupt 模拟无预览退出。"""

    def read_latest(
        self,
        after_sequence: int | None = None,
        timeout: float | None = None,
    ) -> CameraFrame:
        """首轮返回帧，下一轮触发终端中断。"""

        if self.sequence_id > 0:
            raise KeyboardInterrupt
        return super().read_latest(after_sequence, timeout)


class FakeWriter:
    """收集帧对并记录启动、写入和关闭顺序。"""

    instances: list[FakeWriter] = []

    def __init__(
        self,
        config: CameraConfig,
        raw_output_path: Path,
        annotated_output_path: Path,
        *,
        overwrite: bool = False,
    ) -> None:
        """保存构造参数并注册实例。"""

        self.config = config
        self.raw_output_path = raw_output_path
        self.annotated_output_path = annotated_output_path
        self.overwrite = overwrite
        self.frame_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        self.events: list[str] = []
        self.instances.append(self)

    def start(self) -> None:
        """记录编码器启动。"""

        self.events.append("start")

    def write_pair(
        self,
        raw_image_bgr: np.ndarray,
        annotated_image_bgr: np.ndarray,
    ) -> None:
        """保存同步帧副本。"""

        self.events.append("write")
        self.frame_pairs.append(
            (raw_image_bgr.copy(), annotated_image_bgr.copy())
        )

    def close(self) -> None:
        """记录视频封装完成。"""

        self.events.append("close")


class FakeResize:
    """返回确定性工作区变换的 Resize 测试替身。"""

    def __init__(self, config: ResizeConfig) -> None:
        """接受与真实 Resize 相同的配置。"""

        self.config = config

    def resize(self, image_bgr: np.ndarray) -> ResizeResult:
        """返回与当前源帧匹配的校正结果。"""

        return create_resize_result(image_bgr)


class FakeVisionDetector:
    """返回固定单帧与稳定结果的统一视觉测试替身。"""

    def __init__(self, config: Detect1Config) -> None:
        """接受与正式生产工厂相同的兼容配置。"""

        self.config = config
        self.last_single_result: Detect1Result | None = None
        self.close_calls = 0

    def detect(self, image_bgr: np.ndarray) -> Detect1Result:
        """返回一个成功结果并同步保存单帧精修诊断。"""

        self.last_single_result = create_detect1_result()
        return self.last_single_result

    def close(self) -> None:
        """记录统一视觉资源释放次数。"""

        self.close_calls += 1


class FakePreview:
    """保存显示帧并在指定帧数请求停止。"""

    def __init__(self, stop_after_frames: int | None = None) -> None:
        """保存可选自动停止帧数。"""

        self.stop_after_frames = stop_after_frames
        self.frames: list[np.ndarray] = []
        self.start_calls = 0
        self.close_calls = 0

    def start(self) -> None:
        """记录预览启动。"""

        self.start_calls += 1

    def show_and_should_stop(self, image_bgr: np.ndarray) -> bool:
        """保存帧并按约定返回停止信号。"""

        self.frames.append(image_bgr.copy())
        return (
            self.stop_after_frames is not None
            and len(self.frames) >= self.stop_after_frames
        )

    def close(self) -> None:
        """记录预览关闭。"""

        self.close_calls += 1


class BlockingPairWriter:
    """阻塞首帧写入以验证异步单槽覆盖和关闭排空。"""

    def __init__(self) -> None:
        """初始化线程事件和写入值。"""

        self.write_started = threading.Event()
        self.release_write = threading.Event()
        self.values: list[int] = []
        self.closed = False

    def start(self) -> None:
        """模拟编码器启动。"""

    def write_pair(
        self,
        raw_image_bgr: np.ndarray,
        annotated_image_bgr: np.ndarray,
    ) -> None:
        """等待释放后记录成对帧的共同像素值。"""

        self.write_started.set()
        if not self.release_write.wait(timeout=1.0):
            raise RuntimeError("测试未及时释放录像写入")
        raw_value = int(raw_image_bgr[0, 0, 0])
        annotated_value = int(annotated_image_bgr[0, 0, 0])
        if raw_value != annotated_value:
            raise ValueError("测试帧对不一致")
        self.values.append(raw_value)

    def close(self) -> None:
        """记录编码器已经封装。"""

        self.closed = True


class CameraInfoTest(unittest.TestCase):
    """验证摄像头信息解析和外部命令错误映射。"""

    def test_parse_camera_info_extracts_active_profile(self) -> None:
        """解析结果应包含设备、720p、MJPG 和 30 fps。"""

        info = parse_camera_info(Path("/dev/video0"), CAMERA_INFO_TEXT)

        self.assertEqual(info.driver_name, "uvcvideo")
        self.assertEqual((info.width, info.height), (1280, 720))
        self.assertEqual(info.pixel_format, "MJPG")
        self.assertEqual(info.fps, 30.0)

    def test_parse_camera_info_rejects_missing_profile(self) -> None:
        """缺少活动格式时应给出明确解析错误。"""

        with self.assertRaisesRegex(ValueError, "无法.*解析"):
            parse_camera_info(Path("/dev/video0"), "Driver name: fake")

    def test_query_camera_info_reports_command_failure(self) -> None:
        """外部查询失败时应包含返回码和诊断文本。"""

        completed = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="device busy",
        )
        with patch("subprocess.run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "device busy"):
                query_camera_info(Path("/dev/video0"))


class DrawingTest(unittest.TestCase):
    """验证原图绘制和脚本级逆透视映射。"""

    def test_draws_workspace_and_mapped_piece_on_copy(self) -> None:
        """标注帧应同时包含工作区和由 A4 映回原图的纸片。"""

        image = np.zeros((120, 160, 3), dtype=np.uint8)
        source = image.copy()
        resize_result = create_resize_result(image)

        annotated = draw_live_detection(
            image,
            resize_result,
            create_detect1_result(),
            sequence_id=3,
        )

        np.testing.assert_array_equal(image, source)
        self.assertFalse(np.array_equal(annotated, image))
        # 工作区左上角和映射后纸片内部均应出现绘制像素。
        self.assertTrue(np.any(annotated[7:14, 7:14] != 0))
        mapped_center = cv2.perspectiveTransform(
            np.array([[[50.0, 35.0]]], dtype=np.float32),
            np.linalg.inv(resize_result.homography),
        )[0, 0]
        x, y = np.rint(mapped_center).astype(int)
        self.assertTrue(np.any(annotated[y - 3 : y + 4, x - 3 : x + 40] != 0))

    def test_detect1_failure_keeps_workspace_boundary(self) -> None:
        """Detect1 失败标注仍应显示已经成功定位的工作区。"""

        image = np.zeros((120, 160, 3), dtype=np.uint8)
        annotated = draw_detection_failure(
            image,
            sequence_id=4,
            stage="detect1",
            resize_result=create_resize_result(image),
        )

        self.assertTrue(np.any(annotated[7:14, 7:14] != 0))


class RecordingOrchestrationTest(unittest.TestCase):
    """验证预览或 Ctrl+C 停止后会排空并关闭编码器。"""

    def setUp(self) -> None:
        """创建所有编排测试共享的强类型配置。"""

        FakeWriter.instances.clear()
        self.camera_config = CameraConfig(
            width=160,
            height=120,
            startup_timeout_seconds=0.1,
            read_timeout_seconds=0.1,
        )
        self.resize_config = ResizeConfig(
            detect_width=160,
            detect_height=120,
            target_width=140,
            target_height=99,
        )
        self.detect1_config = Detect1Config()

    def _patch_pipeline(self, camera_type: type[FakeCamera]) -> object:
        """返回替换相机、编码器和两个检测单元的组合上下文。"""

        return patch.multiple(
            "tests.external.cam.record_detected_video",
            FFmpegCamera=camera_type,
            H264VideoPairWriter=FakeWriter,
            Resize=FakeResize,
            create_production_detector=FakeVisionDetector,
        )

    def test_escape_style_preview_stop_writes_current_frame(self) -> None:
        """预览停止请求应先提交当前帧，再安全完成封装。"""

        preview = FakePreview(stop_after_frames=1)
        with self._patch_pipeline(FakeCamera):
            summary = record_detection_video(
                self.camera_config,
                self.resize_config,
                self.detect1_config,
                create_camera_info(),
                Path("outputs/cam/preview"),
                preview=preview,
            )

        self.assertTrue(summary.stopped_by_user)
        self.assertFalse(summary.interrupted)
        self.assertEqual(summary.processed_frames, 1)
        self.assertEqual(summary.recorded_frames, 1)
        self.assertEqual(preview.close_calls, 1)
        writer = FakeWriter.instances[0]
        self.assertEqual(writer.events, ["start", "write", "close"])
        self.assertFalse(
            np.array_equal(writer.frame_pairs[0][0], writer.frame_pairs[0][1])
        )

    def test_ctrl_c_without_preview_safely_finishes_encoder(self) -> None:
        """无预览 Ctrl+C 应保留首帧并等待编码器关闭。"""

        with self._patch_pipeline(InterruptingCamera):
            summary = record_detection_video(
                self.camera_config,
                self.resize_config,
                self.detect1_config,
                create_camera_info(),
                Path("outputs/cam/interrupt"),
                preview=None,
            )

        self.assertTrue(summary.interrupted)
        self.assertFalse(summary.stopped_by_user)
        self.assertEqual(summary.processed_frames, 1)
        self.assertEqual(summary.recorded_frames, 1)
        self.assertIn("close", FakeWriter.instances[0].events)

    def test_preview_without_recording_does_not_start_writer(self) -> None:
        """快速预览关闭录像时不得创建路径或启动编码器。"""

        preview = FakePreview(stop_after_frames=1)
        with self._patch_pipeline(FakeCamera):
            summary = record_detection_video(
                self.camera_config,
                self.resize_config,
                self.detect1_config,
                create_camera_info(),
                preview=preview,
                record_enabled=False,
            )

        self.assertTrue(summary.stopped_by_user)
        self.assertFalse(summary.recording_enabled)
        self.assertIsNone(summary.raw_output_path)
        self.assertIsNone(summary.annotated_output_path)
        self.assertEqual(summary.recorded_frames, 0)
        self.assertEqual(FakeWriter.instances, [])

    def test_output_paths_share_session_prefix(self) -> None:
        """两路路径应共享会话前缀并使用不同后缀。"""

        raw_path, annotated_path = resolve_output_paths(
            Path("outputs/cam/session")
        )

        self.assertEqual(raw_path, Path("outputs/cam/session_raw.mp4"))
        self.assertEqual(
            annotated_path,
            Path("outputs/cam/session_annotated.mp4"),
        )


class OpenCVPreviewTest(unittest.TestCase):
    """验证 Esc、Q 和关闭窗口的停止语义。"""

    def test_q_does_not_stop_but_escape_stops(self) -> None:
        """Q 应继续运行，随后按 Esc 才返回停止信号。"""

        image = np.zeros((120, 160, 3), dtype=np.uint8)
        preview = OpenCVPreview(display_name=":0", fullscreen=True)
        with (
            patch.dict(os.environ, {}, clear=False),
            patch(
                "ec_26_e.draw._validate_display_session",
                return_value=":0",
            ),
            patch("ec_26_e.draw.cv2.namedWindow"),
            patch("ec_26_e.draw.cv2.setMouseCallback"),
            patch("ec_26_e.draw.cv2.setWindowProperty"),
            patch("ec_26_e.draw.cv2.imshow"),
            patch(
                "ec_26_e.draw.cv2.waitKey",
                side_effect=[ord("q"), 27, -1],
            ),
            patch(
                "ec_26_e.draw.cv2.getWindowProperty",
                return_value=1.0,
            ),
            patch("ec_26_e.draw.cv2.destroyWindow"),
        ):
            preview.start()
            q_stop = preview.show_and_should_stop(image)
            escape_stop = preview.show_and_should_stop(image)
            preview.close()

        self.assertEqual(os.environ["DISPLAY"], ":0")
        self.assertFalse(q_stop)
        self.assertTrue(escape_stop)

    def test_closed_window_requests_stop(self) -> None:
        """窗口不可见时应触发正常停止。"""

        image = np.zeros((120, 160, 3), dtype=np.uint8)
        preview = OpenCVPreview(
            display_name=":0",
            fullscreen=False,
            window_size=(1280, 720),
        )
        with (
            patch(
                "ec_26_e.draw._validate_display_session",
                return_value=":0",
            ),
            patch("ec_26_e.draw.cv2.namedWindow"),
            patch("ec_26_e.draw.cv2.setMouseCallback"),
            patch("ec_26_e.draw.cv2.resizeWindow"),
            patch("ec_26_e.draw.cv2.moveWindow"),
            patch("ec_26_e.draw.cv2.imshow"),
            patch(
                "ec_26_e.draw.cv2.waitKey",
                side_effect=[-1, -1],
            ),
            patch(
                "ec_26_e.draw.cv2.getWindowProperty",
                return_value=0.0,
            ),
            patch("ec_26_e.draw.cv2.destroyWindow"),
        ):
            preview.start()
            should_stop = preview.show_and_should_stop(image)
            preview.close()

        self.assertTrue(should_stop)

    def test_exit_button_click_requests_stop_without_mutating_frame(self) -> None:
        """点击 EXIT 应正常停止，且按钮不得绘制到调用方原始帧。"""

        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        original = image.copy()
        preview = OpenCVPreview(display_name=":0", fullscreen=True)
        callbacks: list[object] = []

        def capture_callback(_window_name: str, callback: object) -> None:
            """保存 OpenCV 注册的鼠标回调。"""

            callbacks.append(callback)

        with (
            patch(
                "ec_26_e.draw._validate_display_session",
                return_value=":0",
            ),
            patch("ec_26_e.draw.cv2.namedWindow"),
            patch(
                "ec_26_e.draw.cv2.setMouseCallback",
                side_effect=capture_callback,
            ),
            patch("ec_26_e.draw.cv2.setWindowProperty"),
            patch("ec_26_e.draw.cv2.imshow"),
            patch("ec_26_e.draw.cv2.waitKey", return_value=-1),
            patch(
                "ec_26_e.draw.cv2.getWindowProperty",
                return_value=1.0,
            ),
            patch("ec_26_e.draw.cv2.destroyWindow"),
        ):
            preview.start()
            self.assertFalse(preview.show_and_should_stop(image))
            self.assertIsNotNone(preview._exit_button_bounds)
            x1, y1, x2, y2 = preview._exit_button_bounds or (0, 0, 0, 0)
            callback = callbacks[0]
            callback(
                cv2.EVENT_LBUTTONUP,
                (x1 + x2) // 2,
                (y1 + y2) // 2,
                0,
                None,
            )
            self.assertTrue(preview.show_and_should_stop(image))
            preview.close()

        np.testing.assert_array_equal(image, original)


class AsyncPairRecorderTest(unittest.TestCase):
    """验证慢编码器不会阻塞检测且关闭时写完最新帧。"""

    def test_pending_pair_is_overwritten_and_latest_is_drained(self) -> None:
        """繁忙时覆盖旧待处理帧，关闭前仍应写入最新帧。"""

        writer = BlockingPairWriter()
        recorder = AsyncPairRecorder(writer)  # type: ignore[arg-type]
        recorder.start()
        first = np.full((1, 1, 3), 1, dtype=np.uint8)
        second = np.full((1, 1, 3), 2, dtype=np.uint8)
        third = np.full((1, 1, 3), 3, dtype=np.uint8)

        recorder.submit(first, first)
        self.assertTrue(writer.write_started.wait(timeout=0.5))
        recorder.submit(second, second)
        recorder.submit(third, third)
        writer.release_write.set()
        recorder.close()

        self.assertEqual(writer.values, [1, 3])
        self.assertEqual(recorder.recorded_frames, 2)
        self.assertEqual(recorder.overwritten_frames, 1)
        self.assertTrue(writer.closed)


if __name__ == "__main__":
    unittest.main()
