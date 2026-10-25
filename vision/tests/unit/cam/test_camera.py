"""验证 FFmpeg 相机配置、最新帧语义、失败传播和资源释放。

该文件使用内存假进程与可控管道，不依赖相机或系统 FFmpeg。
可通过 ``python -m unittest tests.unit.cam.test_camera`` 独立运行。
"""

from __future__ import annotations

from pathlib import Path
import queue
import subprocess
import time
import unittest
from unittest.mock import patch

import numpy as np

from ec_26_e.cam import (
    CameraCaptureError,
    CameraConfig,
    CameraFrame,
    CameraSource,
    CameraTimeoutError,
    FFmpegCamera,
)
from ec_26_e.cam.ffmpeg_camera import _build_ffmpeg_command
from tests.external.cam.helpers import (
    build_h264_encoder_command,
    build_h264_pair_encoder_command,
)


class ControlledStream:
    """提供可分块送入、可阻塞读取的线程安全二进制假管道。"""

    def __init__(self) -> None:
        """初始化空管道，读取会等待 feed、finish 或 close。"""

        self._chunks: queue.Queue[bytes | None] = queue.Queue()
        self._pending = bytearray()
        self._finished = False
        self._closed = False

    def feed(self, data: bytes) -> None:
        """向管道追加一块非空数据。"""

        if not data:
            raise ValueError("测试数据块不能为空")
        self._chunks.put(data)

    def finish(self) -> None:
        """标记生产端结束，使剩余数据读完后返回 EOF。"""

        self._chunks.put(None)

    def read(self, size: int = -1) -> bytes:
        """按请求长度返回数据，并允许一个帧跨越多次读取。"""

        while not self._pending and not self._finished and not self._closed:
            chunk = self._chunks.get(timeout=1.0)
            if chunk is None:
                self._finished = True
            else:
                self._pending.extend(chunk)
        if not self._pending:
            return b""
        count = len(self._pending) if size < 0 else min(size, len(self._pending))
        result = bytes(self._pending[:count])
        del self._pending[:count]
        return result

    def readline(self, size: int = -1) -> bytes:
        """返回一块错误输出；测试数据均按完整行送入。"""

        return self.read(size)

    def close(self) -> None:
        """关闭管道并唤醒可能阻塞的读取线程。"""

        self._closed = True
        self._chunks.put(None)


class FakeProcess:
    """模拟 FFmpeg 子进程的管道、状态和终止调用。"""

    def __init__(self) -> None:
        """创建仍在运行且管道为空的假进程。"""

        self.stdin = None
        self.stdout = ControlledStream()
        self.stderr = ControlledStream()
        self.returncode: int | None = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls = 0

    def poll(self) -> int | None:
        """返回当前假进程退出码。"""

        return self.returncode

    def terminate(self) -> None:
        """记录温和终止并关闭两个输出管道。"""

        self.terminate_calls += 1
        self.returncode = 0
        self.stdout.close()
        self.stderr.close()

    def kill(self) -> None:
        """记录强制终止并关闭两个输出管道。"""

        self.kill_calls += 1
        self.returncode = -9
        self.stdout.close()
        self.stderr.close()

    def wait(self, timeout: float | None = None) -> int:
        """立即返回退出码，模拟已经完成回收的进程。"""

        self.wait_calls += 1
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake-ffmpeg", timeout)
        return self.returncode

    def exit(self, returncode: int) -> None:
        """模拟 FFmpeg 自行退出并结束图像管道。"""

        self.returncode = returncode
        self.stdout.finish()


def create_test_config(**overrides: object) -> CameraConfig:
    """创建尺寸很小、测试等待时间很短的相机配置。"""

    values: dict[str, object] = {
        "width": 2,
        "height": 1,
        "fps": 30,
        "startup_timeout_seconds": 0.5,
        "read_timeout_seconds": 0.1,
    }
    values.update(overrides)
    return CameraConfig(**values)  # type: ignore[arg-type]


def frame_bytes(value: int) -> bytes:
    """构造一张 1×2 的纯色 BGR 测试帧。"""

    return bytes([value] * 6)


def wait_for_sequence(camera: FFmpegCamera, expected: int) -> CameraFrame:
    """轮询最新槽，直到指定序号出现或测试超时。"""

    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        frame = camera.read_latest()
        if frame.sequence_id >= expected:
            return frame
        time.sleep(0.002)
    raise AssertionError(f"未等到帧序号 {expected}")


class CameraConfigTest(unittest.TestCase):
    """验证相机默认配置、固定帧大小和非法参数。"""

    def test_default_profile_matches_camera_capability(self) -> None:
        """默认配置应使用 MJPEG 1080p30 和标准设备路径。"""

        config = CameraConfig()

        self.assertEqual(str(config.device_path), "/dev/video0")
        self.assertEqual((config.width, config.height), (1920, 1080))
        self.assertEqual(config.fps, 30)
        self.assertEqual(config.input_format, "mjpeg")
        self.assertEqual(config.frame_size_bytes, 1920 * 1080 * 3)

    def test_invalid_config_values_are_rejected(self) -> None:
        """布尔、非正数、未知格式和空命令均应在启动前报错。"""

        invalid_cases = (
            ({"width": 0}, (ValueError, "width")),
            ({"height": True}, (TypeError, "height")),
            ({"fps": 0}, (ValueError, "fps")),
            ({"input_format": "rgb24"}, (ValueError, "input_format")),
            ({"ffmpeg_executable": " "}, (ValueError, "ffmpeg_executable")),
            (
                {"startup_timeout_seconds": float("inf")},
                (ValueError, "startup_timeout_seconds"),
            ),
            (
                {"read_timeout_seconds": False},
                (TypeError, "read_timeout_seconds"),
            ),
        )

        for overrides, (error_type, message) in invalid_cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(error_type, message):
                    create_test_config(**overrides)

    def test_camera_frame_is_validated_and_read_only(self) -> None:
        """结构化帧应保持 BGR 数组只读并拒绝错误像素类型。"""

        image = np.zeros((1, 2, 3), dtype=np.uint8)
        frame = CameraFrame(0, 1, image)

        self.assertFalse(frame.image_bgr.flags.writeable)
        with self.assertRaises(ValueError):
            CameraFrame(
                0,
                1,
                np.zeros((1, 2, 3), dtype=np.float32),
            )


class FFmpegCommandTest(unittest.TestCase):
    """验证 FFmpeg 命令使用目标设备配置且不转换帧率。"""

    def test_capture_command_uses_v4l2_mjpeg_and_raw_bgr_output(self) -> None:
        """命令应输入 MJPEG 1080p30，并逐帧输出 BGR24。"""

        command = _build_ffmpeg_command(CameraConfig())

        self.assertEqual(command[0], "ffmpeg")
        self.assertEqual(
            command[command.index("-input_format") + 1],
            "mjpeg",
        )
        self.assertEqual(
            command[command.index("-video_size") + 1],
            "1920x1080",
        )
        self.assertEqual(command[command.index("-framerate") + 1], "30")
        self.assertEqual(command[command.index("-pix_fmt") + 1], "bgr24")
        self.assertEqual(command[command.index("-vsync") + 1], "0")
        self.assertEqual(command[-2:], ["rawvideo", "pipe:1"])

    def test_recorder_command_preserves_wall_time_and_uses_h264(self) -> None:
        """录像命令应按墙钟生成标称 30 fps 的 H.264 MP4。"""

        command = build_h264_encoder_command(
            CameraConfig(),
            CameraConfig().device_path.with_name("capture.mp4"),
            overwrite=False,
        )

        self.assertEqual(
            command[command.index("-use_wallclock_as_timestamps") + 1],
            "1",
        )
        self.assertEqual(command[command.index("-c:v") + 1], "libx264")
        self.assertEqual(command[command.index("-preset") + 1], "ultrafast")
        self.assertEqual(command[command.index("-pix_fmt") + 1], "yuv420p")
        self.assertIn("-n", command)

    def test_pair_command_uses_one_input_and_two_synchronized_outputs(
        self,
    ) -> None:
        """双路命令应从一张拼接帧同步裁出原始和绘制视频。"""

        config = create_test_config()
        command = build_h264_pair_encoder_command(
            config,
            Path("/tmp/session_raw.mp4"),
            Path("/tmp/session_annotated.mp4"),
            overwrite=False,
        )

        self.assertEqual(
            command[command.index("-video_size") + 1],
            "4x1",
        )
        self.assertEqual(command.count("-i"), 1)
        self.assertEqual(command.count("-map"), 2)
        self.assertIn("[raw]", command)
        self.assertIn("[annotated]", command)
        self.assertEqual(command.count("libx264"), 2)


class FFmpegCameraTest(unittest.TestCase):
    """验证最新帧读取、异常状态和资源生命周期。"""

    def start_camera(
        self,
        process: FakeProcess,
        config: CameraConfig | None = None,
    ) -> FFmpegCamera:
        """向假进程预置首帧并启动相机。"""

        process.stdout.feed(frame_bytes(10))
        camera = FFmpegCamera(config or create_test_config())
        patcher = patch(
            "ec_26_e.cam.ffmpeg_camera.subprocess.Popen",
            return_value=process,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        camera.start()
        self.addCleanup(camera.close)
        return camera

    def test_camera_implements_public_protocol(self) -> None:
        """具体采集器应满足运行时可检查的 CameraSource 协议。"""

        self.assertIsInstance(FFmpegCamera(create_test_config()), CameraSource)

    def test_short_reads_form_complete_independent_frames(self) -> None:
        """分块管道应正确拼帧，且后续覆盖不修改已返回图像。"""

        process = FakeProcess()
        first = frame_bytes(10)
        process.stdout.feed(first[:2])
        process.stdout.feed(first[2:])
        camera = FFmpegCamera(create_test_config())
        with patch(
            "ec_26_e.cam.ffmpeg_camera.subprocess.Popen",
            return_value=process,
        ):
            camera.start()
        self.addCleanup(camera.close)

        first_frame = camera.read_latest()
        process.stdout.feed(frame_bytes(20))
        second_frame = camera.read_latest(
            after_sequence=first_frame.sequence_id,
        )

        self.assertEqual(second_frame.sequence_id, first_frame.sequence_id + 1)
        np.testing.assert_array_equal(first_frame.image_bgr, 10)
        np.testing.assert_array_equal(second_frame.image_bgr, 20)
        self.assertFalse(
            np.shares_memory(first_frame.image_bgr, second_frame.image_bgr)
        )
        self.assertFalse(second_frame.image_bgr.flags.writeable)

    def test_latest_slot_overwrites_intermediate_frames(self) -> None:
        """验证停顿的消费者只看到最新帧。

        调用方可由序号推断被覆盖的中间帧数量。
        """

        process = FakeProcess()
        camera = self.start_camera(process)
        first_frame = camera.read_latest()

        process.stdout.feed(frame_bytes(20))
        process.stdout.feed(frame_bytes(30))
        latest = wait_for_sequence(camera, first_frame.sequence_id + 2)

        self.assertEqual(latest.sequence_id - first_frame.sequence_id - 1, 1)
        np.testing.assert_array_equal(latest.image_bgr, 30)

    def test_waiting_for_new_frame_times_out(self) -> None:
        """没有更新帧时应抛出专用超时异常而非无限阻塞。"""

        process = FakeProcess()
        camera = self.start_camera(process)
        current = camera.read_latest()

        with self.assertRaisesRegex(CameraTimeoutError, "超时"):
            camera.read_latest(
                after_sequence=current.sequence_id,
                timeout=0.02,
            )

    def test_startup_timeout_releases_process(self) -> None:
        """首帧启动超时后应终止进程，且实例不得再次启动。"""

        process = FakeProcess()
        camera = FFmpegCamera(
            create_test_config(startup_timeout_seconds=0.02)
        )
        with patch(
            "ec_26_e.cam.ffmpeg_camera.subprocess.Popen",
            return_value=process,
        ):
            with self.assertRaisesRegex(CameraTimeoutError, "首张帧"):
                camera.start()

        self.assertEqual(process.terminate_calls, 1)
        self.assertGreaterEqual(process.wait_calls, 1)
        with self.assertRaisesRegex(CameraCaptureError, "已关闭"):
            camera.start()

    def test_partial_frame_failure_is_reported(self) -> None:
        """FFmpeg 在帧中途关闭管道时应报告实际和预期字节数。"""

        process = FakeProcess()
        camera = self.start_camera(process)
        current = camera.read_latest()
        process.stdout.feed(b"\x01\x02")
        process.stdout.finish()

        with self.assertRaisesRegex(CameraCaptureError, "不完整帧"):
            camera.read_latest(
                after_sequence=current.sequence_id,
                timeout=0.5,
            )

    def test_process_exit_includes_stderr_diagnostics(self) -> None:
        """后台退出错误应携带 FFmpeg 标准错误尾部。"""

        process = FakeProcess()
        camera = self.start_camera(process)
        current = camera.read_latest()
        process.stderr.feed(b"device disconnected\n")
        process.stderr.finish()
        process.exit(7)

        deadline = time.monotonic() + 0.5
        while not camera._stderr_tail and time.monotonic() < deadline:
            time.sleep(0.002)
        with self.assertRaisesRegex(
            CameraCaptureError,
            "device disconnected",
        ):
            camera.read_latest(
                after_sequence=current.sequence_id,
                timeout=0.5,
            )

    def test_lifecycle_rejects_invalid_start_and_closes_once(self) -> None:
        """验证非法生命周期调用会稳定报错。

        覆盖读取未启动实例、重复启动和关闭后重启。
        """

        camera = FFmpegCamera(create_test_config())
        with self.assertRaisesRegex(CameraCaptureError, "尚未启动"):
            camera.read_latest()

        process = FakeProcess()
        camera = self.start_camera(process)
        with self.assertRaisesRegex(CameraCaptureError, "已经启动"):
            camera.start()

        camera.close()
        camera.close()
        self.assertEqual(process.terminate_calls, 1)
        self.assertGreaterEqual(process.wait_calls, 1)
        with self.assertRaisesRegex(CameraCaptureError, "已关闭"):
            camera.start()

    def test_spawn_failure_is_wrapped(self) -> None:
        """FFmpeg 可执行文件缺失时应转换为相机领域异常。"""

        camera = FFmpegCamera(create_test_config())
        with patch(
            "ec_26_e.cam.ffmpeg_camera.subprocess.Popen",
            side_effect=FileNotFoundError("not found"),
        ):
            with self.assertRaisesRegex(CameraCaptureError, "无法启动"):
                camera.start()


if __name__ == "__main__":
    unittest.main()
