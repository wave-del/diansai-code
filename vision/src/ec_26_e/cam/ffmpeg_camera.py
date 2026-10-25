"""通过 FFmpeg/V4L2 实现低延迟、仅保留最新帧的相机采集器。"""

from __future__ import annotations

from collections import deque
import subprocess
import threading
import time
from typing import BinaryIO

import numpy as np

from ec_26_e.cam.contracts import (
    CameraCaptureError,
    CameraConfig,
    CameraFrame,
    CameraTimeoutError,
)

_PROCESS_STOP_TIMEOUT_SECONDS = 2.0
_THREAD_JOIN_TIMEOUT_SECONDS = 2.0
_STDERR_TAIL_LINES = 30


def _build_ffmpeg_command(config: CameraConfig) -> list[str]:
    """构造无帧率转换、输出固定尺寸 BGR24 帧的 FFmpeg 命令。"""

    return [
        config.ffmpeg_executable,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-nostdin",
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-f",
        "v4l2",
        "-input_format",
        config.input_format,
        "-video_size",
        f"{config.width}x{config.height}",
        "-framerate",
        str(config.fps),
        "-i",
        str(config.device_path),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-vsync",
        "0",
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]


def _validate_after_sequence(after_sequence: int | None) -> None:
    """校验调用方用于等待新帧的序号。"""

    if after_sequence is None:
        return
    if isinstance(after_sequence, bool) or not isinstance(after_sequence, int):
        raise TypeError("after_sequence 必须是整数或 None")
    if after_sequence < -1:
        raise ValueError("after_sequence 不能小于 -1")


def _resolve_timeout(config: CameraConfig, timeout: float | None) -> float:
    """解析单次读取超时，并拒绝无限或无效等待。"""

    resolved = config.read_timeout_seconds if timeout is None else timeout
    if isinstance(resolved, bool) or not isinstance(resolved, (int, float)):
        raise TypeError("timeout 必须是实数或 None")
    if not np.isfinite(resolved) or resolved <= 0:
        raise ValueError("timeout 必须是有限正数")
    return float(resolved)


class FFmpegCamera:
    """管理 FFmpeg 子进程，并以线程安全方式发布最新完整帧。

    后台线程始终排空 FFmpeg 标准输出。消费者处理较慢时，中间帧
    会被单槽缓存覆盖，但采集进程不会因业务侧积压旧帧。
    """

    def __init__(self, config: CameraConfig | None = None) -> None:
        """保存强类型配置；此阶段不会访问设备或创建子进程。"""

        if config is not None and not isinstance(config, CameraConfig):
            raise TypeError("config 必须是 CameraConfig 或 None")
        self.config = config or CameraConfig()
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._process: subprocess.Popen[bytes] | None = None
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._latest_frame: CameraFrame | None = None
        self._next_sequence = 0
        self._failure: str | None = None
        self._stderr_tail: deque[str] = deque(maxlen=_STDERR_TAIL_LINES)
        self._started = False
        self._closed = False

    def start(self) -> None:
        """启动 FFmpeg 和后台线程，并在超时前等待首张完整帧。"""

        with self._condition:
            if self._closed:
                raise CameraCaptureError(
                    "相机采集器已关闭，不能重新启动"
                )
            if self._started:
                raise CameraCaptureError("相机采集器已经启动")

        command = _build_ffmpeg_command(self.config)
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except OSError as error:
            raise CameraCaptureError(
                f"无法启动 FFmpeg：{error}"
            ) from error

        if process.stdout is None or process.stderr is None:
            process.kill()
            process.wait()
            raise CameraCaptureError("FFmpeg 未提供所需的输出管道")

        with self._condition:
            self._process = process
            self._started = True
            self._reader_thread = threading.Thread(
                target=self._read_frames,
                args=(process.stdout,),
                name="ffmpeg-camera-reader",
                daemon=True,
            )
            self._stderr_thread = threading.Thread(
                target=self._drain_stderr,
                args=(process.stderr,),
                name="ffmpeg-camera-stderr",
                daemon=True,
            )
            self._stderr_thread.start()
            self._reader_thread.start()

        try:
            self.read_latest(timeout=self.config.startup_timeout_seconds)
        except (CameraCaptureError, CameraTimeoutError):
            self.close()
            raise

    def read_latest(
        self,
        after_sequence: int | None = None,
        timeout: float | None = None,
    ) -> CameraFrame:
        """返回当前最新帧，或等待比 ``after_sequence`` 更新的帧。

        ``timeout=None`` 使用配置中的默认读取超时，不提供无限等待。
        """

        _validate_after_sequence(after_sequence)
        resolved_timeout = _resolve_timeout(self.config, timeout)
        deadline = time.monotonic() + resolved_timeout

        with self._condition:
            if not self._started:
                raise CameraCaptureError("相机采集器尚未启动")

            while True:
                if self._closed:
                    raise CameraCaptureError("相机采集器已关闭")
                if self._failure is not None:
                    raise CameraCaptureError(self._format_failure())
                if self._latest_frame is not None and (
                    after_sequence is None
                    or self._latest_frame.sequence_id > after_sequence
                ):
                    return self._latest_frame

                process = self._process
                if process is not None and process.poll() is not None:
                    self._failure = (
                        f"FFmpeg 已退出，返回码为 {process.returncode}"
                    )
                    raise CameraCaptureError(self._format_failure())

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    expected = (
                        "首张帧"
                        if after_sequence is None
                        else f"序号大于 {after_sequence} 的新帧"
                    )
                    raise CameraTimeoutError(
                        f"等待{expected}超时（{resolved_timeout:.3f} 秒）"
                    )
                self._condition.wait(remaining)

    def close(self) -> None:
        """幂等停止采集，并回收 FFmpeg、管道和后台线程。"""

        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._stop_event.set()
            process = self._process
            reader_thread = self._reader_thread
            stderr_thread = self._stderr_thread
            self._condition.notify_all()

        if process is not None:
            if process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    pass
            self._close_stream(process.stdout)
            self._close_stream(process.stderr)
            try:
                process.wait(timeout=_PROCESS_STOP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except OSError:
                    pass
                process.wait()

        current_thread = threading.current_thread()
        for thread in (reader_thread, stderr_thread):
            if thread is not None and thread is not current_thread:
                thread.join(timeout=_THREAD_JOIN_TIMEOUT_SECONDS)

    def __enter__(self) -> FFmpegCamera:
        """启动采集并返回上下文中的当前实例。"""

        self.start()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object | None,
    ) -> None:
        """退出上下文时始终释放相机资源。"""

        self.close()

    def _read_frames(self, stream: BinaryIO) -> None:
        """持续读取完整 BGR24 帧并覆盖发布槽。"""

        frame_size = self.config.frame_size_bytes
        try:
            while not self._stop_event.is_set():
                buffer = bytearray(frame_size)
                view = memoryview(buffer)
                received = 0
                while received < frame_size and not self._stop_event.is_set():
                    chunk = stream.read(frame_size - received)
                    if not chunk:
                        break
                    view[received : received + len(chunk)] = chunk
                    received += len(chunk)

                if self._stop_event.is_set():
                    return
                if received == 0:
                    self._set_failure("FFmpeg 图像管道已关闭")
                    return
                if received != frame_size:
                    self._set_failure(
                        "FFmpeg 返回不完整帧："
                        f"收到 {received} 字节，预期 {frame_size} 字节"
                    )
                    return

                image = np.frombuffer(buffer, dtype=np.uint8).reshape(
                    self.config.height,
                    self.config.width,
                    3,
                )
                with self._condition:
                    frame = CameraFrame(
                        sequence_id=self._next_sequence,
                        received_at_ns=time.monotonic_ns(),
                        image_bgr=image,
                    )
                    self._next_sequence += 1
                    self._latest_frame = frame
                    self._condition.notify_all()
        except (OSError, ValueError) as error:
            if not self._stop_event.is_set():
                self._set_failure(f"读取 FFmpeg 图像管道失败：{error}")

    def _drain_stderr(self, stream: BinaryIO) -> None:
        """持续排空 FFmpeg 标准错误并保留有限诊断尾部。"""

        try:
            while not self._stop_event.is_set():
                line = stream.readline()
                if not line:
                    return
                decoded = line.decode("utf-8", errors="replace").strip()
                if decoded:
                    with self._condition:
                        self._stderr_tail.append(decoded)
        except (OSError, ValueError):
            return

    def _set_failure(self, message: str) -> None:
        """记录首个后台故障并唤醒所有等待读取者。"""

        with self._condition:
            if self._failure is None and not self._closed:
                self._failure = message
            self._condition.notify_all()

    def _format_failure(self) -> str:
        """把后台故障与 FFmpeg 标准错误尾部组合为可操作信息。"""

        message = self._failure or "相机采集发生未知故障"
        if not self._stderr_tail:
            return message
        return f"{message}；FFmpeg：{' | '.join(self._stderr_tail)}"

    @staticmethod
    def _close_stream(stream: BinaryIO | None) -> None:
        """忽略重复关闭或进程退出导致的管道关闭错误。"""

        if stream is None:
            return
        try:
            stream.close()
        except OSError:
            pass
