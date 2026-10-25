"""提供相机外部测试共享的设备信息查询与 H.264 编码能力。

该模块会调用系统 ``v4l2-ctl`` 和 FFmpeg，仅供显式运行的外部环境
测试使用，不会被默认单元测试主动访问外部设备。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import threading
from typing import BinaryIO

import numpy as np

from ec_26_e.cam import CameraConfig

ENCODER_STOP_TIMEOUT_SECONDS = 15.0
ENCODER_STDERR_TAIL_LINES = 30
DEFAULT_DEVICE_QUERY_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class CameraDeviceInfo:
    """描述 V4L2 当前报告的设备与活动采集格式。"""

    device_path: Path
    driver_name: str
    card_type: str
    bus_info: str
    width: int
    height: int
    pixel_format: str
    fps: float
    raw_text: str

    def to_dict(self) -> dict[str, object]:
        """转换为适合 JSON 输出的精简摄像头信息。"""

        return {
            "device_path": str(self.device_path),
            "driver_name": self.driver_name,
            "card_type": self.card_type,
            "bus_info": self.bus_info,
            "width": self.width,
            "height": self.height,
            "pixel_format": self.pixel_format,
            "fps": self.fps,
        }


def _match_text(raw_text: str, label: str) -> str:
    """从 v4l2-ctl 文本中提取冒号后的单行字段。"""

    match = re.search(
        rf"^\s*{re.escape(label)}\s*:\s*(.+?)\s*$",
        raw_text,
        flags=re.MULTILINE,
    )
    return match.group(1) if match else "未知"


def parse_camera_info(
    device_path: Path,
    raw_text: str,
) -> CameraDeviceInfo:
    """解析 v4l2-ctl 的设备、活动格式和帧率信息。"""

    if not isinstance(device_path, Path):
        raise TypeError("device_path 必须是 pathlib.Path")
    if not isinstance(raw_text, str):
        raise TypeError("raw_text 必须是字符串")

    size_match = re.search(
        r"^\s*Width/Height\s*:\s*(\d+)/(\d+)\s*$",
        raw_text,
        flags=re.MULTILINE,
    )
    format_match = re.search(
        r"^\s*Pixel Format\s*:\s*'([^']+)'",
        raw_text,
        flags=re.MULTILINE,
    )
    fps_match = re.search(
        r"^\s*Frames per second\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        raw_text,
        flags=re.MULTILINE,
    )
    if size_match is None or format_match is None or fps_match is None:
        raise ValueError("无法从 v4l2-ctl 输出解析活动格式或帧率")

    return CameraDeviceInfo(
        device_path=device_path,
        driver_name=_match_text(raw_text, "Driver name"),
        card_type=_match_text(raw_text, "Card type"),
        bus_info=_match_text(raw_text, "Bus info"),
        width=int(size_match.group(1)),
        height=int(size_match.group(2)),
        pixel_format=format_match.group(1),
        fps=float(fps_match.group(1)),
        raw_text=raw_text,
    )


def query_camera_info(
    device_path: Path,
    *,
    v4l2_ctl_executable: str = "v4l2-ctl",
    timeout_seconds: float = DEFAULT_DEVICE_QUERY_TIMEOUT_SECONDS,
) -> CameraDeviceInfo:
    """调用 v4l2-ctl 读取摄像头标识、活动格式与帧率。"""

    if not isinstance(device_path, Path):
        raise TypeError("device_path 必须是 pathlib.Path")
    if not isinstance(v4l2_ctl_executable, str):
        raise TypeError("v4l2_ctl_executable 必须是字符串")
    if not v4l2_ctl_executable.strip():
        raise ValueError("v4l2_ctl_executable 不能为空")
    if isinstance(timeout_seconds, bool) or not isinstance(
        timeout_seconds,
        (int, float),
    ):
        raise TypeError("timeout_seconds 必须是实数")
    if not np.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds 必须是有限正数")

    command = [
        v4l2_ctl_executable,
        f"--device={device_path}",
        "--info",
        "--get-fmt-video",
        "--get-parm",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=float(timeout_seconds),
        )
    except OSError as error:
        raise RuntimeError(
            f"无法启动摄像头信息查询工具：{error}"
        ) from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"读取摄像头信息超时（{timeout_seconds:.3f} 秒）"
        ) from error

    if result.returncode != 0:
        diagnostic = result.stderr.strip() or "无错误输出"
        raise RuntimeError(
            f"读取摄像头信息失败，返回码 {result.returncode}："
            f"{diagnostic}"
        )
    return parse_camera_info(device_path, result.stdout)


def build_h264_encoder_command(
    config: CameraConfig,
    output_path: Path,
    *,
    overwrite: bool,
) -> list[str]:
    """构造使用墙钟时间戳的低延迟 H.264 MP4 编码命令。"""

    return [
        config.ffmpeg_executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-use_wallclock_as_timestamps",
        "1",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{config.width}x{config.height}",
        "-framerate",
        str(config.fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-y" if overwrite else "-n",
        str(output_path),
    ]


def build_h264_pair_encoder_command(
    config: CameraConfig,
    raw_output_path: Path,
    annotated_output_path: Path,
    *,
    overwrite: bool,
) -> list[str]:
    """构造单时间基准、同步输出原始与绘制 MP4 的编码命令。"""

    combined_width = config.width * 2
    filter_graph = (
        "[0:v]split=2[raw_source][annotated_source];"
        f"[raw_source]crop={config.width}:{config.height}:0:0[raw];"
        "[annotated_source]"
        f"crop={config.width}:{config.height}:{config.width}:0[annotated]"
    )
    common_output_options = [
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    return [
        config.ffmpeg_executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y" if overwrite else "-n",
        "-use_wallclock_as_timestamps",
        "1",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{combined_width}x{config.height}",
        "-framerate",
        str(config.fps),
        "-i",
        "pipe:0",
        "-filter_complex",
        filter_graph,
        "-map",
        "[raw]",
        *common_output_options,
        str(raw_output_path),
        "-map",
        "[annotated]",
        *common_output_options,
        str(annotated_output_path),
    ]


class H264VideoWriter:
    """管理外部测试使用的 FFmpeg H.264 编码进程。"""

    def __init__(
        self,
        config: CameraConfig,
        output_path: Path,
        *,
        overwrite: bool = False,
    ) -> None:
        """保存编码配置；构造时不创建目录或启动进程。"""

        if not isinstance(config, CameraConfig):
            raise TypeError("config 必须是 CameraConfig")
        if not isinstance(output_path, Path):
            raise TypeError("output_path 必须是 pathlib.Path")
        self.config = config
        self.output_path = output_path
        self.overwrite = overwrite
        self.written_frames = 0
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_tail: deque[str] = deque(
            maxlen=ENCODER_STDERR_TAIL_LINES
        )
        self._started = False
        self._closed = False

    def start(self) -> None:
        """校验输出路径并启动 FFmpeg 编码进程。"""

        if self._closed:
            raise RuntimeError("H.264 编码器已关闭，不能重新启动")
        if self._started:
            raise RuntimeError("H.264 编码器已经启动")
        if self.output_path.exists() and not self.overwrite:
            raise FileExistsError(f"输出文件已存在：{self.output_path}")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        command = build_h264_encoder_command(
            self.config,
            self.output_path,
            overwrite=self.overwrite,
        )
        self._start_process(command)

    def write(self, image_bgr: np.ndarray) -> None:
        """完整写入一张 BGR 帧，并拒绝尺寸或类型不匹配。"""

        self._validate_frame(image_bgr)
        self._write_array(image_bgr)

    def close(self) -> None:
        """幂等关闭输入管道，等待 MP4 封装并检查退出码。"""

        if self._closed:
            return
        self._closed = True
        process = self._process
        if process is None:
            return

        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            return_code = process.wait(
                timeout=ENCODER_STOP_TIMEOUT_SECONDS
            )
        except subprocess.TimeoutExpired:
            try:
                process.terminate()
            except OSError:
                pass
            try:
                return_code = process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except OSError:
                    pass
                return_code = process.wait()

        if process.stderr is not None:
            try:
                process.stderr.close()
            except OSError:
                pass
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=2.0)
        if return_code != 0:
            raise RuntimeError(self._format_failure(return_code))

    def __enter__(self) -> H264VideoWriter:
        """启动编码器并返回上下文中的当前实例。"""

        self.start()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object | None,
    ) -> None:
        """退出上下文时完成 MP4 封装和编码进程回收。"""

        self.close()

    def _start_process(self, command: list[str]) -> None:
        """启动指定 FFmpeg 命令并建立错误输出排空线程。"""

        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=0,
                # 终端 Ctrl+C 只中断 Python 采集循环；FFmpeg 由 close() 收尾。
                start_new_session=True,
            )
        except OSError as error:
            raise RuntimeError(
                f"无法启动 FFmpeg 编码器：{error}"
            ) from error
        if process.stdin is None or process.stderr is None:
            process.kill()
            process.wait()
            raise RuntimeError("FFmpeg 编码器未提供所需管道")

        self._process = process
        self._started = True
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(process.stderr,),
            name="ffmpeg-recorder-stderr",
            daemon=True,
        )
        self._stderr_thread.start()

    def _validate_frame(self, image_bgr: np.ndarray) -> None:
        """校验单路 BGR 帧与相机配置完全匹配。"""

        expected_shape = (self.config.height, self.config.width, 3)
        if (
            not isinstance(image_bgr, np.ndarray)
            or image_bgr.shape != expected_shape
            or image_bgr.dtype != np.uint8
        ):
            raise ValueError(
                f"录像帧必须是 {expected_shape} 的 uint8 BGR 图像"
            )
        if not image_bgr.flags.c_contiguous:
            raise ValueError("录像帧必须使用连续内存")

    def _write_array(self, image_bgr: np.ndarray) -> None:
        """把任意连续 BGR 数组完整写入已启动的编码管道。"""

        if not self._started or self._process is None:
            raise RuntimeError("H.264 编码器尚未启动")
        if self._closed:
            raise RuntimeError("H.264 编码器已关闭")
        stream = self._process.stdin
        if stream is None:
            raise RuntimeError("FFmpeg 编码输入管道不可用")
        data = memoryview(image_bgr).cast("B")
        written = 0
        while written < len(data):
            try:
                count = stream.write(data[written:])
            except BrokenPipeError as error:
                raise RuntimeError(self._format_failure()) from error
            if count is None or count <= 0:
                raise RuntimeError(self._format_failure())
            written += count
        self.written_frames += 1

    def _drain_stderr(self, stream: BinaryIO) -> None:
        """排空编码错误管道并保留有限诊断尾部。"""

        try:
            for line in iter(stream.readline, b""):
                decoded = line.decode(
                    "utf-8",
                    errors="replace",
                ).strip()
                if decoded:
                    self._stderr_tail.append(decoded)
        except (OSError, ValueError):
            return

    def _format_failure(self, return_code: int | None = None) -> str:
        """组合编码失败状态和 FFmpeg 错误输出。"""

        prefix = "FFmpeg 编码管道已关闭"
        if return_code is not None:
            prefix = f"FFmpeg 编码失败，返回码 {return_code}"
        diagnostic = " | ".join(self._stderr_tail) or "无 FFmpeg 错误输出"
        return f"{prefix}：{diagnostic}"


class H264VideoPairWriter(H264VideoWriter):
    """用单个 FFmpeg 时间基准同步编码原始帧和绘制帧。"""

    def __init__(
        self,
        config: CameraConfig,
        raw_output_path: Path,
        annotated_output_path: Path,
        *,
        overwrite: bool = False,
    ) -> None:
        """保存两路输出路径；构造时不创建文件或进程。"""

        super().__init__(
            config,
            raw_output_path,
            overwrite=overwrite,
        )
        if not isinstance(annotated_output_path, Path):
            raise TypeError("annotated_output_path 必须是 pathlib.Path")
        if raw_output_path == annotated_output_path:
            raise ValueError("原始与绘制视频路径不能相同")
        self.raw_output_path = raw_output_path
        self.annotated_output_path = annotated_output_path

    def start(self) -> None:
        """校验成对路径并启动单输入、双输出 FFmpeg 进程。"""

        if self._closed:
            raise RuntimeError("H.264 编码器已关闭，不能重新启动")
        if self._started:
            raise RuntimeError("H.264 编码器已经启动")
        for output_path in (
            self.raw_output_path,
            self.annotated_output_path,
        ):
            if output_path.exists() and not self.overwrite:
                raise FileExistsError(f"输出文件已存在：{output_path}")
            output_path.parent.mkdir(parents=True, exist_ok=True)

        command = build_h264_pair_encoder_command(
            self.config,
            self.raw_output_path,
            self.annotated_output_path,
            overwrite=self.overwrite,
        )
        self._start_process(command)

    def write_pair(
        self,
        raw_image_bgr: np.ndarray,
        annotated_image_bgr: np.ndarray,
    ) -> None:
        """以同一帧 ID 写入原始与绘制画面，保持两路严格同步。"""

        self._validate_frame(raw_image_bgr)
        self._validate_frame(annotated_image_bgr)
        combined = np.concatenate(
            (raw_image_bgr, annotated_image_bgr),
            axis=1,
        )
        self._write_array(combined)

    def __enter__(self) -> H264VideoPairWriter:
        """启动双路编码器并返回当前实例。"""

        self.start()
        return self
