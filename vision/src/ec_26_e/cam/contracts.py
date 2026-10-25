"""定义相机采集模块的稳定公共契约与结构化帧数据。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


SUPPORTED_INPUT_FORMATS = frozenset({"mjpeg", "yuyv422"})


def _validate_positive_integer(name: str, value: int) -> None:
    """校验配置值是正整数，并拒绝布尔值冒充整数。"""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} 必须是整数")
    if value <= 0:
        raise ValueError(f"{name} 必须大于 0")


def _validate_positive_number(name: str, value: float) -> None:
    """校验配置值是有限正数。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 必须是实数")
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} 必须是有限正数")


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """描述 FFmpeg/V4L2 相机采集所需的强类型配置。"""

    device_path: Path = Path("/dev/video0")
    width: int = 1920
    height: int = 1080
    fps: int = 30
    input_format: str = "mjpeg"
    ffmpeg_executable: str = "ffmpeg"
    startup_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 2.0

    def __post_init__(self) -> None:
        """在访问设备前校验采集参数和超时边界。"""

        if not isinstance(self.device_path, Path):
            raise TypeError("device_path 必须是 pathlib.Path")
        if not str(self.device_path):
            raise ValueError("device_path 不能为空")
        _validate_positive_integer("width", self.width)
        _validate_positive_integer("height", self.height)
        _validate_positive_integer("fps", self.fps)
        if not isinstance(self.input_format, str):
            raise TypeError("input_format 必须是字符串")
        if self.input_format not in SUPPORTED_INPUT_FORMATS:
            supported = ", ".join(sorted(SUPPORTED_INPUT_FORMATS))
            raise ValueError(f"input_format 必须是以下值之一：{supported}")
        if not isinstance(self.ffmpeg_executable, str):
            raise TypeError("ffmpeg_executable 必须是字符串")
        if not self.ffmpeg_executable.strip():
            raise ValueError("ffmpeg_executable 不能为空")
        _validate_positive_number(
            "startup_timeout_seconds",
            self.startup_timeout_seconds,
        )
        _validate_positive_number(
            "read_timeout_seconds",
            self.read_timeout_seconds,
        )

    @property
    def frame_size_bytes(self) -> int:
        """返回单张 BGR24 帧在管道中的固定字节数。"""

        return self.width * self.height * 3


@dataclass(frozen=True, slots=True)
class CameraFrame:
    """描述一张由采集线程完整接收的只读 BGR 帧。"""

    sequence_id: int
    received_at_ns: int
    image_bgr: np.ndarray

    def __post_init__(self) -> None:
        """校验帧元数据、形状和像素类型，并锁定写权限。"""

        if isinstance(self.sequence_id, bool) or not isinstance(
            self.sequence_id, int
        ):
            raise TypeError("sequence_id 必须是整数")
        if self.sequence_id < 0:
            raise ValueError("sequence_id 不能小于 0")
        if isinstance(self.received_at_ns, bool) or not isinstance(
            self.received_at_ns, int
        ):
            raise TypeError("received_at_ns 必须是整数")
        if self.received_at_ns <= 0:
            raise ValueError("received_at_ns 必须大于 0")
        if not isinstance(self.image_bgr, np.ndarray):
            raise TypeError("image_bgr 必须是 NumPy 数组")
        if (
            self.image_bgr.ndim != 3
            or self.image_bgr.shape[2] != 3
            or self.image_bgr.size == 0
        ):
            raise ValueError("image_bgr 必须是非空的 H×W×3 图像")
        if self.image_bgr.dtype != np.uint8:
            raise ValueError("image_bgr 必须使用 uint8 像素类型")
        self.image_bgr.setflags(write=False)


class CameraCaptureError(RuntimeError):
    """表示设备访问、FFmpeg 进程或帧管道发生故障。"""


class CameraTimeoutError(CameraCaptureError, TimeoutError):
    """表示在限定时间内没有获得符合条件的相机帧。"""


@runtime_checkable
class CameraSource(Protocol):
    """约束可替换的最新帧相机实现及其显式生命周期。"""

    def start(self) -> None:
        """启动采集并等待首张完整帧。"""

    def read_latest(
        self,
        after_sequence: int | None = None,
        timeout: float | None = None,
    ) -> CameraFrame:
        """返回最新帧，必要时等待指定序号之后的新帧。"""

    def close(self) -> None:
        """停止采集并释放进程、管道和后台线程。"""
