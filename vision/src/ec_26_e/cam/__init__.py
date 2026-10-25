"""公开 FFmpeg 相机采集模块的稳定契约与实现。"""

from ec_26_e.cam.contracts import (
    CameraCaptureError,
    CameraConfig,
    CameraFrame,
    CameraSource,
    CameraTimeoutError,
)
from ec_26_e.cam.ffmpeg_camera import FFmpegCamera

__all__ = [
    "CameraCaptureError",
    "CameraConfig",
    "CameraFrame",
    "CameraSource",
    "CameraTimeoutError",
    "FFmpegCamera",
]
