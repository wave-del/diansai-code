"""使用真实相机录制 cam 最新帧，并输出 H.264 MP4 诊断文件。

该脚本依赖 V4L2 相机和系统 FFmpeg，不属于默认单元测试。
独立运行：``poetry run python tests/external/cam/record_camera.py --help``
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np
import typer

# 文件运行时加入项目根目录，以复用 tests 辅助模块。
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ec_26_e.cam import (
    CameraCaptureError,
    CameraConfig,
    CameraTimeoutError,
    FFmpegCamera,
)
from ec_26_e.config import load_env_file
from tests.external.cam.helpers import H264VideoWriter

load_env_file(PROJECT_ROOT / ".env")

DEFAULT_DEVICE_PATH = Path("/dev/video0")
DEFAULT_OUTPUT_DIRECTORY = Path("outputs/cam")
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 30
DEFAULT_INPUT_FORMAT = "mjpeg"
DEFAULT_DURATION_SECONDS = 10.0
DEFAULT_FFMPEG_EXECUTABLE = "ffmpeg"

app = typer.Typer(
    add_completion=False,
    help="录制 FFmpegCamera 发布的最新 BGR 帧，并统计覆盖丢帧。",
)


@dataclass(frozen=True, slots=True)
class RecordingSummary:
    """描述一次相机录像诊断的关键指标。"""

    output_path: Path
    written_frames: int
    skipped_frames: int
    elapsed_seconds: float
    actual_read_fps: float
    interrupted: bool

    def to_dict(self) -> dict[str, object]:
        """转换为便于命令行输出的 JSON 兼容字典。"""

        return {
            "output_path": str(self.output_path),
            "written_frames": self.written_frames,
            "skipped_frames": self.skipped_frames,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "actual_read_fps": round(self.actual_read_fps, 3),
            "interrupted": self.interrupted,
        }


def _default_output_path() -> Path:
    """生成不会在模块导入或查看帮助时创建文件的时间戳路径。"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return DEFAULT_OUTPUT_DIRECTORY / f"camera_{timestamp}.mp4"


def record_camera(
    config: CameraConfig,
    output_path: Path | None = None,
    *,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
    overwrite: bool = False,
) -> RecordingSummary:
    """录制指定时长的最新帧，并返回帧率和覆盖丢帧统计。"""

    if not isinstance(config, CameraConfig):
        raise TypeError("config 必须是 CameraConfig")
    if isinstance(duration_seconds, bool) or not isinstance(
        duration_seconds, (int, float)
    ):
        raise TypeError("duration_seconds 必须是实数")
    if not np.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds 必须是有限正数")

    resolved_output = output_path or _default_output_path()
    written_frames = 0
    skipped_frames = 0
    last_sequence: int | None = None
    interrupted = False
    recording_start = 0.0
    recording_elapsed = 0.0

    with FFmpegCamera(config) as camera:
        with H264VideoWriter(
            config,
            resolved_output,
            overwrite=overwrite,
        ) as writer:
            recording_start = time.monotonic()
            deadline = recording_start + float(duration_seconds)

            try:
                while time.monotonic() < deadline:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        frame = camera.read_latest(
                            after_sequence=last_sequence,
                            timeout=min(
                                config.read_timeout_seconds,
                                remaining,
                            ),
                        )
                    except CameraTimeoutError:
                        if time.monotonic() >= deadline:
                            break
                        raise

                    if last_sequence is not None:
                        skipped_frames += max(
                            0,
                            frame.sequence_id - last_sequence - 1,
                        )
                    writer.write(frame.image_bgr)
                    written_frames += 1
                    last_sequence = frame.sequence_id
            except KeyboardInterrupt:
                interrupted = True
            recording_elapsed = time.monotonic() - recording_start

    elapsed = max(recording_elapsed, 1e-9)
    return RecordingSummary(
        output_path=resolved_output,
        written_frames=written_frames,
        skipped_frames=skipped_frames,
        elapsed_seconds=elapsed,
        actual_read_fps=written_frames / elapsed,
        interrupted=interrupted,
    )


@app.command()
def main(
    device_path: Path = typer.Option(
        DEFAULT_DEVICE_PATH,
        "--device",
        envvar="CAMERA_DEVICE_PATH",
        help="V4L2 相机设备路径；可由 .env 的 CAMERA_DEVICE_PATH 配置。",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output",
        help="MP4 输出路径；省略时在 outputs/cam 下生成时间戳文件。",
    ),
    duration_seconds: float = typer.Option(
        DEFAULT_DURATION_SECONDS,
        "--duration",
        min=0.1,
        help="录像时长，单位为秒。",
    ),
    width: int = typer.Option(
        DEFAULT_WIDTH,
        "--width",
        envvar="CAMERA_WIDTH",
        min=1,
        help="采集宽度；可由 .env 的 CAMERA_WIDTH 配置。",
    ),
    height: int = typer.Option(
        DEFAULT_HEIGHT,
        "--height",
        envvar="CAMERA_HEIGHT",
        min=1,
        help="采集高度；可由 .env 的 CAMERA_HEIGHT 配置。",
    ),
    fps: int = typer.Option(
        DEFAULT_FPS,
        min=1,
        help="采集与录像标称帧率。",
    ),
    input_format: str = typer.Option(
        DEFAULT_INPUT_FORMAT,
        help="相机输入格式，支持 mjpeg 或 yuyv422。",
    ),
    ffmpeg_executable: str = typer.Option(
        DEFAULT_FFMPEG_EXECUTABLE,
        "--ffmpeg",
        help="FFmpeg 可执行文件名或路径。",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite/--no-overwrite",
        help="是否允许覆盖显式指定的已有输出文件。",
    ),
) -> None:
    """构造强类型配置，执行录像并打印 JSON 诊断摘要。"""

    try:
        config = CameraConfig(
            device_path=device_path,
            width=width,
            height=height,
            fps=fps,
            input_format=input_format,
            ffmpeg_executable=ffmpeg_executable,
        )
        summary = record_camera(
            config,
            output_path,
            duration_seconds=duration_seconds,
            overwrite=overwrite,
        )
    except (
        CameraCaptureError,
        FileExistsError,
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as error:
        typer.echo(f"录像失败：{error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
