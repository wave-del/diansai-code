"""以录制视频验证 Task1 视觉、规划和串口执行全链路。

该外部测试将 MP4 帧适配为正式 ``CameraSource``，复用 Task1 的
Resize、YOLO+传统视觉精修、双层稳定审查和规划逻辑，再由正式 ``MovePlanActuator``
生成并发送运动串口报文。默认 ``--dry-run`` 不打开串口、GPIO 或电机；
显式传入 ``--execute`` 才执行真实硬件动作。独立运行：
``poetry run python tests/external/task1_video_serial.py --dry-run``。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import time
import cv2
import typer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ec_26_e.cam import CameraCaptureError, CameraFrame
from ec_26_e.config import (
    DEFAULT_TASK1_CONFIG_PATH,
    Task1Config,
    load_env_file,
    load_task1_config,
)
from ec_26_e.motion import (
    DEFAULT_MOTION_BAUDRATE,
    DEFAULT_MOTION_DEVICE_PATH,
    DEFAULT_MOTION_READ_TIMEOUT_SECONDS,
    DEFAULT_MOTION_REPLY_TIMEOUT_SECONDS,
    DEFAULT_MOTION_WRITE_TIMEOUT_SECONDS,
)
from ec_26_e.control import MovePlan
from ec_26_e.tasks import TaskContext
from ec_26_e.tasks.task1.application import execute_task1
from tests.external.motion.helpers import create_serial_config, emit_summary
from tests.external.motion.move_plan_execution import run_move_plan


DEFAULT_VIDEO_PATH = (
    PROJECT_ROOT
    / "outputs/cam/detection_20260731_065326_138775_raw.mp4"
)

load_env_file(PROJECT_ROOT / ".env")

app = typer.Typer(
    add_completion=False,
    help="将录制视频接入 Task1，并预览或发送最终运动串口报文。",
)


class VideoCamera:
    """把本地视频逐帧发布为满足 Task1 契约的相机源。"""

    def __init__(self, video_path: Path) -> None:
        """保存视频路径；构造阶段不读取文件。"""

        if not isinstance(video_path, Path):
            raise TypeError("video_path 必须是 pathlib.Path")
        self._video_path = video_path
        self._capture: cv2.VideoCapture | None = None
        self._sequence_id = 0
        self._started = False
        self._closed = False
        self._frame_interval_ns = 0
        self._started_at_ns = 0

    def start(self) -> None:
        """打开视频并读取元数据，但不预取以免跳过首帧。"""

        if self._closed:
            raise CameraCaptureError("视频相机已关闭，不能重新启动")
        if self._started:
            raise CameraCaptureError("视频相机已经启动")
        if not self._video_path.is_file():
            raise CameraCaptureError(f"测试视频不存在：{self._video_path}")
        capture = cv2.VideoCapture(str(self._video_path))
        if not capture.isOpened():
            capture.release()
            raise CameraCaptureError(f"无法打开测试视频：{self._video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not fps > 0.0:
            capture.release()
            raise CameraCaptureError("测试视频未提供有效 FPS")
        self._capture = capture
        self._frame_interval_ns = max(1, round(1_000_000_000 / fps))
        self._started_at_ns = time.monotonic_ns()
        self._started = True

    def read_latest(
        self,
        after_sequence: int | None = None,
        timeout: float | None = None,
    ) -> CameraFrame:
        """同步读取下一视频帧，并以视频时间轴提供稳定审查时间戳。"""

        del timeout
        if not self._started or self._capture is None:
            raise CameraCaptureError("视频相机尚未启动")
        if self._closed:
            raise CameraCaptureError("视频相机已关闭")
        if after_sequence is not None and after_sequence < -1:
            raise ValueError("after_sequence 不能小于 -1")
        ok, image_bgr = self._capture.read()
        if not ok or image_bgr is None:
            raise CameraCaptureError("测试视频已读完，未获得满足稳定条件的画面")
        frame = CameraFrame(
            sequence_id=self._sequence_id,
            received_at_ns=(
                self._started_at_ns
                + self._sequence_id * self._frame_interval_ns
            ),
            image_bgr=image_bgr,
        )
        self._sequence_id += 1
        return frame

    def close(self) -> None:
        """幂等释放 OpenCV 视频解码器。"""

        if self._closed:
            return
        self._closed = True
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class VideoPlanActuator:
    """把 Task1 规划结果交给真实执行器或其无硬件预览实现。"""

    def __init__(self, config: Task1Config, *, execute: bool) -> None:
        """保存执行配置和可供命令行打印的最终摘要。"""

        self._config = config
        self._execute = execute
        self.summary: dict[str, object] | None = None

    def execute(self, plan: MovePlan) -> None:
        """使用正式 MovePlanActuator 执行计划，并保留串口写入摘要。"""

        self.summary = run_move_plan(
            self._config.actuation,
            plan,
            execute=self._execute,
        )


def run_video_task1(
    *,
    video_path: Path,
    task_config_path: Path,
    device_path: Path,
    baudrate: int,
    read_timeout_seconds: float,
    write_timeout_seconds: float,
    reply_timeout_seconds: float,
    execute: bool,
) -> dict[str, object]:
    """运行视频到 Task1 再到串口的完整链路，并返回结构化摘要。"""

    if not isinstance(execute, bool):
        raise TypeError("execute 必须是 bool")
    config = load_task1_config(task_config_path)
    serial_config = create_serial_config(
        device_path,
        baudrate,
        read_timeout_seconds,
        write_timeout_seconds,
        reply_timeout_seconds,
    )
    config = replace(
        config,
        actuation=replace(config.actuation, serial=serial_config),
    )
    camera = VideoCamera(video_path)
    actuator = VideoPlanActuator(config, execute=execute)
    execute_task1(
        config,
        TaskContext(draw_enabled=False),
        camera_factory=lambda unused: camera,
        actuator_factory=lambda unused: actuator,
    )
    if actuator.summary is None:
        raise RuntimeError("Task1 未生成运动执行摘要")
    return {
        "operation": "task1-video-serial",
        "video_path": str(video_path),
        "task_config_path": str(task_config_path),
        **actuator.summary,
    }


@app.command()
def main(
    video_path: Path = typer.Option(
        DEFAULT_VIDEO_PATH,
        "--video",
        help="送入正式视觉检测链路的 MP4 视频。",
    ),
    task_config_path: Path = typer.Option(
        PROJECT_ROOT / DEFAULT_TASK1_CONFIG_PATH,
        "--task-config",
        help="Task1 的视觉、规划和执行配置 JSON。",
    ),
    device_path: Path = typer.Option(
        DEFAULT_MOTION_DEVICE_PATH,
        "--device",
        envvar="MOTION_SERIAL_DEVICE",
        help="Servo42C 串口设备；可由 MOTION_SERIAL_DEVICE 配置。",
    ),
    baudrate: int = typer.Option(
        DEFAULT_MOTION_BAUDRATE,
        "--baudrate",
        envvar="MOTION_SERIAL_BAUDRATE",
        min=1,
        help="串口波特率。",
    ),
    read_timeout_seconds: float = typer.Option(
        DEFAULT_MOTION_READ_TIMEOUT_SECONDS,
        "--read-timeout",
        envvar="MOTION_SERIAL_READ_TIMEOUT_SECONDS",
        min=0.001,
        help="单次串口读取超时，单位秒。",
    ),
    write_timeout_seconds: float = typer.Option(
        DEFAULT_MOTION_WRITE_TIMEOUT_SECONDS,
        "--write-timeout",
        envvar="MOTION_SERIAL_WRITE_TIMEOUT_SECONDS",
        min=0.001,
        help="单次串口写入超时，单位秒。",
    ),
    reply_timeout_seconds: float = typer.Option(
        DEFAULT_MOTION_REPLY_TIMEOUT_SECONDS,
        "--reply-timeout",
        envvar="MOTION_SERIAL_REPLY_TIMEOUT_SECONDS",
        min=0.001,
        help="串口回包超时，单位秒。",
    ),
    execute: bool = typer.Option(
        False,
        "--execute/--dry-run",
        help="默认 dry-run 只输出待发送报文；--execute 才控制真实硬件。",
    ),
) -> None:
    """收集 CLI 参数并运行视频驱动的 Task1 外部测试。"""

    try:
        emit_summary(
            run_video_task1(
                video_path=video_path,
                task_config_path=task_config_path,
                device_path=device_path,
                baudrate=baudrate,
                read_timeout_seconds=read_timeout_seconds,
                write_timeout_seconds=write_timeout_seconds,
                reply_timeout_seconds=reply_timeout_seconds,
                execute=execute,
            )
        )
    except KeyboardInterrupt:
        typer.echo("Task1 视频测试已由用户中断", err=True)
        raise typer.Exit(code=130) from None
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"Task1 视频测试失败：{error}", err=True)
        raise typer.Exit(code=1) from error


if __name__ == "__main__":
    app()
