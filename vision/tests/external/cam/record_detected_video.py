"""实时采集相机画面、执行视觉检测并保存绘制后的 H.264 视频。

该脚本依赖真实 V4L2 相机、v4l2-ctl 和 FFmpeg，不属于默认单元测试。
独立运行：
``poetry run python tests/external/cam/record_detected_video.py --help``
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import threading
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
    FFmpegCamera,
)
from ec_26_e.config import (
    TASK_CONFIG_PATH_ENV,
    TASK_NAME_ENV,
    Task1Config,
    Task2Config,
    load_runtime_config,
)
from ec_26_e.draw import (
    FramePreview,
    OpenCVPreview,
    draw_detection_failure,
    draw_live_detection,
)
from ec_26_e.vision import (
    Detect1Config,
    Resize,
    ResizeConfig,
    create_production_detector,
)
from ec_26_e.tasks.registry import get_task_registration
from ec_26_e.tasks.execution_lock import TaskExecutionLock
from tests.external.cam.helpers import (
    CameraDeviceInfo,
    H264VideoPairWriter,
    query_camera_info,
)

DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_OUTPUT_DIRECTORY = Path("outputs/cam")
DEFAULT_V4L2_CTL_EXECUTABLE = "v4l2-ctl"
DEFAULT_DISPLAY_NAME = ":0"
PREVIEW_WINDOW_NAME = "EC 26 E - Detection Preview"
PREVIEW_WINDOW_WIDTH = 1280
PREVIEW_WINDOW_HEIGHT = 720

app = typer.Typer(
    add_completion=False,
    help=(
        "读取正式任务的 .env 和 JSON 配置，实时检测多边形纸片并保存绘制视频。"
    ),
)


@dataclass(frozen=True, slots=True)
class DetectionRuntimeConfig:
    """保存与正式任务同源的视觉诊断配置及来源。"""

    task_name: str
    env_path: Path
    task_config_path: Path
    task_config: Task1Config | Task2Config
    camera: CameraConfig
    resize: ResizeConfig
    detect1: Detect1Config


def load_detection_runtime_config(
    env_path: Path = DEFAULT_ENV_PATH,
    *,
    environ: Mapping[str, str] | None = None,
    task_name: str | None = None,
    task_config_path: Path | None = None,
) -> DetectionRuntimeConfig:
    """通过正式任务注册表加载配置，并接受显式任务诊断覆盖。"""

    if task_name is None and task_config_path is not None:
        raise ValueError("显式 --task-config 必须同时提供 --task")
    merged_environ = dict(os.environ if environ is None else environ)
    if task_name is not None:
        selected = task_name.strip()
        if not selected:
            raise ValueError("显式任务名称不能为空")
        merged_environ[TASK_NAME_ENV] = selected
        selected_path = (
            task_config_path
            if task_config_path is not None
            else PROJECT_ROOT / "configs/tasks" / f"{selected}.json"
        )
        merged_environ[TASK_CONFIG_PATH_ENV] = str(selected_path)
    runtime = load_runtime_config(env_path, environ=merged_environ)
    registration = get_task_registration(runtime.task_name)
    task_config = registration.load(runtime)
    if not isinstance(task_config, (Task1Config, Task2Config)):
        raise TypeError(
            "实时视觉脚本仅支持包含 camera、resize 和 detect1 的拼图任务"
        )
    return DetectionRuntimeConfig(
        task_name=runtime.task_name,
        env_path=env_path,
        task_config_path=runtime.task_config_path,
        task_config=task_config,
        camera=task_config.camera,
        resize=task_config.resize,
        detect1=task_config.detect1,
    )


def apply_detection_cli_overrides(
    config: DetectionRuntimeConfig,
    *,
    device_path: Path | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
    input_format: str | None = None,
    ffmpeg_executable: str | None = None,
    workspace_detect_width: int | None = None,
    workspace_detect_height: int | None = None,
    a4_resize_width: int | None = None,
    a4_resize_height: int | None = None,
    workspace_erode_ratio: float | None = None,
    min_workspace_ratio: float | None = None,
    min_piece_area_ratio: float | None = None,
    max_piece_area_ratio: float | None = None,
    polygon_epsilon_ratio: float | None = None,
    min_vertices: int | None = None,
    max_vertices: int | None = None,
    max_pieces: int | None = None,
    piece_mode: int | None = None,
) -> DetectionRuntimeConfig:
    """仅把用户显式提供的诊断参数覆盖到正式任务配置。"""

    camera_updates = {
        name: value
        for name, value in (
            ("device_path", device_path),
            ("width", width),
            ("height", height),
            ("fps", fps),
            ("input_format", input_format),
            ("ffmpeg_executable", ffmpeg_executable),
        )
        if value is not None
    }
    resize_updates = {
        name: value
        for name, value in (
            ("detect_width", workspace_detect_width),
            ("detect_height", workspace_detect_height),
            ("target_width", a4_resize_width),
            ("target_height", a4_resize_height),
            ("min_workspace_ratio", min_workspace_ratio),
        )
        if value is not None
    }
    detect1_updates = {
        name: value
        for name, value in (
            ("workspace_erode_ratio", workspace_erode_ratio),
            ("min_piece_area_ratio", min_piece_area_ratio),
            ("max_piece_area_ratio", max_piece_area_ratio),
            ("polygon_epsilon_ratio", polygon_epsilon_ratio),
            ("min_vertices", min_vertices),
            ("max_vertices", max_vertices),
            ("max_pieces", max_pieces),
            ("piece_mode", piece_mode),
        )
        if value is not None
    }
    camera = replace(config.camera, **camera_updates)
    resize = replace(config.resize, **resize_updates)
    detect1 = replace(config.detect1, **detect1_updates)
    task_config = replace(
        config.task_config,
        camera=camera,
        resize=resize,
        detect1=detect1,
    )
    return replace(
        config,
        task_config=task_config,
        camera=camera,
        resize=resize,
        detect1=detect1,
    )


def effective_config_summary(
    config: DetectionRuntimeConfig,
) -> dict[str, object]:
    """生成便于确认正式任务对齐状态的非敏感配置摘要。"""

    return {
        "task": config.task_name,
        "env": str(config.env_path),
        "task_json": str(config.task_config_path),
        "camera": repr(config.camera),
        "resize": repr(config.resize),
        "detect1": repr(config.detect1),
    }


@dataclass(frozen=True, slots=True)
class DetectionRecordingSummary:
    """描述实时视觉检测录像的设备信息和处理指标。"""

    camera_info: CameraDeviceInfo
    raw_output_path: Path | None
    annotated_output_path: Path | None
    processed_frames: int
    successful_detections: int
    resize_failures: int
    detect1_failures: int
    detected_pieces: int
    skipped_frames: int
    recorded_frames: int
    recorder_overwritten_frames: int
    elapsed_seconds: float
    processing_fps: float
    preview_enabled: bool
    recording_enabled: bool
    stopped_by_user: bool
    interrupted: bool

    def to_dict(self) -> dict[str, object]:
        """转换为适合命令行输出的 JSON 兼容字典。"""

        return {
            "camera_info": self.camera_info.to_dict(),
            "raw_output_path": (
                None
                if self.raw_output_path is None
                else str(self.raw_output_path)
            ),
            "annotated_output_path": (
                None
                if self.annotated_output_path is None
                else str(self.annotated_output_path)
            ),
            "processed_frames": self.processed_frames,
            "successful_detections": self.successful_detections,
            "resize_failures": self.resize_failures,
            "detect1_failures": self.detect1_failures,
            "failed_detections": (
                self.resize_failures + self.detect1_failures
            ),
            "detected_pieces": self.detected_pieces,
            "skipped_frames": self.skipped_frames,
            "recorded_frames": self.recorded_frames,
            "recorder_overwritten_frames": (
                self.recorder_overwritten_frames
            ),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "processing_fps": round(self.processing_fps, 3),
            "preview_enabled": self.preview_enabled,
            "recording_enabled": self.recording_enabled,
            "stopped_by_user": self.stopped_by_user,
            "interrupted": self.interrupted,
        }


class AsyncPairRecorder:
    """在后台编码最新一对原始与绘制帧，避免阻塞实时预览。"""

    def __init__(self, writer: H264VideoPairWriter) -> None:
        """保存双路写入器并初始化单槽覆盖状态。"""

        self.writer = writer
        self.recorded_frames = 0
        self.overwritten_frames = 0
        self._condition = threading.Condition()
        self._pending: tuple[np.ndarray, np.ndarray] | None = None
        self._thread: threading.Thread | None = None
        self._failure: Exception | None = None
        self._started = False
        self._stopping = False
        self._closed = False

    def start(self) -> None:
        """启动 FFmpeg 和后台编码线程。"""

        with self._condition:
            if self._closed:
                raise RuntimeError("异步录像器已关闭，不能重新启动")
            if self._started:
                raise RuntimeError("异步录像器已经启动")
        self.writer.start()
        with self._condition:
            self._started = True
            self._thread = threading.Thread(
                target=self._run,
                name="detection-pair-recorder",
                daemon=True,
            )
            self._thread.start()

    def submit(
        self,
        raw_image_bgr: np.ndarray,
        annotated_image_bgr: np.ndarray,
    ) -> None:
        """非阻塞提交最新帧对，并覆盖尚未开始编码的旧帧对。"""

        with self._condition:
            if not self._started:
                raise RuntimeError("异步录像器尚未启动")
            if self._closed or self._stopping:
                raise RuntimeError("异步录像器已经停止接收帧")
            self._raise_failure()
            if self._pending is not None:
                self.overwritten_frames += 1
            self._pending = (raw_image_bgr, annotated_image_bgr)
            self._condition.notify_all()

    def close(self) -> None:
        """写完最后一对待处理帧，回收线程并完成双路 MP4 封装。"""

        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._stopping = True
            thread = self._thread
            self._condition.notify_all()

        if thread is not None:
            thread.join(timeout=30.0)
            if thread.is_alive():
                self.writer.close()
                thread.join(timeout=2.0)
                raise RuntimeError("等待后台双路录像线程结束超时")

        writer_error: Exception | None = None
        try:
            self.writer.close()
        except Exception as error:
            writer_error = error
        if writer_error is not None:
            raise writer_error
        with self._condition:
            self._raise_failure()

    def _run(self) -> None:
        """持续取走单槽中的最新帧对并执行同步编码。"""

        try:
            while True:
                with self._condition:
                    while self._pending is None and not self._stopping:
                        self._condition.wait()
                    if self._pending is None and self._stopping:
                        return
                    pair = self._pending
                    self._pending = None
                if pair is None:
                    continue
                self.writer.write_pair(pair[0], pair[1])
                with self._condition:
                    self.recorded_frames += 1
        except Exception as error:
            with self._condition:
                self._failure = error
                self._pending = None
                self._stopping = True
                self._condition.notify_all()

    def _raise_failure(self) -> None:
        """在调用线程重新抛出后台编码错误。"""

        if self._failure is not None:
            raise RuntimeError(
                f"后台双路录像失败：{self._failure}"
            ) from self._failure


def resolve_output_paths(
    output_prefix: Path | None,
) -> tuple[Path, Path]:
    """由共享前缀生成同一会话的原始与绘制视频路径。"""

    if output_prefix is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        prefix = DEFAULT_OUTPUT_DIRECTORY / f"detection_{timestamp}"
    else:
        if not isinstance(output_prefix, Path):
            raise TypeError("output_prefix 必须是 pathlib.Path 或 None")
        if output_prefix.suffix:
            raise ValueError("output_prefix 不应包含文件扩展名")
        prefix = output_prefix
    return (
        prefix.with_name(f"{prefix.name}_raw.mp4"),
        prefix.with_name(f"{prefix.name}_annotated.mp4"),
    )


def record_detection_video(
    camera_config: CameraConfig,
    resize_config: ResizeConfig,
    detect1_config: Detect1Config,
    camera_info: CameraDeviceInfo,
    output_prefix: Path | None = None,
    *,
    overwrite: bool = False,
    preview: FramePreview | None = None,
    record_enabled: bool = True,
) -> DetectionRecordingSummary:
    """持续采集检测，并按需预览或同步保存双路视频。"""

    if not isinstance(camera_config, CameraConfig):
        raise TypeError("camera_config 必须是 CameraConfig")
    if not isinstance(resize_config, ResizeConfig):
        raise TypeError("resize_config 必须是 ResizeConfig")
    if not isinstance(detect1_config, Detect1Config):
        raise TypeError("detect1_config 必须是 Detect1Config")
    if not isinstance(camera_info, CameraDeviceInfo):
        raise TypeError("camera_info 必须是 CameraDeviceInfo")
    if not isinstance(record_enabled, bool):
        raise TypeError("record_enabled 必须是 bool")
    if not record_enabled and preview is None:
        raise ValueError("关闭录像时必须启用预览，避免无可见输出的无限运行")

    if record_enabled:
        raw_output_path, annotated_output_path = resolve_output_paths(
            output_prefix
        )
    else:
        raw_output_path = None
        annotated_output_path = None
    resizer = Resize(resize_config)
    detector = create_production_detector(detect1_config)
    processed_frames = 0
    successful_detections = 0
    resize_failures = 0
    detect1_failures = 0
    detected_pieces = 0
    skipped_frames = 0
    last_sequence: int | None = None
    interrupted = False
    stopped_by_user = False
    processing_start = 0.0
    processing_elapsed = 0.0
    recorded_frames = 0
    recorder_overwritten_frames = 0

    try:
        if preview is not None:
            preview.start()
        with FFmpegCamera(camera_config) as camera:
            recorder: AsyncPairRecorder | None = None
            try:
                if record_enabled:
                    if raw_output_path is None or annotated_output_path is None:
                        raise RuntimeError("录像输出路径未初始化")
                    writer = H264VideoPairWriter(
                        camera_config,
                        raw_output_path,
                        annotated_output_path,
                        overwrite=overwrite,
                    )
                    recorder = AsyncPairRecorder(writer)
                    recorder.start()
                processing_start = time.monotonic()
                try:
                    while True:
                        frame = camera.read_latest(
                            after_sequence=last_sequence,
                            timeout=camera_config.read_timeout_seconds,
                        )

                        if last_sequence is not None:
                            skipped_frames += max(
                                0,
                                frame.sequence_id - last_sequence - 1,
                            )
                        try:
                            resize_result = resizer.resize(frame.image_bgr)
                        except ValueError:
                            annotated = draw_detection_failure(
                                frame.image_bgr,
                                sequence_id=frame.sequence_id,
                                stage="workspace",
                            )
                            resize_failures += 1
                        else:
                            try:
                                stable_result = detector.detect(
                                    resize_result.image_bgr
                                )
                            except ValueError:
                                annotated = draw_detection_failure(
                                    frame.image_bgr,
                                    sequence_id=frame.sequence_id,
                                    stage="detect1",
                                    resize_result=resize_result,
                                )
                                detect1_failures += 1
                            else:
                                # 稳定结果尚未形成时仍显示单帧精修边，便于实时判断贴边精度。
                                single_result = detector.last_single_result
                                display_result = (
                                    stable_result
                                    if stable_result.pieces
                                    else single_result
                                )
                                if display_result is None:
                                    raise RuntimeError(
                                        "统一视觉未生成单帧诊断结果"
                                    )
                                annotated = draw_live_detection(
                                    frame.image_bgr,
                                    resize_result,
                                    display_result,
                                    sequence_id=frame.sequence_id,
                                )
                                successful_detections += 1
                                detected_pieces += len(
                                    display_result.pieces
                                )

                        stop_requested = (
                            preview.show_and_should_stop(annotated)
                            if preview is not None
                            else False
                        )
                        if recorder is not None:
                            recorder.submit(frame.image_bgr, annotated)
                        processed_frames += 1
                        last_sequence = frame.sequence_id
                        if stop_requested:
                            stopped_by_user = True
                            break
                except KeyboardInterrupt:
                    interrupted = True
            finally:
                processing_elapsed = time.monotonic() - processing_start
                if recorder is not None:
                    recorder.close()
                    recorded_frames = recorder.recorded_frames
                    recorder_overwritten_frames = (
                        recorder.overwritten_frames
                    )
    finally:
        detector.close()
        if preview is not None:
            preview.close()

    elapsed = max(processing_elapsed, 1e-9)
    return DetectionRecordingSummary(
        camera_info=camera_info,
        raw_output_path=raw_output_path,
        annotated_output_path=annotated_output_path,
        processed_frames=processed_frames,
        successful_detections=successful_detections,
        resize_failures=resize_failures,
        detect1_failures=detect1_failures,
        detected_pieces=detected_pieces,
        skipped_frames=skipped_frames,
        recorded_frames=recorded_frames,
        recorder_overwritten_frames=recorder_overwritten_frames,
        elapsed_seconds=elapsed,
        processing_fps=processed_frames / elapsed,
        preview_enabled=preview is not None,
        recording_enabled=record_enabled,
        stopped_by_user=stopped_by_user,
        interrupted=interrupted,
    )


@app.command()
def main(
    env_path: Path = typer.Option(
        DEFAULT_ENV_PATH,
        "--env",
        help=(
            "正式任务 .env 路径；其中 EC_TASK 和 EC_TASK_CONFIG 选择任务配置。"
        ),
    ),
    task_name: str | None = typer.Option(
        None,
        "--task",
        help="临时选择 task1 或 task2；省略时使用 .env 的 EC_TASK。",
    ),
    task_config_path: Path | None = typer.Option(
        None,
        "--task-config",
        help="配合 --task 临时指定对应 JSON；省略时使用仓库同名 JSON。",
    ),
    device_path: Path | None = typer.Option(
        None,
        "--device",
        help="临时覆盖正式任务的 V4L2 相机设备路径。",
    ),
    output_prefix: Path | None = typer.Option(
        None,
        "--output-prefix",
        help=(
            "双路视频共享前缀；省略时在 outputs/cam 下生成时间戳。"
        ),
    ),
    width: int | None = typer.Option(
        None,
        "--width",
        min=1,
        help="临时覆盖正式任务的相机采集宽度。",
    ),
    height: int | None = typer.Option(
        None,
        "--height",
        min=1,
        help="临时覆盖正式任务的相机采集高度。",
    ),
    fps: int | None = typer.Option(
        None,
        min=1,
        help="临时覆盖正式任务的相机采集与输出视频标称帧率。",
    ),
    input_format: str | None = typer.Option(
        None,
        help="临时覆盖正式任务的相机输入格式。",
    ),
    ffmpeg_executable: str | None = typer.Option(
        None,
        "--ffmpeg",
        help="临时覆盖正式任务的 FFmpeg 可执行文件名或路径。",
    ),
    v4l2_ctl_executable: str = typer.Option(
        DEFAULT_V4L2_CTL_EXECUTABLE,
        "--v4l2-ctl",
        help="v4l2-ctl 可执行文件名或路径。",
    ),
    preview_enabled: bool = typer.Option(
        True,
        "--preview/--no-preview",
        help="是否在本机显示器实时显示检测绘制结果。",
    ),
    display_name: str = typer.Option(
        DEFAULT_DISPLAY_NAME,
        "--display",
        help="X11 显示会话；当前本机显示器默认为 :0。",
    ),
    fullscreen: bool = typer.Option(
        True,
        "--fullscreen/--windowed",
        help="预览使用全屏或 1280×720 窗口；Esc 可结束录像。",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite/--no-overwrite",
        help="是否允许覆盖显式指定的已有输出文件。",
    ),
    record_enabled: bool = typer.Option(
        True,
        "--record/--no-record",
        help="是否同步保存原始和标注视频；快速现场预览应关闭录像。",
    ),
    workspace_detect_width: int | None = typer.Option(
        None,
        "--workspace-detect-width",
        min=1,
        help="临时覆盖正式任务的蓝色工作区定位宽度。",
    ),
    workspace_detect_height: int | None = typer.Option(
        None,
        "--workspace-detect-height",
        min=1,
        help="临时覆盖正式任务的蓝色工作区定位高度。",
    ),
    a4_resize_width: int | None = typer.Option(
        None,
        "--a4-resize-width",
        min=1,
        help="临时覆盖正式任务的透视校正后横向 A4 宽度。",
    ),
    a4_resize_height: int | None = typer.Option(
        None,
        "--a4-resize-height",
        min=1,
        help="临时覆盖正式任务的透视校正后横向 A4 高度。",
    ),
    workspace_erode_ratio: float | None = typer.Option(
        None,
        min=0.0,
        max=1.0,
        help="临时覆盖任务 JSON 的工作区边缘内缩比例。",
    ),
    min_workspace_ratio: float | None = typer.Option(
        None,
        min=0.000001,
        max=1.0,
        help="临时覆盖任务 JSON 的蓝色工作区最小占比。",
    ),
    min_piece_area_ratio: float | None = typer.Option(
        None,
        min=0.000001,
        max=1.0,
        help="临时覆盖任务 JSON 的纸片最小面积比例。",
    ),
    max_piece_area_ratio: float | None = typer.Option(
        None,
        min=0.000001,
        max=1.0,
        help="临时覆盖任务 JSON 的纸片最大面积比例。",
    ),
    polygon_epsilon_ratio: float | None = typer.Option(
        None,
        min=0.000001,
        max=1.0,
        help="临时覆盖任务 JSON 的轮廓多边形逼近精度。",
    ),
    min_vertices: int | None = typer.Option(
        None,
        min=3,
        help="临时覆盖任务 JSON 接受的最小顶点数。",
    ),
    max_vertices: int | None = typer.Option(
        None,
        min=3,
        help="临时覆盖任务 JSON 接受的最大顶点数。",
    ),
    max_pieces: int | None = typer.Option(
        None,
        min=1,
        help="临时覆盖任务 JSON 单帧最多返回的纸片数量。",
    ),
    piece_mode: int | None = typer.Option(
        None,
        "--piece-mode",
        min=1,
        max=2,
        help=(
            "临时覆盖任务纸片模式：1=纯色白纸时序防跳，"
            "2=带图案纸片暗色补偿。"
        ),
    ),
) -> None:
    """加载正式任务配置，执行实时检测录像并打印诊断摘要。"""

    execution_lock = TaskExecutionLock()
    try:
        execution_lock.acquire()
        if not record_enabled and not preview_enabled:
            raise ValueError("--no-record 不能与 --no-preview 同时使用")
        runtime_config = load_detection_runtime_config(
            env_path,
            task_name=task_name,
            task_config_path=task_config_path,
        )
        runtime_config = apply_detection_cli_overrides(
            runtime_config,
            device_path=device_path,
            width=width,
            height=height,
            fps=fps,
            input_format=input_format,
            ffmpeg_executable=ffmpeg_executable,
            workspace_detect_width=workspace_detect_width,
            workspace_detect_height=workspace_detect_height,
            a4_resize_width=a4_resize_width,
            a4_resize_height=a4_resize_height,
            workspace_erode_ratio=workspace_erode_ratio,
            min_workspace_ratio=min_workspace_ratio,
            min_piece_area_ratio=min_piece_area_ratio,
            max_piece_area_ratio=max_piece_area_ratio,
            polygon_epsilon_ratio=polygon_epsilon_ratio,
            min_vertices=min_vertices,
            max_vertices=max_vertices,
            max_pieces=max_pieces,
            piece_mode=piece_mode,
        )
        camera_config = runtime_config.camera
        resize_config = runtime_config.resize
        detect1_config = runtime_config.detect1
        camera_info = query_camera_info(
            camera_config.device_path,
            v4l2_ctl_executable=v4l2_ctl_executable,
        )
        preview = (
            OpenCVPreview(
                display_name=display_name,
                fullscreen=fullscreen,
                window_name=PREVIEW_WINDOW_NAME,
                window_size=(
                    (PREVIEW_WINDOW_WIDTH, PREVIEW_WINDOW_HEIGHT)
                    if not fullscreen
                    else None
                ),
            )
            if preview_enabled
            else None
        )
        typer.echo(
            json.dumps(
                {
                    "effective_config": effective_config_summary(
                        runtime_config
                    ),
                    "camera_info": camera_info.to_dict(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        summary = record_detection_video(
            camera_config,
            resize_config,
            detect1_config,
            camera_info,
            output_prefix,
            overwrite=overwrite,
            preview=preview,
            record_enabled=record_enabled,
        )
    except (
        CameraCaptureError,
        FileExistsError,
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as error:
        typer.echo(f"实时检测录像失败：{error}", err=True)
        raise typer.Exit(code=1) from error
    finally:
        execution_lock.release()

    typer.echo(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
