"""录制真实 Task1 全程，并保存与执行动作对齐的控制时间线。

此脚本是显式外部硬件诊断入口。它复用正式的视觉、规划与
``MovePlanActuator``，但不会在稳定识别结束时关闭相机；相机持续录制到
全部执行与安全清理完成。默认 ``--execute`` 为 false，避免误动作。
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np
import typer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ec_26_e.actuation import ActuationConfig
from ec_26_e.cam import CameraConfig, CameraSource, FFmpegCamera
from ec_26_e.config import load_env_file, load_task1_config
from ec_26_e.control import MovePlan
from ec_26_e.tasks.task1.actuation import MovePlanActuator
from ec_26_e.tasks.task1.application import _default_actuator_factory
from ec_26_e.tasks.task1.runner import run_task1
from ec_26_e.vision import Resize
from ec_26_e.vision import create_production_detector
from tests.external.cam.helpers import H264VideoWriter


load_env_file(PROJECT_ROOT / ".env")
app = typer.Typer(add_completion=False)


class Timeline:
    """以同一单调时钟写入视频帧和控制动作。"""

    def __init__(self) -> None:
        """创建线程安全的空时间线。"""

        self.origin_ns = time.monotonic_ns()
        self._events: list[dict[str, object]] = []
        self._lock = threading.Lock()
        self.latest_frame_sequence: int | None = None

    def add(self, event: str, **details: object) -> None:
        """把事件和最近录像帧序号写入时间线。"""

        item: dict[str, object] = {
            "t_seconds": round((time.monotonic_ns() - self.origin_ns) / 1e9, 6),
            "event": event,
            "latest_recorded_frame_sequence": self.latest_frame_sequence,
            **details,
        }
        with self._lock:
            self._events.append(item)

    def document(self) -> dict[str, object]:
        """返回可序列化的时间基准与事件快照。"""

        with self._lock:
            return {
                "timebase": "monotonic seconds from recording start",
                "events": list(self._events),
            }


class RecordingCamera:
    """把正式相机源包装为持续录像源；``close`` 延迟到整个任务结束。"""

    def __init__(self, config: CameraConfig, video_path: Path, timeline: Timeline) -> None:
        """构造正式相机、录像器和共享时间线。"""

        self._camera: CameraSource = FFmpegCamera(config)
        self._writer = H264VideoWriter(config, video_path)
        self._timeline = timeline
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._last_sequence: int | None = None
        self._recording_error: BaseException | None = None

    def start(self) -> None:
        """启动相机、编码器和后台持续录像线程。"""

        if self._started:
            return
        self._camera.start()
        self._writer.start()
        self._started = True
        self._thread = threading.Thread(
            target=self._record,
            name="task1-video-recorder",
            daemon=True,
        )
        self._thread.start()
        self._timeline.add("camera_recording_started")

    def read_latest(self, after_sequence: int | None = None, timeout: float | None = None):
        """向正式 runner 转发最新帧读取并传播录像失败。"""

        if self._recording_error is not None:
            raise RuntimeError(f"录像线程失败：{self._recording_error}") from self._recording_error
        return self._camera.read_latest(after_sequence=after_sequence, timeout=timeout)

    def close(self) -> None:
        """兼容正式 runner：识别完成时保持相机，不结束录像。"""

    def finish(self) -> None:
        """停止录像线程并释放编码器和真实相机。"""

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=20.0)
            if self._thread.is_alive():
                raise RuntimeError("录像线程未能停止")
        try:
            self._writer.close()
        finally:
            self._camera.close()
        if self._recording_error is not None:
            raise RuntimeError(f"录像线程失败：{self._recording_error}") from self._recording_error
        self._timeline.add(
            "camera_recording_finished",
            encoded_frames=self._writer.written_frames,
        )

    def _record(self) -> None:
        """持续取得相机最新帧并提交给编码器。"""

        try:
            while not self._stop.is_set():
                frame = self._camera.read_latest(after_sequence=self._last_sequence, timeout=1.0)
                self._writer.write(frame.image_bgr)
                self._last_sequence = frame.sequence_id
                self._timeline.latest_frame_sequence = frame.sequence_id
        except BaseException as error:
            if not self._stop.is_set():
                self._recording_error = error


class TracedMovePlanActuator(MovePlanActuator):
    """正式执行器加上不改变动作顺序的高层控制时间线。"""

    def __init__(self, config: ActuationConfig, timeline: Timeline) -> None:
        """构造正式执行器并绑定共享时间线。"""

        super().__init__(config)
        self._timeline = timeline

    def execute(self, plan: MovePlan):
        """执行方案并记录开始、结束或失败状态。"""

        self._timeline.add("actuation_started", plan=asdict(plan))
        try:
            result = super().execute(plan)
        except BaseException as error:
            self._timeline.add(
                "actuation_failed",
                error=str(error),
                state=self.state.name,
            )
            raise
        self._timeline.add(
            "actuation_finished",
            state=result.state.name,
            completed_cycles=result.completed_cycles,
        )
        return result

    def _move_z_to(self, target_degrees, transport, xy_controller) -> None:
        """记录 Z 目标后复用正式动作。"""

        self._timeline.add("z_target", target_degrees=target_degrees)
        return super()._move_z_to(target_degrees, transport, xy_controller)

    def _move_xy_to(self, x_0p01mm, y_0p01mm, xy_controller) -> None:
        """记录 XY 目标后复用正式动作。"""

        self._timeline.add(
            "xy_target",
            x_mm=x_0p01mm / 100,
            y_mm=y_0p01mm / 100,
        )
        return super()._move_xy_to(x_0p01mm, y_0p01mm, xy_controller)

    def _move_xy_and_rotate(self, cycle, xy_controller, rotation) -> None:
        """记录放置平移与旋转后复用正式并行动作。"""

        self._timeline.add(
            "place_xy_and_rotate",
            x_mm=cycle.place_x_0p01mm / 100,
            y_mm=cycle.place_y_0p01mm / 100,
            clockwise_degrees=cycle.rotation_degrees_clockwise,
        )
        return super()._move_xy_and_rotate(cycle, xy_controller, rotation)

    def _move_next_pick_and_return_rotation_to_zero(self, cycle, next_cycle, xy_controller, rotation) -> None:
        """记录回零与下一拾取目标后复用正式并行动作。"""

        next_pick = (
            None
            if next_cycle is None
            else [
                next_cycle.pick_x_0p01mm / 100,
                next_cycle.pick_y_0p01mm / 100,
            ]
        )
        self._timeline.add(
            "rotation_return_to_zero",
            clockwise_degrees=cycle.rotation_degrees_clockwise,
            next_pick=next_pick,
        )
        return super()._move_next_pick_and_return_rotation_to_zero(
            cycle,
            next_cycle,
            xy_controller,
            rotation,
        )


def _write_json(path: Path, data: dict[str, object]) -> None:
    """以 UTF-8 和稳定缩进写入诊断 JSON。"""

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _render_predicted_placement(
    image_bgr: np.ndarray,
    plan: MovePlan,
    resize_config: object,
    output_path: Path,
) -> None:
    """在动作开始前的画面上绘制本轮理论最终放置轮廓。"""

    result = Resize(resize_config).resize(image_bgr)
    width, height = result.target_size
    source_corners = np.asarray(result.source_corners_px, dtype=np.float32)
    rectified_corners = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(
        rectified_corners, source_corners
    )
    overlay = image_bgr.copy()
    colors = ((0, 255, 255), (255, 0, 255), (0, 255, 0), (0, 128, 255))
    for move, color in zip(plan.moves, colors):
        # 控制层是竖向 A4，视觉校正图则是横向 A4。
        landscape = [
            [(1.0 - y / 292.0) * (width - 1), (x / 207.0) * (height - 1)]
            for x, y in move.target_vertices_a4_mm
        ]
        polygon = cv2.perspectiveTransform(
            np.asarray([landscape], dtype=np.float32), homography
        )[0].astype(np.int32)
        cv2.polylines(overlay, [polygon], True, color, 3, cv2.LINE_AA)
        center = polygon.mean(axis=0).astype(int)
        cv2.putText(
            overlay,
            f"predicted {move.order}",
            tuple(center),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), overlay):
        raise RuntimeError(f"无法写入预测图：{output_path}")


@app.command()
def main(
    execute: bool = typer.Option(False, "--execute/--dry-run", help="仅 --execute 才真实驱动电机和磁铁。"),
    output_directory: Path = typer.Option(PROJECT_ROOT / "outputs/task1", "--output-directory"),
) -> None:
    """运行一次 Task1 并产生 MP4 与 controls.json。"""

    config = load_task1_config(PROJECT_ROOT / "configs/tasks/task1.json")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_directory.mkdir(parents=True, exist_ok=True)
    video_path = output_directory / f"task1_record_{timestamp}.mp4"
    control_path = output_directory / f"task1_record_{timestamp}_controls.json"
    predicted_path = (
        PROJECT_ROOT
        / "outputs/analysis"
        / f"task1_record_{timestamp}_predicted_placement.jpg"
    )
    timeline = Timeline()
    camera = RecordingCamera(config.camera, video_path, timeline)
    timeline.add("run_started", execute=execute, video_path=str(video_path))
    error: BaseException | None = None
    try:
        camera.start()
        detector = create_production_detector(config.detect1)
        try:
            plan = run_task1(
                camera,
                Resize(config.resize),
                detector,
                options=config.options,
            )
        finally:
            detector.close()
        timeline.add("stable_scene_accepted")
        reference = camera.read_latest(
            timeout=config.options.frame_read_timeout_seconds
        )
        _render_predicted_placement(
            reference.image_bgr,
            plan,
            config.resize,
            predicted_path,
        )
        timeline.add("predicted_placement_rendered", path=str(predicted_path))
        if execute:
            TracedMovePlanActuator(config.actuation, timeline).execute(plan)
        else:
            timeline.add("dry_run_plan", plan=asdict(plan))
    except BaseException as caught:
        error = caught
        timeline.add("run_failed", error=str(caught))
    finally:
        try:
            camera.finish()
        finally:
            _write_json(control_path, {"video_path": str(video_path), "execute": execute, **timeline.document()})
    if error is not None:
        raise typer.Exit(code=1) from error
    typer.echo(json.dumps({"video_path": str(video_path), "control_timeline": str(control_path), "predicted_placement": str(predicted_path)}, ensure_ascii=False))


if __name__ == "__main__":
    app()
