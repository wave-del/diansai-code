"""实时采集 1280×720，A4校正后运行统一视觉并保存双路H.264。

独立运行：
``poetry run python tests/external/vision/record_camera.py --help``
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import json
import sys
import time

import cv2
import numpy as np
import typer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ec_26_e.cam import CameraConfig, FFmpegCamera
from ec_26_e.draw import OpenCVPreview
from ec_26_e.vision import (
    Detect1Config,
    Resize,
    create_production_detector,
)
from ec_26_e.vision.drawing import draw_vision_detection
from ec_26_e.vision.evaluation import percentile_summary
from tests.external.cam.helpers import H264VideoPairWriter, query_camera_info

app = typer.Typer(
    add_completion=False,
    help="实时显示 YOLO mask、精修边、稳定交点并保存 H.264。",
)


def _fit_to_source(
    a4_annotated: np.ndarray,
    source_shape: tuple[int, int, int],
) -> np.ndarray:
    """将 A4 诊断画面等比放到与原始帧同尺寸的黑色画布。"""

    height, width = source_shape[:2]
    scale = min(width / a4_annotated.shape[1], height / a4_annotated.shape[0])
    target = (
        round(a4_annotated.shape[1] * scale),
        round(a4_annotated.shape[0] * scale),
    )
    resized = cv2.resize(a4_annotated, target, interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros(source_shape, dtype=np.uint8)
    x = (width - target[0]) // 2
    y = (height - target[1]) // 2
    canvas[y : y + target[1], x : x + target[0]] = resized
    return canvas


@app.command()
def run(
    device: Path = typer.Option(Path("/dev/video0"), "--device"),
    output_prefix: Path | None = typer.Option(None, "--output-prefix"),
    display: str = typer.Option(":0", "--display"),
    preview: bool = typer.Option(True, "--preview/--no-preview"),
    save: bool = typer.Option(True, "--save/--no-save"),
    overwrite: bool = typer.Option(
        False,
        "--overwrite/--no-overwrite",
        help="允许覆盖显式指定的已有录像文件。",
    ),
    max_frames: int = typer.Option(0, min=0, help="0 表示直到 Esc/Ctrl+C。"),
) -> None:
    """以正式生产视觉链路实时检测、显示并同步保存录像。"""

    if output_prefix is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_prefix = Path("outputs/vision") / f"live_{stamp}"
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    raw_video_path = output_prefix.with_name(
        output_prefix.name + "_raw.mp4"
    )
    annotated_video_path = output_prefix.with_name(
        output_prefix.name + "_annotated.mp4"
    )
    report_path = output_prefix.with_suffix(".json")
    output_paths = (
        (raw_video_path, annotated_video_path, report_path)
        if save
        else (report_path,)
    )
    if not overwrite and any(path.exists() for path in output_paths):
        existing = ", ".join(
            str(path) for path in output_paths if path.exists()
        )
        raise typer.BadParameter(f"输出文件已存在：{existing}")

    camera_config = CameraConfig(
        device_path=device,
        width=1280,
        height=720,
        fps=30,
        input_format="mjpeg",
    )
    device_info = query_camera_info(device)
    camera = FFmpegCamera(camera_config)
    resizer = Resize()
    display_window = OpenCVPreview(
        window_name="EC 26 E - YOLO11-seg refinement",
        display_name=display,
        window_size=(1280, 720),
    ) if preview else None
    writer = H264VideoPairWriter(
        camera_config,
        raw_video_path,
        annotated_video_path,
        overwrite=overwrite,
    ) if save else None
    # 工厂参数保留兼容上层签名；内部使用与 Task1、Task2 相同的固定生产参数。
    detector = create_production_detector(Detect1Config())
    required_frames = detector.config.required_frames
    timings: dict[str, list[float]] = defaultdict(list)
    processed = 0
    captured = 0
    resize_failures = 0
    skipped = 0
    last_sequence: int | None = None
    interrupted = False
    started = time.perf_counter()
    try:
        camera.start()
        if display_window is not None:
            display_window.start()
        if writer is not None:
            writer.start()
        while max_frames == 0 or captured < max_frames:
            frame = camera.read_latest(after_sequence=last_sequence)
            captured += 1
            if last_sequence is not None:
                skipped += max(0, frame.sequence_id - last_sequence - 1)
            last_sequence = frame.sequence_id
            raw = np.asarray(frame.image_bgr)
            resize_started = time.perf_counter()
            try:
                a4 = resizer.resize(raw).image_bgr
            except ValueError:
                resize_failures += 1
                annotated = raw.copy()
                cv2.putText(
                    annotated,
                    "A4 resize failed - check board boundary",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                if writer is not None:
                    writer.write_pair(raw, annotated)
                if display_window is not None and display_window.show_and_should_stop(annotated):
                    break
                continue
            timings["a4_resize"].append((time.perf_counter() - resize_started) * 1000.0)
            stable_result = detector.detect(a4)
            refined = detector.last_single_result
            if refined is None:
                raise RuntimeError("统一视觉未生成单帧诊断结果")
            diagnostics = detector.detector.last_diagnostics
            timings["stabilizer"].append(detector.last_stabilizer_ms)
            draw_started = time.perf_counter()
            a4_annotated = draw_vision_detection(
                a4,
                refined,
                diagnostics,
                stable_result,
            )
            annotated = _fit_to_source(a4_annotated, raw.shape)
            timings["drawing"].append((time.perf_counter() - draw_started) * 1000.0)
            write_started = time.perf_counter()
            if writer is not None:
                writer.write_pair(raw, annotated)
            timings["video_submit"].append((time.perf_counter() - write_started) * 1000.0)
            for key, value in diagnostics.native_timing.to_dict().items():
                timings[key].append(value)
            for key, value in diagnostics.stage_ms.items():
                timings[key].append(value)
            processed += 1
            if display_window is not None and display_window.show_and_should_stop(annotated):
                break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        detector.close()
        camera.close()
        if writer is not None:
            writer.close()
        if display_window is not None:
            display_window.close()
    elapsed = time.perf_counter() - started
    complete_fps = processed / elapsed if elapsed else 0.0
    report = {
        "camera": device_info.to_dict(),
        "processed_frames": processed,
        "captured_frames": captured,
        "resize_failures": resize_failures,
        "skipped_camera_frames": skipped,
        "elapsed_seconds": round(elapsed, 3),
        "complete_fps": round(complete_fps, 3),
        "raw_video": str(raw_video_path) if save else None,
        "annotated_video": str(annotated_video_path) if save else None,
        "stable_first_result_latency_frames": required_frames,
        "stable_first_result_latency_seconds_at_camera_30fps_lower_bound": (
            round(required_frames / 30.0, 3)
        ),
        "stable_first_result_latency_seconds_at_observed_fps": (
            round(required_frames / complete_fps, 3)
            if complete_fps
            else None
        ),
        "interrupted": interrupted,
        "timing_ms": {
            key: percentile_summary(values[20:])
            for key, values in timings.items()
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
