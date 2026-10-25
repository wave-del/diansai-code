"""录像并归档随机拼图的逐帧视觉坐标，或离线复用归档进行随机求解。

该外部诊断工具只访问相机、FFmpeg 和视觉模型，绝不导入执行器、串口、
GPIO 或运动模块。录制模式同时生成原始/标注 MP4、JSONL 帧档案及
配置清单；离线模式只读取 JSONL 中已经保存的 ``vertices_mm``，不会
重新运行视觉模型。

独立运行：
``poetry run python tests/external/vision/archive_random_dataset.py --help``。
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import cv2
import numpy as np
import typer


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ec_26_e.cam import CameraConfig, FFmpegCamera
from ec_26_e.config import load_task2_config
from ec_26_e.control import PlanningError, PlanningOptions, PuzzlePiece
from ec_26_e.control import plan_sheet_moves
from ec_26_e.tasks.task1.review import ReviewedFrame, review_visual_frame
from ec_26_e.vision import Detect1Result, Resize, create_production_detector
from ec_26_e.vision.drawing import draw_vision_detection
from tests.external.cam.helpers import H264VideoPairWriter


ARCHIVE_SCHEMA_VERSION = "1.0"
DEFAULT_OUTPUT_DIRECTORY = Path("outputs/diagnostics")
DEFAULT_MAX_FRAMES = 300

app = typer.Typer(
    add_completion=False,
    help="录制随机拼图视觉档案，或直接复用已保存顶点进行离线求解。",
)


def _json_default(value: object) -> object:
    """把配置中少量 Path、NumPy 标量转换为 JSON 兼容值。"""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"不支持 JSON 序列化的类型：{type(value).__name__}")


def _fit_to_source(
    a4_annotated: np.ndarray,
    source_shape: tuple[int, int, int],
) -> np.ndarray:
    """将 A4 绘制图等比放回原始录像尺寸，不修改原始帧。"""

    height, width = source_shape[:2]
    ratio = min(
        width / a4_annotated.shape[1],
        height / a4_annotated.shape[0],
    )
    target_size = (
        round(a4_annotated.shape[1] * ratio),
        round(a4_annotated.shape[0] * ratio),
    )
    resized = cv2.resize(
        a4_annotated,
        target_size,
        interpolation=cv2.INTER_LINEAR,
    )
    canvas = np.zeros(source_shape, dtype=np.uint8)
    left = (width - target_size[0]) // 2
    top = (height - target_size[1]) // 2
    canvas[top : top + target_size[1], left : left + target_size[0]] = (
        resized
    )
    return canvas


def _reviewed_json(reviewed: ReviewedFrame | None) -> dict[str, object] | None:
    """将正式审查结果转成可直接复用为随机拼图输入的稳定 JSON。"""

    if reviewed is None:
        return None
    return {
        "piece_count": len(reviewed.pieces),
        "piece_confidences": list(reviewed.piece_confidences),
        "pieces": [
            {
                "piece_id": piece.piece_id,
                "vertices_mm": [list(point) for point in piece.vertices_mm],
            }
            for piece in reviewed.pieces
        ],
    }


def _review_if_present(
    frame: Any,
    resize_result: Any,
    result: Detect1Result,
    rotation: str,
) -> ReviewedFrame | None:
    """仅在存在纸片时调用正式毫米坐标审查，保留空帧为 null。"""

    if not result.pieces:
        return None
    return review_visual_frame(frame, resize_result, result, rotation)  # type: ignore[arg-type]


def _archive_paths(prefix: Path) -> dict[str, Path]:
    """从同一个会话前缀派生可相互关联的四种档案文件路径。"""

    if prefix.suffix:
        raise ValueError("--output-prefix 不应包含扩展名")
    return {
        "raw_video": prefix.with_name(f"{prefix.name}_raw.mp4"),
        "annotated_video": prefix.with_name(
            f"{prefix.name}_annotated.mp4"
        ),
        "detections": prefix.with_name(f"{prefix.name}_detections.jsonl"),
        "manifest": prefix.with_name(f"{prefix.name}_manifest.json"),
    }


def _write_jsonl(handle: Any, record: dict[str, object]) -> None:
    """追加一条完整帧记录并立即刷新，降低意外中断时的数据损失。"""

    handle.write(json.dumps(
        record,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_json_default,
    ) + "\n")
    handle.flush()


def record_archive(
    *,
    output_prefix: Path,
    max_frames: int,
    duration_seconds: float | None,
    overwrite: bool,
) -> dict[str, object]:
    """录制原始/标注视频和逐帧坐标档案，不创建任何运动执行对象。"""

    if max_frames <= 0:
        raise ValueError("max_frames 必须为正整数")
    if duration_seconds is not None and duration_seconds <= 0.0:
        raise ValueError("duration_seconds 必须为正数或 None")
    paths = _archive_paths(output_prefix)
    if not overwrite:
        existing = [path for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError(
                "输出文件已存在：" + ", ".join(map(str, existing))
            )

    config = load_task2_config()
    camera = FFmpegCamera(config.camera)
    resizer = Resize(config.resize)
    detector = create_production_detector(config.detect1)
    writer = H264VideoPairWriter(
        config.camera,
        paths["raw_video"],
        paths["annotated_video"],
        overwrite=overwrite,
    )
    processed = 0
    resize_failures = 0
    detection_failures = 0
    stable_frames = 0
    started_at_ns = time.time_ns()
    recording_started = time.monotonic()
    deadline = (
        None
        if duration_seconds is None
        else recording_started + duration_seconds
    )
    last_sequence: int | None = None
    try:
        writer.start()
        camera.start()
        with paths["detections"].open("w", encoding="utf-8") as handle:
            for frame_index in range(max_frames):
                if deadline is not None and time.monotonic() >= deadline:
                    break
                previous_sequence = last_sequence
                frame = camera.read_latest(
                    after_sequence=last_sequence,
                    timeout=config.camera.read_timeout_seconds,
                )
                last_sequence = frame.sequence_id
                record: dict[str, object] = {
                    "schema_version": ARCHIVE_SCHEMA_VERSION,
                    "record_type": "frame",
                    "frame_index": frame_index,
                    "camera_sequence_id": frame.sequence_id,
                    # 双路 H.264 保留相机的 30 FPS 时间轴，慢于相机的
                    # 视觉处理会跳过中间帧。该字段让离线工具能精确 seek
                    # 回本次检测对应的录像帧，而不是错误按 JSONL 行号取帧。
                    "video_frame_index": frame.sequence_id,
                    "skipped_camera_frames_since_previous": (
                        0
                        if previous_sequence is None
                        else max(
                            0, frame.sequence_id - previous_sequence - 1
                        )
                    ),
                    "received_at_ns": frame.received_at_ns,
                    "source_size": [
                        int(frame.image_bgr.shape[1]),
                        int(frame.image_bgr.shape[0]),
                    ],
                }
                try:
                    resize_result = resizer.resize(frame.image_bgr)
                except ValueError as error:
                    resize_failures += 1
                    record.update({"status": "resize_failed", "error": str(error)})
                    annotated = frame.image_bgr.copy()
                    cv2.putText(
                        annotated, "A4 resize failed", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,
                        cv2.LINE_AA,
                    )
                else:
                    record["resize"] = resize_result.to_dict()
                    try:
                        stable_result = detector.detect(resize_result.image_bgr)
                        single_result = detector.last_single_result
                        if single_result is None:
                            raise RuntimeError("视觉模块未提供单帧精修结果")
                        reviewed_single = _review_if_present(
                            frame,
                            resize_result,
                            single_result,
                            config.options.landscape_rotation,
                        )
                        reviewed_stable = _review_if_present(
                            frame,
                            resize_result,
                            stable_result,
                            config.options.landscape_rotation,
                        )
                        record.update({
                            "status": "detected",
                            "single_frame": single_result.to_dict(),
                            "stable_frame": stable_result.to_dict(),
                            # 诊断对象的 to_dict() 已主动排除大体积掩膜；
                            # 原始像素由同步 raw.mp4 保存，需要时可重建掩膜。
                            "vision_diagnostics": (
                                detector.detector.last_diagnostics.to_dict()
                            ),
                            "reviewed_single": _reviewed_json(reviewed_single),
                            "reviewed_stable": _reviewed_json(reviewed_stable),
                        })
                        if reviewed_stable is not None:
                            stable_frames += 1
                        annotated_a4 = draw_vision_detection(
                            resize_result.image_bgr,
                            single_result,
                            detector.detector.last_diagnostics,
                            stable_result,
                        )
                        annotated = _fit_to_source(
                            annotated_a4,
                            frame.image_bgr.shape,
                        )
                    except (RuntimeError, ValueError) as error:
                        detection_failures += 1
                        record.update({
                            "status": "detection_failed", "error": str(error),
                        })
                        annotated = frame.image_bgr.copy()
                        cv2.putText(
                            annotated, "Detection failed", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,
                            cv2.LINE_AA,
                        )
                _write_jsonl(handle, record)
                writer.write_pair(frame.image_bgr, annotated)
                processed += 1
    finally:
        try:
            camera.close()
        finally:
            try:
                writer.close()
            finally:
                detector.close()

    manifest = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "record_type": "manifest",
        "created_at_ns": started_at_ns,
        "motor_or_serial_called": False,
        "camera": asdict(config.camera),
        "resize": asdict(config.resize),
        "detect1": asdict(config.detect1),
        "task_options": asdict(config.options),
        "paths": {name: str(path) for name, path in paths.items()},
        "video_frame_mapping": {
            "frame_index": "JSONL 连续检测记录序号",
            "video_frame_index": (
                "raw.mp4 与 annotated.mp4 中对应的 0 基相机帧序号"
            ),
            "note": (
                "视觉慢于相机时，中间相机帧未运行检测；录像为保持实时"
                "时间轴可能重复最近可用画面。离线关联必须使用"
                "video_frame_index，不能使用 JSONL 行号。"
            ),
        },
        "frames": {
            "processed": processed,
            "resize_failures": resize_failures,
            "detection_failures": detection_failures,
            "stable_frames": stable_frames,
        },
        "recording_elapsed_seconds": round(
            time.monotonic() - recording_started, 3
        ),
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return manifest


def _read_records(path: Path) -> Iterable[dict[str, object]]:
    """逐行验证读取 JSONL 档案，避免静默使用错误格式的数据。"""

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"第 {line_number} 行不是 JSON 对象")
            if record.get("schema_version") != ARCHIVE_SCHEMA_VERSION:
                raise ValueError(f"第 {line_number} 行 schema_version 不匹配")
            if record.get("record_type") != "frame":
                raise ValueError(f"第 {line_number} 行 record_type 不是 frame")
            yield record


def _pieces_from_record(
    record: dict[str, object],
    *,
    stable_only: bool,
) -> tuple[PuzzlePiece, ...] | None:
    """从一帧保存的毫米坐标重建随机拼图强类型输入。"""

    key = "reviewed_stable" if stable_only else "reviewed_single"
    reviewed = record.get(key)
    if not isinstance(reviewed, dict):
        return None
    rows = reviewed.get("pieces")
    if not isinstance(rows, list) or not 2 <= len(rows) <= 4:
        return None
    pieces: list[PuzzlePiece] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("归档纸片必须是对象")
        piece_id = row.get("piece_id")
        vertices = row.get("vertices_mm")
        if not isinstance(piece_id, str) or not isinstance(vertices, list):
            raise ValueError("归档纸片缺少 piece_id 或 vertices_mm")
        pieces.append(PuzzlePiece(
            piece_id,
            tuple((float(point[0]), float(point[1])) for point in vertices),
        ))
    return tuple(pieces)


def solve_archive(
    *,
    detections: Path,
    output: Path,
    frame_index: int | None,
    stable_only: bool,
    max_frames: int,
) -> dict[str, object]:
    """离线复用已保存顶点求解随机拼图，不读取视频或启动视觉模型。"""

    if max_frames <= 0:
        raise ValueError("max_frames 必须为正整数")
    if not detections.is_file():
        raise FileNotFoundError(f"检测档案不存在：{detections}")
    results: list[dict[str, object]] = []
    for record in _read_records(detections):
        current_index = record.get("frame_index")
        if not isinstance(current_index, int):
            raise ValueError("归档帧缺少整数 frame_index")
        if frame_index is not None and current_index != frame_index:
            continue
        pieces = _pieces_from_record(record, stable_only=stable_only)
        if pieces is None:
            continue
        try:
            plan = plan_sheet_moves(
                pieces,
                PlanningOptions(mode="random", minimum_execution_confidence=0.0),
            )
        except PlanningError as error:
            results.append({
                "frame_index": current_index,
                "camera_sequence_id": record["camera_sequence_id"],
                "solved": False,
                "error": str(error),
            })
        else:
            results.append({
                "frame_index": current_index,
                "camera_sequence_id": record["camera_sequence_id"],
                "solved": True,
                "plan": plan.to_dict(),
            })
        if len(results) >= max_frames:
            break
    report = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "motor_or_serial_called": False,
        "detections": str(detections),
        "stable_only": stable_only,
        "requested_frame_index": frame_index,
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


@app.command()
def record(
    output_prefix: Path = typer.Option(
        DEFAULT_OUTPUT_DIRECTORY / "random_archive",
        "--output-prefix",
        help="四个输出文件共用的前缀，不能含扩展名。",
    ),
    max_frames: int = typer.Option(
        DEFAULT_MAX_FRAMES,
        "--max-frames",
        min=1,
        help="最多采集帧数；默认约十秒，避免无界录像。",
    ),
    duration_seconds: float | None = typer.Option(
        None,
        "--duration-seconds",
        min=0.001,
        help="按实际墙钟时间停止；与 max_frames 同时生效，先到者停止。",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite/--no-overwrite",
        help="是否覆盖同前缀的已有档案。",
    ),
) -> None:
    """录制相机和逐帧坐标档案，不访问任何运动硬件。"""

    if output_prefix == DEFAULT_OUTPUT_DIRECTORY / "random_archive":
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_prefix = DEFAULT_OUTPUT_DIRECTORY / f"random_archive_{stamp}"
    try:
        report = record_archive(
            output_prefix=output_prefix,
            max_frames=max_frames,
            duration_seconds=duration_seconds,
            overwrite=overwrite,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"随机拼图档案录制失败：{error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))


@app.command("solve")
def solve_command(
    detections: Path = typer.Option(
        ...,
        "--detections",
        exists=True,
        dir_okay=False,
        readable=True,
        help="record 命令生成的 *_detections.jsonl。",
    ),
    output: Path = typer.Option(
        Path("outputs/diagnostics/random_archive_solutions.json"),
        "--output",
        help="离线随机拼接结果 JSON。",
    ),
    frame_index: int | None = typer.Option(
        None,
        "--frame-index",
        min=0,
        help="指定唯一帧；省略时按录像顺序选择可用帧。",
    ),
    stable_only: bool = typer.Option(
        True,
        "--stable-only/--single-frame",
        help="默认仅使用稳定器已接受的坐标。",
    ),
    max_frames: int = typer.Option(
        1,
        "--max-frames",
        min=1,
        help="最多复算多少可用帧；默认一帧，避免无意耗尽求解预算。",
    ),
) -> None:
    """直接复用 JSONL 中的 vertices_mm 离线运行随机拼接。"""

    try:
        report = solve_archive(
            detections=detections,
            output=output,
            frame_index=frame_index,
            stable_only=stable_only,
            max_frames=max_frames,
        )
    except (OSError, PlanningError, TypeError, ValueError) as error:
        typer.echo(f"随机拼图档案离线求解失败：{error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
