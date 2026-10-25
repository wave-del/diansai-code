"""离线比较 YOLO 粗诊断、统一视觉精修和稳定结果。

独立运行：
``poetry run python tests/external/vision/analyze_video.py --help``
"""

from __future__ import annotations

from collections import defaultdict
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

from ec_26_e.cam import CameraConfig
from ec_26_e.vision import (
    Detect1Result,
    PieceDetection,
    Resize,
    StableVisionConfig,
    StableVisionDetector,
    VisionConfig,
    VisionDetector,
)
from ec_26_e.vision.drawing import draw_vision_detection
from ec_26_e.vision.evaluation import GeometryAccumulator, percentile_summary
from ec_26_e.vision.geometry import canonicalize_corners, relative_point
from tests.external.cam.helpers import H264VideoPairWriter

app = typer.Typer(
    add_completion=False,
    help="离线运行统一视觉并输出毫米误差与分阶段耗时。",
)


def _raw_yolo_result(
    detector: VisionDetector,
    image_size: tuple[int, int],
) -> Detect1Result:
    """把粗实例 mask 转成仅用于诊断对照的多边形结果。"""

    width, height = image_size
    pieces = []
    for item in detector.last_diagnostics.instances:
        contours, _ = cv2.findContours(item.coarse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True).reshape(-1, 2)
        if len(polygon) < 3:
            continue
        polygon = canonicalize_corners(polygon)
        moments = cv2.moments(polygon)
        center = (
            moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
        ) if abs(moments["m00"]) > 1e-9 else tuple(np.mean(polygon, axis=0))
        corners = tuple((float(x), float(y)) for x, y in polygon)
        center_px = (float(center[0]), float(center[1]))
        area = float(cv2.contourArea(contour))
        pieces.append(PieceDetection(
            len(pieces), corners, center_px,
            tuple(relative_point(point, width, height) for point in corners),
            relative_point(center_px, width, height),
            area, area / (width * height), 1.0, 0.0, item.score,
        ))
    return Detect1Result(image_size, tuple(pieces), detector.config.score_threshold, width * height)


def _load_references(path: Path | None) -> dict[int, dict[str, object]]:
    """读取先冻结、后运行算法的伪真值 JSON。"""

    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(item["frame_index"]): item for item in payload.get("frames", [])}


def _draw_reference_errors(
    image_bgr: np.ndarray,
    reference: dict[str, object] | None,
    result: Detect1Result,
) -> None:
    """叠加冻结参考线、对应误差连线和毫米标注。"""

    if reference is None or not bool(reference.get("evaluate_geometry", True)):
        return
    width, height = result.image_size
    mm_px = 0.5 * (292.0 / (width - 1) + 207.0 / (height - 1))
    candidates = [np.asarray(piece.corners_px, dtype=np.float64) for piece in result.pieces]
    remaining = set(range(len(candidates)))
    for piece in reference.get("pieces", []):
        points = np.asarray(piece["corners_px"], dtype=np.float64)
        cv2.polylines(
            image_bgr,
            [np.rint(points).astype(np.int32).reshape(-1, 1, 2)],
            True,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )
        compatible = [index for index in remaining if len(candidates[index]) == len(points)]
        if not compatible:
            continue
        candidate_index = min(
            compatible,
            key=lambda index: float(np.linalg.norm(np.mean(candidates[index], axis=0) - np.mean(points, axis=0))),
        )
        remaining.remove(candidate_index)
        candidate = candidates[candidate_index]
        aligned = min(
            (np.roll(candidate, shift, axis=0) for shift in range(len(candidate))),
            key=lambda item: float(np.mean(np.linalg.norm(item - points, axis=1))),
        )
        for expected, actual in zip(points, aligned):
            expected_px = tuple(np.rint(expected).astype(int))
            actual_px = tuple(np.rint(actual).astype(int))
            cv2.line(image_bgr, expected_px, actual_px, (0, 0, 255), 1, cv2.LINE_AA)
            midpoint = np.rint(0.5 * (expected + actual)).astype(int)
            error_mm = float(np.linalg.norm(actual - expected) * mm_px)
            cv2.putText(
                image_bgr,
                f"{error_mm:.2f}mm",
                tuple(midpoint + np.array([3, -3])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )


@app.command()
def run(
    video: Path = typer.Option(
        Path("outputs/cam/detection_20260730_180643_595024_raw.mp4"),
        "--video",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    output_prefix: Path = typer.Option(
        Path("outputs/vision/offline"),
        "--output-prefix",
    ),
    reference_json: Path | None = typer.Option(None, "--reference-json", exists=True, dir_okay=False),
    start_frame: int = typer.Option(0, min=0),
    frame_step: int = typer.Option(1, min=1),
    max_frames: int = typer.Option(0, min=0, help="0 表示处理到视频末尾。"),
    warmup_frames: int = typer.Option(20, min=0),
    score: float = typer.Option(0.15, min=0.0, max=1.0),
    save_video: bool = typer.Option(True, "--save-video/--no-save-video"),
) -> None:
    """运行离线统一检测、保存同步 A4 视频与 JSON 指标。"""

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    references = _load_references(reference_json)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise typer.BadParameter(f"无法打开视频：{video}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    resizer = Resize()
    config = VisionConfig(score_threshold=score)
    detector = VisionDetector(config)
    stable = StableVisionDetector(detector, StableVisionConfig())
    writer = None
    if save_video:
        writer_config = CameraConfig(width=764, height=540, fps=max(1, round(source_fps / frame_step)))
        writer = H264VideoPairWriter(
            writer_config,
            output_prefix.with_name(output_prefix.name + "_a4.mp4"),
            output_prefix.with_name(output_prefix.name + "_annotated.mp4"),
            overwrite=True,
        )
        writer.start()

    timings: dict[str, list[float]] = defaultdict(list)
    geometry = {
        name: GeometryAccumulator.create()
        for name in ("yolo_coarse", "refined", "stable")
    }
    temporal_jitter: dict[str, list[float]] = defaultdict(list)
    previous_corners: dict[str, list[np.ndarray]] = {}
    semantic = defaultdict(int)
    frame_records = []
    processed = 0
    frame_index = start_frame
    started = time.perf_counter()
    try:
        while max_frames == 0 or processed < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if (frame_index - start_frame) % frame_step:
                frame_index += 1
                continue
            stage_started = time.perf_counter()
            try:
                resize_result = resizer.resize(frame)
            except ValueError:
                semantic["resize_failures"] += 1
                frame_index += 1
                continue
            resize_ms = (time.perf_counter() - stage_started) * 1000.0
            a4 = resize_result.image_bgr
            refined = detector.detect(a4)
            masks = tuple(item.refined_mask for item in detector.last_diagnostics.instances if item.failure is None)
            stable_started = time.perf_counter()
            stable_result = stable.update(refined, masks)
            stable_ms = (time.perf_counter() - stable_started) * 1000.0
            coarse = _raw_yolo_result(detector, refined.image_size)
            for name, result in (
                ("yolo_coarse", coarse),
                ("refined", refined),
                ("stable", stable_result),
            ):
                current = [np.asarray(piece.corners_px, dtype=np.float64) for piece in result.pieces]
                previous = previous_corners.get(name)
                if previous is not None and len(previous) == len(current):
                    mm_px = 0.5 * (292.0 / (result.image_size[0] - 1) + 207.0 / (result.image_size[1] - 1))
                    for old, new in zip(previous, current):
                        if len(old) == len(new):
                            temporal_jitter[name].extend((np.linalg.norm(new - old, axis=1) * mm_px).tolist())
                previous_corners[name] = current
            reference = references.get(frame_index)
            annotated = draw_vision_detection(
                a4,
                refined,
                detector.last_diagnostics,
                stable_result,
            )
            _draw_reference_errors(annotated, reference, refined)
            drawing_ms = (time.perf_counter() - stable_started) * 1000.0 - stable_ms
            write_started = time.perf_counter()
            if writer is not None:
                writer.write_pair(a4, annotated)
            write_ms = (time.perf_counter() - write_started) * 1000.0

            if processed >= warmup_frames:
                timings["a4_resize"].append(resize_ms)
                for key, value in detector.last_diagnostics.native_timing.to_dict().items():
                    timings[key].append(value)
                for key, value in detector.last_diagnostics.stage_ms.items():
                    timings[key].append(value)
                timings["stabilizer"].append(stable_ms)
                timings["drawing"].append(drawing_ms)
                timings["video_submit"].append(write_ms)
                timings["complete_frame"].append(
                    resize_ms
                    + detector.last_diagnostics.stage_ms["total"]
                    + stable_ms
                    + drawing_ms
                    + write_ms
                )
            if reference is not None:
                reference_pieces = [piece["corners_px"] for piece in reference.get("pieces", [])]
                if bool(reference.get("evaluate_geometry", True)):
                    for name, result in (
                        ("yolo_coarse", coarse),
                        ("refined", refined),
                        ("stable", stable_result),
                    ):
                        geometry[name].add(reference_pieces, result)
                expected_count = int(reference.get("instance_count", len(reference_pieces)))
                is_negative = bool(reference.get("is_glare_negative", False))
                if is_negative:
                    semantic["glare_negative_frames"] += 1
                    semantic["yolo_glare_false_positive"] += int(bool(coarse.pieces))
                else:
                    semantic["positive_reference_instances"] += expected_count
                    semantic["yolo_true_positive"] += min(len(coarse.pieces), expected_count)
                    semantic["yolo_false_positive"] += max(0, len(coarse.pieces) - expected_count)
                    semantic["yolo_false_negative"] += max(0, expected_count - len(coarse.pieces))
                    semantic["count_correct_frames"] += int(len(coarse.pieces) == expected_count)
            frame_records.append({
                "frame_index": frame_index,
                "yolo_count": len(coarse.pieces),
                "refined_count": len(refined.pieces),
                "stable_count": len(stable_result.pieces),
                "timing": detector.last_diagnostics.to_dict(),
            })
            processed += 1
            frame_index += 1
    finally:
        capture.release()
        detector.close()
        if writer is not None:
            writer.close()

    elapsed = time.perf_counter() - started
    true_positive = semantic.get("yolo_true_positive", 0)
    false_positive = semantic.get("yolo_false_positive", 0)
    false_negative = semantic.get("yolo_false_negative", 0)
    if true_positive + false_positive:
        semantic["yolo_precision"] = round(true_positive / (true_positive + false_positive), 4)
    if true_positive + false_negative:
        semantic["yolo_recall"] = round(true_positive / (true_positive + false_negative), 4)
    report = {
        "video": str(video),
        "processed_frames": processed,
        "elapsed_seconds": round(elapsed, 3),
        "observed_core_fps": round(processed / elapsed, 3) if elapsed else 0.0,
        "warmup_frames_excluded": warmup_frames,
        "timing_ms": {key: percentile_summary(values) for key, values in timings.items()},
        "geometry": {key: value.to_dict() for key, value in geometry.items()},
        "temporal_jitter_mm": {key: percentile_summary(values) for key, values in temporal_jitter.items()},
        "semantic": dict(semantic),
        "reference_uncertainty": "人工伪真值约 0.5～1 px，另叠加 A4 单应性定位误差；不是计量真值。",
        "frames": frame_records,
    }
    report_path = output_prefix.with_suffix(".json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(json.dumps({key: value for key, value in report.items() if key != "frames"}, ensure_ascii=False, indent=2))
    typer.echo(f"完整逐帧报告：{report_path}")


if __name__ == "__main__":
    app()
