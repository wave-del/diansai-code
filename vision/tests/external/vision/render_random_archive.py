"""从随机拼图视觉归档离线求解，并渲染成功方案或失败候选。

该诊断工具只读取 ``*_detections.jsonl`` 中已经保存的毫米轮廓，
不导入相机、运动、电磁铁、串口或 GPIO 模块。成功求解时复用正式求解器
的局部拼图渲染；求解失败时把安全门前保留的完整候选绘制为分页图库。

独立运行：
``poetry run python tests/external/vision/render_random_archive.py --help``。
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import typer


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solver2._engine.solver.solve import solve_observations


ARCHIVE_SCHEMA_VERSION = "1.0"
DEFAULT_DETECTIONS = Path(
    "outputs/diagnostics/random_archive_detections.jsonl"
)
DEFAULT_OUTPUT = Path("outputs/diagnostics/random_archive_render")
DEFAULT_CANDIDATES_PER_PAGE = 6

app = typer.Typer(
    add_completion=False,
    help="仅用视觉 JSON 离线渲染随机拼图成功方案或失败候选。",
)


def _load_frame(path: Path, frame_index: int) -> dict[str, Any]:
    """读取并验证指定归档帧，不扫描或加载对应视频。"""

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
            if record.get("frame_index") == frame_index:
                return record
    raise ValueError(f"归档中不存在 frame_index={frame_index}")


def _build_observations(
    record: dict[str, Any],
    *,
    stable_only: bool,
) -> dict[str, Any]:
    """将保存的审查轮廓转换成正式随机求解器输入。"""

    source_name = "reviewed_stable" if stable_only else "reviewed_single"
    reviewed = record.get(source_name)
    if not isinstance(reviewed, dict):
        raise ValueError(
            f"frame {record.get('frame_index')} 没有 {source_name} 坐标"
        )
    rows = reviewed.get("pieces")
    if not isinstance(rows, list) or not 2 <= len(rows) <= 4:
        raise ValueError("随机求解输入必须包含 2～4 片有效轮廓")
    pieces: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("纸片记录必须是 JSON 对象")
        piece_id = row.get("piece_id")
        vertices = row.get("vertices_mm")
        if not isinstance(piece_id, str) or not isinstance(vertices, list):
            raise ValueError("纸片记录缺少 piece_id 或 vertices_mm")
        pieces.append({"id": piece_id, "vertices_mm": vertices})
    return {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "mode": "random",
        "noise_bound_mm": 0.0,
        "search_top_n": 100,
        "pieces": pieces,
    }


def _load_json(path: Path) -> dict[str, Any]:
    """读取求解器生成的 JSON 对象。"""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} 不是 JSON 对象")
    return value


def _candidate_bounds(candidate: dict[str, Any]) -> tuple[float, float, float, float]:
    """计算一个失败候选全部纸片的绘图范围。"""

    points = [
        point
        for pose in candidate.get("poses", [])
        for point in pose.get("vertices", [])
    ]
    if not points:
        raise ValueError("失败候选没有可绘制的 poses")
    array = np.asarray(points, dtype=float)
    return (
        float(array[:, 0].min()),
        float(array[:, 0].max()),
        float(array[:, 1].min()),
        float(array[:, 1].max()),
    )


def _map_points(
    points: Any,
    bounds: tuple[float, float, float, float],
    image_size: tuple[int, int],
    *,
    top_margin: int,
    side_margin: int,
) -> np.ndarray:
    """把毫米坐标等比映射到 OpenCV 图像坐标。"""

    left, right, top, bottom = bounds
    width, height = image_size
    available_width = max(1, width - 2 * side_margin)
    available_height = max(1, height - top_margin - side_margin)
    span_x = max(right - left, 1e-6)
    span_y = max(bottom - top, 1e-6)
    scale = min(available_width / span_x, available_height / span_y)
    offset_x = side_margin + 0.5 * (available_width - span_x * scale)
    offset_y = top_margin + 0.5 * (available_height - span_y * scale)
    array = np.asarray(points, dtype=float)
    mapped = np.column_stack((
        offset_x + (array[:, 0] - left) * scale,
        offset_y + (array[:, 1] - top) * scale,
    ))
    return np.rint(mapped).astype(np.int32)


def _draw_polygon(
    image: np.ndarray,
    vertices: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    """以半透明填充和实线边界绘制一片候选铁片。"""

    overlay = image.copy()
    cv2.fillPoly(overlay, [vertices], color)
    cv2.addWeighted(overlay, 0.28, image, 0.72, 0.0, dst=image)
    cv2.polylines(image, [vertices], True, color, 2, cv2.LINE_AA)


def _render_rejected_candidate(
    panel: np.ndarray,
    candidate: dict[str, Any],
    candidate_number: int,
) -> dict[str, Any]:
    """在单个子图中绘制完整落位但被安全门拒绝的候选。"""

    colors = (
        (60, 120, 220),
        (80, 180, 80),
        (220, 130, 50),
        (180, 80, 180),
    )
    left, right, top, bottom = _candidate_bounds(candidate)
    width = float(candidate.get("width_mm", right - left))
    height = float(candidate.get("height_mm", bottom - top))
    bounds = (left, max(right, left + width), top, max(bottom, top + height))
    panel_height, panel_width = panel.shape[:2]
    for piece_index, pose in enumerate(candidate.get("poses", [])):
        vertices = _map_points(
            pose["vertices"],
            bounds,
            (panel_width, panel_height),
            top_margin=104,
            side_margin=24,
        )
        color = colors[piece_index % len(colors)]
        _draw_polygon(panel, vertices, color)
        center = np.rint(np.mean(vertices, axis=0)).astype(int)
        cv2.putText(
            panel,
            str(pose["piece_id"]),
            (int(center[0]) - 5, int(center[1]) + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

    rectangle = _map_points(
        ((left, top), (left + width, top + height)),
        bounds,
        (panel_width, panel_height),
        top_margin=104,
        side_margin=24,
    )
    cv2.rectangle(
        panel,
        tuple(rectangle[0]),
        tuple(rectangle[1]),
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    reasons = [str(value) for value in candidate.get("rejection_reasons", [])]
    short_reasons = ", ".join(reasons[:3])
    if len(reasons) > 3:
        short_reasons += f" +{len(reasons) - 3}"
    fill_ratio = float(candidate.get("fill_ratio", 0.0))
    rectangle_iou = float(candidate.get("rectangle_iou", 0.0))
    title_lines = (
        f"Candidate {candidate_number} | {width:.1f} x {height:.1f} mm",
        f"fill={fill_ratio:.3f}, IoU={rectangle_iou:.3f}",
        short_reasons,
    )
    for line_index, line in enumerate(title_lines):
        cv2.putText(
            panel,
            line,
            (14, 24 + 27 * line_index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48 if line_index == 2 else 0.58,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    return {
        "candidate_number": candidate_number,
        "width_mm": width,
        "height_mm": height,
        "fill_ratio": fill_ratio,
        "rectangle_iou": rectangle_iou,
        "rejection_reasons": reasons,
    }


def render_rejected_gallery(
    diagnostics_path: Path,
    output_directory: Path,
    *,
    candidates_per_page: int = DEFAULT_CANDIDATES_PER_PAGE,
) -> list[dict[str, Any]]:
    """把失败诊断中保留的候选渲染为分页 PNG，并返回索引。"""

    if candidates_per_page <= 0:
        raise ValueError("candidates_per_page 必须为正整数")
    diagnostics = _load_json(diagnostics_path)
    candidates = diagnostics.get("geometry_candidate_rejection_samples", [])
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("失败诊断中没有保留可渲染候选")
    output_directory.mkdir(parents=True, exist_ok=True)
    columns = 2
    rows = math.ceil(candidates_per_page / columns)
    page_count = math.ceil(len(candidates) / candidates_per_page)
    panel_width = 720
    panel_height = 520
    index: list[dict[str, Any]] = []
    for page_index in range(page_count):
        start = page_index * candidates_per_page
        page_candidates = candidates[start : start + candidates_per_page]
        page = np.full(
            (rows * panel_height + 54, columns * panel_width, 3),
            255,
            dtype=np.uint8,
        )
        cv2.putText(
            page,
            "Rejected random-assembly candidates | black box = candidate rectangle",
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        for slot, candidate in enumerate(page_candidates):
            if not isinstance(candidate, dict):
                continue
            candidate_number = start + slot + 1
            panel = np.full(
                (panel_height, panel_width, 3), 255, dtype=np.uint8
            )
            row = _render_rejected_candidate(
                panel, candidate, candidate_number
            )
            row.update({
                "page": page_index + 1,
                "page_file": f"page_{page_index + 1:02d}_of_{page_count:02d}.png",
            })
            index.append(row)
            row_index = slot // columns
            column_index = slot % columns
            y0 = 54 + row_index * panel_height
            x0 = column_index * panel_width
            page[y0 : y0 + panel_height, x0 : x0 + panel_width] = panel
            cv2.rectangle(
                page,
                (x0, y0),
                (x0 + panel_width - 1, y0 + panel_height - 1),
                (210, 210, 210),
                1,
            )
        image_path = (
            output_directory
            / f"page_{page_index + 1:02d}_of_{page_count:02d}.png"
        )
        if not cv2.imwrite(str(image_path), page):
            raise OSError(f"无法写入候选图库：{image_path}")
    (output_directory / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return index


def render_successful_assembly(
    solution_path: Path,
    image_path: Path,
) -> None:
    """把成功方案的目标矩形和最终铁片位置绘制成拼凑图。"""

    solution = _load_json(solution_path)
    target = solution.get("target_rectangle")
    moves = solution.get("moves")
    if not isinstance(target, dict) or not isinstance(moves, list) or not moves:
        raise ValueError("成功 solution.json 缺少目标矩形或移动结果")
    center_x, center_y = (float(value) for value in target["center_a4_mm"])
    width = float(target["width_mm"])
    height = float(target["height_mm"])
    left = center_x - width / 2.0
    top = center_y - height / 2.0
    bounds = (left, left + width, top, top + height)
    canvas_width = 1100
    canvas_height = 850
    canvas = np.full(
        (canvas_height, canvas_width, 3), 255, dtype=np.uint8
    )
    colors = (
        (60, 120, 220),
        (80, 180, 80),
        (220, 130, 50),
        (180, 80, 180),
    )
    for index, move in enumerate(moves):
        # 拼凑图展示 clearance 前严丝合缝的估计轮廓；公开移动契约中的
        # target_vertices 已是带 5 mm 留缝的执行位置。
        planned = move.get(
            "estimated_physical_vertices_before_clearance_a4_mm"
        )
        if not isinstance(planned, list):
            raise ValueError("成功移动缺少 clearance 前估计轮廓")
        vertices = _map_points(
            planned,
            bounds,
            (canvas_width, canvas_height),
            top_margin=110,
            side_margin=70,
        )
        color = colors[index % len(colors)]
        _draw_polygon(canvas, vertices, color)
        center = np.rint(np.mean(vertices, axis=0)).astype(int)
        cv2.putText(
            canvas,
            str(move["piece_id"]),
            (int(center[0]) - 7, int(center[1]) + 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
    rectangle = _map_points(
        ((left, top), (left + width, top + height)),
        bounds,
        (canvas_width, canvas_height),
        top_margin=110,
        side_margin=70,
    )
    cv2.rectangle(
        canvas,
        tuple(rectangle[0]),
        tuple(rectangle[1]),
        (20, 20, 20),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Successful structural assembly | {width:.2f} x {height:.2f} mm",
        (45, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.92,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"confidence={float(solution.get('confidence', 0.0)):.3f}",
        (45, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(image_path), canvas):
        raise OSError(f"无法写入成功拼凑图：{image_path}")


def render_archive_frame(
    *,
    detections: Path,
    output_directory: Path,
    frame_index: int,
    stable_only: bool,
    candidates_per_page: int = DEFAULT_CANDIDATES_PER_PAGE,
) -> dict[str, Any]:
    """离线求解一个归档帧，并按成功或失败分支生成图片。"""

    if not detections.is_file():
        raise FileNotFoundError(f"检测归档不存在：{detections}")
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"输出目录不是空目录：{output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    record = _load_frame(detections, frame_index)
    observations = _build_observations(record, stable_only=stable_only)
    observations_path = output_directory / "observations.json"
    observations_path.write_text(
        json.dumps(observations, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    solver_directory = output_directory / "solver"
    try:
        result = solve_observations(
            observations_path,
            solver_directory,
            print_plan=False,
            runtime_config={
                "execution_profile": "production",
                "diagnostics_level": "full",
                "execution_policy": "competition_force_output",
                "minimum_execution_confidence": 0.0,
                "top_n": 100,
            },
        )
    finally:
        logger = logging.getLogger("solver2")
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

    rendered_files: list[str]
    rejected_candidates = 0
    if bool(result.get("success")):
        assembly_path = output_directory / "successful_assembly.png"
        render_successful_assembly(
            solver_directory / "solution.json", assembly_path
        )
        rendered_files = [str(assembly_path)]
    else:
        gallery_directory = output_directory / "rejected_candidates"
        index = render_rejected_gallery(
            solver_directory / "interface_diagnostics.json",
            gallery_directory,
            candidates_per_page=candidates_per_page,
        )
        rejected_candidates = len(index)
        rendered_files = sorted(
            str(path) for path in gallery_directory.glob("page_*.png")
        )
    summary = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "detections": str(detections),
        "frame_index": frame_index,
        "source": "reviewed_stable" if stable_only else "reviewed_single",
        "camera_called": False,
        "motor_called": False,
        "serial_called": False,
        "success": bool(result.get("success")),
        "failure_reason": result.get("failure_reason"),
        "confidence": float(result.get("confidence", 0.0)),
        "rejected_candidates_rendered": rejected_candidates,
        "rendered_files": rendered_files,
    }
    (output_directory / "render_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


@app.command()
def run(
    detections: Path = typer.Option(
        DEFAULT_DETECTIONS,
        "--detections",
        exists=True,
        dir_okay=False,
        readable=True,
        help="随机采集工具生成的 *_detections.jsonl。",
    ),
    output_directory: Path = typer.Option(
        DEFAULT_OUTPUT,
        "--output-directory",
        help="必须为空或尚不存在的离线渲染输出目录。",
    ),
    frame_index: int = typer.Option(
        0,
        "--frame-index",
        min=0,
        help="JSONL 中要复算并渲染的精确 frame_index。",
    ),
    stable_only: bool = typer.Option(
        True,
        "--stable-only/--single-frame",
        help="读取稳定审查坐标；显式回退时才使用单帧坐标。",
    ),
    candidates_per_page: int = typer.Option(
        DEFAULT_CANDIDATES_PER_PAGE,
        "--candidates-per-page",
        min=1,
        max=12,
        help="失败候选图库每页绘制的候选数。",
    ),
) -> None:
    """离线求解并渲染指定视觉归档帧，不访问任何硬件。"""

    try:
        summary = render_archive_frame(
            detections=detections,
            output_directory=output_directory,
            frame_index=frame_index,
            stable_only=stable_only,
            candidates_per_page=candidates_per_page,
        )
    except (OSError, TypeError, ValueError) as error:
        typer.echo(f"随机归档渲染失败：{error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
