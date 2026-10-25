"""把已验证的 YOLO 鲁棒视觉管线接到同事版 Task1 拼图规划器。

本模块只负责相机采集、稳定视觉结果到控制契约的坐标转换，以及纯规划。
它不导入本机运动控制器，也不会连接写字机、F32C 或电磁铁。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import math
from pathlib import Path
import sys
import threading
import time
from typing import Any

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent
A4_ROBUST_DIR = PROJECT_DIR / "tests" / "a4_robust"
# 项目随系统卡迁移时不能依赖开发机用户名或绝对目录。旧控制工程仍只被
# 用于稳定的 MovePlan/clearance/执行校验工具，白色随机题的搜索本身改由
# 相邻 workspace 中的 solver 工程完成。
COLLEAGUE_SRC = PROJECT_DIR / "vision" / "src"
REFERENCE_SRC = PROJECT_DIR / "reference-vision" / "ec_26_e" / "src"
DEFAULT_PLAN_OUTPUT_DIR = PROJECT_DIR / "handover-puzzle" / "outputs" / "task1_plans"

A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
RANDOM_A4_WIDTH_MM = 207.0
RANDOM_A4_HEIGHT_MM = 292.0
RANDOM_PIECE_COUNT_RANGE = range(2, 5)
# 随机求解不设时间上限；网页任务可通过取消按钮终止独立进程。
RANDOM_SOLVER_TOTAL_BUDGET_SECONDS = None
RANDOM_QUICK_PREMATCH_SECONDS = 6.0
FROZEN_PREVIEW_SCHEMA_VERSION = "1.0"

# 正式任务与上位机空闲预览使用同一组蓝纸色域候选，但正式链路仍要求
# seed_a4_robust、透视精修和连续五帧稳定全部通过；不会采用橙色调试框。
_FORMAL_A4_FALLBACKS: tuple[tuple[str, dict[str, int | float]], ...] = (
    (
        "高饱和蓝",
        {
            "hue_min": 90,
            "hue_max": 130,
            "sat_min": 160,
            "value_min": 80,
            "min_a4_area": 0.20,
            "max_refine_shift": 70.0,
        },
    ),
    (
        "中等饱和亮蓝",
        {
            "hue_min": 90,
            "hue_max": 132,
            "sat_min": 135,
            "value_min": 115,
            "min_a4_area": 0.20,
            "max_refine_shift": 70.0,
        },
    ),
    (
        "浅亮蓝",
        {
            "hue_min": 88,
            "hue_max": 135,
            "sat_min": 95,
            "value_min": 150,
            "min_a4_area": 0.20,
            "max_refine_shift": 70.0,
        },
    ),
)


def _formal_a4_candidates(
    params: dict[str, float | int],
) -> list[tuple[str, argparse.Namespace]]:
    """Build formal A4 candidates without relaxing its geometry criteria."""

    formal = {
        "hue_min": int(params["hue_min"]),
        "hue_max": int(params["hue_max"]),
        "sat_min": int(params["sat_min"]),
        "value_min": int(params["value_min"]),
        "min_a4_area": 0.25,
        "max_refine_shift": 35.0,
    }
    values = [("正式参数", formal), *_FORMAL_A4_FALLBACKS]
    return [
        (
            name,
            argparse.Namespace(
                hue_min=int(candidate["hue_min"]),
                hue_max=int(candidate["hue_max"]),
                sat_min=int(candidate["sat_min"]),
                value_min=int(candidate["value_min"]),
                min_a4_area=float(candidate["min_a4_area"]),
                max_refine_shift=float(candidate["max_refine_shift"]),
            ),
        )
        for name, candidate in values
    ]


class _LatestFrameReader:
    """持续读取相机，只保留最新帧，避免推理期间积压旧画面。"""

    def __init__(self, capture: Any) -> None:
        self._capture = capture
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Any | None = None
        self._sequence = 0
        self._thread = threading.Thread(
            target=self._run, name="tangram-task-camera-latest", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._capture.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            with self._lock:
                self._latest = frame
                self._sequence += 1

    def latest(self) -> tuple[int, Any | None]:
        with self._lock:
            return self._sequence, self._latest

    def close(self) -> None:
        self._stop.set()
        self._thread.join(2.0)
        if self._thread.is_alive():
            self._capture.release()
            self._thread.join(0.5)
            return
        self._capture.release()


@dataclass(frozen=True, slots=True)
class TimedMovePlan:
    """向本机适配层附带求解耗时，而不改变同事的 MovePlan 契约。"""

    plan: Any
    solver_elapsed_seconds: float

    def __getattr__(self, name: str) -> Any:
        return getattr(self.plan, name)


def _piece_count_is_plannable(mode: object, count: int) -> bool:
    """返回当前任务模式是否可将该片数交给对应求解器。"""

    if mode == "fixed":
        return count == 4
    if mode == "random":
        return count in RANDOM_PIECE_COUNT_RANGE
    raise ValueError(f"不支持的拼图模式：{mode!r}")


def _points_as_json(points: Any) -> list[list[float]]:
    """把 OpenCV/Numpy 坐标转为可由上位机直接读取的 JSON。"""

    array = np.asarray(points, dtype=float).reshape(-1, 2)
    return [[float(x), float(y)] for x, y in array]


def _frozen_preview_payload(
    result: Any,
    cv2_module: Any,
    *,
    accepted_pieces: tuple[Any, ...] | None = None,
    stabilization_quality: str = "strict_five_frame",
) -> dict[str, Any]:
    """提取一次稳定 YOLO 结果的原始相机像素叠加层。

    该文件只是跨进程交接冻结结果，绝不触发第二次检测或参与求解决策。
    """

    inverse = np.linalg.inv(result.resize_result.homography)
    width, height = result.detect_result.image_size
    divider = np.asarray(
        [[width * 0.52, 0.0], [width * 0.52, height - 1.0]],
        dtype=np.float32,
    ).reshape(1, -1, 2)
    divider_source = cv2_module.perspectiveTransform(divider, inverse)[0]

    def source_piece(piece: Any) -> dict[str, Any]:
        corners = np.asarray(piece.corners_px, dtype=np.float32).reshape(1, -1, 2)
        center = np.asarray(piece.center_px, dtype=np.float32).reshape(1, 1, 2)
        return {
            "piece_id": str(piece.group_id),
            "corners_px": _points_as_json(
                cv2_module.perspectiveTransform(corners, inverse)[0]
            ),
            "center_px": _points_as_json(
                cv2_module.perspectiveTransform(center, inverse)[0]
            )[0],
        }

    return {
        "schema_version": FROZEN_PREVIEW_SCHEMA_VERSION,
        "source_image_size": [
            int(result.resize_result.source_size[0]),
            int(result.resize_result.source_size[1]),
        ],
        "seed_corners_px": _points_as_json(result.seed.corners),
        "seed_reliability": [bool(value) for value in result.seed.reliability],
        "refined_a4_corners_px": _points_as_json(
            result.resize_result.source_corners_px
        ),
        "divider_source_px": _points_as_json(divider_source),
        "accepted_pieces": [
            source_piece(piece)
            for piece in (
                result.detect_result.pieces
                if accepted_pieces is None
                else accepted_pieces
            )
        ],
        "rejected_pieces": [
            source_piece(piece) for piece in result.rejected_pieces
        ],
        "stabilization_quality": stabilization_quality,
    }


def _write_frozen_preview_payload(
    config: "YoloTaskConfig", payload: dict[str, Any]
) -> Path:
    """原子写入上位机恢复原始预览时使用的冻结叠加结果。"""

    path = config.plan_output_dir / "latest_frozen_detection.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _write_patterned_scene(
    config: "YoloTaskConfig", result: Any, cv2_module: Any
) -> Path:
    """保存与 ``_portrait_vertices`` 完全同向的冻结 A4 图像。

    几何顶点已从横向校正图转到了竖向 207×292 mm 坐标；花纹采样必须使用
    同一坐标系，否则即使候选几何正确也会在错误像素位置取样。
    """
    image = np.asarray(result.resize_result.image_bgr)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("冻结校正图格式无效")
    rotate_code = (
        cv2_module.ROTATE_90_CLOCKWISE
        if config.landscape_rotation == "clockwise"
        else cv2_module.ROTATE_90_COUNTERCLOCKWISE
    )
    portrait = cv2_module.rotate(image, rotate_code)
    path = config.plan_output_dir / "latest_patterned_scene.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2_module.imwrite(str(path), portrait):
        raise ValueError(f"无法保存冻结花纹图：{path}")
    return path


def _solver_branch_name(mode: object, count: int) -> str:
    """给现场日志使用的同事新版求解器分支名称。"""

    if mode == "fixed" and count == 4:
        return "固定四片模板"
    if mode == "random" and count == 2:
        return "随机两片边链"
    if mode == "random" and count == 3:
        return "随机三片拓扑"
    if mode == "random" and count == 4:
        return "随机四片拓扑"
    raise ValueError(f"模式 {mode!r} 不支持 {count} 片")


@dataclass(frozen=True, slots=True)
class YoloTaskConfig:
    """鲁棒视觉采集和同事版纯规划所需的最小配置。"""

    camera: str
    width: int
    height: int
    fps: int
    process_fps: float
    acquisition_timeout_seconds: float
    params_name: str
    landscape_rotation: str
    planning: dict[str, object]
    plan_output_dir: Path
    draw: bool
    windowed: bool
    allow_competition_compliant: bool
    reference_width_mm: float = A4_WIDTH_MM
    reference_height_mm: float = A4_HEIGHT_MM
    surface_mode: str = "white"
    use_new_random_solver: bool = False
    use_patterned_solver: bool = False


def _read_env(path: Path) -> dict[str, str]:
    """读取本项目使用的简单 KEY=VALUE 环境文件。"""

    values: dict[str, str] = {}
    for line_number, source_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = source_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: 环境配置必须为 KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{path}:{line_number}: 环境变量名不能为空")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def _positive_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 必须是实数")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} 必须是正的有限数")
    return result


def load_yolo_task_config(
    env_path: Path,
    task_config_path: Path,
    *,
    params_name: str,
    process_fps: float,
    acquisition_timeout_seconds: float | None,
    plan_output_dir: Path,
    draw: bool,
    windowed: bool,
    allow_competition_compliant: bool = False,
    landscape_rotation: str | None = None,
    reference_width_mm: float = A4_WIDTH_MM,
    reference_height_mm: float = A4_HEIGHT_MM,
    surface_mode: str = "white",
    use_new_random_solver: bool = False,
    use_patterned_solver: bool = False,
    planning_start_xy_mm: tuple[float, float] | None = None,
    camera_override: str | None = None,
) -> YoloTaskConfig:
    """读取现有 Task1 配置，但不导入任一同名 ``ec_26_e`` 包。"""

    env = _read_env(env_path)
    raw = json.loads(task_config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Task1 JSON 顶层必须是对象")
    task = raw.get("task")
    planning = raw.get("planning")
    if not isinstance(task, dict) or not isinstance(planning, dict):
        raise ValueError("Task1 JSON 必须包含 task 和 planning 对象")
    rotation = (
        task.get("landscape_rotation", "counterclockwise")
        if landscape_rotation is None
        else landscape_rotation
    )
    if rotation not in {"clockwise", "counterclockwise"}:
        raise ValueError("landscape_rotation 必须为 clockwise 或 counterclockwise")
    if params_name not in {"current", "new"}:
        raise ValueError("params_name 必须为 current 或 new")

    required_env = (
        "CAMERA_DEVICE_PATH",
        "CAMERA_WIDTH",
        "CAMERA_HEIGHT",
        "CAMERA_FPS",
    )
    missing = [name for name in required_env if name not in env]
    if missing:
        raise ValueError("相机环境配置缺少：" + ", ".join(missing))
    timeout = (
        task.get("acquisition_timeout_seconds", 30.0)
        if acquisition_timeout_seconds is None
        else acquisition_timeout_seconds
    )
    if not (
        math.isfinite(reference_width_mm)
        and math.isfinite(reference_height_mm)
        and reference_width_mm > 0.0
        and reference_height_mm > 0.0
    ):
        raise ValueError("参考坐标宽高必须为正的有限数")
    if surface_mode not in {"white", "patterned"}:
        raise ValueError("surface_mode 必须为 white 或 patterned")
    planning_values = dict(planning)
    if planning_start_xy_mm is not None:
        planning_values["start_xy_a4_mm"] = list(planning_start_xy_mm)
    return YoloTaskConfig(
        # 上位机可在热插拔后由操作员选择设备；仅覆盖本次任务，不写回
        # 环境文件，从而保留原工程的可移植默认配置。
        camera=(
            str(camera_override).strip()
            if camera_override is not None and str(camera_override).strip()
            else env["CAMERA_DEVICE_PATH"]
        ),
        width=int(env["CAMERA_WIDTH"]),
        height=int(env["CAMERA_HEIGHT"]),
        fps=int(env["CAMERA_FPS"]),
        process_fps=_positive_number("process_fps", process_fps),
        acquisition_timeout_seconds=_positive_number(
            "acquisition_timeout_seconds", timeout
        ),
        params_name=params_name,
        landscape_rotation=str(rotation),
        planning=planning_values,
        plan_output_dir=plan_output_dir,
        draw=bool(draw),
        windowed=bool(windowed),
        allow_competition_compliant=bool(allow_competition_compliant),
        reference_width_mm=float(reference_width_mm),
        reference_height_mm=float(reference_height_mm),
        surface_mode=surface_mode,
        use_new_random_solver=bool(use_new_random_solver),
        use_patterned_solver=bool(use_patterned_solver),
    )


def _portrait_vertices(
    vertices: tuple[tuple[float, float], ...],
    rotation: str,
    reference_width_mm: float = A4_WIDTH_MM,
    reference_height_mm: float = A4_HEIGHT_MM,
) -> tuple[tuple[float, float], ...]:
    """把横向校正图的相对坐标转为竖向 A4 毫米坐标。"""

    converted: list[tuple[float, float]] = []
    for point in vertices:
        if not isinstance(point, tuple) or len(point) != 2:
            raise TypeError("拼图顶点必须是二维元组")
        u, v = float(point[0]), float(point[1])
        if not math.isfinite(u) or not math.isfinite(v):
            raise ValueError("拼图顶点必须是有限数")
        if not 0.0 <= u <= 1.0 or not 0.0 <= v <= 1.0:
            raise ValueError("拼图相对坐标必须位于 [0, 1]")
        if rotation == "clockwise":
            converted.append((reference_width_mm * (1.0 - v), reference_height_mm * u))
        elif rotation == "counterclockwise":
            converted.append((reference_width_mm * v, reference_height_mm * (1.0 - u)))
        else:
            raise ValueError("rotation 必须为 clockwise 或 counterclockwise")
    return tuple(converted)


def _prepare_import_paths() -> None:
    if not A4_ROBUST_DIR.is_dir():
        raise RuntimeError(f"未找到实测视觉管线：{A4_ROBUST_DIR}")
    if not COLLEAGUE_SRC.is_dir():
        raise RuntimeError(f"未找到同事工程：{COLLEAGUE_SRC}")
    # YOLO 管线与旧 reference-vision 都含同名 ec_26_e 包。先移除旧路径，
    # 再清除已经错误加载的同名模块，确保规划器和视觉来自同事新版工程。
    reference_value = str(REFERENCE_SRC)
    while reference_value in sys.path:
        sys.path.remove(reference_value)
    for path in (A4_ROBUST_DIR, COLLEAGUE_SRC):
        value = str(path)
        while value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)
    loaded = sys.modules.get("ec_26_e")
    loaded_file = getattr(loaded, "__file__", None)
    if loaded_file is not None and COLLEAGUE_SRC not in Path(loaded_file).resolve().parents:
        for name in tuple(sys.modules):
            if name == "ec_26_e" or name.startswith("ec_26_e."):
                sys.modules.pop(name, None)


def _planning_options(raw: dict[str, object]) -> Any:
    from ec_26_e.control import PlanningOptions

    values = dict(raw)
    start = values.get("start_xy_a4_mm")
    if isinstance(start, list):
        values["start_xy_a4_mm"] = tuple(start)
    return PlanningOptions(**values)


def _competition_override_allowed(
    result: dict[str, Any],
    *,
    enabled: bool,
) -> bool:
    """只放行无歧义且通过竞赛名义几何门的固定题测试方案。"""

    return bool(
        enabled
        and result.get("competition_gate_passed")
        and result.get("machine_executable")
        and result.get("search_candidate_valid")
        and not result.get("ambiguous")
        and result.get("execution_quality") == "competition_compliant"
    )


def _plan_with_artifacts(
    pieces: tuple[Any, ...],
    options: Any,
    output_root: Path,
    *,
    allow_competition_compliant: bool = False,
    surface_mode: str = "white",
    use_new_random_solver: bool = False,
    use_patterned_solver: bool = False,
    patterned_scene_path: Path | None = None,
) -> object:
    """运行同事求解器，并在执行门判定前持久化结果图和诊断文件。"""

    from ec_26_e.control import MovePlan, PlanningError, PlanningOptions, PuzzlePiece
    from ec_26_e.control._engine.evaluation.clearance_compaction_smoke import (
        _render_layout,
    )
    from ec_26_e.control.planner import _to_move

    if not isinstance(options, PlanningOptions):
        raise TypeError("options 必须是 PlanningOptions")
    if not pieces or any(not isinstance(piece, PuzzlePiece) for piece in pieces):
        raise TypeError("pieces 必须是非空 PuzzlePiece 元组")
    if not _piece_count_is_plannable(options.mode, len(pieces)):
        if options.mode == "fixed":
            raise ValueError("fixed 模式必须恰好输入 4 片纸片")
        raise ValueError("random 模式只支持 2～4 片纸片")
    if len({piece.piece_id for piece in pieces}) != len(pieces):
        raise ValueError("piece_id 不能重复")

    payload = {
        "schema_version": "1.0",
        "mode": options.mode,
        "surface_mode": surface_mode,
        "noise_bound_mm": options.noise_bound_mm,
        "search_top_n": options.search_top_n,
        "pieces": [
            {
                "id": piece.piece_id,
                "vertices_mm": [list(point) for point in piece.vertices_mm],
            }
            for piece in pieces
        ],
    }
    # 随机适配层使用与公开 planner 相同的轮廓和赛题约束。
    if not (options.mode == "random" and (use_new_random_solver or use_patterned_solver)):
        from ec_26_e.control._engine.solver.preprocess import preprocess_observations

        preprocess_observations(payload)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    input_path = output_dir / "observations.json"
    input_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if options.mode == "random" and (use_new_random_solver or use_patterned_solver):
        if use_patterned_solver:
            policy = "不设时间上限；边链搜索、联合优化、现场花纹消歧及最终检查"
        else:
            policy = "不设时间上限；先验证整边链，再按需展开部分边和 T 形匹配"
        print(
            "新随机求解时间策略：" + policy
            + "；候选统一通过几何、留缝和最终执行检查。",
            flush=True,
        )
    try:
        if options.mode == "random" and use_patterned_solver:
            from random_patterned_solver_adapter import solve_random_patterned

            result = solve_random_patterned(
                pieces, output_dir, scene_path=patterned_scene_path, timeout_s=None,
                noise_bound_mm=options.noise_bound_mm,
                search_top_n=options.search_top_n,
                minimum_execution_confidence=options.minimum_execution_confidence,
            )
        elif options.mode == "random" and use_new_random_solver:
            # 白色和花纹任务共用迁入的几何引擎及最终检查。
            from random_white_solver_adapter import solve_random_white

            result = solve_random_white(
                pieces,
                output_dir,
                timeout_s=RANDOM_SOLVER_TOTAL_BUDGET_SECONDS,
                noise_bound_mm=options.noise_bound_mm,
                search_top_n=options.search_top_n,
                minimum_execution_confidence=options.minimum_execution_confidence,
            )
        else:
            from ec_26_e.control._engine.solver.solve import solve_observations

            result = solve_observations(
                input_path,
                output_dir,
                print_plan=False,
                runtime_config={
                    "execution_profile": "production",
                    "top_n": options.search_top_n,
                    "minimum_execution_confidence": (
                        options.minimum_execution_confidence
                    ),
                    "execution_policy": "strict",
                    "unlimited_search": True,
                },
            )
    finally:
        logger = logging.getLogger("ec_26_e.control")
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

    if not result.get("success"):
        raise PlanningError(str(result.get("failure_reason", "未找到完整拼图方案")))

    image_path = output_dir / "solution_after.png"
    _render_layout(result, image_path)
    print(f"求解结果图已保存：{image_path}", flush=True)
    reconstruction = result.get("pattern_reconstruction_path")
    if reconstruction:
        print(f"花纹复原图已保存：{reconstruction}", flush=True)
    print(f"完整求解诊断已保存：{output_dir}", flush=True)

    if options.mode == "fixed" and not result.get("executable"):
        competition_override_allowed = _competition_override_allowed(
            result,
            enabled=allow_competition_compliant,
        )
        if not competition_override_allowed:
            reasons = result.get("execution_block_reasons", [])
            raise PlanningError(f"拼图方案未通过执行校验：{reasons}")
        warnings = result.get("risk_warnings", [])
        print(
            "⚠ 测试放行：方案通过竞赛名义几何门，但未通过严格鲁棒门；"
            f"风险={warnings}",
            flush=True,
        )
    elif options.mode == "random" and not result.get("executable"):
        reasons = result.get("execution_block_reasons", [])
        raise PlanningError(f"随机拼图方案未通过新版执行校验：{reasons}")
    target = result.get("target_rectangle")
    if not isinstance(target, dict):
        raise PlanningError("求解器返回了无效的目标矩形")
    return TimedMovePlan(
        plan=MovePlan(
            mode=options.mode,
            confidence=float(result["confidence"]),
            ambiguous=bool(result["ambiguous"]),
            target_width_mm=float(target["width_mm"]),
            target_height_mm=float(target["height_mm"]),
            target_center_a4_mm=tuple(target["center_a4_mm"]),
            moves=tuple(_to_move(move) for move in result["moves"]),
        ),
        solver_elapsed_seconds=float(result.get("elapsed_seconds", 0.0)),
    )


def acquire_yolo_move_plan(config: YoloTaskConfig) -> object:
    """取得五帧稳定结果并调用同事版求解器，期间不访问运动硬件。"""

    if not isinstance(config, YoloTaskConfig):
        raise TypeError("config 必须是 YoloTaskConfig")
    _prepare_import_paths()

    import cv2
    import test_a4_detection as a4_detection
    import ec_26_e
    from ec_26_e.control import PuzzlePiece
    from run_live_yolo import _draw
    from seed import seed_a4_robust
    from yolo_pipeline import YoloRobustDetector

    package_file = Path(ec_26_e.__file__).resolve()
    if COLLEAGUE_SRC not in package_file.parents:
        raise RuntimeError(
            "YOLO 分支加载了错误的同名 ec_26_e 包："
            f"{package_file}；请使用独立进程运行 traditional 与 yolo"
        )

    preset_args = argparse.Namespace(
        params=config.params_name,
        hue_min=None,
        hue_max=None,
        sat_min=None,
        value_min=None,
    )
    params = a4_detection._resolve_params(preset_args)
    a4_candidates = _formal_a4_candidates(params)
    camera_args = argparse.Namespace(
        camera=config.camera,
        width=config.width,
        height=config.height,
        fps=config.fps,
    )
    detector: Any | None = None
    frames: _LatestFrameReader | None = None
    window = "Tangram Task1 - YOLO Robust Detection"
    frozen: tuple[Any, ...] | None = None
    stabilization_quality = "strict_five_frame"
    next_process = -math.inf
    latest = None
    status = "等待五帧稳定视觉结果..."
    last_error = "尚未完成首帧检测"
    processed_frames = 0
    last_frame_sequence = -1
    frozen_preview_payload: dict[str, Any] | None = None
    frozen_preview_error: Exception | None = None
    patterned_scene_path: Path | None = None
    active_a4_name: str | None = None
    last_a4_diagnostic: str | None = None

    try:
        detector = YoloRobustDetector(params, a4_candidates[0][1], stable=True)
        capture = a4_detection._open_camera(camera_args)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        frames = _LatestFrameReader(capture)
        frames.start()
        # 采集时限只从相机和检测器均已准备好后开始计算。比赛 UI 计时仍在
        # 按键时立即开始；这里仅避免初始化消耗掉“等待解除遮挡”的容忍时间。
        deadline = time.monotonic() + config.acquisition_timeout_seconds
        if config.draw:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            if config.windowed:
                cv2.resizeWindow(window, 1280, 720)
            else:
                cv2.setWindowProperty(
                    window,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN,
                )
        while time.monotonic() < deadline:
            sequence, frame = frames.latest()
            if frame is None or sequence == last_frame_sequence:
                time.sleep(0.01)
                last_error = "摄像头未返回有效画面"
                continue
            last_frame_sequence = sequence
            now = time.monotonic()
            if now >= next_process:
                next_process = now + 1.0 / config.process_fps
                processed_frames += 1
                try:
                    valid_a4: list[tuple[str, argparse.Namespace, Any]] = []
                    diagnostics: list[str] = []
                    for name, a4_args in a4_candidates:
                        try:
                            seed = seed_a4_robust(frame, a4_args)
                        except ValueError as error:
                            diagnostics.append(f"{name}=失败({error})")
                            continue
                        valid_a4.append((name, a4_args, seed))
                        diagnostics.append(
                            f"{name}=通过(mode={seed.mode}, "
                            f"角点={sum(bool(item) for item in seed.reliability)}/4)"
                        )
                    diagnostic = "A4 候选诊断：" + "；".join(diagnostics)
                    if diagnostic != last_a4_diagnostic:
                        print(diagnostic, flush=True)
                        last_a4_diagnostic = diagnostic
                    if not valid_a4:
                        latest = None
                        last_error = diagnostic
                        status = "FAIL: 所有正式 A4 HSV 候选均未通过矩形验证"
                        continue
                    by_name = {name: (args, seed) for name, args, seed in valid_a4}
                    if active_a4_name in by_name:
                        selected_name = active_a4_name
                        selected_args, selected_seed = by_name[selected_name]
                    else:
                        selected_name, selected_args, selected_seed = max(
                            valid_a4,
                            key=lambda item: (
                                sum(bool(value) for value in item[2].reliability),
                                -float(item[2].aspect_error),
                                float(item[2].area_ratio),
                            ),
                        )
                    if selected_name != active_a4_name:
                        detector._a4_args = selected_args
                        detector.reset()
                        active_a4_name = selected_name
                        print(
                            f"正式 A4 候选已锁定：{selected_name} "
                            f"(mode={selected_seed.mode}, "
                            f"角点={sum(bool(item) for item in selected_seed.reliability)}/4, "
                            f"面积={selected_seed.area_ratio:.3f}, "
                            f"比例误差={selected_seed.aspect_error:.3f})",
                            flush=True,
                        )
                    latest = detector.detect(frame)
                    if detector.reference_just_locked:
                        print(
                            "A4 参考四角已由 3 帧中位数锁定；"
                            "后续拼图稳定采集使用固定透视坐标。",
                            flush=True,
                        )
                    stable = tuple(latest.detect_result.pieces)
                    single_count = (
                        len(latest.single_result.pieces)
                        if latest.single_result is not None
                        else 0
                    )
                    mode = config.planning.get("mode", "random")
                    expected = "4" if mode == "fixed" else "2～4（自动识别）"
                    status = (
                        f"OK: seed={latest.seed.mode} "
                        f"稳定片数={len(stable)} 单帧片数={single_count} "
                        f"期望={expected} 总耗时={latest.timing_ms['total']:.0f}ms"
                    )
                    # 上位机据此显示正式任务的采集过程。预览阶段已验证稳定
                    # 并不代表新进程可直接复用其历史，正式任务仍须独立收集五帧。
                    print(
                        f"视觉采集中：已处理={processed_frames} 帧；"
                        f"稳定={len(stable)} 片；单帧={single_count} 片；"
                        f"剩余={max(0.0, deadline - now):.1f} 秒。",
                        flush=True,
                    )
                    last_error = ""
                    count_valid = _piece_count_is_plannable(mode, len(stable))
                    if count_valid:
                        frozen = stable
                        stabilization_quality = "strict_five_frame"
                except Exception as error:
                    latest = None
                    last_error = f"检测阶段失败：{error}"
                    status = f"FAIL: {error}"
            if config.draw:
                cv2.imshow(window, _draw(frame, latest, status))
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    raise KeyboardInterrupt("用户通过视觉预览窗口取消")
            if frozen is not None:
                break
        if frozen is None and detector is not None and latest is not None:
            # 轻微抖动不应让整个比赛链路直接退出：关联、片数和多边形质量
            # 仍由稳定器检查，只放宽严格五帧的 4 mm 顶点门到 8 mm。
            fallback_result = detector.best_effort_result(
                max_vertex_jitter_mm=8.0
            )
            fallback = (
                tuple(fallback_result.pieces)
                if fallback_result is not None
                else ()
            )
            mode = config.planning.get("mode", "random")
            if _piece_count_is_plannable(mode, len(fallback)):
                frozen = fallback
                stabilization_quality = "relaxed_multiframe"
                print(
                    "⚠ 严格五帧稳定未达标；使用关联一致的多帧聚合兜底"
                    "（顶点抖动上限 8 mm）继续解算。",
                    flush=True,
                )
            else:
                single = latest.single_result
                single_pieces = tuple(single.pieces) if single is not None else ()
                if _piece_count_is_plannable(mode, len(single_pieces)):
                    frozen = single_pieces
                    stabilization_quality = "single_frame_fallback"
                    print(
                        "⚠ 多帧稳定未达标；使用最后一帧有效识别结果兜底继续解算。",
                        flush=True,
                    )
        if frozen is None:
            raise TimeoutError(
                f"{config.acquisition_timeout_seconds:.1f} 秒内未获得可规划的视觉结果；"
                f"已处理帧={processed_frames}；最后状态={status}；最后错误={last_error}"
            )
        try:
            frozen_preview_payload = _frozen_preview_payload(
                latest,
                cv2,
                accepted_pieces=frozen,
                stabilization_quality=stabilization_quality,
            )
        except Exception as error:
            # 预览文件只用于上位机展示，不能妨碍已经冻结的正式拼图流程。
            frozen_preview_error = error
        if config.use_patterned_solver:
            try:
                patterned_scene_path = _write_patterned_scene(config, latest, cv2)
                print(f"花纹现场图已冻结：{patterned_scene_path}", flush=True)
            except Exception as error:
                # 花纹是消歧增益而非几何执行前提，保存失败仍必须继续输出。
                print(f"⚠ 无法保存冻结花纹图，改用几何兜底：{error}", flush=True)
    finally:
        if frames is not None:
            frames.close()
        if detector is not None:
            detector.close()
        if config.draw:
            try:
                cv2.destroyWindow(window)
                cv2.waitKey(1)
            except cv2.error:
                pass

    if frozen_preview_payload is not None:
        try:
            frozen_preview_path = _write_frozen_preview_payload(
                config, frozen_preview_payload
            )
            print(f"视觉冻结预览已就绪：{frozen_preview_path}", flush=True)
        except Exception as error:
            print(f"⚠ 无法保存冻结视觉预览：{error}", flush=True)
    elif frozen_preview_error is not None:
        print(f"⚠ 无法生成冻结视觉预览：{frozen_preview_error}", flush=True)

    pieces = tuple(
        PuzzlePiece(
            piece_id=str(piece.group_id),
            vertices_mm=_portrait_vertices(
                tuple(tuple(point) for point in piece.corners_relative),
                config.landscape_rotation,
                config.reference_width_mm,
                config.reference_height_mm,
            ),
        )
        for piece in frozen
    )
    branch = _solver_branch_name(config.planning.get("mode"), len(pieces))
    solver_label = (
        "新扑克牌花纹随机" if config.use_patterned_solver
        else "新白色随机" if config.use_new_random_solver else "原有"
    )
    print(
        f"视觉已冻结：{len(pieces)} 片，自动选择{solver_label}“{branch}”求解器...",
        flush=True,
    )
    return _plan_with_artifacts(
        pieces,
        _planning_options(config.planning),
        config.plan_output_dir,
        allow_competition_compliant=config.allow_competition_compliant,
        surface_mode=config.surface_mode,
        use_new_random_solver=config.use_new_random_solver,
        use_patterned_solver=config.use_patterned_solver,
        patterned_scene_path=patterned_scene_path,
    )
