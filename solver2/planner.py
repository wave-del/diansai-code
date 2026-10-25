"""以完整拼图求解器生成与既有控制契约兼容的移动方案。"""

from __future__ import annotations

from collections.abc import Sequence
import json
import logging
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from solver2.acceleration import ClearanceScanBackend, RigidFitBackend
from solver2.candidate_acceleration import CandidateScanBackend, candidate_acceleration

from solver2._engine.geometry import clean_vertices
from solver2._engine.solver.preprocess import (
    preprocess_observations,
)
from solver2._engine.solver.solve import solve_observations
from solver2.contracts import (
    MovePlan,
    PlannedMove,
    PlanningError,
    PlanningOptions,
    PuzzlePiece,
)


def _point(value: object) -> tuple[float, float]:
    """将求解器 JSON 坐标转换为既有的强类型二维坐标。"""

    x, y = value  # type: ignore[misc]
    return float(x), float(y)


def _vertices(value: object) -> tuple[tuple[float, float], ...]:
    """将求解器 JSON 顶点列表转换为既有顶点元组。"""

    return tuple(_point(point) for point in value)  # type: ignore[arg-type]


def _to_move(move: dict[str, object]) -> PlannedMove:
    """把新求解器的单步 JSON 结果适配为稳定公开契约。"""

    transform = move["rigid_transform"]
    if not isinstance(transform, dict):
        raise PlanningError("求解器返回了无效的刚体变换")
    # 求解器用 [-180, 180) 表示等价刚体旋转；执行器只接受顺时针
    # [0, 360] 目标，取模不会改变该刚体变换的最终姿态。
    rotation_degrees_clockwise = (
        float(move["rotation_deg_clockwise"]) % 360.0
    )
    return PlannedMove(
        order=int(move["order"]),
        piece_id=str(move["piece_id"]),
        pick_xy_a4_mm=_point(move["pick_xy_a4_mm"]),
        pick_clearance_mm=float(move["pick_clearance_mm"]),
        rotation_deg_clockwise=rotation_degrees_clockwise,
        place_xy_a4_mm=_point(move["place_xy_a4_mm"]),
        translation_a4_mm=_point(transform["translation_a4_mm"]),
        observed_vertices_a4_mm=_vertices(move["observed_vertices_a4_mm"]),
        target_vertices_a4_mm=_vertices(move["target_vertices_a4_mm"]),
        template_id=(
            str(move["template_id"])
            if move.get("template_id") is not None
            else None
        ),
    )


def _validate_public_input(pieces: tuple[PuzzlePiece, ...]) -> None:
    """保留公开接口的输入校验语义，不参与实际求解。"""

    if not 2 <= len(pieces) <= 4:
        raise ValueError("控制输入必须包含 2～4 片纸片")
    ids = [piece.piece_id for piece in pieces]
    if len(set(ids)) != len(ids):
        raise ValueError("piece_id 不能重复")
    for piece in pieces:
        cleaned = clean_vertices(piece.vertices_mm)
        if len(cleaned) != len(piece.vertices_mm):
            raise ValueError(f"{piece.piece_id}: 顶点不能重复或重复首尾点")


def plan_sheet_moves(
    pieces: Sequence[PuzzlePiece],
    options: PlanningOptions = PlanningOptions(),
    *,
    rigid_backend: RigidFitBackend | None = None,
    clearance_backend: ClearanceScanBackend | None = None,
    candidate_backend: CandidateScanBackend | None = None,
) -> MovePlan:
    """同步计算纸片拼成矩形所需的完整移动方案。

    调用方只传入强类型内存对象并接收强类型方案。内部求解器使用
    自动清理的临时文件，不访问视觉、相机或硬件资源。
    """

    if isinstance(pieces, (str, bytes)) or not isinstance(pieces, Sequence):
        raise TypeError("pieces 必须是 PuzzlePiece 序列")
    if not isinstance(options, PlanningOptions):
        raise TypeError("options 必须是 PlanningOptions")
    typed_pieces = tuple(pieces)
    if any(not isinstance(piece, PuzzlePiece) for piece in typed_pieces):
        raise TypeError("pieces 中的每一项都必须是 PuzzlePiece")
    if options.mode == "fixed" and len(typed_pieces) != 4:
        raise ValueError("fixed 模式必须恰好输入 4 片纸片")

    payload = {
        "schema_version": "1.0",
        "mode": options.mode,
        "noise_bound_mm": options.noise_bound_mm,
        "search_top_n": options.search_top_n,
        "pieces": [
            {
                "id": piece.piece_id,
                "vertices_mm": [list(point) for point in piece.vertices_mm],
            }
            for piece in typed_pieces
        ],
    }
    # 保留公开接口对输入结构错误的 ValueError 语义；完整求解仍只使用
    # 新引擎，预处理不参与候选搜索或姿态计算。
    _validate_public_input(typed_pieces)
    if options.mode == 'random' and options.competition_rules:
        for piece in typed_pieces:
            vertices = np.asarray(piece.vertices_mm)
            if not 3 <= len(vertices) <= 5:
                raise PlanningError(f'{piece.piece_id}: 赛题要求每片 3～5 条边')
            lengths = np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1)
            # Each measured endpoint may deviate by noise_bound_mm.
            if lengths.min() + 2 * options.noise_bound_mm < 20.0 - 1e-6:
                raise PlanningError(f'{piece.piece_id}: 最短边不满足 20 mm（含端点观测误差）')
    preprocess_observations(payload)
    with TemporaryDirectory(prefix="ec_26_e_puzzle_") as temporary:
        root = Path(temporary)
        input_path = root / "observations.json"
        input_path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            with candidate_acceleration(candidate_backend):
                result = _solve_with_options(input_path, root, options, rigid_backend, clearance_backend)
        finally:
            # 求解器创建文件日志处理器；临时目录删除前必须显式关闭。
            logger = logging.getLogger("solver2")
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
                handler.close()

    logging.getLogger(__name__).info("SOLVER_PERFORMANCE %s", json.dumps(result.get("performance_profile", {}), sort_keys=True))
    if not result.get("success"):
        raise PlanningError(str(result.get("failure_reason", "未找到完整拼图方案")))
    if not result.get("executable") or not result.get("competition_gate_passed"):
        raise PlanningError(f"候选解未通过执行检查：{result.get('execution_block_reasons', [])}")
    if len(result.get("moves", [])) != len(typed_pieces):
        raise PlanningError("候选解未覆盖全部纸片")
    target = result.get("target_rectangle")
    if not isinstance(target, dict):
        raise PlanningError("求解器返回了无效的目标矩形")
    return MovePlan(
        mode=options.mode,
        confidence=float(result["confidence"]),
        ambiguous=bool(result["ambiguous"]),
        target_width_mm=float(target["width_mm"]),
        target_height_mm=float(target["height_mm"]),
        target_center_a4_mm=_point(target["center_a4_mm"]),
        moves=tuple(_to_move(move) for move in result["moves"]),
    )


def _solve_with_options(input_path, root, options, rigid_backend, clearance_backend):
    return solve_observations(
        input_path,
        root / "solution",
        print_plan=False,
        rigid_backend=rigid_backend,
        clearance_backend=clearance_backend,
        runtime_config={
            "execution_profile": "production",
            "unlimited_search": True,
            "top_n": options.search_top_n,
            # 几何复刻不要求图案唯一，但必须通过几何及机械检查。
            "execution_policy": "strict",
            "minimum_execution_confidence": options.minimum_execution_confidence,
            "random_competition_rules": options.competition_rules,
        },
    )
