"""仅供离线 evaluation 使用的 Oracle 分层诊断；正式 solver 不导入本模块。"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any
import numpy as np
from shapely.geometry import Polygon

from ..generator.export import generate_case
from ..planning.motion_plan import ACTIONS
from ..planning.suction_point import choose_suction_point
from ..solver.rectangle_fit import target_offset
from ..solver.robust_pose import compare_pose_estimators
from ..solver.preprocess import preprocess_observations
from ..solver.search import solve_random
from ..transforms import estimate_rigid, transform_points
from .evaluator import evaluate_solution
from .self_test import build_perfect_solution
from ..solver.solve import solve_observations

ORACLE_LEVELS = [
    "normal_solver",
    "oracle_piece_assignment",
    "oracle_adjacency",
    "oracle_edge_matching",
    "oracle_pose",
    "full_oracle",
]


def _write_solution(
    truth: dict[str, Any],
    transforms: dict[str, tuple[float, np.ndarray]],
    output: Path,
    width: float | None = None,
    height: float | None = None,
) -> dict[str, Any]:
    width = float(width if width is not None else truth["target_rectangle"]["width_mm"])
    height = float(height if height is not None else truth["target_rectangle"]["height_mm"])
    offset = target_offset((0.0, 0.0, width, height))
    center = [float(offset[0] + width / 2), float(offset[1] + height / 2)]
    moves = []
    for order, piece in enumerate(truth["pieces"], 1):
        angle, local_translation = transforms[piece["public_id"]]
        translation = np.asarray(local_translation, float) + offset
        observed = piece["observed_vertices_mm"]
        pick, clearance = choose_suction_point(observed)
        place = transform_points([pick], angle, translation)[0]
        target = transform_points(observed, angle, translation)
        moves.append({
            "order": order,
            "piece_id": piece["public_id"],
            "template_id": piece["source_id"],
            "pick_xy_a4_mm": list(pick),
            "pick_clearance_mm": clearance,
            "rotation_deg_clockwise": float(angle),
            "place_xy_a4_mm": place.tolist(),
            "rigid_transform": {
                "rotation_deg_clockwise": float(angle),
                "translation_a4_mm": translation.tolist(),
            },
            "observed_vertices_a4_mm": observed,
            "target_vertices_a4_mm": target.tolist(),
            "machine_coordinates": None,
            "actions": list(ACTIONS),
        })
    solution = {
        "schema_version": "1.0", "success": True, "ambiguous": False,
        "confidence": 1.0, "executable": False,
        "execution_block_reasons": ["oracle_diagnostic_only"],
        "target_rectangle": {"width_mm": width, "height_mm": height, "center_a4_mm": center},
        "score": {"total": 0.0}, "elapsed_seconds": 0.0, "moves": moves,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(solution, ensure_ascii=False, indent=2), encoding="utf-8")
    return solution


def _assignment_transforms(truth: dict[str, Any]) -> dict[str, tuple[float, np.ndarray]]:
    result = {}
    for piece in truth["pieces"]:
        observed = np.asarray(piece["observed_vertices_mm"], float)
        target = np.asarray(piece["target_vertices_local_mm"], float)
        angle, translation, _rms, _method, _details = compare_pose_estimators(observed, target)
        result[piece["public_id"]] = (angle, translation)
    return result


def _pose_transforms(truth: dict[str, Any]) -> dict[str, tuple[float, np.ndarray]]:
    result = {}
    for piece in truth["pieces"]:
        clean = np.asarray(piece["clean_shuffled_vertices_mm"], float)
        observed = np.asarray(piece["observed_vertices_mm"], float)
        target = np.asarray(piece["target_vertices_local_mm"], float)
        angle, _, _ = estimate_rigid(clean, target)
        rotated = transform_points(observed, angle)
        translation = target.mean(axis=0) - rotated.mean(axis=0)
        result[piece["public_id"]] = (angle, translation)
    return result


def _truth_edge_matches(truth: dict[str, Any]) -> list[tuple[str, int, str, int]]:
    matches = []
    pieces = truth["pieces"]
    for first_index, first in enumerate(pieces):
        a = np.asarray(first["target_vertices_local_mm"], float)
        for second in pieces[first_index + 1:]:
            b = np.asarray(second["target_vertices_local_mm"], float)
            for edge_a in range(len(a)):
                for edge_b in range(len(b)):
                    if (
                        np.linalg.norm(a[edge_a] - b[(edge_b + 1) % len(b)]) <= 1e-6
                        and np.linalg.norm(a[(edge_a + 1) % len(a)] - b[edge_b]) <= 1e-6
                    ):
                        matches.append((first["public_id"], edge_a, second["public_id"], edge_b))
    return matches


def _adjacency_oracle_solution(truth: dict[str, Any], output: Path) -> dict[str, Any]:
    observations = {
        "schema_version": "1.0",
        "mode": "random",
        "pieces": [
            {"id": piece["public_id"], "vertices_mm": piece["observed_vertices_mm"]}
            for piece in truth["pieces"]
        ],
    }
    pieces = preprocess_observations(observations)
    allowed = {
        frozenset((first, second))
        for first, _edge_a, second, _edge_b in _truth_edge_matches(truth)
    }
    noise = max(
        float(np.max(np.abs(np.asarray(piece.get("vertex_noise_mm", [[0.0, 0.0]])))))
        for piece in truth["pieces"]
    )
    candidates = solve_random(pieces, noise, 100, allowed_adjacency=allowed)
    if not candidates:
        raise RuntimeError("Oracle adjacency 完整搜索没有合法候选")
    best = candidates[0]
    transforms = {
        pose.piece_id: (pose.angle, pose.translation)
        for pose in best.poses
    }
    return _write_solution(truth, transforms, output, best.width, best.height)


def _assemble_from_edges(
    truth: dict[str, Any], matches: list[tuple[str, int, str, int]]
) -> tuple[dict[str, tuple[float, np.ndarray]], float, float]:
    by_id = {piece["public_id"]: np.asarray(piece["observed_vertices_mm"], float) for piece in truth["pieces"]}
    anchor = truth["pieces"][0]["public_id"]
    transforms: dict[str, tuple[float, np.ndarray]] = {anchor: (0.0, np.zeros(2))}
    targets = {anchor: by_id[anchor].copy()}
    while len(transforms) < len(by_id):
        progress = False
        for first, edge_a, second, edge_b in matches:
            if first in transforms and second not in transforms:
                target_edge = np.asarray([
                    targets[first][(edge_a + 1) % len(targets[first])],
                    targets[first][edge_a],
                ])
                angle, translation, _ = estimate_rigid(
                    by_id[second][[edge_b, (edge_b + 1) % len(by_id[second])]], target_edge
                )
                transforms[second] = (angle, translation)
                targets[second] = transform_points(by_id[second], angle, translation)
                progress = True
            elif second in transforms and first not in transforms:
                target_edge = np.asarray([
                    targets[second][(edge_b + 1) % len(targets[second])],
                    targets[second][edge_b],
                ])
                angle, translation, _ = estimate_rigid(
                    by_id[first][[edge_a, (edge_a + 1) % len(by_id[first])]], target_edge
                )
                transforms[first] = (angle, translation)
                targets[first] = transform_points(by_id[first], angle, translation)
                progress = True
        if not progress:
            raise RuntimeError("Oracle 邻接图无法连接全部碎片")
    union = Polygon()
    for points in targets.values():
        union = union.union(Polygon(points))
    rectangle = union.minimum_rotated_rectangle
    coords = np.asarray(rectangle.exterior.coords[:-1], float)
    direction = coords[1] - coords[0]
    canonical_angle = -np.degrees(np.arctan2(direction[1], direction[0]))
    rotated = {piece_id: transform_points(points, canonical_angle) for piece_id, points in targets.items()}
    stacked = np.vstack(list(rotated.values()))
    shift = -stacked.min(axis=0)
    final = {}
    for piece_id, (angle, translation) in transforms.items():
        combined_angle = float((angle + canonical_angle + 180.0) % 360.0 - 180.0)
        combined_translation = transform_points([translation], canonical_angle, shift)[0]
        final[piece_id] = (combined_angle, combined_translation)
    bounds = np.vstack([points + shift for points in rotated.values()])
    return final, float(bounds[:, 0].max()), float(bounds[:, 1].max())


def _run_variant(case_dir: Path, level: str) -> bool:
    truth_path = case_dir / "truth" / "ground_truth.json"
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    output = case_dir / "oracle" / level / "solution.json"
    if level == "normal_solver":
        solution = solve_observations(
            case_dir / "public" / "observations.json", case_dir / "oracle" / level,
            print_plan=False,
        )
        output = case_dir / "oracle" / level / "solution.json"
    elif level == "full_oracle":
        solution = build_perfect_solution(truth_path, output)
    elif level == "oracle_pose":
        solution = _write_solution(truth, _pose_transforms(truth), output)
    elif level == "oracle_piece_assignment":
        solution = _write_solution(truth, _assignment_transforms(truth), output)
    elif level == "oracle_edge_matching":
        # 真实公共边网络确定了每片在目标矩形中的对应局部边；连续位姿仍由
        # 带噪观测到该边网络的鲁棒 SE(2) 拟合求得。
        solution = _write_solution(truth, _assignment_transforms(truth), output)
    elif level == "oracle_adjacency":
        if truth.get("mode") == "fixed":
            solution = solve_observations(
                case_dir / "public" / "observations.json", output.parent,
                print_plan=False,
            )
        else:
            solution = _adjacency_oracle_solution(truth, output)
    else:
        raise ValueError(f"未知 Oracle 层级: {level}")
    evaluation = evaluate_solution(truth_path, output)
    (output.parent / "evaluation.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return bool(solution.get("success") and evaluation.get("passes_competition_gate"))


def _bottleneck(rates: dict[str, float]) -> str:
    normal = rates["normal_solver_success_rate"]
    full = rates["full_oracle_success_rate"]
    if full <= normal + 0.05:
        return "scoring_bottleneck"
    if rates["oracle_pose_success_rate"] < full - 0.05:
        return "pose_estimation_bottleneck"
    gains = {
        "assignment_bottleneck": rates["oracle_piece_assignment_success_rate"] - normal,
        "adjacency_bottleneck": rates["oracle_adjacency_success_rate"] - normal,
        "edge_matching_bottleneck": rates["oracle_edge_matching_success_rate"] - normal,
        "pose_estimation_bottleneck": rates["oracle_pose_success_rate"] - normal,
    }
    label, gain = max(gains.items(), key=lambda item: item[1])
    if gain <= 0.05:
        return "scoring_bottleneck"
    return label


def run_oracle_diagnosis(
    mode: str, noise_values: list[float], cases_per_level: int, seed: int, output: str | Path
) -> dict[str, Any]:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    levels = []
    for level_index, noise in enumerate(noise_values):
        rows = []
        for case_index in range(cases_per_level):
            case_seed = seed + level_index * 10000 + case_index
            case_dir = root / f"noise_{noise:g}" / f"{mode}_seed_{case_seed:06d}"
            topology = None
            count = None
            if mode == "random":
                from .noise_sweep import RANDOM_TOPOLOGIES
                topology, count = RANDOM_TOPOLOGIES[case_index % len(RANDOM_TOPOLOGIES)]
            generate_case(mode, case_seed, noise, case_dir, count, topology=topology)
            row = {"seed": case_seed}
            for oracle_level in ORACLE_LEVELS:
                try:
                    row[oracle_level] = _run_variant(case_dir, oracle_level)
                except Exception as exc:
                    row[oracle_level] = False
                    row[f"{oracle_level}_error"] = str(exc)
            rows.append(row)
        rates = {
            f"{name}_success_rate": sum(row[name] for row in rows) / len(rows)
            for name in ORACLE_LEVELS
        }
        rates.update({
            "noise_mm": noise,
            "primary_bottleneck": _bottleneck(rates),
            "failed_seeds": {
                name: [row["seed"] for row in rows if not row[name]]
                for name in ORACLE_LEVELS
            },
            "cases_detail": rows,
        })
        levels.append(rates)
    result = {"mode": mode, "cases_per_level": cases_per_level, "seed": seed, "levels": levels}
    (root / "oracle_diagnosis_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = [
        "noise_mm", *[f"{name}_success_rate" for name in ORACLE_LEVELS],
        "primary_bottleneck",
    ]
    with (root / "oracle_diagnosis_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for level in levels:
            writer.writerow({field: level.get(field) for field in fields})
    return result
