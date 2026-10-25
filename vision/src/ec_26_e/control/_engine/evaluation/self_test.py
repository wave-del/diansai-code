"""使用 ground truth 构造完美答案，验证 evaluator 本身。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..generator.export import generate_case
from ..planning.motion_plan import ACTIONS
from ..planning.suction_point import choose_suction_point
from ..solver.rectangle_fit import target_offset
from ..transforms import estimate_rigid, transform_points
from .evaluator import evaluate_solution


def build_perfect_solution(truth_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    truth = json.loads(Path(truth_path).read_text(encoding="utf-8"))
    width = float(truth["target_rectangle"]["width_mm"])
    height = float(truth["target_rectangle"]["height_mm"])
    offset = target_offset((0.0, 0.0, width, height))
    center = [float(offset[0] + width / 2.0), float(offset[1] + height / 2.0)]
    moves = []
    for order, piece in enumerate(truth["pieces"], 1):
        clean = piece["clean_shuffled_vertices_mm"]
        observed = piece["observed_vertices_mm"]
        expected_clean = [
            (float(x + offset[0]), float(y + offset[1]))
            for x, y in piece["target_vertices_local_mm"]
        ]
        angle, translation, rms = estimate_rigid(clean, expected_clean)
        if rms > 1e-7:
            raise AssertionError(f"truth 刚体闭环残差异常：{piece['public_id']} rms={rms}")
        pick, clearance = choose_suction_point(observed)
        place = transform_points([pick], angle, translation)[0]
        target_observed = transform_points(observed, angle, translation)
        moves.append(
            {
                "order": order,
                "piece_id": piece["public_id"],
                "template_id": piece["source_id"],
                "pick_xy_a4_mm": list(pick),
                "pick_clearance_mm": clearance,
                "rotation_deg_clockwise": angle,
                "place_xy_a4_mm": [float(place[0]), float(place[1])],
                "rigid_transform": {
                    "rotation_deg_clockwise": angle,
                    "translation_a4_mm": [float(translation[0]), float(translation[1])],
                },
                "observed_vertices_a4_mm": observed,
                "target_vertices_a4_mm": target_observed.tolist(),
                "machine_coordinates": None,
                "actions": list(ACTIONS),
            }
        )
    solution = {
        "schema_version": "1.0",
        "success": True,
        "ambiguous": False,
        "confidence": 1.0,
        "coordinate_system": {
            "unit": "mm",
            "origin": "a4_top_left",
            "x_direction": "right",
            "y_direction": "down",
            "positive_rotation": "clockwise",
        },
        "target_rectangle": {
            "width_mm": width,
            "height_mm": height,
            "center_a4_mm": center,
        },
        "score": {"total": 0.0},
        "elapsed_seconds": 0.0,
        "moves": moves,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(solution, ensure_ascii=False, indent=2), encoding="utf-8")
    return solution


def run_evaluator_self_test(mode: str, cases: int, seed: int, output: str | Path) -> dict[str, Any]:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    tolerance = 1e-4
    for index in range(cases):
        case_seed = seed + index
        case_dir = root / f"{mode}_seed_{case_seed:04d}"
        generate_case(mode, case_seed, 5.0, case_dir)
        perfect_path = case_dir / "perfect_solution.json"
        build_perfect_solution(case_dir / "truth" / "ground_truth.json", perfect_path)
        evaluation = evaluate_solution(case_dir / "truth" / "ground_truth.json", perfect_path)
        checks = {
            "iou": abs(1.0 - evaluation["union_rectangle_iou"]) <= tolerance,
            "overlap": evaluation["overlap_area_mm2"] <= tolerance,
            "gap": evaluation["gap_area_mm2"] <= tolerance,
            "rotation": (evaluation["rotation_error_deg_mean"] or 0.0) <= tolerance,
            "translation": (evaluation["translation_error_mm_mean"] or 0.0) <= tolerance,
            "adjacency": abs(1.0 - evaluation["adjacency_accuracy"]) <= tolerance,
            "correspondence": abs(1.0 - evaluation["piece_correspondence_accuracy"]) <= tolerance,
            "competition_gate": bool(evaluation["passes_competition_gate"]),
        }
        rows.append(
            {
                "seed": case_seed,
                "success": all(checks.values()),
                "checks": checks,
                "evaluation": evaluation,
            }
        )
    summary = {
        "mode": mode,
        "cases": cases,
        "passed": sum(row["success"] for row in rows),
        "failed": sum(not row["success"] for row in rows),
        "success_rate": sum(row["success"] for row in rows) / cases if cases else 0.0,
        "failed_seeds": [row["seed"] for row in rows if not row["success"]],
        "cases_detail": rows,
    }
    (root / "evaluator_self_test_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
