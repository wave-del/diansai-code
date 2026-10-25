"""Bounded fixed-mode smoke for variable samples on the large triangle."""

from __future__ import annotations

import csv
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..generator.export import generate_case
from ..planning.motion_plan import build_moves
from ..runtime_config import RuntimeConfig
from ..solver.execution_gate import final_candidate_revalidation
from ..solver.solve import solve_observations
from .evaluator import evaluate_solution


CASES = {
    "case_3_vertices": [],
    "case_4_vertices_one_sample": [0.50],
    "case_5_vertices_original": [0.70, 0.20],
    "case_5_vertices_shifted_symmetric": [0.90, 0.10],
    "case_5_vertices_shifted_asymmetric": [0.85, 0.50],
    "case_6_vertices_multiple_samples": [0.88, 0.62, 0.25],
    "case_samples_slightly_off_line": [0.70, 0.20],
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _replace_triangle_samples(
    observations: dict[str, Any],
    truth: dict[str, Any],
    parameters: list[float],
    *,
    off_line_mm: float = 0.0,
) -> str:
    source = next(
        item for item in truth["pieces"]
        if item["source_id"] == "fixed_piece_3"
    )
    public_id = source["public_id"]
    clean = np.asarray(source["clean_shuffled_vertices_mm"], dtype=float)
    assert len(clean) == 3
    start, end = clean[0], clean[2]
    direction = end - start
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    normal = np.asarray([-direction[1], direction[0]], dtype=float)
    samples = []
    for index, parameter in enumerate(parameters):
        point = start + float(parameter) * (end - start)
        if off_line_mm:
            point = point + normal * off_line_mm * (
                1.0 if index % 2 == 0 else -1.0
            )
        samples.append(point.tolist())
    public_piece = next(
        item for item in observations["pieces"]
        if item["id"] == public_id
    )
    public_piece["vertices_mm"] = [
        *clean.tolist(),
        *samples,
    ]
    return public_id


def _mismatch_diagnostic() -> dict[str, Any]:
    record = {
        "piece_id": "synthetic",
        "observed": np.asarray([[10, 10], [30, 10], [20, 30]], float),
        "angle": 0.0,
        "translation": np.asarray([40, 160], float),
        "target": np.asarray([[51, 170], [71, 170], [61, 190]], float),
        "virtual_assembly_anchors": [],
    }
    moves = build_moves([record], 1.0)
    reasons = final_candidate_revalidation(
        "fixed",
        [record],
        moves,
        config=RuntimeConfig(),
        recovery_outcome=None,
        search_diagnostics={},
    )
    return {
        "case": "case_target_transform_mismatch",
        "success": False,
        "motion_plan_generated": False,
        "final_transform_closure": False,
        "rejection_reasons": reasons,
    }


def run_fixed_physical_contour_anchor_smoke(
    output: str | Path = "outputs/fixed_physical_contour_anchor_smoke",
) -> dict[str, Any]:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    config = RuntimeConfig(
        execution_profile="production",
        diagnostics_level="summary",
        transport_collision_mode="safe_z_clearance",
    )
    rows: list[dict[str, Any]] = []
    for index, (name, parameters) in enumerate(CASES.items()):
        case = root / name
        generate_case("fixed", 4100 + index, 0.0, case, render=False)
        observations = _load(case / "public/observations.json")
        truth = _load(case / "truth/ground_truth.json")
        public_id = _replace_triangle_samples(
            observations,
            truth,
            parameters,
            off_line_mm=(0.4 if "off_line" in name else 0.0),
        )
        if "off_line" in name:
            observations["measurement_error_radius_mm"] = 0.5
            observations["noise_bound_mm"] = 0.5
        _dump(case / "public/observations.json", observations)
        solution = solve_observations(
            case / "public/observations.json",
            case,
            print_plan=False,
            runtime_config=config,
        )
        evaluation = evaluate_solution(
            case / "truth/ground_truth.json",
            case / "solution.json",
        )
        _dump(case / "evaluation.json", evaluation)
        move = next(
            (
                item for item in solution.get("moves", [])
                if item["piece_id"] == public_id
            ),
            {},
        )
        diagnostics = move.get("physical_contour_diagnostics", {})
        anchors = move.get("virtual_assembly_anchors", [])
        row = {
            "case": name,
            "raw_vertex_count": 3 + len(parameters),
            "structural_corner_count": diagnostics.get(
                "structural_corner_count"
            ),
            "physical_edge_count": diagnostics.get(
                "structural_corner_count"
            ),
            "removed_edge_sample_count": diagnostics.get(
                "removed_collinear_sample_count"
            ),
            "selected_template_piece": move.get("template_id"),
            "rotation_error_deg": evaluation.get(
                "strict_rotation_error_deg_mean"
            ),
            "translation_error_mm": evaluation.get(
                "strict_translation_error_mm_mean"
            ),
            "physical_corner_max_error_mm": evaluation.get(
                "strict_corresponding_vertex_max_distance_mm"
            ),
            "support_line_rms_mm": diagnostics.get(
                "maximum_support_line_rms_mm"
            ),
            "virtual_anchor_count": len(anchors),
            "optimization_before_rms_mm": (
                solution.get("optimization") or {}
            ).get("before_rms_mm"),
            "optimization_after_rms_mm": (
                solution.get("optimization") or {}
            ).get("after_rms_mm"),
            "final_transform_closure": (
                solution.get("final_candidate_revalidation", {})
                .get("passed", False)
            ),
            "motion_plan_generated": bool(solution.get("moves")),
            "competition_gate_passed": solution.get(
                "competition_gate_passed"
            ),
            "evaluator_competition_gate": evaluation.get(
                "passes_competition_gate"
            ),
            "elapsed_seconds": solution.get("elapsed_seconds"),
        }
        rows.append(row)
    mismatch = _mismatch_diagnostic()
    mismatch_dir = root / mismatch["case"]
    mismatch_dir.mkdir(parents=True, exist_ok=True)
    _dump(mismatch_dir / "mismatch_diagnostic.json", mismatch)

    with (root / "cases.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    unresolved = [
        row for row in rows
        if not row["evaluator_competition_gate"]
    ]
    summary = {
        "solver_run_count": len(rows),
        "solver_run_limit": 16,
        "cases": rows,
        "target_transform_mismatch": mismatch,
        "all_variable_sample_cases_solved": not unresolved,
        "large_benchmark_not_run": True,
    }
    _dump(root / "summary.json", summary)
    lines = ["# Fixed physical contour unresolved cases", ""]
    if unresolved:
        for row in unresolved:
            lines.extend([
                f"## {row['case']}",
                "",
                f"- evaluator competition: `{row['evaluator_competition_gate']}`",
                f"- closure: `{row['final_transform_closure']}`",
                "",
            ])
    else:
        lines.append("All seven variable-sample cases passed.")
    (root / "unresolved_cases.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(
        run_fixed_physical_contour_anchor_smoke(),
        ensure_ascii=False,
        indent=2,
    ))
