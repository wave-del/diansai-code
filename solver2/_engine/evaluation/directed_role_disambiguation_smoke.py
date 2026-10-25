"""非镜像碎片角色交换的小规模集成验证（最多 6 次真实 solver）。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolygonPatch
import numpy as np

from ..generator.export import generate_case
from ..runtime_config import RuntimeConfig
from ..solver.directed_roles import (
    fit_rectangle_frame,
    physically_equivalent_permutation,
)
from ..solver.observation_uncertainty import ObservationUncertaintyProfile
from ..solver.solve import solve_observations
from .evaluator import evaluate_solution


def _profile(sigma: float = 0.5) -> ObservationUncertaintyProfile:
    return ObservationUncertaintyProfile(
        3.0, sigma, None, None, 1.0, "vision", "high", True
    )


def _render(solution: dict[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(figsize=(7.2, 5.4))
    for index, move in enumerate(solution.get("moves", [])):
        vertices = move.get(
            "planned_final_vertices_a4_mm",
            move.get("target_vertices_a4_mm", []),
        )
        if not vertices:
            continue
        axes.add_patch(PolygonPatch(
            vertices, closed=True, alpha=0.45,
            color=plt.get_cmap("tab10")(index),
            label=str(move.get("piece_id", index)),
        ))
    axes.set_aspect("equal")
    axes.autoscale()
    axes.grid(alpha=0.2)
    if solution.get("moves"):
        axes.legend(loc="best")
    axes.set_title(
        "directed-role result | "
        f"competition={solution.get('competition_gate_passed')}"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _selected_admission(
    case: Path, solution: dict[str, Any]
) -> dict[str, Any]:
    path = case / "candidate_scores.json"
    if not path.exists():
        return (
            solution.get("search_diagnostics", {})
            .get("selected_geometry_admissibility")
            or {}
        )
    rows = json.loads(path.read_text(encoding="utf-8"))
    selected = next(
        (row for row in rows if row.get("selected_after_optimization")),
        rows[0] if rows else {},
    )
    return (
        selected.get("geometry_admissibility")
        or solution.get("search_diagnostics", {}).get(
            "selected_geometry_admissibility"
        )
        or {}
    )


def _artificial_checks() -> dict[str, Any]:
    parallelogram = {
        "left": np.asarray([[0, 0], [50, 0], [60, 60], [10, 60]], float),
        "right": np.asarray([[50, 0], [100, 0], [110, 60], [60, 60]], float),
    }
    frame, _roles, consistency = fit_rectangle_frame(
        parallelogram,
        _profile(0.3),
        excluded_piece_edges={("left", 1), ("right", 3)},
    )
    near_right = {
        "all": np.asarray([[0, 0], [100, 0], [101, 60], [0, 60]], float)
    }
    near_frame, _, near_consistency = fit_rectangle_frame(
        near_right, _profile(0.8)
    )
    first = {
        "left": np.asarray([[0, 0], [50, 0], [50, 60], [0, 60]], float),
        "right": np.asarray(
            [[50, 0], [100, 0], [100, 60], [50, 60]], float
        ),
    }
    swapped = {"x": first["right"], "y": first["left"]}
    return {
        "parallelogram_rejected": not frame.valid,
        "parallelogram_like": consistency.parallelogram_like,
        "parallelogram_reasons": list(consistency.reasons),
        "near_90_rectangle_passed": (
            near_frame.valid and near_consistency.valid
        ),
        "near_90_orthogonality_error_deg": (
            near_frame.orthogonality_error_deg
        ),
        "white_physical_permutation_equivalent": (
            physically_equivalent_permutation(first, swapped)
        ),
    }


def run_directed_role_disambiguation_smoke(
    output: str | Path = "outputs/directed_role_disambiguation_smoke",
    *,
    solve_cases: bool = True,
) -> dict[str, Any]:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    specs = [
        ("seed_23105", 23105, 4, "template", "difficult_similar", "white"),
        ("seed_23106", 23106, 4, "recursive", None, "white"),
        ("three_t", 23102, 3, "template", "three_y", "white"),
        ("four_center", 23103, 4, "template", "four_fan", "white"),
        ("two_polyline", 23101, 2, "template", "two_polyline", "white"),
        (
            "patterned_similar", 23105, 4, "template",
            "difficult_similar", "patterned",
        ),
    ]
    rows: list[dict[str, Any]] = []
    config = RuntimeConfig(
        execution_profile="production",
        diagnostics_level="summary",
        failure_diagnostics_level="full",
        execution_policy="strict",
    )
    for name, seed, count, generation_mode, topology, surface_mode in specs:
        case = root / name
        if solve_cases:
            generate_case(
                "random", seed, 0.0, case,
                piece_count=count,
                generation_mode=generation_mode,
                topology=topology,
                surface_mode=surface_mode,
                render=False,
            )
            solution = solve_observations(
                case / "public/observations.json",
                case,
                print_plan=False,
                runtime_config=config,
            )
        else:
            solution = json.loads(
                (case / "solution.json").read_text(encoding="utf-8")
            )
        evaluation = evaluate_solution(
            case / "truth/ground_truth.json", case / "solution.json"
        )
        (case / "evaluation.json").write_text(
            json.dumps(evaluation, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _render(solution, case / "final_layout.png")
        admission = _selected_admission(case, solution)
        frame = admission.get("rectangle_frame_fit") or {}
        role = admission.get("rectangle_role_consistency") or {}
        junctions = admission.get("junction_closure") or []
        opposite_rows = [
            interface.get("opposite_side_result")
            for interface in (solution.get("selected_interfaces") or [])
            if interface.get("opposite_side_result")
        ]
        row = {
            "case": name,
            "seed": seed,
            "surface_mode": surface_mode,
            "success": bool(solution.get("success")),
            "solver_competition_gate": bool(
                solution.get("competition_gate_passed")
            ),
            "evaluator_competition_gate": bool(
                evaluation.get("passes_competition_gate")
            ),
            "geometry_success": bool(evaluation.get("geometry_success")),
            "topology_correct": bool(evaluation.get("topology_correct")),
            "piece_correspondence_accuracy": evaluation.get(
                "piece_correspondence_accuracy"
            ),
            "adjacency_accuracy": evaluation.get("adjacency_accuracy"),
            "rectangle_iou": evaluation.get("union_rectangle_iou"),
            "max_vertex_error_mm": evaluation.get(
                "common_vertex_max_distance_mm"
            ),
            "directed_endpoint_context_valid": admission.get(
                "directed_endpoint_context_valid"
            ),
            "opposite_side_valid": admission.get("opposite_side_valid"),
            "rectangle_frame_fit_valid": admission.get(
                "rectangle_frame_fit_valid"
            ),
            "rectangle_role_consistency_valid": admission.get(
                "rectangle_role_consistency_valid"
            ),
            "orthogonality_error_deg": frame.get(
                "orthogonality_error_deg"
            ),
            "parallel_error_deg": frame.get(
                "opposite_side_parallel_error_deg"
            ),
            "support_line_rms_mm": frame.get("support_line_rms_mm"),
            "outer_side_coverage_ratio": role.get("side_coverage_ratio"),
            "junction_closure_valid": admission.get(
                "junction_closure_valid"
            ),
            "junction_count": len(junctions),
            "same_side_interface_count": sum(
                row and row.get("reason") == "same_side_interface"
                for row in opposite_rows
            ),
            "elapsed_seconds": solution.get("elapsed_seconds"),
            "termination_reason": (
                solution.get("search_diagnostics", {})
                .get("termination_reason")
            ),
        }
        rows.append(row)

    with (root / "cases.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "actual_solver_run_count": len(rows),
        "actual_solver_run_limit": 15,
        "cases": rows,
        "artificial_checks": _artificial_checks(),
        "large_benchmark_not_run": True,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    unresolved = [
        row for row in rows
        if (
            row["surface_mode"] == "white"
            and not row["evaluator_competition_gate"]
        )
    ]
    lines = ["# Directed role disambiguation unresolved cases", ""]
    for row in unresolved:
        lines.extend([
            f"## {row['case']}",
            "",
            f"- topology_correct: `{row['topology_correct']}`",
            f"- adjacency_accuracy: `{row['adjacency_accuracy']}`",
            f"- rectangle_iou: `{row['rectangle_iou']}`",
            f"- image: [final_layout.png]({row['case']}/final_layout.png)",
            "",
        ])
    if not unresolved:
        lines.append("全部 white 小规模案例通过 evaluator competition gate。")
    (root / "unresolved_cases.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(
        run_directed_role_disambiguation_smoke(),
        ensure_ascii=False,
        indent=2,
    ))
