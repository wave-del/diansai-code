"""Bounded integration smoke for fully-placed geometry admission."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolygonPatch

from ..generator.export import generate_case
from ..runtime_config import RuntimeConfig
from ..solver.solve import solve_observations
from .evaluator import evaluate_solution


def _render(solution: dict[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(figsize=(7, 5))
    for index, move in enumerate(solution.get("moves", [])):
        vertices = move.get(
            "planned_final_vertices_a4_mm",
            move.get("target_vertices_a4_mm", []),
        )
        if vertices:
            axes.add_patch(PolygonPatch(
                vertices,
                closed=True,
                alpha=0.38,
                color=plt.get_cmap("tab10")(index),
                label=move.get("piece_id", f"P{index}"),
            ))
    axes.set_aspect("equal")
    axes.autoscale()
    axes.grid(alpha=0.2)
    if solution.get("moves"):
        axes.legend(loc="best")
    axes.set_title(
        f"geometry candidate | executable={solution.get('executable')}"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run_geometry_candidate_admission_smoke(
    output: str | Path = "outputs/geometry_candidate_admission_smoke",
) -> dict[str, Any]:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    specs = [
        {
            "name": "seed_12010", "seed": 12010, "piece_count": 4,
            "generation_mode": "mixed", "topology": None,
        },
        {
            "name": "two_polyline", "seed": 23101, "piece_count": 2,
            "generation_mode": "template", "topology": "two_polyline",
        },
        {
            "name": "three_t", "seed": 23102, "piece_count": 3,
            "generation_mode": "template", "topology": "three_y",
        },
        {
            "name": "four_center", "seed": 23103, "piece_count": 4,
            "generation_mode": "template", "topology": "four_fan",
        },
        {
            "name": "guillotine", "seed": 23104, "piece_count": 4,
            "generation_mode": "template", "topology": "guillotine",
        },
        {
            "name": "similar_shapes", "seed": 23105, "piece_count": 4,
            "generation_mode": "template", "topology": "difficult_similar",
        },
        {
            "name": "recursive", "seed": 23106, "piece_count": 4,
            "generation_mode": "recursive", "topology": None,
        },
    ]
    rows: list[dict[str, Any]] = []
    for spec in specs:
        case = root / str(spec["name"])
        if not (case / "public/observations.json").exists():
            generate_case(
                "random",
                int(spec["seed"]),
                0.0,
                case,
                piece_count=int(spec["piece_count"]),
                topology=spec["topology"],
                generation_mode=str(spec["generation_mode"]),
                render=False,
            )
        solution = solve_observations(
            case / "public/observations.json",
            case,
            print_plan=False,
            runtime_config=RuntimeConfig(
                execution_profile="production",
                diagnostics_level="summary",
                failure_diagnostics_level="full",
                execution_policy="strict",
            ),
        )
        evaluation = evaluate_solution(
            case / "truth/ground_truth.json", case / "solution.json"
        )
        (case / "evaluation.json").write_text(
            json.dumps(evaluation, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _render(solution, case / "final_layout.png")
        diagnostics = solution.get("search_diagnostics", {})
        statistics = diagnostics.get("search_statistics", {})
        selected_admission = diagnostics.get(
            "selected_geometry_admissibility"
        )
        candidate_path = case / "candidate_scores.json"
        if candidate_path.exists():
            candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
            selected = next(
                (
                    item for item in candidates
                    if item.get("selected_after_optimization")
                ),
                candidates[0] if candidates else None,
            )
            selected_admission = (
                selected.get("geometry_admissibility") if selected else None
            )
        row = {
            "case": spec["name"],
            "seed": spec["seed"],
            "generation_mode": spec["generation_mode"],
            "topology": spec["topology"] or "auto",
            "success": solution.get("success"),
            "executable": solution.get("executable"),
            "competition_gate_passed": solution.get(
                "competition_gate_passed"
            ),
            "evaluator_geometry_success": evaluation.get("geometry_success"),
            "evaluator_competition_pass": evaluation.get(
                "passes_competition_gate"
            ),
            "fully_placed_state_count": statistics.get(
                "fully_placed_state_count",
                diagnostics.get("fully_placed_state_count", 0),
            ),
            "fully_placed_rejected_count": statistics.get(
                "fully_placed_rejected_count",
                diagnostics.get("fully_placed_rejected_count", 0),
            ),
            "geometry_candidate_count": statistics.get(
                "geometry_candidate_count",
                diagnostics.get("geometry_candidate_count", 0),
            ),
            "geometry_candidate_pool_size": statistics.get(
                "geometry_candidate_pool_size",
                diagnostics.get("geometry_candidate_pool_size", 0),
            ),
            "pool_ever_full": statistics.get(
                "pool_ever_full",
                diagnostics.get("candidate_pool_ever_full", False),
            ),
            "search_continued_after_pool_full": statistics.get(
                "search_continued_after_pool_full",
                diagnostics.get("search_continued_after_pool_full", False),
            ),
            "optimized_candidate_count": solution.get(
                "optimized_candidate_count", 0
            ),
            "rectangle_fill_ratio": (
                selected_admission or {}
            ).get("rectangle_fill_ratio"),
            "estimated_rectangle_iou": (
                selected_admission or {}
            ).get("union_rectangle_iou_estimate"),
            "overlap_area_mm2": (
                selected_admission or {}
            ).get("overlap_area_mm2"),
            "outside_area_mm2": (
                selected_admission or {}
            ).get("outside_area_mm2"),
            "uncovered_area_mm2": (
                selected_admission or {}
            ).get("uncovered_area_mm2"),
            "declared_interface_count": (
                selected_admission or {}
            ).get("declared_interface_count"),
            "discovered_interface_count": (
                selected_admission or {}
            ).get("discovered_interface_count"),
            "interface_explanation_complete": (
                selected_admission or {}
            ).get("interface_explanation_complete"),
            "elapsed_seconds": solution.get("elapsed_seconds"),
            "failure_type": solution.get("failure_type"),
            "rejection_reasons": json.dumps(
                diagnostics.get(
                    "geometry_candidate_rejection_reasons", {}
                ),
                ensure_ascii=False,
                sort_keys=True,
            ),
        }
        rows.append(row)

    baseline = {
        "selected_candidate": "candidate_025:raw",
        "retained_fully_placed_candidates": 26,
        "piece_correspondence_accuracy": 0.50,
        "union_rectangle_iou": 0.6305,
        "estimated_overlap_area_mm2": 134.190841,
        "common_vertex_max_distance_mm": 98.44,
        "source": "previous seed 12010 offline diagnosis",
    }
    summary = {
        "case_count": len(rows),
        "actual_solver_case_count": len(rows),
        "actual_solver_run_limit": 16,
        "seed_12010_before": baseline,
        "seed_12010_after": next(
            row for row in rows if row["case"] == "seed_12010"
        ),
        "success_count": sum(bool(row["success"]) for row in rows),
        "competition_pass_count": sum(
            bool(row["evaluator_competition_pass"]) for row in rows
        ),
        "cases": rows,
        "large_benchmark_not_run": True,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (root / "cases.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    unresolved = [row for row in rows if not row["evaluator_competition_pass"]]
    lines = ["# Geometry candidate admission unresolved cases", ""]
    if not unresolved:
        lines.append("All bounded smoke cases passed the evaluator competition gate.")
    else:
        for row in unresolved:
            lines.extend([
                f"## {row['case']}",
                "",
                f"- failure_type: `{row['failure_type']}`",
                f"- rejection_reasons: `{row['rejection_reasons']}`",
                f"- image: [final_layout.png]({row['case']}/final_layout.png)",
                "",
            ])
    (root / "unresolved_cases.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(
        run_geometry_candidate_admission_smoke(),
        ensure_ascii=False,
        indent=2,
    ))
