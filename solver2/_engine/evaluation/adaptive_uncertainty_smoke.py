"""Small, bounded smoke matrix for the unified random search path."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from ..generator.export import generate_case
from ..runtime_config import RuntimeConfig
from ..solver.solve import solve_observations
from .evaluator import evaluate_solution


def _nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def run_adaptive_uncertainty_smoke(
    output: str | Path = "outputs/adaptive_uncertainty_smoke",
    *,
    seeds: tuple[int, ...] = (17001, 17002, 17003, 13010, 13060),
    noise_values: tuple[float, ...] = (0.0, 1.0, 2.0),
) -> dict[str, Any]:
    """Run at most 15 solves and hide the generator noise label from solver."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    config = RuntimeConfig(
        execution_profile="benchmark",
        diagnostics_level="summary",
    )
    for seed in seeds:
        for noise in noise_values:
            case = root / f"seed_{seed}_noise_{noise:g}"
            generate_case(
                "random",
                seed,
                noise,
                case,
                noise_model="disk_bounded_vertex",
                generation_mode="mixed",
                render=False,
            )
            public_path = case / "public" / "observations.json"
            public = json.loads(public_path.read_text(encoding="utf-8"))
            public.pop("noise_bound_mm", None)
            public.pop("noise_mm", None)
            hidden_path = case / "solver_observations.json"
            hidden_path.write_text(
                json.dumps(public, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            solution_dir = case / "solution"
            result = solve_observations(
                hidden_path,
                solution_dir,
                print_plan=False,
                runtime_config=config,
            )
            evaluation = evaluate_solution(
                case / "truth" / "ground_truth.json",
                solution_dir / "solution.json",
            )
            search = result.get("search_diagnostics", {})
            recovery = result.get("recovery", {})
            stage_runs = search.get("stage_runs", [])
            rows.append({
                "seed": seed,
                "noise_mm_evaluator_only": noise,
                "success": bool(result.get("success")),
                "executable": bool(result.get("executable")),
                "competition_gate": bool(
                    evaluation.get("passes_competition_gate")
                ),
                "strict_truth": bool(
                    evaluation.get("passes_strict_exact")
                ),
                "elapsed_seconds": float(
                    result.get("elapsed_seconds", 0.0)
                ),
                "search_stage": (
                    stage_runs[-1].get("stage")
                    if stage_runs else None
                ),
                "hard_interface_pool_count": max(
                    (
                        int(item.get("hard_interface_pool_count", 0))
                        for item in stage_runs
                    ),
                    default=0,
                ),
                "opened_interface_count": max(
                    (
                        int(item.get("opened_interface_count", 0))
                        for item in stage_runs
                    ),
                    default=0,
                ),
                "candidate_count": int(
                    search.get("candidate_count", 0)
                ),
                "optimized_candidate_count": int(
                    search.get("optimized_candidate_count", 0)
                ),
                "anchor_attempt_count": int(
                    search.get("anchor_attempt_count", 0)
                ),
                "candidate_cache_hits": int(
                    search.get("candidate_cache_hits", 0)
                ),
                "termination_reason": (
                    search.get("termination_reason")
                    or recovery.get("termination_reason")
                ),
                "overlap_area_mm2": float(
                    evaluation.get(
                        "estimated_physical_overlap_area_mm2", 0.0
                    )
                ),
                "robust_max_vertex_distance_mm": float(
                    _nested(
                        result,
                        "execution_gate",
                        "robust_max_vertex_distance_mm",
                        default=result.get(
                            "robust_max_vertex_distance_mm", 0.0
                        ),
                    )
                ),
            })
    by_noise: list[dict[str, Any]] = []
    for noise in noise_values:
        group = [
            row for row in rows
            if row["noise_mm_evaluator_only"] == noise
        ]
        by_noise.append({
            "noise_mm": noise,
            "cases": len(group),
            "solver_success_rate": mean(
                row["success"] for row in group
            ),
            "executable_rate": mean(
                row["executable"] for row in group
            ),
            "competition_gate_rate": mean(
                row["competition_gate"] for row in group
            ),
            "strict_truth_rate": mean(
                row["strict_truth"] for row in group
            ),
            "mean_elapsed_seconds": mean(
                row["elapsed_seconds"] for row in group
            ),
            "mean_candidate_count": mean(
                row["candidate_count"] for row in group
            ),
            "mean_optimized_candidate_count": mean(
                row["optimized_candidate_count"] for row in group
            ),
        })
    summary = {
        "solver_noise_label_hidden": True,
        "algorithm_entry_identical_for_all_noise_levels": True,
        "total_solves": len(rows),
        "by_noise": by_noise,
        "failed_cases": [
            {
                "seed": row["seed"],
                "noise_mm": row["noise_mm_evaluator_only"],
                "termination_reason": row["termination_reason"],
            }
            for row in rows if not row["executable"]
        ],
    }
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (root / "cases.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    unresolved = [
        "# Adaptive uncertainty unresolved cases",
        "",
        "The solver input copies omit generator noise labels.",
        "",
    ]
    unresolved.extend(
        f"- seed {item['seed']}, noise {item['noise_mm']}: "
        f"{item['termination_reason']}"
        for item in summary["failed_cases"]
    )
    if not summary["failed_cases"]:
        unresolved.append("- None.")
    (root / "unresolved_cases.md").write_text(
        "\n".join(unresolved) + "\n", encoding="utf-8"
    )
    return summary

