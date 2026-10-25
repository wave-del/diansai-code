"""Small fixed-seed overlap/clearance regression; intentionally capped at 20."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from ..generator.export import generate_case
from ..runtime_config import RuntimeConfig
from ..solver.solve import solve_observations
from .evaluator import evaluate_solution


CASE_SPECS = (
    {
        "name": "ordinary_two_piece",
        "seed": 31001,
        "piece_count": 2,
        "topology": "two_polyline",
        "generation_mode": "template",
    },
    {
        "name": "three_piece_t",
        "seed": 31002,
        "piece_count": 3,
        "topology": "three_y",
        "generation_mode": "template",
    },
    {
        "name": "four_piece_center",
        "seed": 31003,
        "piece_count": 4,
        "topology": "four_fan",
        "generation_mode": "template",
    },
    {
        "name": "historical_13010",
        "seed": 13010,
        "piece_count": None,
        "topology": None,
        "generation_mode": "mixed",
    },
    {
        "name": "historical_13060",
        "seed": 13060,
        "piece_count": None,
        "topology": None,
        "generation_mode": "mixed",
    },
)


def _clean_geometry_signature(truth: dict[str, Any]) -> str:
    payload = {
        "target_rectangle": truth["target_rectangle"],
        "pieces": [
            {
                "public_id": piece["public_id"],
                "source_id": piece["source_id"],
                "target": piece["target_vertices_local_mm"],
                "clean": piece["clean_shuffled_vertices_mm"],
            }
            for piece in truth["pieces"]
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def run_overlap_clearance_smoke(
    output: str | Path = "outputs/overlap_clearance_smoke",
    *,
    reuse_existing: bool = False,
) -> dict[str, Any]:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    runtime = RuntimeConfig(
        execution_profile="benchmark",
        diagnostics_level="summary",
        failure_diagnostics_level="full",
        worker_count=1,
        parallel_mode="off",
    )
    rows: list[dict[str, Any]] = []
    signatures: dict[str, list[str]] = {}

    for spec in CASE_SPECS:
        for radius in (0.0, 1.0, 2.0, 3.0):
            case_root = root / spec["name"] / f"R_{int(radius)}"
            solution_path = case_root / "solution" / "solution.json"
            if not (reuse_existing and solution_path.exists()):
                generate_case(
                    "random",
                    spec["seed"],
                    radius,
                    case_root,
                    piece_count=spec["piece_count"],
                    topology=spec["topology"],
                    generation_mode=spec["generation_mode"],
                    noise_model="disk_bounded_vertex",
                    render=False,
                )
            truth_path = case_root / "truth" / "ground_truth.json"
            truth = json.loads(truth_path.read_text(encoding="utf-8"))
            signatures.setdefault(spec["name"], []).append(
                _clean_geometry_signature(truth)
            )
            if reuse_existing and solution_path.exists():
                solution = json.loads(
                    solution_path.read_text(encoding="utf-8")
                )
                elapsed = float(
                    solution.get("elapsed_seconds", 0.0)
                )
            else:
                started = perf_counter()
                solution = solve_observations(
                    case_root / "public" / "observations.json",
                    case_root / "solution",
                    print_plan=False,
                    runtime_config=runtime,
                )
                elapsed = perf_counter() - started
            evaluation = (
                evaluate_solution(
                    truth_path,
                    solution_path,
                )
                if solution.get("success")
                else {}
            )
            actual_overlap = evaluation.get(
                "ground_truth_actual_overlap_area_mm2"
            )
            max_vertex = evaluation.get(
                "max_corresponding_vertex_distance_mm"
            )
            failure_reasons = list(
                solution.get("execution_block_reasons", [])
            )
            if actual_overlap is not None and actual_overlap > 1e-5:
                failure_reasons.append("ground_truth_actual_overlap")
            if max_vertex is not None and max_vertex > 20.0:
                failure_reasons.append("maximum_vertex_distance_exceeded")
            if (
                solution.get("success")
                and not evaluation.get("competition_white_success", False)
            ):
                failure_reasons.append("competition_geometry_failed")
            row = {
                "case": spec["name"],
                "seed": spec["seed"],
                "radius_mm": radius,
                "solver_completed": bool(solution.get("success")),
                "competition_white_success": bool(
                    evaluation.get("competition_white_success", False)
                ),
                "ground_truth_actual_overlap_area_mm2": actual_overlap,
                "estimated_physical_overlap_area_mm2": solution.get(
                    "estimated_physical_overlap_area_mm2"
                ),
                "robust_buffer_overlap_area_mm2": solution.get(
                    "robust_buffer_overlap_area_mm2"
                ),
                "max_corresponding_vertex_distance_mm": max_vertex,
                "planned_clearance_mm": solution.get(
                    "planned_clearance_mm"
                ),
                "minimum_pairwise_clearance_mm": solution.get(
                    "minimum_pairwise_clearance_mm"
                ),
                "clearance_feasible": solution.get(
                    "clearance_feasible", False
                ),
                "executable": solution.get("executable", False),
                "solve_time_seconds": elapsed,
                "failure_reason": ",".join(sorted(set(failure_reasons))),
            }
            rows.append(row)

            if failure_reasons and not reuse_existing:
                diagnostic_runtime = RuntimeConfig(
                    execution_profile="debug",
                    diagnostics_level="full",
                    failure_diagnostics_level="full",
                    worker_count=1,
                    parallel_mode="off",
                )
                solve_observations(
                    case_root / "public" / "observations.json",
                    case_root / "failure_diagnostics",
                    print_plan=False,
                    runtime_config=diagnostic_runtime,
                )

    geometry_reused = {
        name: len(set(items)) == 1
        for name, items in signatures.items()
    }
    summary = {
        "case_count": len(rows),
        "noise_model": "disk_bounded_vertex",
        "radii_mm": [0.0, 1.0, 2.0, 3.0],
        "clean_geometry_reused_across_radii": geometry_reused,
        "all_clean_geometry_reused": all(geometry_reused.values()),
        "solver_completed_count": sum(
            row["solver_completed"] for row in rows
        ),
        "competition_success_count": sum(
            row["competition_white_success"] for row in rows
        ),
        "actual_overlap_case_count": sum(
            (row["ground_truth_actual_overlap_area_mm2"] or 0.0) > 1e-5
            for row in rows
        ),
        "vertex_distance_exceeded_count": sum(
            (row["max_corresponding_vertex_distance_mm"] or 0.0) > 20.0
            for row in rows
        ),
        "executable_count": sum(row["executable"] for row in rows),
        "rows": rows,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fields = list(rows[0])
    with (root / "cases.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return summary


if __name__ == "__main__":
    print(json.dumps(
        run_overlap_clearance_smoke(),
        ensure_ascii=False,
        indent=2,
    ))
