"""批量生成、独立求解、评估并保留失败案例。"""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any
import numpy as np

from ..generator.export import generate_case
from ..solver.solve import solve_observations
from ..profiling.profile_report import portable_profile
from ..runtime_config import RuntimeConfig, resolve_runtime_config
from .evaluator import evaluate_solution
from .failure import write_failure_classification

EXHAUSTIVE_DIAGNOSTIC_SEEDS = {2000, 2001, 2008, 2013, 2014, 2016, 2018}


def finalize_legacy_benchmark_artifacts(output: str | Path) -> None:
    """为旧版已完成 summary 补齐可由原始逐案数据恢复的标准文件。

    旧版未记录的搜索/内存字段明确写 unavailable，不反向猜测。
    """
    root = Path(output)
    summary_path = root / "benchmark_summary.json"
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = data.get("cases_detail", [])
    times = [float(row.get("elapsed_seconds", 0.0)) for row in rows]
    data["runtime_seconds"] = {
        "mean": float(np.mean(times)) if times else 0.0,
        "median": float(np.median(times)) if times else 0.0,
        "p50": float(np.percentile(times, 50)) if times else 0.0,
        "p90": float(np.percentile(times, 90)) if times else 0.0,
        "p95": float(np.percentile(times, 95)) if times else 0.0,
        "p99": float(np.percentile(times, 99)) if times else 0.0,
        "max": max(times, default=0.0),
    }
    data["legacy_missing_metrics"] = [
        "per_case_stage_times", "generated_states", "expanded_states",
        "duplicate_pruned", "rectangle_pruned", "overlap_pruned",
        "cycle_pruned", "interface_conflict_pruned", "peak_memory_mb",
    ]
    summary_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = [
        "seed", "topology", "generator_mode", "success", "solver_success",
        "geometry_success", "competition_white_success",
        "strict_truth_success", "elapsed_seconds", "failure_type", "reason",
    ]
    with (root / "benchmark_cases.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)
    with (root / "runtime_distribution.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["seed", "elapsed_seconds"]
        )
        writer.writeheader()
        writer.writerows({
            "seed": row.get("seed"),
            "elapsed_seconds": row.get("elapsed_seconds"),
        } for row in rows)
    (root / "search_distribution.csv").write_text(
        "status,reason\r\nunavailable,legacy benchmark did not persist per-case search diagnostics\r\n",
        encoding="utf-8-sig",
    )
    (root / "failed_seeds.json").write_text(
        json.dumps({
            "solver_failed_seeds": data.get("failed_seeds", []),
            "strict_failed_seeds": data.get("strict_failed_seeds", []),
            "missing_metrics": data["legacy_missing_metrics"],
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_exhaustive_log(case_dir: Path, evaluation: dict, seed: int) -> None:
    truth = json.loads((case_dir / "truth" / "ground_truth.json").read_text(encoding="utf-8"))
    candidates_path = case_dir / "solution" / "candidate_scores.json"
    candidates = json.loads(candidates_path.read_text(encoding="utf-8")) if candidates_path.exists() else []
    from .oracle_diagnosis import _truth_edge_matches
    correct_edges = _truth_edge_matches(truth)
    observed_by_id = {
        piece["public_id"]: piece["observed_vertices_mm"] for piece in truth["pieces"]
    }
    correct_edges_in_candidate_set = True
    edge_length_errors = []
    for first, edge_a, second, edge_b in correct_edges:
        import numpy as np
        a, b = np.asarray(observed_by_id[first]), np.asarray(observed_by_id[second])
        la = float(np.linalg.norm(a[(edge_a + 1) % len(a)] - a[edge_a]))
        lb = float(np.linalg.norm(b[(edge_b + 1) % len(b)] - b[edge_b]))
        edge_length_errors.append(abs(la - lb))
        correct_edges_in_candidate_set &= abs(la - lb) <= 1e-6
    retained_correct = False
    correct_set = {
        frozenset(((a, int(ea)), (b, int(eb)))) for a, ea, b, eb in correct_edges
    }
    correct_score = None
    for candidate in candidates:
        candidate_set = {
            frozenset(((item["piece_a"], int(item["edge_a"])), (item["piece_b"], int(item["edge_b"]))))
            for item in candidate.get("matched_edges", [])
        }
        if candidate_set.issubset(correct_set):
            retained_correct = True
            correct_score = candidate.get("total_score")
            break
    best_score = candidates[0].get("total_score") if candidates else None
    lines = [
        f"seed={seed}",
        f"correct_edges_entered_candidate_set={str(correct_edges_in_candidate_set).lower()}",
        f"correct_edge_max_length_error_mm={max(edge_length_errors) if edge_length_errors else 0.0:.9g}",
        "correct_adjacency_graph_enumerated=true",
        "beam_search_used=false",
        f"correct_candidate_retained={str(retained_correct).lower()}",
        f"correct_candidate_score={correct_score}",
        f"best_candidate_score={best_score}",
        f"legal_complete_candidate_count={candidates[0].get('legal_complete_candidate_count', 0) if candidates else 0}",
        f"geometry_success={str(bool(evaluation.get('geometry_success'))).lower()}",
        f"strict_truth_success={str(bool(evaluation.get('passes_competition_gate'))).lower()}",
        "true_geometric_multiple_solutions="
        + str(bool(evaluation.get("geometry_success") and not evaluation.get("passes_competition_gate"))).lower(),
        "pruning_stage=none" if candidates else "pruning_stage=hard_geometry_validation",
    ]
    (case_dir / "solution" / "exhaustive_search.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark(
    mode: str,
    cases: int,
    noise_mm: float,
    seed: int,
    output: str | Path,
    piece_count: int | None = None,
    noise_model: str = "disk_bounded_vertex",
    generation_mode: str = "mixed",
    runtime_config: RuntimeConfig | dict[str, Any] | str | Path | None = None,
    surface_mode: str = "white",
) -> dict:
    config = resolve_runtime_config(runtime_config)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(cases):
        solution: dict[str, Any] = {}
        case_seed = seed + index
        case_dir = root / f"{mode}_seed_{case_seed:04d}"
        started = perf_counter()
        try:
            generate_case(
                mode, case_seed, noise_mm, case_dir, piece_count,
                noise_model=noise_model, generation_mode=generation_mode,
                render=config.execution_profile == "debug",
                surface_mode=surface_mode,
            )
            solution = solve_observations(
                case_dir / "public" / "observations.json", case_dir / "solution",
                print_plan=False,
                runtime_config=config,
                surface_mode=surface_mode,
            )
            evaluation = evaluate_solution(case_dir / "truth" / "ground_truth.json", case_dir / "solution" / "solution.json")
            truth = json.loads((case_dir / "truth" / "ground_truth.json").read_text(encoding="utf-8"))
            topology = truth.get("topology", "fixed")
            partition_metadata = truth.get("partition_metadata", {})
            target_area = float(
                truth["target_rectangle"]["width_mm"]
                * truth["target_rectangle"]["height_mm"]
            )
            classification = write_failure_classification(
                case_dir, solution, evaluation, target_area
            )
            if mode == "random" and noise_mm == 0 and (
                case_seed in EXHAUSTIVE_DIAGNOSTIC_SEEDS
                or not evaluation.get("passes_competition_gate")
            ):
                _write_exhaustive_log(case_dir, evaluation, case_seed)
            solver_success = bool(solution.get("success"))
            failure_type = classification["primary_failure_type"]
            strict_success = bool(
                solver_success and evaluation.get("strict_truth_success")
            )
            operational_success = bool(
                solver_success and evaluation.get("competition_white_success")
            )
            geometry_success = bool(solver_success and evaluation.get("geometry_success"))
            competition_white_success = bool(
                solver_success and evaluation.get("competition_white_success")
            )
            strict_truth_success = bool(
                solver_success and evaluation.get("strict_truth_success")
            )
            success = solver_success
            reason = None if strict_success else evaluation.get("reason", "未达到严格推荐几何阈值")
        except Exception as exc:
            success, solver_success, strict_success, operational_success, geometry_success, competition_white_success, strict_truth_success, reason, evaluation, topology, partition_metadata, failure_type = (
                False, False, False, False, False, False, False, str(exc), {}, "unknown", {}, "no_complete_search_candidate"
            )
        elapsed = perf_counter() - started
        rows.append(
            {
                "seed": case_seed,
                "topology": topology,
                "generator_mode": partition_metadata.get("generator_mode", topology),
                "adjacency_canonical_signature": partition_metadata.get(
                    "adjacency_canonical_signature"
                ),
                "has_t_junction": partition_metadata.get("has_t_junction"),
                "has_multiple_internal_nodes": partition_metadata.get(
                    "has_multiple_internal_nodes"
                ),
                "partition_metadata": partition_metadata,
                "success": success,
                "solver_success": solver_success,
                "operational_thresholds_passed": operational_success,
                "geometry_success": geometry_success,
                "competition_white_success": competition_white_success,
                "strict_truth_success": strict_truth_success,
                "equivalent_global_rotation": bool(
                    evaluation.get("equivalent_global_rotation")
                ),
                "positive_length_adjacency_connected": partition_metadata.get(
                    "positive_length_adjacency_connected", True
                ),
                "positive_length_component_count": partition_metadata.get(
                    "positive_length_component_count", 1
                ),
                "point_contact_count": partition_metadata.get(
                    "point_contact_count", 0
                ),
                "one_to_many_candidate_count": (
                    solution.get("search_diagnostics", {}).get(
                        "one_to_many_candidate_count", 0
                    )
                    if solver_success else 0
                ),
                "failure_type": failure_type,
                "executable": bool(
                    solution.get("executable", False)
                    if "solution" in locals() else False
                ),
                "search_budget_limited": bool(
                    solution.get("search_budget_limited", False)
                    if "solution" in locals() else False
                ),
                "ambiguous": bool(
                    solution.get("ambiguous", False)
                    if "solution" in locals() else True
                ),
                "search_nodes": int(
                    solution.get("search_nodes", 0)
                    if "solution" in locals() else 0
                ),
                "search_diagnostics": (
                    solution.get("search_diagnostics", {})
                    if "solution" in locals() else {}
                ),
                "performance_profile": (
                    solution.get("performance_profile", {})
                    if "solution" in locals() else {}
                ),
                "strict_recommended_thresholds_passed": strict_success,
                "elapsed_seconds": elapsed,
                "reason": reason,
                "evaluation": evaluation,
            }
        )
    times = [row["elapsed_seconds"] for row in rows]
    def percentile(values: list[float], quantile: float) -> float:
        return float(np.percentile(values, quantile)) if values else 0.0

    peak_memories = [
        float(
            row.get("performance_profile", {})
            .get("memory", {})
            .get("peak_memory_mb", 0.0)
        )
        for row in rows
    ]
    executable_rows = [row for row in rows if row["executable"]]
    stage_names = sorted({
        name
        for row in rows
        for name in row.get("performance_profile", {})
        .get("stage_times", {})
    })
    result = {
        "mode": mode,
        "noise_model": noise_model,
        "generation_mode": generation_mode,
        "cases": cases,
        "success_count": sum(row["success"] for row in rows),
        "failure_count": sum(not row["success"] for row in rows),
        "success_rate": sum(row["success"] for row in rows) / cases if cases else 0.0,
        "operational_pass_count": sum(row["operational_thresholds_passed"] for row in rows),
        "operational_pass_rate": sum(row["operational_thresholds_passed"] for row in rows) / cases if cases else 0.0,
        "strict_pass_count": sum(row["strict_recommended_thresholds_passed"] for row in rows),
        "geometry_success_count": sum(row["geometry_success"] for row in rows),
        "geometry_success_rate": sum(row["geometry_success"] for row in rows) / cases if cases else 0.0,
        "competition_white_success_count": sum(
            row["competition_white_success"] for row in rows
        ),
        "competition_white_success_rate": sum(
            row["competition_white_success"] for row in rows
        ) / cases if cases else 0.0,
        "strict_truth_success_count": sum(
            row["strict_truth_success"] for row in rows
        ),
        "strict_truth_success_rate": sum(
            row["strict_truth_success"] for row in rows
        ) / cases if cases else 0.0,
        "positive_edge_connected_success_rate": (
            sum(
                row["competition_white_success"]
                for row in rows
                if row["positive_length_adjacency_connected"]
            )
            / sum(
                row["positive_length_adjacency_connected"] for row in rows
            )
            if any(row["positive_length_adjacency_connected"] for row in rows)
            else None
        ),
        "point_contact_challenge_success_rate": (
            sum(
                row["competition_white_success"]
                for row in rows
                if not row["positive_length_adjacency_connected"]
            )
            / sum(
                not row["positive_length_adjacency_connected"] for row in rows
            )
            if any(
                not row["positive_length_adjacency_connected"] for row in rows
            )
            else None
        ),
        "t_junction_success_rate": (
            sum(
                row["competition_white_success"]
                for row in rows if row.get("has_t_junction")
            )
            / sum(bool(row.get("has_t_junction")) for row in rows)
            if any(row.get("has_t_junction") for row in rows)
            else None
        ),
        "one_to_many_interface_success_rate": (
            sum(
                row["competition_white_success"]
                for row in rows
                if row["one_to_many_candidate_count"] > 0
            )
            / sum(row["one_to_many_candidate_count"] > 0 for row in rows)
            if any(row["one_to_many_candidate_count"] > 0 for row in rows)
            else None
        ),
        "equivalent_global_rotation_count": sum(
            row["equivalent_global_rotation"] for row in rows
        ),
        "no_complete_search_candidate_count": sum(
            row["failure_type"] == "no_complete_search_candidate"
            for row in rows
        ),
        "strict_pass_rate": sum(row["strict_recommended_thresholds_passed"] for row in rows) / cases if cases else 0.0,
        "average_seconds": mean(times) if times else 0.0,
        "median_seconds": median(times) if times else 0.0,
        "slowest_seconds": max(times) if times else 0.0,
        "runtime_seconds": {
            "mean": mean(times) if times else 0.0,
            "median": median(times) if times else 0.0,
            "p50": percentile(times, 50),
            "p90": percentile(times, 90),
            "p95": percentile(times, 95),
            "p99": percentile(times, 99),
            "max": max(times, default=0.0),
        },
        "solver_completed_rate": (
            sum(row["solver_success"] for row in rows) / cases if cases else 0.0
        ),
        "executable_rate": (
            len(executable_rows) / cases if cases else 0.0
        ),
        "executable_true_actual_success_rate": (
            sum(row["competition_white_success"] for row in executable_rows)
            / len(executable_rows) if executable_rows else None
        ),
        "ambiguity_rate": (
            sum(row["ambiguous"] for row in rows) / cases if cases else 0.0
        ),
        "budget_limited_rate": (
            sum(row["search_budget_limited"] for row in rows) / cases
            if cases else 0.0
        ),
        "search_statistics": {
            name: {
                "mean": mean(values) if values else 0.0,
                "median": median(values) if values else 0.0,
                "p95": percentile(values, 95),
                "p99": percentile(values, 99),
                "max": max(values, default=0),
            }
            for name in (
                "search_nodes", "generated_states", "expanded_states",
                "candidate_count", "legal_complete_candidate_count",
                "duplicate_pruned", "rectangle_pruned", "overlap_pruned",
                "cycle_pruned", "interface_conflict_pruned",
                "pairwise_interface_placement_count",
                "sourced_atomic_interval_count",
            )
            if (
                values := [
                    float(
                        row.get("search_diagnostics", {}).get(
                            name, row.get(name, 0)
                        )
                    )
                    for row in rows
                ]
            )
        },
        "memory_mb": {
            "median_peak": median(peak_memories) if peak_memories else 0.0,
            "p95_peak": percentile(peak_memories, 95),
            "max_peak": max(peak_memories, default=0.0),
        },
        "stage_mean_seconds": {
            name: mean([
                float(
                    row.get("performance_profile", {})
                    .get("stage_times", {})
                    .get(name, {})
                    .get("seconds", 0.0)
                )
                for row in rows
            ])
            for name in stage_names
        },
        "fallback_trigger_rate": (
            sum(bool(row.get("search_diagnostics", {}).get(
                "generic_partial_fallback_used", False
            )) for row in rows) / cases if cases else 0.0
        ),
        "search_level_distribution": dict(Counter(
            str(row.get("search_diagnostics", {}).get(
                "search_level_reached", 1
            ))
            for row in rows
        )),
        "runtime_config": config.as_dict(),
        "failed_seeds": [row["seed"] for row in rows if not row["success"]],
        "strict_failed_seeds": [row["seed"] for row in rows if not row["strict_recommended_thresholds_passed"]],
        "cases_detail": rows,
    }
    if mode == "random":
        result["by_topology"] = {
            topology: {
                "cases": len(group),
                "competition_gate_rate": sum(row["operational_thresholds_passed"] for row in group) / len(group),
                "competition_white_success_rate": sum(
                    row["competition_white_success"] for row in group
                ) / len(group),
                "geometry_success_rate": sum(row["geometry_success"] for row in group) / len(group),
                "strict_truth_success_rate": sum(
                    row["strict_truth_success"] for row in group
                ) / len(group),
                "failed_seeds": [
                    row["seed"] for row in group
                    if not row["competition_white_success"]
                ],
            }
            for topology in sorted({row["topology"] for row in rows})
            if (group := [row for row in rows if row["topology"] == topology])
        }
        result["by_adjacency_signature"] = {
            signature: {
                "cases": len(group),
                "competition_gate_rate": sum(
                    row["operational_thresholds_passed"] for row in group
                ) / len(group),
                "competition_white_success_rate": sum(
                    row["competition_white_success"] for row in group
                ) / len(group),
                "geometry_success_rate": sum(
                    row["geometry_success"] for row in group
                ) / len(group),
                "strict_truth_success_rate": sum(
                    row["strict_truth_success"] for row in group
                ) / len(group),
            }
            for signature in sorted(
                {
                    row["adjacency_canonical_signature"]
                    for row in rows
                    if row["adjacency_canonical_signature"]
                }
            )
            if (
                group := [
                    row
                    for row in rows
                    if row["adjacency_canonical_signature"] == signature
                ]
            )
        }
        signatures = Counter(
            row["adjacency_canonical_signature"]
            for row in rows
            if row["adjacency_canonical_signature"]
        )
        interior_nodes = Counter(
            row.get("partition_metadata", {}).get("interior_node_count")
            for row in rows
            if row.get("partition_metadata")
        )
        piece_edges = Counter(
            edge_count
            for row in rows
            for edge_count in row.get("partition_metadata", {}).get(
                "piece_edge_counts", []
            )
        )
        result["partition_structure_summary"] = {
            "adjacency_signature_distribution": dict(sorted(signatures.items())),
            "interior_node_count_distribution": {
                str(key): value
                for key, value in sorted(interior_nodes.items())
            },
            "t_junction_count": sum(
                bool(row.get("partition_metadata", {}).get("has_t_junction"))
                for row in rows
            ),
            "multiple_internal_node_count": sum(
                bool(
                    row.get("partition_metadata", {}).get(
                        "has_multiple_internal_nodes"
                    )
                )
                for row in rows
            ),
            "piece_edge_count_distribution": {
                str(key): value
                for key, value in sorted(piece_edges.items())
            },
        }
    (root / "benchmark_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    case_fields = [
        "seed", "topology", "generator_mode", "success",
        "solver_success", "geometry_success", "competition_white_success",
        "strict_truth_success", "executable", "ambiguous",
        "search_budget_limited", "elapsed_seconds", "search_nodes",
        "failure_type", "reason",
    ]
    with (root / "benchmark_cases.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=case_fields)
        writer.writeheader()
        writer.writerows(
            {key: row.get(key) for key in case_fields} for row in rows
        )
    with (root / "runtime_distribution.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["seed", "elapsed_seconds", "peak_memory_mb"],
        )
        writer.writeheader()
        writer.writerows({
            "seed": row["seed"],
            "elapsed_seconds": row["elapsed_seconds"],
            "peak_memory_mb": (
                row.get("performance_profile", {})
                .get("memory", {}).get("peak_memory_mb", 0.0)
            ),
        } for row in rows)
    search_fields = [
        "seed", "search_nodes", "generated_states", "expanded_states",
        "candidate_count", "duplicate_pruned", "rectangle_pruned",
        "overlap_pruned", "cycle_pruned", "interface_conflict_pruned",
        "pairwise_interface_placement_count", "sourced_atomic_interval_count",
    ]
    with (root / "search_distribution.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=search_fields)
        writer.writeheader()
        for row in rows:
            diagnostics = row.get("search_diagnostics", {})
            writer.writerow({
                name: (
                    row["seed"] if name == "seed"
                    else diagnostics.get(name, row.get(name, 0))
                )
                for name in search_fields
            })
    (root / "failed_seeds.json").write_text(
        json.dumps({
            "solver_failed_seeds": result["failed_seeds"],
            "competition_failed_seeds": [
                row["seed"] for row in rows
                if not row["competition_white_success"]
            ],
            "strict_failed_seeds": result["strict_failed_seeds"],
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (root / "portable_profile.json").write_text(
        json.dumps(
            portable_profile(
                config.as_dict(),
                {
                    **result["runtime_seconds"],
                    "peak_memory_mb": result["memory_mb"]["max_peak"],
                },
            ),
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    return result
