"""按噪声等级运行生成、求解、评估和拓扑分组统计。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any
from ..runtime_config import RuntimeConfig, resolve_runtime_config

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..generator.export import generate_case
from ..solver.solve import solve_observations
from .evaluator import evaluate_solution
from .failure import write_failure_classification

RANDOM_TOPOLOGIES: list[tuple[str, int]] = [
    ("two_polyline", 2),
    ("three_y", 3),
    ("four_fan", 4),
    ("guillotine", 4),
    ("nonparallel", 4),
    ("difficult_similar", 4),
]


def _safe_mean(values: list[float | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    return mean(filtered) if filtered else None


def _safe_median(values: list[float | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    return median(filtered) if filtered else None


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    evaluations = [row["evaluation"] for row in rows if row["evaluation"].get("success")]
    solve_times = [row["solve_time"] for row in rows if row["solve_time"] is not None]
    positive_edge_rows = [
        row for row in rows
        if row.get("positive_length_adjacency_connected", True)
    ]
    point_contact_rows = [
        row for row in rows
        if not row.get("positive_length_adjacency_connected", True)
    ]
    t_rows = [row for row in rows if row.get("has_t_junction")]
    one_to_many_rows = [
        row for row in rows if row.get("one_to_many_candidate_count", 0) > 0
    ]
    executable_rows = [row for row in rows if row.get("executable", False)]
    return {
        "total_cases": total,
        "solver_completed_rate": sum(row["solver_completed"] for row in rows) / total if total else 0.0,
        "executable_rate": sum(row.get("executable", False) for row in rows) / total if total else 0.0,
        "executable_true_actual_success_rate": (
            sum(
                row.get("competition_white_success", False)
                for row in executable_rows
            ) / len(executable_rows)
            if executable_rows else None
        ),
        "competition_gate_rate": sum(row["competition_gate"] for row in rows) / total if total else 0.0,
        "competition_white_success_rate": sum(
            row.get("competition_white_success", False) for row in rows
        ) / total if total else 0.0,
        "geometry_success_rate": sum(row.get("geometry_success", False) for row in rows) / total if total else 0.0,
        "strict_exact_rate": sum(row["strict_exact"] for row in rows) / total if total else 0.0,
        "strict_truth_success_rate": sum(
            row.get("strict_truth_success", False) for row in rows
        ) / total if total else 0.0,
        "positive_edge_connected_success_rate": (
            sum(row.get("competition_white_success", False) for row in positive_edge_rows)
            / len(positive_edge_rows)
            if positive_edge_rows else None
        ),
        "point_contact_challenge_success_rate": (
            sum(row.get("competition_white_success", False) for row in point_contact_rows)
            / len(point_contact_rows)
            if point_contact_rows else None
        ),
        "t_junction_success_rate": (
            sum(row.get("competition_white_success", False) for row in t_rows)
            / len(t_rows)
            if t_rows else None
        ),
        "one_to_many_interface_success_rate": (
            sum(row.get("competition_white_success", False) for row in one_to_many_rows)
            / len(one_to_many_rows)
            if one_to_many_rows else None
        ),
        "equivalent_global_rotation_count": sum(
            row.get("equivalent_global_rotation", False) for row in rows
        ),
        "no_complete_search_candidate_count": sum(
            row.get("failure_type") == "no_complete_search_candidate"
            for row in rows
        ),
        "tolerance_2mm_connected_rate": sum(e.get("connected_with_tolerance_2mm", False) for e in evaluations) / total if total else 0.0,
        "tolerance_5mm_connected_rate": sum(e.get("connected_with_tolerance_5mm", False) for e in evaluations) / total if total else 0.0,
        "tolerance_10mm_connected_rate": sum(e.get("connected_with_tolerance_10mm", False) for e in evaluations) / total if total else 0.0,
        "exact_connected_rate": sum(e.get("exact_connected", False) for e in evaluations) / total if total else 0.0,
        "mean_iou": _safe_mean([e.get("union_rectangle_iou") for e in evaluations]),
        "median_iou": _safe_median([e.get("union_rectangle_iou") for e in evaluations]),
        "mean_overlap_ratio": _safe_mean([row.get("overlap_ratio") for row in rows]),
        "mean_gap_ratio": _safe_mean([row.get("gap_ratio") for row in rows]),
        "max_vertex_error_mean": _safe_mean([e.get("corresponding_vertex_max_distance_mm") for e in evaluations]),
        "rotation_error_mean": _safe_mean([e.get("rotation_error_deg_mean") for e in evaluations]),
        "translation_error_mean": _safe_mean([e.get("translation_error_mm_mean") for e in evaluations]),
        "mean_solve_time": _safe_mean(solve_times),
        "median_solve_time": _safe_median(solve_times),
        "max_solve_time": max(solve_times) if solve_times else None,
        "failed_seeds": [
            row["seed"]
            for row in rows
            if not row.get("competition_white_success", False)
        ],
        "optimization_before_rms_mean": _safe_mean([row.get("optimization_before_rms") for row in rows]),
        "optimization_after_rms_mean": _safe_mean([row.get("optimization_after_rms") for row in rows]),
    }


def run_noise_sweep(
    mode: str,
    noise_values: list[float],
    cases_per_level: int,
    seed: int,
    output: str | Path,
    noise_model: str = "disk_bounded_vertex",
    generation_mode: str = "mixed",
    runtime_config: RuntimeConfig | dict[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    config = resolve_runtime_config(runtime_config)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    levels = []
    for level_index, noise in enumerate(noise_values):
        rows = []
        level_dir = root / f"noise_{noise:g}"
        for case_index in range(cases_per_level):
            # 各噪声等级复用同一批底层几何，避免把样本难度变化误当作噪声影响。
            case_seed = seed + case_index
            topology = None
            piece_count = None
            if mode == "random" and generation_mode == "template":
                topology, piece_count = RANDOM_TOPOLOGIES[case_index % len(RANDOM_TOPOLOGIES)]
            case_dir = level_dir / f"{mode}_seed_{case_seed:06d}"
            try:
                generate_case(
                    mode, case_seed, noise, case_dir, piece_count,
                    topology=topology, noise_model=noise_model,
                    generation_mode=generation_mode,
                    render=config.execution_profile == "debug",
                )
                solution = solve_observations(
                    case_dir / "public" / "observations.json", case_dir / "solution",
                    print_plan=False,
                    runtime_config=config,
                )
                evaluation = evaluate_solution(
                    case_dir / "truth" / "ground_truth.json",
                    case_dir / "solution" / "solution.json",
                )
                truth = json.loads((case_dir / "truth" / "ground_truth.json").read_text(encoding="utf-8"))
                partition_metadata = truth.get("partition_metadata", {})
                area = float(truth["target_rectangle"]["width_mm"] * truth["target_rectangle"]["height_mm"])
                classification = write_failure_classification(case_dir, solution, evaluation, area)
                candidate_path = case_dir / "solution" / "candidate_scores.json"
                candidates = json.loads(candidate_path.read_text(encoding="utf-8")) if candidate_path.exists() else []
                optimization = candidates[0].get("optimization") if candidates else None
                rows.append(
                    {
                        "seed": case_seed,
                        "topology": truth.get("topology", "fixed"),
                        "generator_mode": partition_metadata.get(
                            "generator_mode", "fixed"
                        ),
                        "adjacency_canonical_signature": partition_metadata.get(
                            "adjacency_canonical_signature"
                        ),
                        "solver_completed": bool(solution.get("success")),
                        "executable": bool(solution.get("executable")),
                        "competition_gate": bool(evaluation.get("passes_competition_gate")),
                        "competition_white_success": bool(
                            evaluation.get("competition_white_success")
                        ),
                        "geometry_success": bool(evaluation.get("geometry_success")),
                        "strict_exact": bool(evaluation.get("passes_strict_exact")),
                        "strict_truth_success": bool(
                            evaluation.get("strict_truth_success")
                        ),
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
                        "has_t_junction": partition_metadata.get(
                            "has_t_junction", False
                        ),
                        "one_to_many_candidate_count": solution.get(
                            "search_diagnostics", {}
                        ).get("one_to_many_candidate_count", 0),
                        "solve_time": solution.get("elapsed_seconds"),
                        "overlap_ratio": evaluation.get("overlap_area_mm2", 0.0) / area,
                        "gap_ratio": evaluation.get("gap_area_mm2", 0.0) / area,
                        "optimization_before_rms": optimization.get("before_rms_mm") if optimization else None,
                        "optimization_after_rms": optimization.get("after_rms_mm") if optimization else None,
                        "failure_type": classification["primary_failure_type"],
                        "evaluation": evaluation,
                    }
                )
            except Exception as exc:
                failed_topology = topology or (
                    "fixed" if mode == "fixed" else "unknown"
                )
                failed_generator_mode = (
                    "fixed" if mode == "fixed" else generation_mode
                )
                failed_signature = None
                truth_path = case_dir / "truth" / "ground_truth.json"
                if truth_path.exists():
                    failed_truth = json.loads(
                        truth_path.read_text(encoding="utf-8")
                    )
                    failed_metadata = failed_truth.get(
                        "partition_metadata", {}
                    )
                    failed_topology = failed_truth.get(
                        "topology", failed_topology
                    )
                    failed_generator_mode = failed_metadata.get(
                        "generator_mode", failed_generator_mode
                    )
                    failed_signature = failed_metadata.get(
                        "adjacency_canonical_signature"
                    )
                rows.append(
                    {
                        "seed": case_seed,
                        "topology": failed_topology,
                        "generator_mode": failed_generator_mode,
                        "adjacency_canonical_signature": failed_signature,
                        "solver_completed": False, "executable": False,
                        "competition_gate": False,
                        "competition_white_success": False,
                        "geometry_success": False, "strict_exact": False,
                        "strict_truth_success": False,
                        "equivalent_global_rotation": False,
                        "positive_length_adjacency_connected": (
                            failed_metadata.get(
                                "positive_length_adjacency_connected", True
                            )
                            if truth_path.exists() else True
                        ),
                        "positive_length_component_count": (
                            failed_metadata.get(
                                "positive_length_component_count", 1
                            )
                            if truth_path.exists() else 1
                        ),
                        "point_contact_count": 0,
                        "has_t_junction": False,
                        "one_to_many_candidate_count": 0,
                        "solve_time": None, "overlap_ratio": None, "gap_ratio": None,
                        "optimization_before_rms": None, "optimization_after_rms": None,
                        "failure_type": "no_complete_search_candidate",
                        "error": str(exc), "evaluation": {},
                    }
                )
        summary = _summarize(rows)
        summary["noise_mm"] = noise
        if mode == "random":
            summary["by_topology"] = {
                topology: _summarize([row for row in rows if row["topology"] == topology])
                for topology in sorted({row["topology"] for row in rows})
            }
            summary["by_adjacency_signature"] = {
                signature: _summarize(
                    [
                        row
                        for row in rows
                        if row.get("adjacency_canonical_signature") == signature
                    ]
                )
                for signature in sorted(
                    {
                        row.get("adjacency_canonical_signature")
                        for row in rows
                        if row.get("adjacency_canonical_signature")
                    }
                )
            }
        summary["cases_detail"] = rows
        levels.append(summary)
    result = {
        "mode": mode, "noise_model": noise_model,
        "generation_mode": generation_mode,
        "cases_per_level": cases_per_level, "seed": seed, "levels": levels,
    }
    (root / "noise_sweep_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    csv_fields = [key for key in levels[0] if key not in {"cases_detail", "by_topology"}] if levels else []
    with (root / "noise_sweep_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for level in levels:
            writer.writerow({key: level.get(key) for key in csv_fields})
    if levels:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = [level["noise_mm"] for level in levels]
        ax.plot(x, [level["solver_completed_rate"] for level in levels], "o-", label="solver completed")
        ax.plot(x, [level["competition_gate_rate"] for level in levels], "o-", label="competition gate")
        ax.plot(
            x,
            [level["competition_white_success_rate"] for level in levels],
            "o-",
            label="competition white",
        )
        ax.plot(
            x,
            [level["strict_truth_success_rate"] for level in levels],
            "o-",
            label="strict truth",
        )
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("vertex noise bound (mm)")
        ax.set_ylabel("rate")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(root / "noise_sweep_plot.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    return result
