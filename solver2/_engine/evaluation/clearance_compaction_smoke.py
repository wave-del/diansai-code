"""Small, bounded integration smoke for clearance compaction policies."""

from __future__ import annotations

import csv
import json
import shutil
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


def _render_layout(solution: dict[str, Any], path: Path) -> None:
    """绘制最终 *执行* 布局，而非零缝的理论候选图。

    ``moves`` 的 ``planned_final_vertices_a4_mm`` 已是 clearance 后的最终
    坐标。这里刻意使用深色底和高对比描边，让 3--6 mm 的物理拼缝在上位机
    缩放显示时仍清晰可见；只影响 PNG 诊断图，不参与求解、校验或 MovePlan。
    """
    clearance = solution.get("clearance", {})
    clearance_ok = bool(clearance.get("feasible", False))
    min_gap = float(clearance.get("minimum_gap_mm", 0.0))
    max_vertex = float(
        clearance.get(
            "max_vertex_distance_mm",
            solution.get("nominal_max_vertex_distance_mm", 0.0),
        )
    )
    overlap = float(clearance.get("overlap_mm2", 0.0))
    figure, axes = plt.subplots(figsize=(7, 5))
    figure.patch.set_facecolor("#111827")
    axes.set_facecolor("#1f2937")
    for index, move in enumerate(solution.get("moves", [])):
        vertices = move.get(
            "planned_final_vertices_a4_mm",
            move.get("target_vertices_a4_mm", []),
        )
        axes.add_patch(PolygonPatch(
            vertices, closed=True, alpha=0.82,
            facecolor=plt.get_cmap("tab10")(index),
            edgecolor="#f8fafc", linewidth=1.6,
            label=move["piece_id"],
        ))
    axes.set_aspect("equal")
    axes.autoscale()
    axes.grid(alpha=0.24, color="#94a3b8")
    axes.legend(loc="best")
    for spine in axes.spines.values():
        spine.set_color("#cbd5e1")
    axes.tick_params(colors="#e2e8f0")
    if clearance_ok:
        title = (
            "EXECUTION PREVIEW — clearance OK\n"
            f"min gap={min_gap:.2f} mm | vertex≤{max_vertex:.2f} mm | "
            f"overlap≈{overlap:.2f} mm²"
        )
    else:
        title = "EXECUTION PREVIEW | ⚠ zero-gap fallback: clearance unavailable"
    axes.set_title(title, color="#f8fafc", fontsize=10, fontweight="bold", pad=10)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _copy_case_input(source: Path, destination: Path) -> None:
    for name in ("public", "truth"):
        target = destination / name
        if target.exists():
            continue
        shutil.copytree(source / name, target)


def run_clearance_compaction_smoke(
    output: str | Path = "outputs/clearance_compaction_smoke",
) -> dict[str, Any]:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    historical_root = Path(
        "outputs/benchmark_random_global_interfaces_final2_13000"
    )
    cases = [
        {
            "name": "seed_13010", "source": historical_root / "random_seed_13010",
            "policy": "competition_force_output", "kind": "historical_hard",
        },
        {
            "name": "seed_13060", "source": historical_root / "random_seed_13060",
            "policy": "competition_force_output", "kind": "historical_hard",
        },
        {
            "name": "ordinary_two_piece", "seed": 22101, "piece_count": 2,
            "generation_mode": "recursive",
            "policy": "competition_best_effort", "kind": "ordinary_two",
        },
        {
            "name": "three_piece_t", "seed": 22102, "piece_count": 3,
            "generation_mode": "template", "topology": "three_y",
            "policy": "competition_force_output", "kind": "t_junction",
        },
        {
            "name": "four_piece_center", "seed": 22103, "piece_count": 4,
            "generation_mode": "template", "topology": "four_fan",
            "policy": "competition_best_effort", "kind": "center_junction",
        },
        {
            "name": "patterned_ambiguity", "seed": 22104, "piece_count": 4,
            "generation_mode": "template", "topology": "difficult_similar",
            "surface_mode": "patterned",
            "policy": "competition_force_output", "kind": "patterned",
        },
    ]
    rows: list[dict[str, Any]] = []
    for spec in cases:
        case_dir = root / spec["name"]
        if "source" in spec:
            source = Path(spec["source"])
            _copy_case_input(source, case_dir)
        else:
            source = case_dir
            if not (source / "public/observations.json").exists():
                generate_case(
                    "random",
                    int(spec["seed"]),
                    0.0,
                    source,
                    piece_count=int(spec["piece_count"]),
                    topology=spec.get("topology"),
                    generation_mode=str(spec["generation_mode"]),
                    surface_mode=str(spec.get("surface_mode", "white")),
                    render=False,
                )
        solution_path = case_dir / "solution.json"
        if not solution_path.exists():
            solution = solve_observations(
                case_dir / "public/observations.json",
                case_dir,
                print_plan=False,
                runtime_config=RuntimeConfig(
                    execution_profile="production",
                    diagnostics_level="summary",
                    execution_policy=str(spec["policy"]),
                ),
            )
        else:
            solution = json.loads(solution_path.read_text(encoding="utf-8"))
        evaluation = evaluate_solution(
            case_dir / "truth/ground_truth.json", solution_path
        )
        (case_dir / "evaluation.json").write_text(
            json.dumps(evaluation, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _render_layout(solution, case_dir / "final_layout.png")
        clearance = solution.get("clearance", {})
        baseline: dict[str, Any] = {}
        if spec["name"].startswith("seed_"):
            seed = spec["name"].removeprefix("seed_")
            old_path = Path(
                f"outputs/adaptive_uncertainty_smoke/"
                f"seed_{seed}_noise_0/solution/solution.json"
            )
            if old_path.exists():
                baseline = json.loads(old_path.read_text(encoding="utf-8"))
        rows.append({
            "case": spec["name"],
            "kind": spec["kind"],
            "execution_policy": spec["policy"],
            "elapsed_seconds": solution.get("elapsed_seconds"),
            "machine_executable": solution.get("machine_executable"),
            "competition_gate_passed": solution.get(
                "competition_gate_passed"
            ),
            "strict_robust_gate_passed": solution.get(
                "strict_robust_gate_passed"
            ),
            "execution_quality": solution.get("execution_quality"),
            "motion_count": len(solution.get("moves", [])),
            "pre_compaction_max_gap_mm": clearance.get(
                "pre_compaction_max_gap_mm"
            ),
            "post_compaction_max_gap_mm": clearance.get(
                "post_compaction_max_gap_mm"
            ),
            "nominal_max_vertex_distance_mm": solution.get(
                "nominal_max_vertex_distance_mm"
            ),
            "estimated_risk_vertex_distance_mm": solution.get(
                "estimated_actual_risk_vertex_distance_mm"
            ),
            "worst_case_robust_vertex_distance_mm": solution.get(
                "robust_max_vertex_distance_mm"
            ),
            "estimated_overlap_area_mm2": solution.get(
                "estimated_physical_overlap_area_mm2"
            ),
            "baseline_target_clearance_mm": baseline.get(
                "clearance_target_mm"
            ),
            "baseline_nominal_max_vertex_distance_mm": baseline.get(
                "nominal_max_vertex_distance_mm"
            ),
            "baseline_robust_max_vertex_distance_mm": baseline.get(
                "robust_max_vertex_distance_mm"
            ),
            "baseline_executable": baseline.get("executable"),
            "topology_correct": evaluation.get("topology_correct"),
            "adjacency_accuracy": evaluation.get("adjacency_accuracy"),
            "evaluator_competition_pass": evaluation.get(
                "passes_competition_gate"
            ),
            "evaluator_strict_pass": evaluation.get("passes_strict_exact"),
            "risk_warnings": "|".join(solution.get("risk_warnings", [])),
        })
    summary = {
        "case_count": len(rows),
        "actual_solver_run_limit": 18,
        "actual_solver_case_count": len(rows),
        "cases": rows,
        "machine_executable_count": sum(
            bool(row["machine_executable"]) for row in rows
        ),
        "competition_pass_count": sum(
            bool(row["evaluator_competition_pass"]) for row in rows
        ),
        "strict_exact_count": sum(
            bool(row["evaluator_strict_pass"]) for row in rows
        ),
        "large_synthetic_cases_omitted": True,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (root / "cases.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    unresolved = [
        row for row in rows
        if (
            not row["evaluator_competition_pass"]
            or not row["competition_gate_passed"]
        )
    ]
    lines = ["# Clearance compaction unresolved cases", ""]
    if not unresolved:
        lines.append("小规模集成中的全部案例均通过 evaluator 赛题门槛。")
    for row in unresolved:
        lines.extend([
            f"## {row['case']}", "",
            f"- quality: {row['execution_quality']}",
            f"- topology_correct: {row['topology_correct']}",
            f"- warnings: {row['risk_warnings']}",
            f"- [最终布局](./{row['case']}/final_layout.png)",
            "",
        ])
    (root / "unresolved_cases.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(
        run_clearance_compaction_smoke(), ensure_ascii=False, indent=2
    ))
