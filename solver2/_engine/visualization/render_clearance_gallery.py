"""Render saved clearance smoke cases without rerunning the solver."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as PolygonPatch
from matplotlib.patches import Rectangle
import numpy as np
from shapely.geometry import Polygon

from ..evaluation.evaluator import evaluate_solution
from ..transforms import transform_points


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _render_case(
    truth_path: Path,
    solution_path: Path,
    image_path: Path,
    label: str,
) -> dict:
    truth = _load(truth_path)
    solution = _load(solution_path)
    evaluation = evaluate_solution(truth_path, solution_path)
    truth_by_public = {
        piece["public_id"]: piece for piece in truth["pieces"]
    }
    colors = plt.get_cmap("tab10").colors
    fig, ax = plt.subplots(figsize=(8.6, 6.2))

    target = solution["target_rectangle"]
    center = np.asarray(target["center_a4_mm"], dtype=float)
    tx = center[0] - float(target["width_mm"]) / 2.0
    ty = center[1] - float(target["height_mm"]) / 2.0
    ax.add_patch(Rectangle(
        (tx, ty),
        float(target["width_mm"]),
        float(target["height_mm"]),
        fill=False,
        edgecolor="black",
        linestyle="--",
        linewidth=1.2,
    ))

    actual_polygons: list[Polygon] = []
    all_points: list[np.ndarray] = []
    rendered: list[dict] = []
    for index, move in enumerate(solution["moves"]):
        color = colors[index % len(colors)]
        transform = move["rigid_transform"]
        angle = float(transform["rotation_deg_clockwise"])
        translation = transform["translation_a4_mm"]
        observed = transform_points(
            move["observed_vertices_a4_mm"], angle, translation
        )
        estimated_before = np.asarray(
            move.get(
                "estimated_physical_vertices_before_clearance_a4_mm",
                move["target_vertices_a4_mm"],
            ),
            dtype=float,
        )
        planned = np.asarray(
            move.get(
                "planned_final_vertices_a4_mm",
                move["target_vertices_a4_mm"],
            ),
            dtype=float,
        )
        actual = transform_points(
            truth_by_public[move["piece_id"]][
                "clean_shuffled_vertices_mm"
            ],
            angle,
            translation,
        )
        buffer_radius = float(
            move.get("uncertainty_buffer_radius_mm", 0.0)
        )
        buffered = Polygon(planned).buffer(buffer_radius)
        if not buffered.is_empty:
            ax.add_patch(PolygonPatch(
                list(buffered.exterior.coords),
                closed=True,
                facecolor=color,
                edgecolor=color,
                alpha=0.10,
                linewidth=0.7,
            ))
        ax.add_patch(PolygonPatch(
            observed,
            closed=True,
            fill=False,
            edgecolor="0.45",
            linestyle=":",
            linewidth=1.0,
        ))
        ax.add_patch(PolygonPatch(
            estimated_before,
            closed=True,
            fill=False,
            edgecolor=color,
            linestyle="--",
            linewidth=1.0,
        ))
        ax.add_patch(PolygonPatch(
            planned,
            closed=True,
            fill=False,
            edgecolor=color,
            linewidth=1.5,
        ))
        actual_poly = Polygon(actual)
        actual_polygons.append(actual_poly)
        ax.add_patch(PolygonPatch(
            actual,
            closed=True,
            facecolor=color,
            edgecolor=color,
            alpha=0.26,
            linewidth=2.6,
        ))
        centroid = actual_poly.centroid
        ax.text(
            centroid.x,
            centroid.y,
            move["piece_id"],
            ha="center",
            va="center",
            fontsize=8,
            weight="bold",
            bbox={"facecolor": "white", "alpha": 0.65, "pad": 1},
        )
        all_points.extend((observed, estimated_before, planned, actual))
        rendered.append({"piece_id": move["piece_id"]})

    overlap_count = 0
    for first in range(len(actual_polygons)):
        for second in range(first + 1, len(actual_polygons)):
            intersection = actual_polygons[first].intersection(
                actual_polygons[second]
            )
            if intersection.area > 1e-5:
                overlap_count += 1
                if intersection.geom_type == "Polygon":
                    ax.add_patch(PolygonPatch(
                        list(intersection.exterior.coords),
                        closed=True,
                        facecolor="red",
                        edgecolor="darkred",
                        alpha=0.75,
                        linewidth=2.0,
                    ))

    points = np.vstack(all_points)
    pad = 10.0
    ax.set_xlim(float(points[:, 0].min()) - pad, float(points[:, 0].max()) + pad)
    ax.set_ylim(float(points[:, 1].max()) + pad, float(points[:, 1].min()) - pad)
    ax.set_aspect("equal")
    ax.grid(alpha=0.15)
    ax.set_xlabel("A4 x (mm)")
    ax.set_ylabel("A4 y (mm)")
    actual_overlap = float(
        evaluation["ground_truth_actual_overlap_area_mm2"]
    )
    ax.set_title(
        f"{label}\n"
        f"actual overlap={actual_overlap:.6g} mm² | "
        f"max vertex={evaluation['max_corresponding_vertex_distance_mm']:.3f} mm | "
        f"planned clearance={solution.get('planned_clearance_mm', 0.0):.1f} mm | "
        f"executable={bool(solution.get('executable', False))}"
    )
    ax.legend(handles=[
        Line2D([0], [0], color="0.45", linestyle=":", label="Observed after output transform"),
        Line2D([0], [0], color=colors[0], linestyle="--", label="Estimated before clearance"),
        Line2D([0], [0], color=colors[0], linewidth=1.5, label="Planned estimated outline"),
        Line2D([0], [0], color=colors[0], linewidth=2.6, alpha=0.6, label="Clean actual outline"),
        Line2D([0], [0], color=colors[0], linewidth=7, alpha=0.12, label="Uncertainty buffer"),
        Line2D([0], [0], color="red", linewidth=7, alpha=0.75, label="Positive-area actual overlap"),
    ], loc="best", fontsize=7)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(image_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {
        "actual_overlap_area_mm2": actual_overlap,
        "max_vertex_distance_mm": float(
            evaluation["max_corresponding_vertex_distance_mm"]
        ),
        "planned_clearance_mm": float(
            solution.get("planned_clearance_mm", 0.0)
        ),
        "executable": bool(solution.get("executable", False)),
        "competition_success": bool(
            evaluation.get("competition_white_success", False)
        ),
        "overlapping_pair_count": overlap_count,
    }


def render_clearance_gallery(
    root: str | Path = "outputs/overlap_clearance_smoke",
) -> Path:
    root = Path(root)
    summary = _load(root / "summary.json")
    image_dir = root / "all_rendered"
    rows: list[dict] = []
    for row in summary["rows"]:
        case = row["case"]
        radius = int(row["radius_mm"])
        case_root = root / case / f"R_{radius}"
        image_name = f"{case}_R{radius}.png"
        metrics = _render_case(
            case_root / "truth" / "ground_truth.json",
            case_root / "solution" / "solution.json",
            image_dir / image_name,
            f"{case} | R={radius} mm",
        )
        rows.append({
            "case": case,
            "radius": radius,
            "image": image_name,
            **metrics,
        })

    integration_specs = (
        (
            "fixed_integration_R1",
            Path("outputs/clearance_integration_fixed"),
        ),
        (
            "random_integration_R1",
            Path("outputs/clearance_integration_random"),
        ),
    )
    integration_rows = []
    for name, case_root in integration_specs:
        if (case_root / "solution" / "solution.json").exists():
            image_name = f"{name}.png"
            metrics = _render_case(
                case_root / "truth" / "ground_truth.json",
                case_root / "solution" / "solution.json",
                image_dir / image_name,
                name,
            )
            integration_rows.append({
                "case": name,
                "image": image_name,
                **metrics,
            })

    lines = [
        "# Overlap / clearance 全部测试图",
        "",
        "图例：灰色点线=observed；彩色虚线=clearance前估计轮廓；"
        "彩色细实线=规划轮廓；彩色粗填充=clean actual；"
        "半透明外圈=uncertainty buffer；红色=actual正面积重叠。",
        "",
        "注意：`actual overlap > 1e-5 mm²` 的案例均应为 "
        "`executable=false`。",
        "",
        "## 20个 random smoke 案例",
        "",
    ]
    for case in dict.fromkeys(row["case"] for row in rows):
        lines.extend([f"### {case}", ""])
        for row in [item for item in rows if item["case"] == case]:
            status = "OVERLAP" if row["actual_overlap_area_mm2"] > 1e-5 else "NO OVERLAP"
            lines.extend([
                f"#### R={row['radius']} mm — {status}",
                "",
                f"- actual overlap: `{row['actual_overlap_area_mm2']:.6g} mm²`",
                f"- max corresponding vertex distance: `{row['max_vertex_distance_mm']:.3f} mm`",
                f"- planned clearance: `{row['planned_clearance_mm']:.1f} mm`",
                f"- executable: `{str(row['executable']).lower()}`",
                f"- competition success: `{str(row['competition_success']).lower()}`",
                "",
                f"![{case} R={row['radius']}](all_rendered/{row['image']})",
                "",
            ])
    lines.extend(["## 本轮 fixed/random 集成案例", ""])
    for row in integration_rows:
        status = "OVERLAP" if row["actual_overlap_area_mm2"] > 1e-5 else "NO OVERLAP"
        lines.extend([
            f"### {row['case']} — {status}",
            "",
            f"- actual overlap: `{row['actual_overlap_area_mm2']:.6g} mm²`",
            f"- max corresponding vertex distance: `{row['max_vertex_distance_mm']:.3f} mm`",
            f"- executable: `{str(row['executable']).lower()}`",
            "",
            f"![{row['case']}](all_rendered/{row['image']})",
            "",
        ])
    markdown_path = root / "ALL_CASES_GALLERY.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return markdown_path


if __name__ == "__main__":
    print(render_clearance_gallery().resolve())

