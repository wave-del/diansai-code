"""强制 solution_overlay 与局部拼图可视化。"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as PolygonPatch, Rectangle
from shapely.geometry import Polygon

from ..config import A4_HEIGHT_MM, A4_WIDTH_MM, DIVIDER_Y_MM
from ..models import Move, Score


def render_solution(
    output: Path,
    moves: list[Move],
    target: dict,
    score: Score,
    confidence: float,
    safety: dict | None = None,
) -> None:
    colors = plt.get_cmap("tab10").colors
    fig, ax = plt.subplots(figsize=(7.2, 10.2))
    ax.add_patch(Rectangle((0, 0), A4_WIDTH_MM, A4_HEIGHT_MM, facecolor="white", edgecolor="black"))
    ax.axhline(DIVIDER_Y_MM, color="black", linestyle="--", linewidth=1)
    tx = target["center_a4_mm"][0] - target["width_mm"] / 2
    ty = target["center_a4_mm"][1] - target["height_mm"] / 2
    ax.add_patch(Rectangle((tx, ty), target["width_mm"], target["height_mm"], fill=False, edgecolor="black", linestyle="--", linewidth=1.5))
    safety_moves = {
        item["piece_id"]: item
        for item in (safety or {}).get("moves", [])
    }
    for i, move in enumerate(moves):
        color = colors[i % len(colors)]
        ax.add_patch(PolygonPatch(move.observed_vertices_a4_mm, closed=True, fill=False, edgecolor="0.45", linestyle="--"))
        safety_move = safety_moves.get(move.piece_id, {})
        estimated = safety_move.get(
            "estimated_physical_vertices_before_clearance_a4_mm",
            move.target_vertices_a4_mm,
        )
        planned = safety_move.get(
            "planned_final_vertices_a4_mm",
            move.target_vertices_a4_mm,
        )
        buffer_radius = float(
            safety_move.get("uncertainty_buffer_radius_mm", 0.0)
        )
        if buffer_radius > 0.0:
            buffered = Polygon(planned).buffer(buffer_radius)
            ax.add_patch(PolygonPatch(
                list(buffered.exterior.coords),
                closed=True,
                facecolor=color,
                alpha=0.10,
                edgecolor=color,
                linewidth=0.8,
            ))
        ax.add_patch(PolygonPatch(
            estimated,
            closed=True,
            fill=False,
            edgecolor=color,
            linewidth=1.0,
        ))
        ax.add_patch(PolygonPatch(
            planned,
            closed=True,
            facecolor=color,
            alpha=0.25,
            edgecolor=color,
            linewidth=2.4,
        ))
        px, py = move.pick_xy_a4_mm
        qx, qy = move.place_xy_a4_mm
        ax.plot(px, py, "o", color=color, markersize=5)
        ax.plot(qx, qy, "x", color=color, markersize=7, markeredgewidth=2)
        ax.annotate("", xy=(qx, qy), xytext=(px, py), arrowprops=dict(arrowstyle="->", color=color, lw=1.2))
        ax.annotate(
            f"Step {move.order} {move.piece_id}\nPick=({px:.2f}, {py:.2f}) mm\n"
            f"Place=({qx:.2f}, {qy:.2f}) mm\nRotate={move.rotation_deg_clockwise:+.2f}° CW",
            (px, py),
            xytext=(0.01 + 0.247 * i, 0.49),
            textcoords="axes fraction",
            fontsize=5.8,
            arrowprops=dict(arrowstyle="-", color=color, alpha=0.45, lw=0.7),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.72),
        )
    legend = [
        Line2D([0], [0], color="0.45", linestyle="--", label="Original observed outline"),
        Line2D([0], [0], color=colors[0], linewidth=1, label="Estimated physical outline"),
        Line2D([0], [0], color=colors[0], linewidth=2.4, label="Planned final outline"),
        Line2D([0], [0], marker="o", color="black", linestyle="", label="Pick"),
        Line2D([0], [0], marker="x", color="black", linestyle="", label="Place"),
        Line2D([0], [0], color="black", marker=">", label="XY movement"),
        Line2D([0], [0], color="black", linestyle="--", label="Target rectangle"),
    ]
    ax.legend(handles=legend, loc="lower left", fontsize=7)
    ax.set_xlim(-5, A4_WIDTH_MM + 5)
    ax.set_ylim(A4_HEIGHT_MM + 5, -5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    status = ""
    if safety is not None:
        status = (
            f"\nclearance={safety.get('planned_clearance_mm', 0.0):.2f} mm, "
            f"max vertex={safety.get('max_corresponding_vertex_distance_mm', 0.0):.2f} mm, "
            f"estimated overlap={safety.get('estimated_physical_overlap_area_mm2', 0.0):.3g} mm², "
            f"robust overlap={safety.get('robust_buffer_overlap_area_mm2', 0.0):.3g} mm², "
            f"executable={bool(safety.get('executable', False))}"
        )
    ax.set_title(
        "Puzzle motion plan: observed / estimated / planned / buffer"
        + status
    )
    fig.savefig(output / "solution_overlay.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, move in enumerate(moves):
        local = [(x - tx, y - ty) for x, y in move.target_vertices_a4_mm]
        ax.add_patch(PolygonPatch(local, closed=True, facecolor=colors[i], alpha=0.35, edgecolor=colors[i], label=move.piece_id))
    ax.add_patch(Rectangle((0, 0), target["width_mm"], target["height_mm"], fill=False, edgecolor="black", linewidth=2))
    ax.set_xlim(-5, target["width_mm"] + 5)
    ax.set_ylim(target["height_mm"] + 5, -5)
    ax.set_aspect("equal")
    ax.set_title(
        f"Local solution | score={score.total:.3f} | confidence={confidence:.3f}\n"
        f"edge RMS={score.matched_edge_rms_mm:.3f} mm, outer RMS={score.outer_edge_rms_mm:.3f} mm"
    )
    ax.legend(fontsize=7)
    fig.savefig(output / "solution_local.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
