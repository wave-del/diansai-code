"""生成阶段的四张隔离检查图。"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolygonPatch, Rectangle

from ..config import A4_HEIGHT_MM, A4_WIDTH_MM, DIVIDER_Y_MM


def _a4_axes(title: str):
    fig, ax = plt.subplots(figsize=(6.2, 8.7))
    ax.add_patch(Rectangle((0, 0), A4_WIDTH_MM, A4_HEIGHT_MM, facecolor="white", edgecolor="black"))
    ax.axhline(DIVIDER_Y_MM, color="black", linestyle="--", linewidth=1)
    ax.set_xlim(-5, A4_WIDTH_MM + 5)
    ax.set_ylim(A4_HEIGHT_MM + 5, -5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title)
    return fig, ax


def render_generated_case(root: Path, observations: dict, truth: dict) -> None:
    colors = plt.get_cmap("tab10").colors
    fig, ax = _a4_axes("Shuffled contest view")
    for piece in observations["pieces"]:
        ax.add_patch(PolygonPatch(piece["vertices_mm"], closed=True, facecolor="0.92", edgecolor="black"))
    fig.savefig(root / "public" / "shuffled_contest_view.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = _a4_axes("Observed vertices (public)")
    for i, piece in enumerate(observations["pieces"]):
        verts = piece["vertices_mm"]
        ax.add_patch(PolygonPatch(verts, closed=True, facecolor=colors[i], alpha=0.28, edgecolor=colors[i], label=piece["id"]))
        for j, (x, y) in enumerate(verts):
            ax.plot(x, y, "o", color=colors[i], markersize=3)
            ax.annotate(f"v{j} ({x:.1f},{y:.1f})", (x, y), xytext=(3, 3), textcoords="offset points", fontsize=6)
    ax.legend(fontsize=7)
    fig.savefig(root / "public" / "shuffled_debug.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    width = truth["target_rectangle"]["width_mm"]
    height = truth["target_rectangle"]["height_mm"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, piece in enumerate(truth["pieces"]):
        verts = piece["target_vertices_local_mm"]
        ax.add_patch(PolygonPatch(verts, closed=True, facecolor=colors[i], alpha=0.35, edgecolor=colors[i], label=piece["source_id"]))
    ax.add_patch(Rectangle((0, 0), width, height, fill=False, edgecolor="black", linewidth=2))
    ax.set_xlim(-5, width + 5)
    ax.set_ylim(height + 5, -5)
    ax.set_aspect("equal")
    ax.set_title(f"Clean target {width:.2f} mm x {height:.2f} mm")
    ax.legend(fontsize=7)
    fig.savefig(root / "truth" / "target_clean.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = _a4_axes("Clean shuffled reference (truth only)")
    for i, piece in enumerate(truth["pieces"]):
        verts = piece["clean_shuffled_vertices_mm"]
        ax.add_patch(PolygonPatch(verts, closed=True, facecolor=colors[i], alpha=0.3, edgecolor=colors[i]))
        x, y = verts[0]
        ax.annotate(
            f"{piece['source_id']}\nR={piece['shuffle_rotation_deg_clockwise']:.2f} deg",
            (x, y), fontsize=6,
        )
    fig.savefig(root / "truth" / "shuffled_clean_reference.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

