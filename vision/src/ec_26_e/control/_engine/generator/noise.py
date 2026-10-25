"""可复现的视觉几何噪声模型。"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

NOISE_MODELS = {
    "disk_bounded_vertex",
    "square_axis_bounded_stress",
    # Backward-compatible alias for previously generated command lines.
    "uniform_independent_vertex",
    "gaussian_vertex",
    "rigid_pose_noise",
    "rigid_plus_line",
    "line_fit_noise",
    "mixed_realistic",
}


def sample_disk_bounded_offsets(
    count: int,
    radius_mm: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample uniformly by area in a closed 2-D disk."""
    angles = rng.uniform(0.0, 2.0 * np.pi, size=int(count))
    radii = float(radius_mm) * np.sqrt(
        rng.uniform(0.0, 1.0, size=int(count))
    )
    offsets = np.column_stack((
        radii * np.cos(angles),
        radii * np.sin(angles),
    ))
    if np.any(
        np.linalg.norm(offsets, axis=1)
        > float(radius_mm) + 1e-9
    ):
        raise AssertionError("disk_bounded_vertex exceeded radius")
    return offsets


def _line_intersection(
    p1: np.ndarray,
    d1: np.ndarray,
    p2: np.ndarray,
    d2: np.ndarray,
    fallback_vertex: np.ndarray,
) -> np.ndarray:
    matrix = np.column_stack((d1, -d2))
    if abs(float(np.linalg.det(matrix))) < 0.12:
        # 连续共线分段不是稳定角点；分别把原顶点投影到两条拟合线后取均值，
        # 避免近平行直线求交把亚毫米拟合误差放大成数百毫米。
        projections = [
            p + d * float(np.dot(fallback_vertex - p, d))
            for p, d in ((p1, d1), (p2, d2))
        ]
        return (projections[0] + projections[1]) / 2.0
    return p1 + d1 * np.linalg.solve(matrix, p2 - p1)[0]


def _line_fit_vertices(clean: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    lines: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(len(clean)):
        a, b = clean[index], clean[(index + 1) % len(clean)]
        direction = b - a
        direction /= np.linalg.norm(direction)
        angle = rng.normal(0.0, np.deg2rad(0.12 * scale))
        c, s = np.cos(angle), np.sin(angle)
        direction = np.array([c * direction[0] - s * direction[1], s * direction[0] + c * direction[1]])
        normal = np.array([-direction[1], direction[0]])
        midpoint = (a + b) / 2.0 + normal * rng.normal(0.0, 0.18 * scale)
        lines.append((midpoint, direction))
    return np.asarray([
        _line_intersection(
            lines[index - 1][0], lines[index - 1][1],
            lines[index][0], lines[index][1], clean[index],
        )
        for index in range(len(clean))
    ])


def add_vertex_noise(
    vertices: list[tuple[float, float]],
    noise_mm: float,
    rng: np.random.Generator,
    model: str = "disk_bounded_vertex",
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    if model not in NOISE_MODELS:
        raise ValueError(f"未知噪声模型: {model}")
    clean = np.asarray(vertices, dtype=float)
    for _ in range(1000):
        if model == "disk_bounded_vertex":
            offsets = sample_disk_bounded_offsets(
                len(clean), noise_mm, rng
            )
            observed = clean + offsets
        elif model in {
            "square_axis_bounded_stress",
            "uniform_independent_vertex",
        }:
            observed = clean + rng.uniform(-noise_mm, noise_mm, size=clean.shape)
        elif model == "gaussian_vertex":
            observed = clean + rng.normal(0.0, noise_mm, size=clean.shape)
        elif model == "rigid_pose_noise":
            angle = rng.normal(0.0, 0.12 * noise_mm)
            radians = np.deg2rad(angle)
            rotation = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]])
            center = clean.mean(axis=0)
            observed = (clean - center) @ rotation.T + center + rng.normal(0.0, 0.35 * noise_mm, size=2)
        elif model == "line_fit_noise":
            observed = _line_fit_vertices(clean, noise_mm, rng)
        elif model == "rigid_plus_line":
            observed = _line_fit_vertices(clean, noise_mm, rng)
            angle = rng.normal(0.0, 0.08 * noise_mm)
            radians = np.deg2rad(angle)
            rotation = np.array([
                [np.cos(radians), -np.sin(radians)],
                [np.sin(radians), np.cos(radians)],
            ])
            center = clean.mean(axis=0)
            observed = (
                (observed - center) @ rotation.T + center
                + rng.normal(0.0, 0.2 * noise_mm, size=2)
            )
        else:
            observed = _line_fit_vertices(clean, noise_mm, rng)
            angle = rng.normal(0.0, 0.06 * noise_mm)
            radians = np.deg2rad(angle)
            rotation = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]])
            center = clean.mean(axis=0)
            observed = (observed - center) @ rotation.T + center + rng.normal(0.0, 0.15 * noise_mm, size=2)
            if noise_mm > 0 and rng.random() < 0.18:
                observed[int(rng.integers(len(clean)))] += rng.normal(0.0, 0.8 * noise_mm, size=2)
        noise = observed - clean
        if model == "disk_bounded_vertex":
            radii = np.linalg.norm(noise, axis=1)
            if np.any(radii > noise_mm + 1e-9):
                raise AssertionError("disk_bounded_vertex exceeded radius")
        poly = Polygon(observed)
        lengths = np.linalg.norm(np.roll(observed, -1, axis=0) - observed, axis=1)
        if poly.is_valid and poly.area > 1.0 and float(lengths.min()) > 2.0:
            return (
                [(float(x), float(y)) for x, y in observed],
                [(float(x), float(y)) for x, y in noise],
            )
    raise RuntimeError("1000 次采样后仍无法获得有效的加噪多边形")


def noise_parameters(model: str, noise_mm: float) -> dict[str, float | str]:
    """把模型语义与标称尺度写入输出，防止跨模型混合统计。"""
    return {
        "model": model,
        "noise_level_mm": float(noise_mm),
        "coordinate_unit": "mm",
        "distribution": {
            "disk_bounded_vertex": "uniform by area inside radius-bounded disk",
            "square_axis_bounded_stress": "Uniform(-level,+level) per x/y stress model",
            "uniform_independent_vertex": "Uniform(-level,+level) per x/y",
            "gaussian_vertex": "Normal(0,level) per x/y",
            "rigid_pose_noise": "rigid pose perturbation",
            "rigid_plus_line": "line fit plus rigid pose perturbation",
            "line_fit_noise": "independent fitted support lines",
            "mixed_realistic": "line fit, rigid perturbation and sparse outlier",
        }[model],
    }
