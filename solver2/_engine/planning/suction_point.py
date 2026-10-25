"""安全拾取点。"""

from __future__ import annotations

from ..geometry import max_inscribed_point


def choose_suction_point(vertices: list[tuple[float, float]]) -> tuple[tuple[float, float], float]:
    return max_inscribed_point(vertices)

