"""A4 毫米坐标到写字机坐标的独立仿射变换。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np


def load_calibration(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    matrix = np.asarray(data.get("matrix"), dtype=float)
    if data.get("transform_type") != "affine" or matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError("机器标定必须是含 2x3 matrix 的 affine 配置")
    return data


def a4_to_machine(point: tuple[float, float], calibration: dict[str, Any]) -> tuple[float, float]:
    matrix = np.asarray(calibration["matrix"], dtype=float)
    result = matrix[:, :2] @ np.asarray(point, dtype=float) + matrix[:, 2]
    return float(result[0]), float(result[1])


def machine_rotation(angle_clockwise: float, calibration: dict[str, Any]) -> float:
    direction = calibration.get("machine_positive_rotation", "clockwise")
    if direction not in {"clockwise", "counterclockwise"}:
        raise ValueError("machine_positive_rotation 必须为 clockwise 或 counterclockwise")
    return float(angle_clockwise if direction == "clockwise" else -angle_clockwise)

