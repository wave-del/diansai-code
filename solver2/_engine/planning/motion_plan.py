"""运动计划生成与贪心空载路径排序。"""

from __future__ import annotations

import csv
import os
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon

from ..config import OVERLAP_NUMERICAL_EPSILON_MM2
from ..models import Move
from ..transforms import rotation_matrix, transform_points
from .machine_transform import a4_to_machine, machine_rotation
from .suction_point import choose_suction_point

ACTIONS = [
    "MOVE_XY_PICK",
    "PICK_DOWN",
    "MAGNET_ON",
    "PICK_UP",
    "ROTATE",
    "MOVE_XY_PLACE",
    "PICK_DOWN",
    "MAGNET_OFF",
    "PICK_UP",
]


def validate_transport_paths(
    pose_records: list[dict],
    moves: list[Move],
    mode: str,
    *,
    overlap_epsilon_mm2: float = OVERLAP_NUMERICAL_EPSILON_MM2,
    samples: int = 24,
) -> list[str]:
    """Validate carried-piece motion for the configured writing-machine model."""
    if mode == "safe_z_clearance":
        return []
    if mode == "head_only":
        return []
    if mode != "swept_polygon":
        return ["unknown_transport_collision_mode"]

    records = {record["piece_id"]: record for record in pose_records}
    remaining = {
        piece_id: Polygon(np.asarray(record["observed"], dtype=float))
        for piece_id, record in records.items()
    }
    placed: list[Polygon] = []
    reasons: list[str] = []
    for move in moves:
        record = records[move.piece_id]
        observed = np.asarray(record["observed"], dtype=float)
        pick = np.asarray(move.pick_xy_a4_mm, dtype=float)
        place = np.asarray(move.place_xy_a4_mm, dtype=float)
        remaining.pop(move.piece_id, None)
        obstacles = list(remaining.values()) + placed

        def collides(points: np.ndarray) -> bool:
            polygon = Polygon(points)
            if not polygon.is_valid:
                return True
            return any(
                polygon.intersection(obstacle).area > overlap_epsilon_mm2
                for obstacle in obstacles
            )

        # The platform rotates about the suction point before XY transport.
        rotated = observed
        for fraction in np.linspace(0.0, 1.0, max(2, samples)):
            matrix = rotation_matrix(
                float(move.rotation_deg_clockwise) * float(fraction)
            )
            rotated = (observed - pick) @ matrix.T + pick
            if collides(rotated):
                reasons.append(
                    f"transport_swept_collision:{move.piece_id}:rotation"
                )
                break
        delta = place - pick
        for fraction in np.linspace(0.0, 1.0, max(2, samples)):
            carried = rotated + delta * float(fraction)
            if collides(carried):
                reasons.append(
                    f"transport_swept_collision:{move.piece_id}:translation"
                )
                break
        placed.append(Polygon(np.asarray(record["target"], dtype=float)))
    return sorted(set(reasons))


def build_moves(
    pose_records: list[dict],
    confidence: float,
    calibration: dict | None = None,
    start: tuple[float, float] = (10.0, 10.0),
) -> list[Move]:
    pending: list[Move] = []
    for record in pose_records:
        observed = [(float(x), float(y)) for x, y in record["observed"]]
        pick, clearance = choose_suction_point(observed)
        angle = float(record["angle"])
        translation = tuple(float(v) for v in record["translation"])
        place_arr = transform_points([pick], angle, translation)[0]
        place = (float(place_arr[0]), float(place_arr[1]))
        machine = None
        if calibration is not None:
            machine = {
                "pick_xy_machine": a4_to_machine(pick, calibration),
                "place_xy_machine": a4_to_machine(place, calibration),
                "machine_unit": calibration.get("machine_unit", "mm"),
                "rotation_deg_machine": machine_rotation(angle, calibration),
            }
        pending.append(
            Move(
                0,
                record["piece_id"],
                pick,
                clearance,
                angle,
                place,
                translation,
                observed,
                [(float(x), float(y)) for x, y in record["target"]],
                confidence,
                record.get("template_id"),
                machine,
                list(ACTIONS),
            )
        )
    ordered: list[Move] = []
    current = np.asarray(start, dtype=float)
    while pending:
        index = min(
            range(len(pending)),
            key=lambda i: float(np.linalg.norm(np.asarray(pending[i].pick_xy_a4_mm) - current)),
        )
        move = pending.pop(index)
        move.order = len(ordered) + 1
        ordered.append(move)
        current = np.asarray(move.place_xy_a4_mm)
    return ordered


def write_motion_csv(path: Path, moves: list[Move], confidence: float, executable: bool = False) -> None:
    machine_enabled = any(move.machine_coordinates is not None for move in moves)
    fields = [
        "order", "piece_id", "pick_x_a4_mm", "pick_y_a4_mm", "pick_clearance_mm",
        "rotation_deg_clockwise", "place_x_a4_mm", "place_y_a4_mm",
        "translation_x_a4_mm", "translation_y_a4_mm", "confidence", "executable",
    ]
    if machine_enabled:
        fields += [
            "pick_x_machine", "pick_y_machine", "place_x_machine", "place_y_machine",
            "rotation_deg_machine", "machine_unit",
        ]
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for move in moves:
            row = {
                "order": move.order,
                "piece_id": move.piece_id,
                "pick_x_a4_mm": f"{move.pick_xy_a4_mm[0]:.2f}",
                "pick_y_a4_mm": f"{move.pick_xy_a4_mm[1]:.2f}",
                "pick_clearance_mm": f"{move.pick_clearance_mm:.2f}",
                "rotation_deg_clockwise": f"{move.rotation_deg_clockwise:.2f}",
                "place_x_a4_mm": f"{move.place_xy_a4_mm[0]:.2f}",
                "place_y_a4_mm": f"{move.place_xy_a4_mm[1]:.2f}",
                "translation_x_a4_mm": f"{move.translation_a4_mm[0]:.2f}",
                "translation_y_a4_mm": f"{move.translation_a4_mm[1]:.2f}",
                "confidence": f"{confidence:.4f}",
                "executable": str(bool(executable)).lower(),
            }
            if machine_enabled and move.machine_coordinates:
                mc = move.machine_coordinates
                row.update(
                    {
                        "pick_x_machine": f"{mc['pick_xy_machine'][0]:.3f}",
                        "pick_y_machine": f"{mc['pick_xy_machine'][1]:.3f}",
                        "place_x_machine": f"{mc['place_xy_machine'][0]:.3f}",
                        "place_y_machine": f"{mc['place_xy_machine'][1]:.3f}",
                        "rotation_deg_machine": f"{mc['rotation_deg_machine']:.3f}",
                        "machine_unit": mc["machine_unit"],
                    }
                )
            writer.writerow(row)
    os.replace(temporary, path)
