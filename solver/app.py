#!/usr/bin/env python3
"""Standalone visual workbench for the E-question rectangle puzzle solver."""
from __future__ import annotations

import json
import logging
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from threading import Event

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QAction, QColor, QImage, QPainter, QPainterPath, QPen, QPolygonF, QTransform
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import split, unary_union

from core.geometry import identity, matrix_pose_error, polygon_edges, rotation_translation, transform_point, transform_polygon
from core.models import Candidate, PuzzlePiece, SolveResult
from core.solver import RectanglePuzzleSolver


ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("solver-workbench")


def matrix_to_list(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def qtransform(matrix: np.ndarray) -> QTransform:
    return QTransform(matrix[0, 0], matrix[1, 0], matrix[0, 1], matrix[1, 1], matrix[0, 2], matrix[1, 2])


def qpath(poly: Polygon) -> QPainterPath:
    path = QPainterPath()
    coords = list(poly.exterior.coords)
    if not coords:
        return path
    path.moveTo(*coords[0])
    for point in coords[1:]:
        path.lineTo(*point)
    path.closeSubpath()
    return path


def qpolygon(poly: Polygon) -> QPolygonF:
    return QPolygonF([QPointF(x, y) for x, y in list(poly.exterior.coords)])


class PuzzleDocument:
    """Ground truth and the scattered, solver-visible piece geometry."""

    table_width = 190.0
    table_height = 135.0

    def __init__(self, width_mm: float = 100.0, height_mm: float = 60.0) -> None:
        self.width_mm = width_mm
        self.height_mm = height_mm
        self.mode = "白底"
        self.texture_path: str | None = None
        self.texture: QImage | None = None
        self.pieces: dict[int, Polygon] = {}
        self.scatter_poses: dict[int, np.ndarray] = {}
        self.observed: dict[int, Polygon] = {}
        self.history: list[dict[int, Polygon]] = []
        self.seed = 20260802
        self.reset(width_mm, height_mm)

    def reset(self, width_mm: float | None = None, height_mm: float | None = None) -> None:
        if width_mm is not None:
            self.width_mm = width_mm
        if height_mm is not None:
            self.height_mm = height_mm
        self.pieces = {0: Polygon([(0, 0), (self.width_mm, 0), (self.width_mm, self.height_mm), (0, self.height_mm)])}
        self.scatter_poses = {0: identity()}
        self.observed = {0: self.pieces[0]}
        self.history = []

    def set_texture(self, filename: str) -> bool:
        image = QImage(filename)
        if image.isNull():
            return False
        self.texture = image.convertToFormat(QImage.Format.Format_RGBA8888)
        self.texture_path = filename
        self.mode = "扑克牌"
        return True

    def cut(self, piece_id: int, p0: tuple[float, float], p1: tuple[float, float]) -> tuple[bool, str]:
        if len(self.pieces) >= 4:
            return False, "题目最多允许 4 块碎片。"
        poly = self.pieces[piece_id]
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        if math.hypot(dx, dy) < 2.0:
            return False, "切线过短。"
        # The selected piece is split by the infinite line through the drag.  This
        # makes endpoint snapping robust and enables a later cut to terminate at a T junction.
        norm = math.hypot(dx, dy)
        ux, uy = dx / norm, dy / norm
        far = max(self.width_mm, self.height_mm) * 5.0
        line = LineString([(p0[0] - ux * far, p0[1] - uy * far), (p1[0] + ux * far, p1[1] + uy * far)])
        result = split(poly, line)
        parts = [geom for geom in result.geoms if isinstance(geom, Polygon) and geom.area > 1e-5]
        if len(parts) != 2:
            return False, "拖拽线没有有效穿过选中的碎片。"
        trial = {**self.pieces, piece_id: parts[0], max(self.pieces) + 1: parts[1]}
        warning = self._validate_pieces(trial)
        if warning:
            return False, warning
        self.history.append(dict(self.pieces))
        self.pieces = trial
        self.scatter_poses = {piece: identity() for piece in self.pieces}
        self.observed = dict(self.pieces)
        return True, f"切分成功：当前 {len(self.pieces)} 块。"

    def cut_polyline(self, piece_id: int, points: list[tuple[float, float]]) -> tuple[bool, str]:
        """Split one piece with a finite two-segment polyline.

        Unlike ``cut()``, this preserves the user-selected endpoints and bend;
        it does not extend either segment into an infinite line.  The middle
        point selects the physical piece to cut, while both endpoints must be
        outside (or on the boundary of) that piece.
        """
        if len(self.pieces) >= 4:
            return False, "题目最多允许 4 块碎片。"
        if len(points) != 3:
            return False, "折线切割需要起点、折点、终点三个点。"
        poly = self.pieces[piece_id]
        start, bend, end = (Point(point) for point in points)
        if not poly.contains(bend):
            return False, "折点必须位于目标碎片内部。"
        if poly.contains(start) or poly.contains(end):
            return False, "折线的起点和终点必须位于目标碎片外部或边界上。"
        cutter = LineString(points)
        if not cutter.is_simple or cutter.length < 2.0:
            return False, "折线无效或过短。"
        result = split(poly, cutter)
        parts = [geom for geom in result.geoms if isinstance(geom, Polygon) and geom.area > 1e-5]
        if len(parts) != 2:
            return False, "折线必须从目标碎片一侧进入、经内部折点后从另一侧离开。"
        trial = {**self.pieces, piece_id: parts[0], max(self.pieces) + 1: parts[1]}
        warning = self._validate_pieces(trial)
        if warning:
            return False, warning
        self.history.append(dict(self.pieces))
        self.pieces = trial
        self.scatter_poses = {piece: identity() for piece in self.pieces}
        self.observed = dict(self.pieces)
        return True, f"折线切分成功：当前 {len(self.pieces)} 块。"

    def undo_cut(self) -> tuple[bool, str]:
        if not self.history:
            return False, "没有可撤销的切分。"
        self.pieces = self.history.pop()
        self.scatter_poses = {piece: identity() for piece in self.pieces}
        self.observed = dict(self.pieces)
        return True, f"已撤销切分：当前 {len(self.pieces)} 块。"

    @staticmethod
    def _validate_pieces(pieces: dict[int, Polygon]) -> str | None:
        for piece_id, poly in pieces.items():
            edges = polygon_edges(poly, piece_id)
            if len(edges) > 5:
                return "切分后有碎片超过 5 条边，不符合题目约束。"
            if min(edge.length for edge in edges) < 19.8:
                return "切分后存在短于 2 cm 的边，不符合题目约束。"
            if not poly.is_valid:
                return "切分产生了无效多边形。"
        return None

    def scatter(self, seed: int, jitter_mm: float = 0.0) -> str:
        self.seed = seed
        rng = random.Random(seed)
        poses: dict[int, np.ndarray] = {}
        observed: dict[int, Polygon] = {}
        ordered = sorted(self.pieces)
        for index, piece_id in enumerate(ordered):
            canonical = self.pieces[piece_id]
            cx, cy = canonical.centroid.x, canonical.centroid.y
            placed = False
            for _ in range(500):
                angle = rng.uniform(-math.pi, math.pi)
                rotate = rotation_translation(angle, 0.0, 0.0, (cx, cy))
                rotated = transform_polygon(canonical, rotate)
                minx, miny, maxx, maxy = rotated.bounds
                if maxx - minx > self.table_width - 10 or maxy - miny > self.table_height - 10:
                    continue
                tx = rng.uniform(5.0 - minx, self.table_width - 5.0 - maxx)
                ty = rng.uniform(5.0 - miny, self.table_height - 5.0 - maxy)
                pose = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]]) @ rotate
                candidate = transform_polygon(canonical, pose)
                if all(candidate.intersection(existing).area <= 0.05 for existing in observed.values()):
                    poses[piece_id] = pose
                    observed[piece_id] = candidate
                    placed = True
                    break
            if not placed:
                # Deterministic fallback still gives an independent input pose.
                pose = rotation_translation((index + 1) * 0.57, 12.0 + 42.0 * index, 18.0 + 23.0 * (index % 2), (cx, cy))
                poses[piece_id] = pose
                observed[piece_id] = transform_polygon(canonical, pose)
        self.scatter_poses = poses
        self.observed = {pid: self._jitter_polygon(poly, jitter_mm, seed + pid * 97) for pid, poly in observed.items()}
        return f"已按种子 {seed} 散落 {len(self.pieces)} 块碎片，顶点噪声 {jitter_mm:.1f} mm。"

    @staticmethod
    def _jitter_polygon(poly: Polygon, jitter: float, seed: int) -> Polygon:
        if jitter <= 0:
            return poly
        rng = random.Random(seed)
        raw = list(poly.exterior.coords)[:-1]
        for _ in range(8):
            points = [(x + rng.uniform(-jitter, jitter), y + rng.uniform(-jitter, jitter)) for x, y in raw]
            candidate = Polygon(points)
            if candidate.is_valid and candidate.area > poly.area * 0.75:
                return candidate
        return poly

    def solver_pieces(self) -> list[PuzzlePiece]:
        return [PuzzlePiece(piece_id, self.observed[piece_id]) for piece_id in sorted(self.observed)]

    def true_pose(self, piece_id: int) -> np.ndarray:
        return np.linalg.inv(self.scatter_poses[piece_id])

    def save(self, filename: str) -> None:
        payload = {
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "mode": self.mode,
            "texture_path": self.texture_path,
            "seed": self.seed,
            "pieces": {str(pid): [[float(x), float(y)] for x, y in list(poly.exterior.coords)[:-1]] for pid, poly in self.pieces.items()},
            "scatter_poses": {str(pid): matrix_to_list(pose) for pid, pose in self.scatter_poses.items()},
        }
        Path(filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, filename: str) -> None:
        payload = json.loads(Path(filename).read_text(encoding="utf-8"))
        self.width_mm = float(payload["width_mm"])
        self.height_mm = float(payload["height_mm"])
        self.mode = payload.get("mode", "白底")
        self.texture_path = payload.get("texture_path")
        self.texture = None
        if self.texture_path and Path(self.texture_path).exists():
            self.set_texture(self.texture_path)
        self.seed = int(payload.get("seed", 0))
        self.pieces = {int(pid): Polygon(coords) for pid, coords in payload["pieces"].items()}
        self.scatter_poses = {int(pid): np.asarray(pose, dtype=float) for pid, pose in payload["scatter_poses"].items()}
        self.observed = {pid: transform_polygon(self.pieces[pid], self.scatter_poses[pid]) for pid in self.pieces}
        self.history = []


class PuzzleCanvas(QWidget):
    cut_completed = Signal(bool, str)

    colors = [QColor("#60a5fa"), QColor("#f59e0b"), QColor("#34d399"), QColor("#f472b6")]

    def __init__(self, document: PuzzleDocument, view: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.document = document
        self.view = view
        self.candidate: Candidate | None = None
        self.cut_mode = "straight"
        self._drag_start: tuple[int, tuple[float, float]] | None = None
        self._drag_current: tuple[float, float] | None = None
        self._polyline_points: list[tuple[float, float]] = []
        self._polyline_piece_id: int | None = None
        self._polyline_hover: tuple[float, float] | None = None
        self._bounds = (0.0, 0.0, 100.0, 60.0)
        self.setMinimumSize(330, 260)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFrameStyle()

    def setFrameStyle(self) -> None:
        self.setStyleSheet("background: #fcfcfd; border: 1px solid #cbd5e1; border-radius: 5px;")

    def set_candidate(self, candidate: Candidate | None) -> None:
        self.candidate = candidate
        self.update()

    def set_cut_mode(self, mode: str) -> None:
        self.cut_mode = mode
        self._clear_cut_state()
        self.update()

    def _clear_cut_state(self) -> None:
        self._drag_start = None
        self._drag_current = None
        self._polyline_points = []
        self._polyline_piece_id = None
        self._polyline_hover = None

    def cancel_polyline(self, announce: bool = True) -> bool:
        active = bool(self._polyline_points)
        self._clear_cut_state()
        if active and announce:
            self.cut_completed.emit(False, "已取消当前折线切割。")
        self.update()
        return active

    def undo_polyline_point(self) -> bool:
        if not self._polyline_points:
            return False
        self._polyline_points.pop()
        if len(self._polyline_points) < 2:
            self._polyline_piece_id = None
        self.cut_completed.emit(False, "已撤回折线的上一个点。")
        self.update()
        return True

    def _scene_polygons(self) -> tuple[dict[int, Polygon], dict[int, np.ndarray]]:
        if self.view == "source":
            return dict(self.document.pieces), {pid: identity() for pid in self.document.pieces}
        if self.view == "scatter":
            return dict(self.document.observed), dict(self.document.scatter_poses)
        if self.candidate is None:
            return {}, {}
        raw_poses = {
            pid: self.candidate.poses[pid] @ self.document.scatter_poses[pid]
            for pid in self.document.pieces
        }
        raw_polygons = {
            pid: transform_polygon(self.document.pieces[pid], pose)
            for pid, pose in raw_poses.items()
        }
        # The geometry solver fixes one input piece as its root, so its global
        # output coordinate system is intentionally arbitrary.  Normalise it
        # only for this workbench display: the long side of the assembled MRR
        # is horizontal and points right, matching the eventual A4 placement.
        # Candidate poses and solver scoring remain untouched.
        upright = self._upright_solution_transform(raw_polygons)
        poses = {pid: upright @ pose for pid, pose in raw_poses.items()}
        polygons = {pid: transform_polygon(self.document.pieces[pid], pose) for pid, pose in poses.items()}
        return polygons, poses

    @staticmethod
    def _upright_solution_transform(polygons: dict[int, Polygon]) -> np.ndarray:
        merged = unary_union(list(polygons.values()))
        if merged.is_empty or merged.area <= 1e-8:
            return identity()
        frame = merged.minimum_rotated_rectangle
        corners = list(frame.exterior.coords)[:-1]
        if len(corners) != 4:
            return identity()
        edges = [
            np.asarray(corners[(index + 1) % 4]) - np.asarray(corners[index])
            for index in range(4)
        ]
        long_edge = max(edges, key=lambda edge: float(np.linalg.norm(edge)))
        if float(np.linalg.norm(long_edge)) <= 1e-8:
            return identity()
        rotate = rotation_translation(-math.atan2(float(long_edge[1]), float(long_edge[0])), 0.0, 0.0)
        rotated = [transform_polygon(poly, rotate) for poly in polygons.values()]
        minx = min(poly.bounds[0] for poly in rotated)
        miny = min(poly.bounds[1] for poly in rotated)
        translate = identity()
        translate[0, 2] = -minx
        translate[1, 2] = -miny
        return translate @ rotate

    def _compute_bounds(self) -> tuple[float, float, float, float]:
        if self.view == "source":
            return (-6.0, -6.0, self.document.width_mm + 6.0, self.document.height_mm + 6.0)
        if self.view == "scatter":
            return (0.0, 0.0, self.document.table_width, self.document.table_height)
        polygons, _ = self._scene_polygons()
        if not polygons:
            return (0.0, 0.0, 100.0, 60.0)
        minx = min(poly.bounds[0] for poly in polygons.values()) - 8.0
        miny = min(poly.bounds[1] for poly in polygons.values()) - 8.0
        maxx = max(poly.bounds[2] for poly in polygons.values()) + 8.0
        maxy = max(poly.bounds[3] for poly in polygons.values()) + 8.0
        return minx, miny, maxx, maxy

    def _scale_and_offset(self) -> tuple[float, float, float]:
        minx, miny, maxx, maxy = self._bounds
        margin = 18.0
        scale = min((self.width() - margin * 2) / max(maxx - minx, 1), (self.height() - margin * 2) / max(maxy - miny, 1))
        return scale, margin - minx * scale, margin - miny * scale

    def _screen_to_scene(self, point: QPointF) -> tuple[float, float]:
        scale, ox, oy = self._scale_and_offset()
        return (float((point.x() - ox) / scale), float((point.y() - oy) / scale))

    def paintEvent(self, event) -> None:  # type: ignore[override]
        self._bounds = self._compute_bounds()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#fcfcfd"))
        scale, ox, oy = self._scale_and_offset()
        painter.save()
        painter.translate(ox, oy)
        painter.scale(scale, scale)
        polygons, poses = self._scene_polygons()
        for piece_id in sorted(polygons):
            self._paint_piece(painter, piece_id, polygons[piece_id], poses[piece_id])
        if self.view == "source":
            self._paint_cut_preview(painter)
        if self.view == "solution" and polygons:
            merged = unary_union(list(polygons.values()))
            if merged.geom_type == "Polygon":
                painter.setPen(QPen(QColor("#0f172a"), 0.9, Qt.PenStyle.DashLine))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPolygon(qpolygon(merged.minimum_rotated_rectangle))
        painter.restore()
        title = {"source": "原始矩形 / 手动切分", "scatter": "散落碎片（解算输入）", "solution": "解算结果 / Top-K（已旋正展示）"}[self.view]
        painter.setPen(QColor("#334155"))
        painter.drawText(10, 20, title)
        if self.view == "source":
            painter.setPen(QColor("#64748b"))
            if self.cut_mode == "polyline":
                stage = len(self._polyline_points)
                hint = (
                    "折线切割：第 1 点请点矩形外（Esc 取消）"
                    if stage == 0
                    else "折线切割：第 2 点请点目标碎片内作为折点"
                    if stage == 1
                    else "折线切割：第 3 点请点矩形外完成（退格撤点，Esc 取消）"
                )
                painter.drawText(10, self.height() - 10, hint)
            else:
                painter.drawText(10, self.height() - 10, "在碎片内按住鼠标拖拽；红色虚线为松开后实际执行的切割线")
        elif self.view == "solution" and self.candidate is None:
            painter.setPen(QColor("#64748b"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "尚未开始解算")
        painter.end()

    def _paint_piece(self, painter: QPainter, piece_id: int, displayed: Polygon, pose: np.ndarray) -> None:
        canonical = self.document.pieces[piece_id]
        color = self.colors[piece_id % len(self.colors)]
        if self.document.texture is not None:
            painter.save()
            painter.setWorldTransform(qtransform(pose), True)
            painter.setClipPath(qpath(canonical))
            painter.drawImage(QRectF(0, 0, self.document.width_mm, self.document.height_mm), self.document.texture)
            painter.restore()
        else:
            painter.setBrush(color.lighter(160))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(qpolygon(displayed))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(color.darker(180), 0.8))
        painter.drawPolygon(qpolygon(displayed))
        c = displayed.centroid
        painter.setPen(QColor("#0f172a"))
        painter.drawText(QPointF(c.x, c.y), f"{piece_id + 1}")

    def _paint_cut_preview(self, painter: QPainter) -> None:
        """Render the same infinite-line intersection used by PuzzleDocument.cut."""
        if self.cut_mode == "polyline":
            self._paint_polyline_preview(painter)
            return
        if self._drag_start is None or self._drag_current is None:
            return
        piece_id, start = self._drag_start
        current = self._drag_current
        dx, dy = current[0] - start[0], current[1] - start[1]
        length = math.hypot(dx, dy)
        if length < 0.5:
            return
        ux, uy = dx / length, dy / length
        far = max(self.document.width_mm, self.document.height_mm) * 5.0
        guide = LineString(
            [(start[0] - ux * far, start[1] - uy * far), (current[0] + ux * far, current[1] + uy * far)]
        )
        intersection = self.document.pieces[piece_id].intersection(guide)
        painter.save()
        painter.setPen(QPen(QColor("#ef4444"), 1.4, Qt.PenStyle.DashLine))
        if intersection.geom_type == "LineString":
            coords = list(intersection.coords)
            if len(coords) >= 2:
                painter.drawLine(QPointF(*coords[0]), QPointF(*coords[-1]))
        elif intersection.geom_type == "MultiLineString":
            for line in intersection.geoms:
                coords = list(line.coords)
                if len(coords) >= 2:
                    painter.drawLine(QPointF(*coords[0]), QPointF(*coords[-1]))
        painter.setPen(QPen(QColor("#2563eb"), 0.8, Qt.PenStyle.SolidLine))
        painter.drawLine(QPointF(*start), QPointF(*current))
        painter.restore()

    def _paint_polyline_preview(self, painter: QPainter) -> None:
        if not self._polyline_points:
            return
        painter.save()
        painter.setBrush(QColor("#f97316"))
        painter.setPen(QPen(QColor("#c2410c"), 0.8))
        for index, point in enumerate(self._polyline_points, start=1):
            painter.drawEllipse(QPointF(*point), 1.2, 1.2)
            painter.drawText(QPointF(point[0] + 1.6, point[1] - 1.6), str(index))
        if len(self._polyline_points) >= 2:
            painter.setPen(QPen(QColor("#f97316"), 1.3, Qt.PenStyle.SolidLine))
            for first, second in zip(self._polyline_points, self._polyline_points[1:]):
                painter.drawLine(QPointF(*first), QPointF(*second))
        if self._polyline_hover is not None:
            painter.setPen(QPen(QColor("#2563eb"), 1.1, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(*self._polyline_points[-1]), QPointF(*self._polyline_hover))
        if self._polyline_piece_id is not None and len(self._polyline_points) >= 2:
            route = list(self._polyline_points)
            if self._polyline_hover is not None:
                route.append(self._polyline_hover)
            if len(route) >= 2:
                clipped = self.document.pieces[self._polyline_piece_id].intersection(LineString(route))
                painter.setPen(QPen(QColor("#ef4444"), 1.6, Qt.PenStyle.DashLine))
                if clipped.geom_type == "LineString":
                    coords = list(clipped.coords)
                    if len(coords) >= 2:
                        painter.drawPolyline(QPolygonF([QPointF(*point) for point in coords]))
                elif clipped.geom_type == "MultiLineString":
                    for line in clipped.geoms:
                        coords = list(line.coords)
                        if len(coords) >= 2:
                            painter.drawPolyline(QPolygonF([QPointF(*point) for point in coords]))
        painter.restore()

    def _point_outside_board(self, point: tuple[float, float]) -> bool:
        return not (0.0 <= point[0] <= self.document.width_mm and 0.0 <= point[1] <= self.document.height_mm)

    def _piece_at(self, point: tuple[float, float]) -> int | None:
        probe = Point(point)
        for piece_id, poly in self.document.pieces.items():
            if poly.contains(probe):
                return piece_id
        return None

    def _polyline_click(self, point: tuple[float, float]) -> None:
        stage = len(self._polyline_points)
        if stage == 0:
            if not self._point_outside_board(point):
                self.cut_completed.emit(False, "折线起点请点在原始矩形外。")
                return
            self._polyline_points.append(point)
        elif stage == 1:
            piece_id = self._piece_at(point)
            if piece_id is None:
                self.cut_completed.emit(False, "折点必须点在某一块碎片内部。")
                return
            self._polyline_piece_id = piece_id
            self._polyline_points.append(point)
        else:
            if not self._point_outside_board(point):
                self.cut_completed.emit(False, "折线终点请点在原始矩形外。")
                return
            piece_id = self._polyline_piece_id
            points = [*self._polyline_points, point]
            self._clear_cut_state()
            if piece_id is None:
                self.cut_completed.emit(False, "没有识别到折线目标碎片。")
                return
            success, message = self.document.cut_polyline(piece_id, points)
            self.cut_completed.emit(success, message)
        self.update()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if self.view != "source":
            return super().mousePressEvent(event)
        self.setFocus()
        if self.cut_mode == "polyline":
            if event.button() == Qt.MouseButton.RightButton:
                self.cancel_polyline()
                event.accept()
                return
            if event.button() == Qt.MouseButton.LeftButton:
                self._polyline_click(self._screen_to_scene(event.position()))
                event.accept()
                return
            return super().mousePressEvent(event)
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        point = self._screen_to_scene(event.position())
        for piece_id, poly in self.document.pieces.items():
            if poly.buffer(0.6).contains(Point(point)):
                self._drag_start = (piece_id, point)
                self._drag_current = point
                self.update()
                return

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self.view == "source" and self.cut_mode == "polyline" and self._polyline_points:
            self._polyline_hover = self._screen_to_scene(event.position())
            self.update()
            event.accept()
            return
        if self.view == "source" and self._drag_start is not None:
            self._drag_current = self._screen_to_scene(event.position())
            self.update()
            event.accept()
            return
        return super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if self.cut_mode == "polyline":
            return super().mouseReleaseEvent(event)
        if self.view != "source" or self._drag_start is None or event.button() != Qt.MouseButton.LeftButton:
            return super().mouseReleaseEvent(event)
        piece_id, start = self._drag_start
        self._drag_start = None
        current = self._screen_to_scene(event.position())
        self._drag_current = None
        success, message = self.document.cut(piece_id, start, current)
        self.cut_completed.emit(success, message)
        self.update()

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if self.view == "source" and self.cut_mode == "polyline":
            if event.key() == Qt.Key.Key_Escape:
                if self.cancel_polyline():
                    event.accept()
                    return
            if event.key() == Qt.Key.Key_Backspace:
                if self.undo_polyline_point():
                    event.accept()
                    return
        return super().keyPressEvent(event)


class SolverWorker(QObject):
    log = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, pieces: list[PuzzlePiece], timeout_s: float, cancel: Event) -> None:
        super().__init__()
        self.pieces = pieces
        self.timeout_s = timeout_s
        self.cancel = cancel

    def run(self) -> None:
        try:
            result = RectanglePuzzleSolver().solve(
                self.pieces,
                self.timeout_s,
                self.log.emit,
                self.cancel,
                allow_best_effort=True,
            )
            self.finished.emit(result)
        except Exception as exc:  # pragma: no cover - surfaced in UI
            LOG.exception("solver worker failed")
            self.failed.emit(repr(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("拼图解算器实验台（独立验证版）")
        self.resize(1500, 900)
        self.document = PuzzleDocument()
        self.result: SolveResult | None = None
        self.cancel_event: Event | None = None
        self.worker_thread: QThread | None = None
        self.solver_started_at = 0.0
        self._build_ui()
        self._append_log("实验台就绪：请在左侧矩形上拖拽切分，再随机散落并解算。")

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        controls = self._make_controls()
        layout.addWidget(controls, 0)
        center = QWidget()
        grid = QGridLayout(center)
        self.source_canvas = PuzzleCanvas(self.document, "source")
        self.scatter_canvas = PuzzleCanvas(self.document, "scatter")
        self.solution_canvas = PuzzleCanvas(self.document, "solution")
        self.source_canvas.cut_completed.connect(self._cut_completed)
        self.source_canvas.set_cut_mode(str(self.cut_mode_box.currentData()))
        grid.addWidget(self.source_canvas, 0, 0)
        grid.addWidget(self.scatter_canvas, 0, 1)
        grid.addWidget(self.solution_canvas, 0, 2)
        self.candidate_list = QListWidget()
        self.candidate_list.currentRowChanged.connect(self._select_candidate)
        grid.addWidget(self.candidate_list, 1, 0, 1, 2)
        self.metrics = QLabel("尚无解算结果")
        self.metrics.setWordWrap(True)
        self.metrics.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.metrics.setStyleSheet("padding: 8px; background: #f8fafc; border: 1px solid #cbd5e1;")
        grid.addWidget(self.metrics, 1, 2)
        layout.addWidget(center, 1)
        logs = QWidget()
        log_layout = QVBoxLayout(logs)
        log_layout.addWidget(QLabel("实时状态日志（命令行同步输出）"))
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(1500)
        log_layout.addWidget(self.log_view)
        logs.setMinimumWidth(340)
        layout.addWidget(logs, 0)
        self.cut_mode_box.currentIndexChanged.connect(self._change_cut_mode)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

    def _make_controls(self) -> QWidget:
        panel = QWidget()
        panel.setMaximumWidth(265)
        box = QVBoxLayout(panel)
        dimensions = QGroupBox("测试矩形")
        form = QFormLayout(dimensions)
        self.width_box = QDoubleSpinBox()
        self.width_box.setRange(90.0, 120.0)
        self.width_box.setValue(100.0)
        self.width_box.setSuffix(" mm")
        self.height_box = QDoubleSpinBox()
        self.height_box.setRange(50.0, 90.0)
        self.height_box.setValue(60.0)
        self.height_box.setSuffix(" mm")
        self.mode_box = QComboBox()
        self.mode_box.addItems(["白底", "扑克牌"])
        self.cut_mode_box = QComboBox()
        self.cut_mode_box.addItem("直线拖拽", "straight")
        self.cut_mode_box.addItem("三点折线", "polyline")
        form.addRow("长边", self.width_box)
        form.addRow("短边", self.height_box)
        form.addRow("模式", self.mode_box)
        form.addRow("切割方式", self.cut_mode_box)
        self.reset_button = QPushButton("新建完整矩形")
        self.reset_button.clicked.connect(self._reset)
        form.addRow(self.reset_button)
        self.undo_button = QPushButton("撤销上一次切分")
        self.undo_button.clicked.connect(self._undo_cut)
        form.addRow(self.undo_button)
        self.fullscreen_button = QPushButton("进入全屏（F11）")
        self.fullscreen_button.clicked.connect(self._toggle_fullscreen)
        form.addRow(self.fullscreen_button)
        self.open_image_button = QPushButton("导入扑克牌图片")
        self.open_image_button.clicked.connect(self._open_image)
        form.addRow(self.open_image_button)
        box.addWidget(dimensions)

        simulation = QGroupBox("散落与解算")
        sim_form = QFormLayout(simulation)
        self.seed_box = QDoubleSpinBox()
        self.seed_box.setDecimals(0)
        self.seed_box.setRange(0, 999999999)
        self.seed_box.setValue(20260802)
        self.jitter_box = QDoubleSpinBox()
        self.jitter_box.setRange(0.0, 3.0)
        self.jitter_box.setSingleStep(0.2)
        self.jitter_box.setSuffix(" mm")
        self.timeout_box = QDoubleSpinBox()
        self.timeout_box.setRange(1.0, 70.0)
        self.timeout_box.setValue(70.0)
        self.timeout_box.setSuffix(" s")
        self.countdown_label = QLabel("未开始解算")
        self.countdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.countdown_label.setStyleSheet("font-weight: 600; color: #0f172a; padding: 6px; background: #e2e8f0; border-radius: 4px;")
        self.scatter_button = QPushButton("随机散落")
        self.scatter_button.clicked.connect(self._scatter)
        self.solve_button = QPushButton("开始解算")
        self.solve_button.clicked.connect(self._solve)
        self.cancel_button = QPushButton("取消解算")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)
        sim_form.addRow("随机种子", self.seed_box)
        sim_form.addRow("顶点噪声", self.jitter_box)
        sim_form.addRow("时间上限", self.timeout_box)
        sim_form.addRow("结算计时", self.countdown_label)
        sim_form.addRow(self.scatter_button)
        sim_form.addRow(self.solve_button)
        sim_form.addRow(self.cancel_button)
        box.addWidget(simulation)

        files = QGroupBox("案例")
        files_layout = QVBoxLayout(files)
        save_button = QPushButton("保存案例 JSON")
        save_button.clicked.connect(self._save_case)
        load_button = QPushButton("加载案例 JSON")
        load_button.clicked.connect(self._load_case)
        files_layout.addWidget(save_button)
        files_layout.addWidget(load_button)
        box.addWidget(files)
        box.addStretch(1)
        return panel

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showMaximized()
            self.fullscreen_button.setText("进入全屏（F11）")
            self._append_log("已退出全屏，窗口保持最大化。")
        else:
            self.showFullScreen()
            self.fullscreen_button.setText("退出全屏（F11）")
            self._append_log("已进入全屏。")

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_F11:
            self._toggle_fullscreen()
            event.accept()
            return
        return super().keyPressEvent(event)

    def _change_cut_mode(self, _index: int | None = None) -> None:
        mode = str(self.cut_mode_box.currentData())
        self.source_canvas.set_cut_mode(mode)
        description = "直线拖拽切割" if mode == "straight" else "三点折线切割：外部起点 → 内部折点 → 外部终点"
        self._append_log(f"已切换为{description}。")

    def _reset(self) -> None:
        self.source_canvas.cancel_polyline(announce=False)
        self.document.mode = self.mode_box.currentText()
        if self.document.mode == "白底":
            self.document.texture = None
            self.document.texture_path = None
        self.document.reset(self.width_box.value(), self.height_box.value())
        self.result = None
        self.candidate_list.clear()
        self.solution_canvas.set_candidate(None)
        self._append_log("已新建完整矩形。")
        self._refresh()

    def _undo_cut(self) -> None:
        self.source_canvas.cancel_polyline(announce=False)
        success, message = self.document.undo_cut()
        self._append_log(message)
        if success:
            self.result = None
            self.candidate_list.clear()
            self.solution_canvas.set_candidate(None)
            self._refresh()

    def _open_image(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "选择扑克牌正面图片", str(ROOT / "assets"), "Image files (*.png *.jpg *.jpeg *.bmp)")
        if not filename:
            return
        if not self.document.set_texture(filename):
            QMessageBox.warning(self, "图片错误", "无法读取该图片。")
            return
        self.mode_box.setCurrentText("扑克牌")
        self._append_log(f"已导入扑克牌测试图：{filename}。完整图仅用于切分与真值，不交给解算核心。")
        self._refresh()

    def _cut_completed(self, success: bool, message: str) -> None:
        self._append_log(message)
        if success:
            self.result = None
            self.candidate_list.clear()
            self.solution_canvas.set_candidate(None)
            self._refresh()

    def _scatter(self) -> None:
        message = self.document.scatter(int(self.seed_box.value()), self.jitter_box.value())
        self.result = None
        self.candidate_list.clear()
        self.solution_canvas.set_candidate(None)
        self._append_log(message)
        self._refresh()

    def _solve(self) -> None:
        if len(self.document.pieces) < 2:
            QMessageBox.information(self, "需要切分", "请先至少切成两块碎片。")
            return
        if self.worker_thread is not None:
            return
        self._append_log("开始后台解算；界面保持可操作，可随时取消。")
        self.result = None
        self.candidate_list.clear()
        self.solution_canvas.set_candidate(None)
        self.cancel_event = Event()
        self.solve_timeout_s = float(self.timeout_box.value())
        self.worker_thread = QThread(self)
        self.worker = SolverWorker(self.document.solver_pieces(), self.solve_timeout_s, self.cancel_event)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.log.connect(self._append_log)
        self.worker.finished.connect(self._solved)
        self.worker.failed.connect(self._solver_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self._thread_finished)
        self.solve_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.solver_started_at = time.monotonic()
        self.countdown_label.setText(f"结算中：剩余 {self.solve_timeout_s:.1f} s\n已用 0.0 s")
        self.countdown_label.setStyleSheet("font-weight: 700; color: #1d4ed8; padding: 6px; background: #dbeafe; border-radius: 4px;")
        self.timer.start(100)
        self.worker_thread.start()

    def _cancel(self) -> None:
        if self.cancel_event is not None:
            self.cancel_event.set()
            self._append_log("已请求取消，等待当前搜索节点安全退出。")

    def _tick(self) -> None:
        if self.worker_thread is not None:
            elapsed = time.monotonic() - self.solver_started_at
            remaining = max(0.0, self.solve_timeout_s - elapsed)
            self.countdown_label.setText(f"结算中：剩余 {remaining:.1f} s\n已用 {elapsed:.1f} s")
            if remaining <= 10.0:
                self.countdown_label.setStyleSheet("font-weight: 700; color: #b91c1c; padding: 6px; background: #fee2e2; border-radius: 4px;")
            self.statusBar().showMessage(f"正在解算：剩余 {remaining:.1f} s，已用 {elapsed:.1f} s")

    def _solved(self, result: SolveResult) -> None:
        if self.document.texture is not None and result.candidates:
            self._append_log("几何候选已生成；开始依据各拼缝两侧的局部纹理连续性重新排序。")
            for candidate in result.candidates:
                continuity = self._texture_continuity(candidate)
                candidate.metrics["texture_continuity"] = continuity
                # The original full image is used only as each physical piece's
                # own texture sampler.  No global-card template comparison occurs.
                candidate.score += 28.0 * continuity
            result.candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        # Include every phase after the worker begins (including texture ranking)
        # in the user-visible competition time.
        result.elapsed_s = time.monotonic() - self.solver_started_at
        self.result = result
        self.countdown_label.setText(f"结算完成：用时 {result.elapsed_s:.2f} s\n状态：{result.status}")
        self.countdown_label.setStyleSheet("font-weight: 700; color: #166534; padding: 6px; background: #dcfce7; border-radius: 4px;")
        self._append_log(f"解算结束：{result.status}，用时 {result.elapsed_s:.2f} s，Top-K={len(result.candidates)}。")
        for rank, candidate in enumerate(result.candidates, start=1):
            truth_text = self._truth_text(candidate)
            item = QListWidgetItem(f"#{rank}  分数 {candidate.score:.2f}  覆盖 {candidate.metrics['coverage']:.4f}  {truth_text}")
            item.setData(Qt.ItemDataRole.UserRole, rank - 1)
            self.candidate_list.addItem(item)
        if result.candidates:
            self.candidate_list.setCurrentRow(0)
        else:
            self.metrics.setText(f"{result.status}\n用时 {result.elapsed_s:.2f} s，展开 {result.expanded_states} 个状态。")
        self._write_report(result)

    def _solver_failed(self, detail: str) -> None:
        elapsed = time.monotonic() - self.solver_started_at
        self.countdown_label.setText(f"结算异常：用时 {elapsed:.2f} s")
        self.countdown_label.setStyleSheet("font-weight: 700; color: #b91c1c; padding: 6px; background: #fee2e2; border-radius: 4px;")
        self._append_log(f"解算器异常：{detail}")
        QMessageBox.critical(self, "解算器异常", detail)

    def _thread_finished(self) -> None:
        if self.worker_thread is not None:
            self.worker_thread.deleteLater()
        self.worker_thread = None
        self.worker = None
        self.cancel_event = None
        self.solve_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.timer.stop()
        self.statusBar().showMessage(self.countdown_label.text().replace("\n", "；"))

    def _select_candidate(self, row: int) -> None:
        if self.result is None or not (0 <= row < len(self.result.candidates)):
            return
        candidate = self.result.candidates[row]
        self.solution_canvas.set_candidate(candidate)
        metrics = candidate.metrics
        texture_line = f"花纹连续性：{metrics['texture_continuity']:.4f}\n" if "texture_continuity" in metrics else ""
        frame_line = f"外框直角：{int(metrics['frame_corners'])} / 4\n" if "frame_corners" in metrics else ""
        outer_line = f"满足外边碎片：{int(metrics['outer_piece_count'])} / {len(candidate.poses)}\n" if "outer_piece_count" in metrics else ""
        topology_line = (
            f"实际相邻块：{int(metrics['neighbor_pairs'])} 对；闭环：{int(metrics['closed_cycles'])}\n"
            if "neighbor_pairs" in metrics
            else ""
        )
        text = (
            f"候选 #{row + 1}\n"
            f"总分：{candidate.score:.3f}\n"
            f"矩形覆盖率：{metrics['coverage']:.5f}\n"
            f"重叠面积：{metrics['overlap_mm2']:.3f} mm²\n"
            f"外框：{metrics['long_mm']:.2f} × {metrics['short_mm']:.2f} mm\n"
            + frame_line
            + outer_line
            + topology_line
            + f"拼缝总长：{metrics['seam_mm']:.2f} mm\n"
            + texture_line
            + f"{self._truth_text(candidate)}\n"
            + "接口：" + ", ".join(seam['kind'] for seam in candidate.seams)
        )
        self.metrics.setText(text)

    def _texture_continuity(self, candidate: Candidate) -> float:
        """Score only pixels belonging to the two pieces adjacent to each seam.

        In a real camera adapter these samples come from each segmented piece crop.
        The workbench reconstructs those crops from the source image but never
        compares a candidate to an intact-card template.
        """
        image = self.document.texture
        if image is None or not candidate.seams:
            return 0.0
        total_weight = 0.0
        total_score = 0.0
        for seam in candidate.seams:
            a_id, b_id = seam["a_piece"], seam["b_piece"]
            a_edges = polygon_edges(self.document.observed[a_id], a_id)
            edge = a_edges[seam["a_edge"]]
            start = transform_point(candidate.poses[a_id], edge.p0)
            end = transform_point(candidate.poses[a_id], edge.p1)
            vector = end - start
            length = float(np.linalg.norm(vector))
            if length < 4.0:
                continue
            normal = np.array([-vector[1], vector[0]]) / length
            a_shape = transform_polygon(self.document.observed[a_id], candidate.poses[a_id])
            b_shape = transform_polygon(self.document.observed[b_id], candidate.poses[b_id])
            seam_error: list[float] = []
            for fraction in np.linspace(0.12, 0.88, 9):
                center = start + vector * fraction
                a_point = center + normal * 0.8
                if not a_shape.buffer(0.05).contains(Point(a_point)):
                    a_point = center - normal * 0.8
                b_point = center - normal * 0.8
                if not b_shape.buffer(0.05).contains(Point(b_point)):
                    b_point = center + normal * 0.8
                color_a = self._piece_color(a_id, candidate, a_point, image)
                color_b = self._piece_color(b_id, candidate, b_point, image)
                if color_a is not None and color_b is not None:
                    seam_error.append(float(np.mean(np.abs(color_a - color_b))) / 255.0)
            if seam_error:
                # A blank white region is neutral rather than decisive; it still
                # contributes a little but cannot overwhelm geometry evidence.
                local = max(0.0, 1.0 - float(np.mean(seam_error)) * 2.0)
                weight = min(length, 60.0)
                total_score += local * weight
                total_weight += weight
        return total_score / total_weight if total_weight else 0.0

    def _piece_color(self, piece_id: int, candidate: Candidate, point_in_solution: np.ndarray, image: QImage) -> np.ndarray | None:
        raw_point = transform_point(np.linalg.inv(candidate.poses[piece_id]), point_in_solution)
        canonical = transform_point(np.linalg.inv(self.document.scatter_poses[piece_id]), raw_point)
        if not self.document.pieces[piece_id].buffer(0.1).contains(Point(canonical)):
            return None
        x = min(image.width() - 1, max(0, int(canonical[0] / self.document.width_mm * image.width())))
        y = min(image.height() - 1, max(0, int(canonical[1] / self.document.height_mm * image.height())))
        color = image.pixelColor(x, y)
        return np.array([color.red(), color.green(), color.blue()], dtype=float)

    def _truth_text(self, candidate: Candidate) -> str:
        if self.jitter_box.value() > 0:
            return "真值比对：已加入噪声，仅供参考"
        root_id = min(candidate.poses)
        global_pose = self.document.true_pose(root_id) @ np.linalg.inv(candidate.poses[root_id])
        errors = [matrix_pose_error(global_pose @ candidate.poses[pid], self.document.true_pose(pid)) for pid in candidate.poses]
        max_mm = max(error[0] for error in errors)
        max_deg = math.degrees(max(error[1] for error in errors))
        label = "真值命中" if max_mm < 0.5 and max_deg < 0.5 else "真值未命中"
        return f"{label}：最大 {max_mm:.2f} mm / {max_deg:.2f}°"

    def _save_case(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(self, "保存案例", str(ROOT / "cases" / "case.json"), "JSON (*.json)")
        if not filename:
            return
        self.document.save(filename)
        self._append_log(f"案例已保存：{filename}")

    def _load_case(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "加载案例", str(ROOT / "cases"), "JSON (*.json)")
        if not filename:
            return
        try:
            self.document.load(filename)
        except Exception as exc:
            QMessageBox.critical(self, "加载失败", repr(exc))
            return
        self.width_box.setValue(self.document.width_mm)
        self.height_box.setValue(self.document.height_mm)
        self.mode_box.setCurrentText(self.document.mode)
        self.result = None
        self.candidate_list.clear()
        self.solution_canvas.set_candidate(None)
        self._append_log(f"案例已加载：{filename}")
        self._refresh()

    def _write_report(self, result: SolveResult) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        filename = REPORT_DIR / f"solve-{stamp}.json"
        payload = {
            "status": result.status,
            "elapsed_s": result.elapsed_s,
            "expanded_states": result.expanded_states,
            "piece_count": len(self.document.pieces),
            "seed": self.document.seed,
            "candidates": [
                {"score": c.score, "metrics": c.metrics, "seams": c.seams, "poses": {str(k): matrix_to_list(v) for k, v in c.poses.items()}}
                for c in result.candidates
            ],
        }
        filename.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._append_log(f"结果报告已保存：{filename}")

    def _append_log(self, text: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {text}"
        LOG.info(text)
        self.log_view.appendPlainText(line)

    def _refresh(self) -> None:
        self.source_canvas.update()
        self.scatter_canvas.update()
        self.solution_canvas.update()


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
