"""对统一视觉轨迹的法线角度和偏移执行 5～10 帧稳健聚合。"""

from __future__ import annotations

from collections import deque
import time

import cv2
import numpy as np

from ec_26_e.vision.contracts import (
    Detect1Result,
    PieceDetection,
    StableVisionConfig,
)
from ec_26_e.vision.detector import VisionDetector
from ec_26_e.vision.geometry import (
    canonicalize_corners,
    is_simple_polygon,
    polygon_corner_depths,
    polygon_edge_lengths,
    polygon_turn_angles_deg,
    relative_point,
)


def _polygon_lines(points: np.ndarray) -> np.ndarray:
    """将顺时针多边形的各边转换为向外法线角和偏移。"""

    lines = []
    for start, end in zip(points, np.roll(points, -1, axis=0)):
        direction = end - start
        direction /= max(float(np.linalg.norm(direction)), 1e-9)
        normal = np.array([direction[1], -direction[0]])
        lines.append((np.arctan2(normal[1], normal[0]), -float(normal @ start)))
    return np.asarray(lines, dtype=np.float64)


def _line_intersection(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
    """由角度、偏移式直线求交。"""

    matrix = np.asarray(
        [[np.cos(first[0]), np.sin(first[0])], [np.cos(second[0]), np.sin(second[0])]],
        dtype=np.float64,
    )
    if abs(float(np.linalg.det(matrix))) < 0.05:
        return None
    return np.linalg.solve(matrix, -np.asarray([first[1], second[1]]))


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    """计算实例关联使用的掩膜 IoU。"""

    intersection = np.count_nonzero((left > 0) & (right > 0))
    union = np.count_nonzero((left > 0) | (right > 0))
    return float(intersection / union) if union else 0.0


class StableVisionDetector:
    """为控制入口提供连续稳定结果，不改变单帧结果契约。"""

    def __init__(
        self,
        detector: VisionDetector,
        config: StableVisionConfig | None = None,
    ) -> None:
        """绑定单帧检测器并创建有界轨迹窗口。"""

        if not isinstance(detector, VisionDetector):
            raise TypeError("detector 必须是 VisionDetector")
        self.detector = detector
        self.config = config or StableVisionConfig()
        self._history: deque[tuple[tuple[PieceDetection, ...], tuple[np.ndarray, ...]]] = deque(
            maxlen=self.config.window_size
        )
        self.last_single_result: Detect1Result | None = None
        self.last_stabilizer_ms = 0.0
        self._closed = False

    def reset(self) -> None:
        """清空所有连续帧轨迹。"""

        self._history.clear()
        self.last_single_result = None

    def close(self) -> None:
        """幂等释放底层统一检测器及其 NPU 资源。"""

        if self._closed:
            return
        self._closed = True
        self.detector.close()

    def __enter__(self) -> StableVisionDetector:
        """进入确定性资源上下文。"""

        return self

    def __exit__(self, *_: object) -> None:
        """退出上下文并释放底层模型。"""

        self.close()

    def _compatible(self, pieces: tuple[PieceDetection, ...], masks: tuple[np.ndarray, ...]) -> bool:
        """检查当前帧是否延续实例数量、边数、位置和 mask 关联。"""

        if not self._history:
            return True
        previous_pieces, previous_masks = self._history[-1]
        if len(previous_pieces) != len(pieces):
            return False
        mm_px = 0.5 * (
            292.0 / max(self.last_single_result.image_size[0] - 1, 1)
            + 207.0 / max(self.last_single_result.image_size[1] - 1, 1)
        )
        for index, (previous, current) in enumerate(zip(previous_pieces, pieces)):
            if len(previous.corners_px) != len(current.corners_px):
                return False
            center_mm = np.linalg.norm(np.asarray(previous.center_px) - np.asarray(current.center_px)) * mm_px
            overlap = _mask_iou(previous_masks[index], masks[index]) if index < len(previous_masks) and index < len(masks) else 0.0
            if center_mm > self.config.max_center_distance_mm:
                return False
            if overlap < self.config.min_mask_iou and center_mm > 5.0:
                return False
        return True

    def _aggregate_piece(
        self,
        index: int,
        width: int,
        height: int,
        *,
        max_vertex_jitter_mm: float | None = None,
    ) -> PieceDetection | None:
        """对匹配轨迹的直线角和法向偏移取中位数后重新求交。"""

        observations = [frame[0][index] for frame in self._history]
        point_frames = [
            canonicalize_corners(
                np.asarray(piece.corners_px, dtype=np.float32)
            ).astype(np.float64)
            for piece in observations
        ]
        reference = _polygon_lines(point_frames[0])
        aligned_lines = []
        for points in point_frames:
            lines = _polygon_lines(points)
            adjusted = lines.copy()
            adjusted[:, 0] = reference[:, 0] + np.angle(np.exp(1j * (lines[:, 0] - reference[:, 0])))
            aligned_lines.append(adjusted)
        stable_lines = np.median(np.stack(aligned_lines), axis=0)
        corners = []
        for edge_index in range(len(stable_lines)):
            point = _line_intersection(stable_lines[edge_index - 1], stable_lines[edge_index])
            if point is None:
                return None
            corners.append(point)
        stable = canonicalize_corners(np.asarray(corners, dtype=np.float32))
        temporal_median = np.median(np.stack(point_frames), axis=0)
        mm_px = 0.5 * (292.0 / max(width - 1, 1) + 207.0 / max(height - 1, 1))
        geometry_config = self.detector.config
        if (
            not geometry_config.min_vertices
            <= len(stable)
            <= geometry_config.max_vertices
            or not is_simple_polygon(stable)
            or float(np.min(polygon_edge_lengths(stable)) * mm_px)
            <= geometry_config.minimum_physical_edge_mm
            or float(np.min(np.abs(polygon_turn_angles_deg(stable))))
            < geometry_config.minimum_corner_turn_deg
            or float(np.min(polygon_corner_depths(stable)) * mm_px)
            < geometry_config.minimum_corner_depth_mm
        ):
            return None
        maximum_jitter = max(
            float(np.max(np.linalg.norm(points - temporal_median, axis=1))) * mm_px
            for points in point_frames
        )
        jitter_limit = (
            self.config.max_vertex_jitter_mm
            if max_vertex_jitter_mm is None
            else max_vertex_jitter_mm
        )
        if maximum_jitter > jitter_limit:
            return None
        moments = cv2.moments(stable)
        center = (
            moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
        ) if abs(moments["m00"]) > 1e-9 else tuple(np.mean(stable, axis=0))
        corners_px = tuple((round(float(x), 2), round(float(y), 2)) for x, y in stable)
        center_px = (round(float(center[0]), 2), round(float(center[1]), 2))
        latest = observations[-1]
        area = float(cv2.contourArea(stable))
        return PieceDetection(
            group_id=index,
            corners_px=corners_px,
            center_px=center_px,
            corners_relative=tuple(
                relative_point(point, width, height)
                for point in corners_px
            ),
            center_relative=relative_point(center_px, width, height),
            area_px=round(area, 2),
            area_relative=round(area / (width * height), 6),
            polygon_iou=round(float(np.median([piece.polygon_iou for piece in observations])), 4),
            line_residual=round(float(np.median([piece.line_residual for piece in observations])), 3),
            confidence=round(float(np.median([piece.confidence for piece in observations])), 4),
        )

    def best_effort_result(
        self,
        *,
        max_vertex_jitter_mm: float = 8.0,
    ) -> Detect1Result | None:
        """在严格稳定未达标时返回仍一致的多帧聚合结果。

        这是正式流程的受控兜底，不会把不同片数、不同顶点数或明显跳变的
        帧混在一起。关联规则仍由 ``_compatible`` 守住；区别仅是允许的顶点
        抖动可比正常控制入口宽一些，避免轻微相机/A4 重投影漂移阻断解算。
        """

        if max_vertex_jitter_mm <= 0.0:
            raise ValueError("max_vertex_jitter_mm 必须大于 0")
        if self.last_single_result is None or len(self._history) < self.config.required_frames:
            return None
        latest_pieces = self._history[-1][0]
        if not latest_pieces:
            return None
        if any(len(frame_pieces) != len(latest_pieces) for frame_pieces, _ in self._history):
            return None
        stable_pieces: list[PieceDetection] = []
        for index in range(len(latest_pieces)):
            aggregate = self._aggregate_piece(
                index,
                *self.last_single_result.image_size,
                max_vertex_jitter_mm=max_vertex_jitter_mm,
            )
            if aggregate is None:
                return None
            stable_pieces.append(aggregate)
        return Detect1Result(
            image_size=self.last_single_result.image_size,
            pieces=tuple(stable_pieces),
            foreground_threshold=self.last_single_result.foreground_threshold,
            workspace_area_px=self.last_single_result.workspace_area_px,
        )

    def detect(self, a4_image_bgr: np.ndarray) -> Detect1Result:
        """返回达到连续稳定要求的结果；未确认时返回同契约空结果。"""

        started = time.perf_counter()
        single = self.detector.detect(a4_image_bgr)
        masks = tuple(
            item.refined_mask for item in self.detector.last_diagnostics.instances if item.failure is None
        )
        result = self.update(single, masks)
        self.last_stabilizer_ms = (
            (time.perf_counter() - started) * 1000.0
            - self.detector.last_diagnostics.stage_ms.get("total", 0.0)
        )
        return result

    def update(
        self,
        single: Detect1Result,
        masks: tuple[np.ndarray, ...] = (),
    ) -> Detect1Result:
        """接收已经计算的单帧结果，供四路离线诊断避免重复 NPU 推理。"""

        started = time.perf_counter()
        if not isinstance(single, Detect1Result):
            raise TypeError("single 必须是 Detect1Result")
        self.last_single_result = single
        pieces = tuple(sorted(single.pieces, key=lambda piece: (piece.center_px[1], piece.center_px[0])))
        if not self._compatible(pieces, masks):
            self._history.clear()
        self._history.append((pieces, masks))
        stable_pieces: list[PieceDetection] = []
        if len(self._history) >= self.config.required_frames and pieces:
            for index in range(len(pieces)):
                aggregate = self._aggregate_piece(index, *single.image_size)
                if aggregate is None:
                    stable_pieces.clear()
                    break
                stable_pieces.append(aggregate)
        self.last_stabilizer_ms = (time.perf_counter() - started) * 1000.0
        return Detect1Result(
            image_size=single.image_size,
            pieces=tuple(stable_pieces),
            foreground_threshold=single.foreground_threshold,
            workspace_area_px=single.workspace_area_px,
        )
