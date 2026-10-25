"""提供 Task1 单元与集成测试共用的强类型内存替身和构造器。"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ec_26_e.cam import CameraFrame, CameraTimeoutError
from ec_26_e.control import MovePlan, PlanningOptions, PuzzlePiece
from ec_26_e.vision import (
    Detect1Result,
    PieceDetection,
    ResizeResult,
)


DEFAULT_RELATIVE_PIECES = (
    (
        (0.12, 0.15),
        (0.24, 0.15),
        (0.24, 0.28),
        (0.12, 0.28),
    ),
    (
        (0.58, 0.55),
        (0.72, 0.55),
        (0.65, 0.72),
    ),
)


def make_camera_frame(
    sequence_id: int,
    received_at_ns: int,
    *,
    image: np.ndarray | None = None,
) -> CameraFrame:
    """构造一张满足相机公共契约的只读 BGR 帧。"""

    source = (
        np.zeros((720, 1280, 3), dtype=np.uint8)
        if image is None
        else image.copy()
    )
    return CameraFrame(sequence_id, received_at_ns, source)


def make_resize_result(
    frame: CameraFrame,
    *,
    corner_offset_px: float = 0.0,
    image_bgr: np.ndarray | None = None,
) -> ResizeResult:
    """构造与指定相机帧尺寸关联的 A4 校正结果。"""

    source_height, source_width = frame.image_bgr.shape[:2]
    corners = (
        (100.0 + corner_offset_px, 80.0),
        (1180.0 + corner_offset_px, 80.0),
        (1180.0 + corner_offset_px, 640.0),
        (100.0 + corner_offset_px, 640.0),
    )
    rectified = (
        np.zeros((540, 764, 3), dtype=np.uint8)
        if image_bgr is None
        else image_bgr
    )
    return ResizeResult(
        image_bgr=rectified,
        source_size=(source_width, source_height),
        target_size=(764, 540),
        source_corners_px=corners,
        homography=np.eye(3, dtype=float),
        workspace_threshold=0.5,
        workspace_ratio=0.6,
        line_residual=0.2,
    )


def make_piece_detection(
    group_id: int,
    vertices_relative: tuple[tuple[float, float], ...],
    *,
    confidence: float = 0.9,
) -> PieceDetection:
    """由横向 A4 相对顶点构造完整 Detect1 纸片对象。"""

    corners_px = tuple(
        (round(u * 763.0, 2), round(v * 539.0, 2))
        for u, v in vertices_relative
    )
    center_relative = (
        sum(point[0] for point in vertices_relative)
        / len(vertices_relative),
        sum(point[1] for point in vertices_relative)
        / len(vertices_relative),
    )
    center_px = (
        round(center_relative[0] * 763.0, 2),
        round(center_relative[1] * 539.0, 2),
    )
    return PieceDetection(
        group_id=group_id,
        corners_px=corners_px,
        center_px=center_px,
        corners_relative=vertices_relative,
        center_relative=center_relative,
        area_px=1000.0,
        area_relative=0.01,
        polygon_iou=0.95,
        line_residual=0.2,
        confidence=confidence,
    )


def make_detect_result(
    pieces: Sequence[
        tuple[int, tuple[tuple[float, float], ...]]
        | PieceDetection
    ] = tuple(
        (index, vertices)
        for index, vertices in enumerate(DEFAULT_RELATIVE_PIECES)
    ),
) -> Detect1Result:
    """构造包含给定纸片且符合冻结契约的 Detect1 结果。"""

    detections = tuple(
        item
        if isinstance(item, PieceDetection)
        else make_piece_detection(item[0], item[1])
        for item in pieces
    )
    return Detect1Result(
        image_size=(764, 540),
        pieces=detections,
        foreground_threshold=0.4,
        workspace_area_px=350000,
    )


def make_move_plan(mode: str = "random") -> MovePlan:
    """构造无需实际求解即可用于编排断言的空移动方案。"""

    return MovePlan(
        mode=mode,  # type: ignore[arg-type]
        confidence=1.0,
        ambiguous=False,
        target_width_mm=100.0,
        target_height_mm=60.0,
        target_center_a4_mm=(105.0, 222.75),
        moves=(),
    )


def portrait_vertices_to_landscape_relative(
    vertices_mm: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    """按顺时针转换的逆变换生成 Detect1 横向相对顶点。"""

    return tuple(
        (y / 292.0, 1.0 - x / 207.0)
        for x, y in vertices_mm
    )


class FakeCamera:
    """按预设序列返回相机帧或异常，并记录生命周期事件。"""

    def __init__(
        self,
        items: Sequence[CameraFrame | BaseException],
    ) -> None:
        """保存预设读取结果并初始化事件记录。"""

        self._items = list(items)
        self.events: list[str] = []
        self.closed = False

    def start(self) -> None:
        """记录相机启动。"""

        self.events.append("start")

    def read_latest(
        self,
        after_sequence: int | None = None,
        timeout: float | None = None,
    ) -> CameraFrame:
        """返回下一项，并在耗尽时模拟读取超时。"""

        del after_sequence, timeout
        self.events.append("read")
        if not self._items:
            raise CameraTimeoutError("测试相机已无新帧")
        item = self._items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self) -> None:
        """记录幂等关闭状态。"""

        self.events.append("close")
        self.closed = True


class SequenceResizer:
    """按调用顺序返回 ResizeResult 或抛出预设异常。"""

    def __init__(
        self,
        template_frame: CameraFrame,
        factory: Callable[[CameraFrame], ResizeResult] | None = None,
        *,
        failures: int = 0,
    ) -> None:
        """保存可选结果工厂和前置失败次数。"""

        self.template_frame = template_frame
        self.factory = factory or make_resize_result
        self.failures = failures
        self.calls = 0

    def resize(self, image_bgr: np.ndarray) -> ResizeResult:
        """返回与同尺寸模板帧关联的校正结果。"""

        del image_bgr
        self.calls += 1
        if self.calls <= self.failures:
            raise ValueError("测试 Resize 失败")
        return self.factory(self.template_frame)


class ConstantDetector:
    """始终返回同一结构化 Detect1 结果的测试替身。"""

    def __init__(self, result: Detect1Result | None = None) -> None:
        """保存检测结果并初始化调用计数。"""

        self.result = result or make_detect_result()
        self.calls = 0

    def detect(self, image_bgr: np.ndarray) -> Detect1Result:
        """返回预设检测结果。"""

        del image_bgr
        self.calls += 1
        return self.result


class RecordingPlanner:
    """记录规划输入并返回预设 MovePlan。"""

    def __init__(
        self,
        camera: FakeCamera | None = None,
        *,
        error: BaseException | None = None,
    ) -> None:
        """保存可选相机状态断言和规划异常。"""

        self.camera = camera
        self.error = error
        self.calls: list[
            tuple[tuple[PuzzlePiece, ...], PlanningOptions]
        ] = []

    def __call__(
        self,
        pieces: Sequence[PuzzlePiece],
        options: PlanningOptions,
    ) -> MovePlan:
        """记录强类型输入，并确认规划发生在相机关闭后。"""

        if self.camera is not None and not self.camera.closed:
            raise AssertionError("规划器调用前必须关闭相机")
        self.calls.append((tuple(pieces), options))
        if self.error is not None:
            raise self.error
        return make_move_plan(options.mode)
