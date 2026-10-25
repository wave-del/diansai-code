"""验证统一视觉语义门控、结果契约和稳定器。"""

from __future__ import annotations

import cv2
import numpy as np
from unittest.mock import patch

from ec_26_e.vision import (
    Detect1Config,
    Detect1Result,
    NativeTiming,
    PieceDetection,
    SegInstance,
    StableVisionConfig,
    StableVisionDetector,
    VisionConfig,
    VisionDetector,
    create_production_detector,
)
from ec_26_e.vision.refine import _choose_polygon
from ec_26_e.vision.geometry import is_simple_polygon
from ec_26_e.vision.detector import _suppress_contained_duplicates


class _FakeInferencer:
    """返回固定语义实例，避免单元测试访问真实 NPU。"""

    def __init__(self, mask: np.ndarray) -> None:
        """保存要按值复制返回的唯一实例掩膜。"""

        self.mask = mask
        self.last_timing = NativeTiming(network_ms=1.0, total_ms=2.0)
        self.driver_version = 123

    def infer(self, image_bgr: np.ndarray) -> tuple[SegInstance, ...]:
        """返回唯一矩形实例。"""

        return (SegInstance((80.0, 80.0, 230.0, 230.0), 0.9, 0, self.mask.copy()),)


class _ClosableVisionDetector(VisionDetector):
    """只记录释放次数的统一视觉测试替身。"""

    def __init__(self) -> None:
        """跳过 NPU 初始化。"""

        self.close_calls = 0

    def close(self) -> None:
        """记录底层释放次数。"""

        self.close_calls += 1


def _scene() -> tuple[np.ndarray, np.ndarray]:
    """构造含两个传统视觉前景但仅一个 YOLO 候选的横向 A4 图。"""

    image = np.zeros((540, 764, 3), dtype=np.uint8)
    image[:] = (180, 90, 30)
    image[90:220, 90:220] = (240, 240, 240)
    image[300:430, 480:620] = (240, 240, 240)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[88:222, 88:222] = 255
    return image, mask


def test_vision_never_adds_candidate_outside_yolo_mask() -> None:
    """传统视觉不能把第二个白色区域新增为实例。"""

    image, mask = _scene()
    detector = VisionDetector(VisionConfig(), _FakeInferencer(mask))
    source = image.copy()
    result = detector.detect(image)
    assert isinstance(result, Detect1Result)
    assert np.array_equal(image, source)
    assert len(result.pieces) == 1
    assert result.pieces[0].center_px[0] < 300
    assert detector.last_diagnostics.yolo_count == 1
    assert detector.last_diagnostics.refined_count == 1
    diagnostic = detector.last_diagnostics.instances[0]
    assert diagnostic.boundary_source == "global_foreground"
    assert diagnostic.edge_sources
    assert all(source.startswith("visible_edge") or source == "foreground_contour" for source in diagnostic.edge_sources)


def test_vision_never_returns_geometry_from_coarse_mask_alone() -> None:
    """YOLO 有 mask 但原图无纸片边缘时不得复制 mask 多边形。"""

    image, mask = _scene()
    image[:] = (180, 90, 30)
    detector = VisionDetector(VisionConfig(), _FakeInferencer(mask))

    result = detector.detect(image)

    assert result.pieces == ()
    assert detector.last_diagnostics.yolo_count == 1
    assert detector.last_diagnostics.refined_count == 0
    assert detector.last_diagnostics.instances[0].failure is not None


def test_contained_smaller_duplicate_is_suppressed() -> None:
    """高度包含的较小 mask 应按面积去重，不依赖置信度高低。"""

    outer = np.zeros((100, 100), dtype=np.uint8)
    outer[10:90, 10:90] = 255
    inner = np.zeros_like(outer)
    inner[10:50, 10:90] = 255
    instances = (
        SegInstance((10.0, 10.0, 90.0, 90.0), 0.48, 0, outer),
        SegInstance((10.0, 10.0, 90.0, 50.0), 0.72, 0, inner),
    )

    kept, removed = _suppress_contained_duplicates(instances)

    assert len(kept) == 1
    assert kept[0] is instances[0]
    assert removed == 1


def test_low_score_noncontained_instance_is_preserved() -> None:
    """孤立低分真实候选不得因全局分数较低而被去重。"""

    left = np.zeros((100, 100), dtype=np.uint8)
    left[10:50, 10:50] = 255
    right = np.zeros_like(left)
    right[50:90, 50:90] = 255
    instances = (
        SegInstance((10.0, 10.0, 50.0, 50.0), 0.91, 0, left),
        SegInstance((50.0, 50.0, 90.0, 90.0), 0.16, 0, right),
    )

    kept, removed = _suppress_contained_duplicates(instances)

    assert len(kept) == 2
    assert all(actual is expected for actual, expected in zip(kept, instances))
    assert removed == 0


def test_partially_contained_instance_is_preserved() -> None:
    """包含率未达到 0.94 的实例仍可能是真实遮挡纸片，应予保留。"""

    outer = np.zeros((100, 100), dtype=np.uint8)
    outer[0:93, :] = 255
    inner = np.zeros_like(outer)
    inner[:, 20:80] = 255
    instances = (
        SegInstance((0.0, 0.0, 100.0, 100.0), 0.91, 0, outer),
        SegInstance((20.0, 0.0, 80.0, 100.0), 0.30, 0, inner),
    )

    kept, removed = _suppress_contained_duplicates(instances)

    assert len(kept) == 2
    assert all(actual is expected for actual, expected in zip(kept, instances))
    assert removed == 0


def test_stable_vision_requires_configured_consecutive_frames() -> None:
    """9 帧窗口达到 6 帧确认前不输出，满足时返回稳定直线交点。"""

    image, mask = _scene()
    detector = VisionDetector(VisionConfig(), _FakeInferencer(mask))
    stable = StableVisionDetector(
        detector,
        StableVisionConfig(window_size=9, required_frames=6),
    )
    for _ in range(5):
        assert stable.detect(image).pieces == ()
    result = stable.detect(image)
    assert len(result.pieces) == 1
    assert len(result.pieces[0].corners_px) == 4


def test_production_factory_uses_task_edge_configuration() -> None:
    """正式工厂应读取任务短边阈值并固定其余生产参数。"""

    detector = _ClosableVisionDetector()
    with patch(
        "ec_26_e.vision.factory.VisionDetector",
        return_value=detector,
    ) as constructor:
        stable = create_production_detector(
            Detect1Config(
                short_edge_merge_mm=9.0,
                minimum_physical_edge_mm=5.5,
                minimum_corner_turn_deg=14.0,
                minimum_corner_depth_mm=3.0,
                minimum_polygon_iou=0.88,
            )
        )

    config = constructor.call_args.args[0]
    assert isinstance(stable, StableVisionDetector)
    assert stable.detector is detector
    assert config.score_threshold == 0.15
    assert config.nms_threshold == 0.45
    assert config.mask_threshold == 0.50
    assert config.dilation_mm == 3.0
    assert config.fallback_dilation_mm == 4.0
    assert config.short_edge_merge_mm == 9.0
    assert config.minimum_physical_edge_mm == 5.5
    assert config.minimum_corner_turn_deg == 14.0
    assert config.minimum_corner_depth_mm == 3.0
    assert config.minimum_polygon_iou == 0.88
    assert (config.min_vertices, config.max_vertices, config.max_pieces) == (3, 5, 4)
    assert (stable.config.window_size, stable.config.required_frames) == (9, 6)


def test_stable_vision_close_is_idempotent() -> None:
    """应用层重复清理不得重复释放 NPU。"""

    detector = _ClosableVisionDetector()
    stable = StableVisionDetector(detector)

    stable.close()
    stable.close()

    assert detector.close_calls == 1


def test_vision_config_rejects_out_of_contract_values() -> None:
    """拒绝超出毫米膨胀和实例数量约束的配置。"""

    try:
        VisionConfig(dilation_mm=4.1)
    except ValueError:
        pass
    else:
        raise AssertionError("应拒绝大于 4 mm 的膨胀")
    try:
        VisionConfig(
            short_edge_merge_mm=5.0,
            minimum_physical_edge_mm=6.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("应拒绝合并阈值小于物理边下限")
    try:
        StableVisionConfig(window_size=4)
    except ValueError:
        pass
    else:
        raise AssertionError("应拒绝小于 5 帧的窗口")


def test_polygon_selection_never_returns_edge_below_physical_limit() -> None:
    """轮廓简化结果不得绕过 20 mm 的最终契约。"""

    contour = np.asarray(
        [[10, 10], [100, 10], [100, 100], [12, 100], [10, 99]],
        dtype=np.int32,
    ).reshape(-1, 1, 2)
    polygon, _ = _choose_polygon(contour, VisionConfig(), 0.39)
    assert polygon is not None
    lengths_mm = np.linalg.norm(polygon - np.roll(polygon, -1, axis=0), axis=1) * 0.39
    assert float(np.min(lengths_mm)) > 20.0


def test_polygon_selection_preserves_significant_concave_corner() -> None:
    """边长、转角、深度和 IoU 合格的凹五边形必须原样保留。"""

    contour = np.asarray(
        [[10, 10], [150, 10], [150, 110], [80, 70], [10, 110]],
        dtype=np.int32,
    ).reshape(-1, 1, 2)

    polygon, _ = _choose_polygon(contour, VisionConfig(), 1.0)

    assert polygon is not None
    assert len(polygon) == 5
    assert is_simple_polygon(polygon)
    assert not cv2.isContourConvex(
        np.rint(polygon).astype(np.int32).reshape(-1, 1, 2)
    )


def test_polygon_selection_removes_shallow_concave_artifact() -> None:
    """不足 4 mm 深的浅凹杂点不得作为物理角保留。"""

    contour = np.asarray(
        [[10, 10], [150, 10], [150, 110], [80, 107], [10, 110]],
        dtype=np.int32,
    ).reshape(-1, 1, 2)

    polygon, _ = _choose_polygon(contour, VisionConfig(), 1.0)

    assert polygon is not None
    assert len(polygon) == 4
    assert cv2.isContourConvex(
        np.rint(polygon).astype(np.int32).reshape(-1, 1, 2)
    )


def test_polygon_selection_enforces_minimum_iou() -> None:
    """候选与原始轮廓 IoU 未达到配置下限时必须拒绝。"""

    contour = cv2.ellipse2Poly((100, 100), (70, 60), 0, 0, 360, 5).reshape(-1, 1, 2)

    polygon, _ = _choose_polygon(
        contour,
        VisionConfig(minimum_polygon_iou=1.0),
        1.0,
    )

    assert polygon is None


def test_simple_polygon_rejects_self_intersection() -> None:
    """简单多边形检查必须拒绝蝴蝶结式自交轮廓。"""

    bow_tie = np.asarray(
        [[0.0, 0.0], [100.0, 100.0], [0.0, 100.0], [100.0, 0.0]],
        dtype=np.float64,
    )

    assert not is_simple_polygon(bow_tie)


def _piece_with_corners(
    corners: tuple[tuple[float, float], ...],
) -> PieceDetection:
    """构造供稳定器纯逻辑测试使用的纸片结果。"""

    center = tuple(np.mean(np.asarray(corners), axis=0).tolist())
    return PieceDetection(
        group_id=0,
        corners_px=corners,
        center_px=center,
        corners_relative=tuple((x / 761.0, y / 539.0) for x, y in corners),
        center_relative=(center[0] / 761.0, center[1] / 539.0),
        area_px=float(cv2.contourArea(np.asarray(corners, dtype=np.float32))),
        area_relative=0.01,
        polygon_iou=0.95,
        line_residual=0.1,
        confidence=0.95,
    )


def test_stabilizer_preserves_valid_concave_polygon() -> None:
    """稳定器聚合后应继续输出满足物理约束的凹五边形。"""

    detector = VisionDetector(VisionConfig(), _FakeInferencer(np.zeros((1, 1), dtype=np.uint8)))
    stable = StableVisionDetector(detector)
    piece = _piece_with_corners(
        ((100.0, 100.0), (400.0, 100.0), (400.0, 350.0), (250.0, 250.0), (100.0, 350.0))
    )
    frame = Detect1Result((762, 540), (piece,), 0.15, 762 * 540)

    for _ in range(5):
        assert stable.update(frame).pieces == ()
    result = stable.update(frame)

    assert len(result.pieces) == 1
    output = np.rint(result.pieces[0].corners_px).astype(np.int32).reshape(-1, 1, 2)
    assert not cv2.isContourConvex(output)


def test_stabilizer_rejects_polygon_with_short_physical_edge() -> None:
    """稳定聚合不得重新放行包含不满 20 mm 边的轮廓。"""

    detector = VisionDetector(VisionConfig(), _FakeInferencer(np.zeros((1, 1), dtype=np.uint8)))
    stable = StableVisionDetector(detector)
    piece = _piece_with_corners(
        ((100.0, 100.0), (140.0, 100.0), (140.0, 300.0), (100.0, 300.0))
    )
    frame = Detect1Result((762, 540), (piece,), 0.15, 762 * 540)

    for _ in range(7):
        result = stable.update(frame)

    assert result.pieces == ()
