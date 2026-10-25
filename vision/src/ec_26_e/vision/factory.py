"""构造正式任务统一使用的 YOLO 语义门控与几何精修检测器。"""

from __future__ import annotations

from ec_26_e.vision.contracts import (
    Detect1Config,
    StableVisionConfig,
    VisionConfig,
)
from ec_26_e.vision.detector import VisionDetector
from ec_26_e.vision.stabilizer import StableVisionDetector


def create_production_detector(
    legacy_config: Detect1Config,
) -> StableVisionDetector:
    """保持旧工厂签名，并返回 9 帧窗口、6 帧确认的统一视觉检测器。

    ``legacy_config`` 只保留用于 Task1/Task2 现有配置和第三方工厂注入
    契约。正式检测参数集中固定在此处，生产路径不再依据旧 Detect1
    配置选择纯传统视觉检测器。
    """

    if not isinstance(legacy_config, Detect1Config):
        raise TypeError("legacy_config 必须是 Detect1Config")
    detector = VisionDetector(
        VisionConfig(
            score_threshold=0.15,
            nms_threshold=0.45,
            mask_threshold=0.50,
            dilation_mm=3.0,
            fallback_dilation_mm=4.0,
            short_edge_merge_mm=legacy_config.short_edge_merge_mm,
            minimum_physical_edge_mm=(
                legacy_config.minimum_physical_edge_mm
            ),
            minimum_corner_turn_deg=(
                legacy_config.minimum_corner_turn_deg
            ),
            minimum_corner_depth_mm=(
                legacy_config.minimum_corner_depth_mm
            ),
            minimum_polygon_iou=legacy_config.minimum_polygon_iou,
            min_vertices=3,
            max_vertices=5,
            max_pieces=4,
        )
    )
    return StableVisionDetector(
        detector,
        StableVisionConfig(window_size=9, required_frames=6),
    )
