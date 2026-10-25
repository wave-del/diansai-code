"""遮挡鲁棒检测管线：seed(部分遮挡回退) → ResizePartial → 参考 Detect1。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import cv2
import numpy as np

try:
    from . import _paths
except ImportError:
    import _paths
_paths.ensure()

import test_a4_detection as a4d

try:
    from .resize_partial import ResizePartial
except ImportError:
    from resize_partial import ResizePartial
try:
    from .seed import SeedResult, seed_a4_robust
except ImportError:
    from seed import SeedResult, seed_a4_robust


@dataclass
class RobustResult:
    """完整检测结果：四角、逐角可靠性、校正图与碎片检测。"""

    corners: np.ndarray
    reliability: tuple[bool, bool, bool, bool]
    seed: SeedResult
    resize_result: Any | None
    detect_result: Any | None
    status: str


def detect_robust(
    frame: np.ndarray,
    params: dict[str, float | int],
    a4_args: Any,
) -> RobustResult:
    """遮挡鲁棒全流程检测。

    - 完整四边形 → 参考 Resize（与现有流程完全一致）；
    - 部分遮挡 → ResizePartial（估计角放宽校验）；
    - 碎片检测始终使用参考 Detect1，参数由 params 指定白片 seed 门槛。
    """
    seed_result = seed_a4_robust(frame, a4_args)
    resize_class, resize_config, detector = a4d._build_reference_detectors(params)
    height, width = frame.shape[:2]
    relative = seed_result.corners / np.asarray((width - 1, height - 1), dtype=np.float32)
    config = replace(
        resize_config,
        reference_corners_relative=tuple((float(x), float(y)) for x, y in relative),
    )
    if all(seed_result.reliability):
        resize_result = resize_class(config).resize(frame)
    else:
        resize_result = ResizePartial(
            config,
            reliability=seed_result.reliability,
        ).resize(frame)
    detect_result = detector.detect(resize_result.image_bgr)
    return RobustResult(
        corners=seed_result.corners,
        reliability=seed_result.reliability,
        seed=seed_result,
        resize_result=resize_result,
        detect_result=detect_result,
        status="OK",
    )
