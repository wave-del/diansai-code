"""实现 YOLO 实例精修所需的蓝底前景分割，不提供独立检测器。"""

from __future__ import annotations

import cv2
import numpy as np

from ec_26_e.vision._blue import (
    normalized_blue_score_u8,
    odd_kernel_size,
    otsu_threshold,
    validate_bgr_image,
)
from ec_26_e.vision.contracts import validate_a4_image_size


_DARK_BRIGHTNESS_RATIO = 0.55
_DARK_BRIGHTNESS_MIN = 75
_DARK_BRIGHTNESS_MAX = 110
_DARK_BRIGHTNESS_FALLBACK = 100
_DARK_ATTACH_BAND_MM = 3.0
_A4_SHORT_SIDE_MM = 207.0
_BACKGROUND_SCORE_MARGIN = 12
_MIN_BACKGROUND_SAMPLE_RATIO = 0.10
_WORKSPACE_ERODE_RATIO = 0.007


def _workspace_mask(image_size: tuple[int, int]) -> np.ndarray:
    """构造与已验证生产参数一致的校正 A4 安全工作区。"""

    width, height = image_size
    margin = int(round(min(width, height) * _WORKSPACE_ERODE_RATIO))
    workspace = np.zeros((height, width), dtype=np.uint8)
    workspace[margin : height - margin, margin : width - margin] = 255
    return workspace


def _adaptive_dark_threshold(
    score_u8: np.ndarray,
    brightness_u8: np.ndarray,
    workspace: np.ndarray,
    near_foreground: np.ndarray,
    threshold_u8: int,
) -> int:
    """从远离纸片的高置信蓝色桌面估计暗色亮度阈值。"""

    reference = (
        (score_u8 >= min(255, threshold_u8 + _BACKGROUND_SCORE_MARGIN))
        & (workspace > 0)
        & (~near_foreground)
    )
    minimum_samples = int(
        round(
            _MIN_BACKGROUND_SAMPLE_RATIO
            * cv2.countNonZero(workspace)
        )
    )
    if np.count_nonzero(reference) < minimum_samples:
        return _DARK_BRIGHTNESS_FALLBACK
    board_brightness = float(np.median(brightness_u8[reference]))
    return int(
        np.clip(
            round(_DARK_BRIGHTNESS_RATIO * board_brightness),
            _DARK_BRIGHTNESS_MIN,
            _DARK_BRIGHTNESS_MAX,
        )
    )


def _foreground_mask(
    score_u8: np.ndarray,
    brightness_u8: np.ndarray,
    workspace: np.ndarray,
) -> tuple[np.ndarray, float]:
    """分割非蓝色纸片并补回与纸片相连的暗色印刷区域。"""

    threshold, threshold_u8 = otsu_threshold(score_u8[workspace > 0])
    foreground_pixels = (score_u8 < threshold_u8) & (workspace > 0)
    short_side = min(score_u8.shape)
    open_size = odd_kernel_size(short_side * 0.0025)
    close_size = odd_kernel_size(short_side * 0.0065)
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (open_size, open_size),
    )
    seed = cv2.morphologyEx(
        foreground_pixels.astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        open_kernel,
    )
    attach_band = max(
        1,
        int(round(short_side * _DARK_ATTACH_BAND_MM / _A4_SHORT_SIDE_MM)),
    )
    near_foreground = cv2.dilate(
        seed,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * attach_band + 1, 2 * attach_band + 1),
        ),
    ) > 0
    dark_threshold = _adaptive_dark_threshold(
        score_u8,
        brightness_u8,
        workspace,
        near_foreground,
        threshold_u8,
    )
    foreground_pixels |= (
        (brightness_u8 < dark_threshold)
        & near_foreground
        & (workspace > 0)
    )
    foreground = foreground_pixels.astype(np.uint8) * 255
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_OPEN,
        open_kernel,
    )
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_size, close_size),
        ),
    )
    contours, _ = cv2.findContours(
        foreground,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    filled = np.zeros_like(foreground)
    cv2.drawContours(filled, contours, -1, 255, thickness=-1)
    return filled, threshold


def prepare_foreground(
    image_bgr: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """校验横向 A4 图并生成精修使用的前景、阈值和工作区。"""

    validate_bgr_image(image_bgr)
    height, width = image_bgr.shape[:2]
    validate_a4_image_size(width, height)
    workspace = _workspace_mask((width, height))
    blurred = cv2.GaussianBlur(image_bgr, (5, 5), 0)
    score_u8 = normalized_blue_score_u8(blurred)
    foreground, threshold = _foreground_mask(
        score_u8,
        np.max(blurred, axis=2),
        workspace,
    )
    return foreground, threshold, workspace
