"""提供工作区校正与纸片检测共享的蓝色分数基础运算。"""

from __future__ import annotations

from typing import Final

import cv2
import numpy as np


def _build_blue_score_u8_lut() -> np.ndarray:
    """构建按 B、G 通道和与 R 通道索引的八位蓝色分数表。"""

    sum_bg = np.arange(511, dtype=np.int32)[:, np.newaxis]
    red = np.arange(256, dtype=np.int32)[np.newaxis, :]
    numerator = (2 * sum_bg - red + 1) * 255
    denominator = 2 * (sum_bg + red + 1)
    return np.clip(
        np.floor_divide(numerator, denominator),
        0,
        255,
    ).astype(np.uint8)


_BLUE_SCORE_U8_LUT: Final[np.ndarray] = _build_blue_score_u8_lut()
_BLUE_SCORE_U8_LUT.setflags(write=False)


def validate_bgr_image(image_bgr: np.ndarray) -> None:
    """校验输入是非空的八位三通道 BGR 图像。"""

    if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3:
        raise ValueError("image_bgr 必须是 H×W×3 的 NumPy 数组")
    if image_bgr.shape[2] != 3 or image_bgr.size == 0:
        raise ValueError("image_bgr 必须是非空的三通道 BGR 图像")
    if image_bgr.dtype != np.uint8:
        raise ValueError("image_bgr 的数据类型必须是 uint8")


def normalized_blue_score_u8(image_bgr: np.ndarray) -> np.ndarray:
    """生成适合 OpenCV 阈值处理的八位归一化蓝色分数。"""

    sum_bg = np.add(
        image_bgr[..., 0],
        image_bgr[..., 1],
        dtype=np.uint16,
    )
    return _BLUE_SCORE_U8_LUT[sum_bg, image_bgr[..., 2]]


def otsu_threshold(values: np.ndarray) -> tuple[float, int]:
    """计算 Otsu 阈值并返回归一化诊断值与八位阈值。"""

    if values.size == 0:
        raise ValueError("没有可用于自适应阈值计算的像素")
    threshold_u8, _ = cv2.threshold(
        values.reshape(-1, 1),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return float(threshold_u8 / 127.5 - 1.0), int(round(threshold_u8))


def odd_kernel_size(value: float, minimum: int = 3) -> int:
    """生成不小于下限的奇数形态学核尺寸。"""

    size = max(minimum, int(round(value)))
    return size if size % 2 else size + 1
