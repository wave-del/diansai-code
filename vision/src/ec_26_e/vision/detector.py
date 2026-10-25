"""编排 YOLO11n-seg 推理和受其约束的单帧传统视觉精修。"""

from __future__ import annotations

import time

import numpy as np

import cv2

from ec_26_e.vision._blue import normalized_blue_score_u8, validate_bgr_image
from ec_26_e.vision.contracts import (
    Detect1Result,
    SegInstance,
    VisionConfig,
    VisionDiagnostics,
    validate_a4_image_size,
)
from ec_26_e.vision.foreground import prepare_foreground
from ec_26_e.vision.npu import NpuYolo11SegInferencer
from ec_26_e.vision.refine import refine_instance


_MIN_MASK_CONTAINMENT = 0.94


def _mask_containment(inner: np.ndarray, outer: np.ndarray) -> float:
    """返回 inner 前景像素落在 outer 内部的比例。"""

    inner_foreground = inner > 0
    inner_area = int(np.count_nonzero(inner_foreground))
    if inner_area == 0:
        return 0.0
    intersection = int(
        np.count_nonzero(inner_foreground & (outer > 0))
    )
    return float(intersection / inner_area)


def _suppress_contained_duplicates(
    instances: tuple[SegInstance, ...],
) -> tuple[tuple[SegInstance, ...], int]:
    """删除同类别较大实例内部高度重合的较小局部副本。

    置信度不能可靠地区分完整实例和局部副本：完整轮廓可能因图案复杂而
    得分更低。这里按 mask 面积确定内外层，并用较小 mask 的包含率判断，
    避免普通框 IoU 把相邻或局部遮挡的真实纸片误删。
    """

    mask_areas = tuple(
        int(np.count_nonzero(instance.mask))
        for instance in instances
    )
    kept: list[SegInstance] = []
    removed = 0
    for candidate_index, candidate in enumerate(instances):
        duplicate = False
        candidate_area = mask_areas[candidate_index]
        for container_index, container in enumerate(instances):
            if container_index == candidate_index:
                continue
            if candidate.class_id != container.class_id:
                continue
            if candidate_area >= mask_areas[container_index]:
                continue
            if _mask_containment(
                candidate.mask,
                container.mask,
            ) >= _MIN_MASK_CONTAINMENT:
                duplicate = True
                break
        if duplicate:
            removed += 1
        else:
            kept.append(candidate)
    return tuple(kept), removed


class VisionDetector:
    """保持无状态输出语义的 YOLO 门控单帧精修器。"""

    def __init__(
        self,
        config: VisionConfig | None = None,
        inferencer: NpuYolo11SegInferencer | object | None = None,
    ) -> None:
        """保存配置；允许测试注入同接口推理器。"""

        self.config = config or VisionConfig()
        self._owns_inferencer = inferencer is None
        self.inferencer = inferencer or NpuYolo11SegInferencer(self.config)
        self.last_diagnostics = VisionDiagnostics()

    def detect(self, a4_image_bgr: np.ndarray) -> Detect1Result:
        """输入现有 Resize 横向 A4 图，输出既有 Detect1Result 契约。"""

        validate_bgr_image(a4_image_bgr)
        height, width = a4_image_bgr.shape[:2]
        validate_a4_image_size(width, height)
        total_started = time.perf_counter()
        yolo_started = time.perf_counter()
        raw_instances = self.inferencer.infer(a4_image_bgr)
        instances, duplicate_count = _suppress_contained_duplicates(
            raw_instances
        )
        yolo_ms = (time.perf_counter() - yolo_started) * 1000.0
        diagnostics = VisionDiagnostics(
            native_timing=self.inferencer.last_timing,
            yolo_count=len(raw_instances),
            contained_duplicate_count=duplicate_count,
            driver_version=getattr(self.inferencer, "driver_version", 0),
        )
        foreground_started = time.perf_counter()
        foreground, _, _ = prepare_foreground(a4_image_bgr)
        blue_score = normalized_blue_score_u8(cv2.GaussianBlur(a4_image_bgr, (5, 5), 0))
        foreground_ms = (time.perf_counter() - foreground_started) * 1000.0
        refine_started = time.perf_counter()
        pieces = []
        for instance in sorted(instances, key=lambda item: (item.box_px[1], item.box_px[0])):
            piece, item_diagnostics = refine_instance(
                a4_image_bgr,
                instance,
                len(pieces),
                self.config,
                foreground,
                blue_score,
            )
            diagnostics.instances.append(item_diagnostics)
            if piece is not None:
                pieces.append(piece)
        refine_ms = (time.perf_counter() - refine_started) * 1000.0
        diagnostics.refined_count = len(pieces)
        diagnostics.stage_ms = {
            "yolo_total": yolo_ms,
            "traditional_foreground": foreground_ms,
            "traditional_refine": refine_ms,
            "total": (time.perf_counter() - total_started) * 1000.0,
        }
        self.last_diagnostics = diagnostics
        return Detect1Result(
            image_size=(width, height),
            pieces=tuple(pieces),
            foreground_threshold=self.config.score_threshold,
            workspace_area_px=width * height,
        )

    def close(self) -> None:
        """释放由本实例创建的 NPU 推理器。"""

        if self._owns_inferencer:
            self.inferencer.close()

    def __enter__(self) -> VisionDetector:
        """进入确定性模型资源上下文。"""

        return self

    def __exit__(self, *_: object) -> None:
        """退出上下文并释放模型。"""

        self.close()
