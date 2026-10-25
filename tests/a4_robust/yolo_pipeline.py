"""组合检测管线：我们的遮挡鲁棒 seed → 同事 Resize → 同事 YOLO 门控精修。

- 参考角 = 每帧 a4_robust seed 输出（机位无关、遮挡鲁棒），
  不再依赖任何固定机位标定；
- 透视校正 = 同事 Resize（含 Canny+Hough 兜底）；
- 拼图检测 = 同事 VisionDetector（YOLO11n-seg NPU 门控 + 传统视觉精修），
  可选五帧稳定器。

注意：本模块导入同事的 ec_26_e 包（与 reference-vision 同名），
因此与 a4_robust 传统管线（detect_robust）不能在同一进程混用；
对同一帧做对比时请用独立进程。
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

try:
    from . import _paths
except ImportError:
    import _paths
_paths.ensure()

import test_a4_detection as a4d

try:
    from .seed import SeedResult, seed_a4_robust
except ImportError:
    from seed import SeedResult, seed_a4_robust
try:
    from .source_region_filter import (
        DEFAULT_SOURCE_MAX_X_RATIO,
        filter_left_source_pieces,
    )
except ImportError:
    from source_region_filter import (
        DEFAULT_SOURCE_MAX_X_RATIO,
        filter_left_source_pieces,
    )

# 同事包路径（src 布局）；随系统卡迁移时不得依赖旧机器绝对目录。
_PROJECT_DIR = Path(__file__).resolve().parents[2]
COLLEAGUE_SRC = _PROJECT_DIR / "vision" / "src"
# _paths.ensure() 同时服务旧参考视觉，会重新加入同名旧 ec_26_e 包路径。
# YOLO 管线必须强制把新版工程放在最前，不能只在“不存在”时插入。
_colleague_src = str(COLLEAGUE_SRC)
while _colleague_src in sys.path:
    sys.path.remove(_colleague_src)
sys.path.insert(0, _colleague_src)
_loaded_ec = sys.modules.get("ec_26_e")
_loaded_file = getattr(_loaded_ec, "__file__", None)
if _loaded_file is not None and COLLEAGUE_SRC not in Path(_loaded_file).resolve().parents:
    for _name in tuple(sys.modules):
        if _name == "ec_26_e" or _name.startswith("ec_26_e."):
            sys.modules.pop(_name, None)

from ec_26_e.vision import Resize, ResizeConfig  # noqa: E402
from ec_26_e.vision.contracts import VisionConfig  # noqa: E402
from ec_26_e.vision.detector import VisionDetector  # noqa: E402
from ec_26_e.vision.stabilizer import StableVisionConfig, StableVisionDetector  # noqa: E402

MODEL_LIB = Path(__file__).resolve().parent.parent.parent / "model-lib"

# 与同事 create_production_detector 相同的正式生产参数
_PRODUCTION_VISION = dict(
    score_threshold=0.15,
    nms_threshold=0.45,
    mask_threshold=0.50,
    dilation_mm=3.0,
    fallback_dilation_mm=4.0,
    # 新版 ec_26_e 将旧 ``min_edge_mm`` 拆为物理最短边和短边合并阈值。
    # 两者均保留为现场已验证的 6 mm，等价于旧鲁棒视觉的最小边过滤。
    short_edge_merge_mm=6.0,
    minimum_physical_edge_mm=6.0,
    min_vertices=3,
    max_vertices=5,
    max_pieces=4,
)


class LeftSourceRegionDetector(VisionDetector):
    """与新版稳定器类型契约兼容的左侧来源区过滤代理。"""

    def __init__(self, detector: VisionDetector, *, max_x_ratio: float) -> None:
        if not isinstance(detector, VisionDetector):
            raise TypeError("底层 detector 必须是 VisionDetector")
        # StableVisionDetector 既做 isinstance 检查，也会读取 config；这里
        # 不重新创建 NPU 资源，只代理并过滤已完成的单帧检测结果。
        self._detector = detector
        self.config = detector.config
        self.max_x_ratio = max_x_ratio
        self.last_rejected_pieces: tuple[Any, ...] = ()

    @property
    def last_diagnostics(self) -> Any:
        return self._detector.last_diagnostics

    def detect(self, image: Any) -> Any:
        result = self._detector.detect(image)
        filtered = filter_left_source_pieces(
            tuple(result.pieces), max_x_ratio=self.max_x_ratio,
        )
        self.last_rejected_pieces = filtered.rejected
        return replace(result, pieces=filtered.accepted)

    def close(self) -> None:
        self._detector.close()


@dataclass
class YoloRobustResult:
    """组合管线单帧结果。"""

    seed: SeedResult
    resize_result: Any
    detect_result: Any            # 稳定器输出（未稳定时为空结果）
    single_result: Any | None     # 未稳定时的单帧精修结果
    rejected_pieces: tuple[Any, ...]  # 位于右侧或跨越来源区边界的候选
    timing_ms: dict[str, float]


class YoloRobustDetector:
    """组合检测器：每帧 seed 给参考角，Resize 精修，YOLO 检测拼图。"""

    def __init__(
        self,
        params: dict[str, float | int],
        a4_args: Any,
        *,
        stable: bool = True,
        source_max_x_ratio: float = DEFAULT_SOURCE_MAX_X_RATIO,
        reference_lock_frames: int = 3,
        stability_vertex_jitter_mm: float = 4.0,
    ) -> None:
        """加载 NPU 模型；stable=True 时使用固定参考的五帧稳定器。"""
        if reference_lock_frames < 1:
            raise ValueError("reference_lock_frames 必须至少为 1")
        if stability_vertex_jitter_mm <= 0.0:
            raise ValueError("stability_vertex_jitter_mm 必须大于 0")
        if str(MODEL_LIB) not in sys.path:
            pass
        self._params = params
        self._a4_args = a4_args
        self._stable = stable
        vision = VisionDetector(
            VisionConfig(
                model_path=MODEL_LIB / "q6a/puzzle_q6a.bin",
                library_path=MODEL_LIB / "q6a/libq6a_qnn.so",
                **_PRODUCTION_VISION,
            )
        )
        self._source_detector = LeftSourceRegionDetector(
            vision, max_x_ratio=source_max_x_ratio,
        )
        if stable:
            self._detector = StableVisionDetector(
                self._source_detector,
                StableVisionConfig(
                    window_size=5,
                    required_frames=5,
                    max_vertex_jitter_mm=stability_vertex_jitter_mm,
                ),
            )
        else:
            self._detector = self._source_detector
        self._vision = vision
        # Resize 基础配置：参考角每帧由 seed 给出
        self._base_resize = dict(
            detect_width=640,
            detect_height=360,
            target_width=764,
            target_height=540,
            require_observed_boundary=True,
            min_workspace_ratio=0.25,
            search_band_ratio=0.14,
        )
        self._reference_lock_frames = reference_lock_frames
        self._reference_samples: list[np.ndarray] = []
        self._locked_reference_corners: np.ndarray | None = None
        self._reference_just_locked = False

    @property
    def reference_locked(self) -> bool:
        """本次采集是否已固定 A4 参考四角。"""

        return self._locked_reference_corners is not None

    @property
    def reference_just_locked(self) -> bool:
        """最近一次 detect 是否刚刚固定了 A4 参考。"""

        return self._reference_just_locked

    def _reference_corners(self, seed: SeedResult) -> np.ndarray:
        """积累可靠 A4 seed，并在早期固定其鲁棒中位数。

        固定后仍会计算每帧 seed 供诊断，但不再让透视变换随 seed 轻微摆动。
        """

        self._reference_just_locked = False
        if self._locked_reference_corners is not None:
            return self._locked_reference_corners
        if sum(bool(value) for value in seed.reliability) >= 3:
            self._reference_samples.append(np.asarray(seed.corners, dtype=np.float32))
        if len(self._reference_samples) >= self._reference_lock_frames:
            samples = np.stack(self._reference_samples[-self._reference_lock_frames :])
            self._locked_reference_corners = np.median(samples, axis=0).astype(np.float32)
            self._reference_samples.clear()
            self._reference_just_locked = True
            # 前两帧可能采用了移动的参考坐标；锁定后从同一坐标系重新计数。
            if self._stable:
                self._detector.reset()
        return (
            self._locked_reference_corners
            if self._locked_reference_corners is not None
            else np.asarray(seed.corners, dtype=np.float32)
        )

    def detect(self, frame: Any) -> YoloRobustResult:
        """单帧组合检测：seed → Resize → YOLO。"""
        started = time.perf_counter()
        seed = seed_a4_robust(frame, self._a4_args)
        seed_ms = (time.perf_counter() - started) * 1000.0
        height, width = frame.shape[:2]
        reference_corners = self._reference_corners(seed)
        relative = reference_corners / np.asarray((width - 1, height - 1))
        resize_config = ResizeConfig(
            **self._base_resize,
            reference_corners_relative=tuple(
                (float(x), float(y)) for x, y in relative
            ),
        )
        resize_started = time.perf_counter()
        resize_result = Resize(resize_config).resize(frame)
        resize_ms = (time.perf_counter() - resize_started) * 1000.0
        detect_started = time.perf_counter()
        detect_result = self._detector.detect(resize_result.image_bgr)
        detect_ms = (time.perf_counter() - detect_started) * 1000.0
        if self._stable:
            single_result = self._detector.last_single_result
        else:
            single_result = detect_result
        return YoloRobustResult(
            seed=seed,
            resize_result=resize_result,
            detect_result=detect_result,
            single_result=single_result,
            rejected_pieces=self._source_detector.last_rejected_pieces,
            timing_ms={
                "seed": seed_ms,
                "resize": resize_ms,
                "yolo_refine": detect_ms,
                "total": (time.perf_counter() - started) * 1000.0,
            },
        )

    @property
    def diagnostics(self) -> Any:
        """最近一帧的 NPU/精修诊断（含分阶段耗时与边支撑）。"""
        return self._vision.last_diagnostics

    def best_effort_result(self, *, max_vertex_jitter_mm: float = 8.0) -> Any | None:
        """返回关联一致的多帧视觉兜底；单帧模式不伪造稳定结果。"""

        if not self._stable:
            return None
        return self._detector.best_effort_result(
            max_vertex_jitter_mm=max_vertex_jitter_mm
        )

    def reset(self) -> None:
        """清空稳定器轨迹（场景变化后调用，如一片拼图被抓走后）。"""
        self._reference_samples.clear()
        self._locked_reference_corners = None
        self._reference_just_locked = False
        if self._stable:
            self._detector.reset()

    def close(self) -> None:
        """释放 NPU 模型上下文（必须在创建线程调用）。"""
        self._vision.close()
