"""显式访问真实 A733 NPU 的模型生命周期冒烟测试。"""

from __future__ import annotations

import numpy as np

from ec_26_e.vision import NpuYolo11SegInferencer


def test_real_npu_model_accepts_a4_sized_bgr() -> None:
    """创建模型、推理连续 A4 图、检查 mask 坐标并确定性关闭。"""

    image = np.zeros((540, 764, 3), dtype=np.uint8)
    inferencer = NpuYolo11SegInferencer()
    try:
        source = image.copy()
        instances = inferencer.infer(image)
        assert np.array_equal(image, source)
        assert inferencer.input_size == 640
        assert all(instance.mask.shape == image.shape[:2] for instance in instances)
        assert inferencer.last_timing.network_ms > 0.0
    finally:
        inferencer.close()
