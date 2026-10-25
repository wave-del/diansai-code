"""公开唯一的 A4 校正、YOLO语义门控、传统精修和稳定检测接口。"""

from ec_26_e.vision.contracts import (
    Detect1Config,
    Detect1Result,
    NativeTiming,
    PieceDetection,
    SegInstance,
    StableVisionConfig,
    VisionConfig,
    VisionDiagnostics,
)
from ec_26_e.vision.detector import VisionDetector
from ec_26_e.vision.factory import create_production_detector
from ec_26_e.vision.npu import NpuYolo11SegInferencer
from ec_26_e.vision.resize import Resize, ResizeConfig, ResizeResult
from ec_26_e.vision.stabilizer import StableVisionDetector

__all__ = [
    "Detect1Config",
    "Detect1Result",
    "NativeTiming",
    "NpuYolo11SegInferencer",
    "PieceDetection",
    "Resize",
    "ResizeConfig",
    "ResizeResult",
    "SegInstance",
    "StableVisionConfig",
    "StableVisionDetector",
    "VisionConfig",
    "VisionDetector",
    "VisionDiagnostics",
    "create_production_detector",
]
