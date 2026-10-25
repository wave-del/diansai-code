"""公开 MovePlan 运动适配层的稳定契约和真实能力工厂。"""

from ec_26_e.actuation.compiler import compile_move_plan
from ec_26_e.actuation.contracts import (
    ActuationConfig,
    ActuationError,
    ActuationExecutionError,
    ActuationResult,
    ActuationShutdownError,
    ActuationValidationError,
    CompletionIndicator,
    MagnetConfig,
    MotionLifecycleState,
    RotationAxis,
    TransferCycle,
)
from ec_26_e.actuation.led import GpiodCompletionLedOutput
from ec_26_e.actuation.magnet import GpiodMagnetOutput

__all__ = [
    "ActuationConfig",
    "ActuationError",
    "ActuationExecutionError",
    "ActuationResult",
    "ActuationShutdownError",
    "ActuationValidationError",
    "CompletionIndicator",
    "GpiodCompletionLedOutput",
    "GpiodMagnetOutput",
    "MagnetConfig",
    "MotionLifecycleState",
    "RotationAxis",
    "TransferCycle",
    "compile_move_plan",
]
