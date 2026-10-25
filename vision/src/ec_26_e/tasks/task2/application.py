"""装配 Task2 的视觉、随机规划和真实运动能力并运行完整生命周期。"""

from __future__ import annotations

from collections.abc import Callable

from ec_26_e.actuation import ActuationError
from ec_26_e.cam import (
    CameraCaptureError,
    CameraConfig,
    CameraSource,
    FFmpegCamera,
)
from ec_26_e.config import Task2Config
from ec_26_e.control import PlanningError, plan_sheet_moves
from ec_26_e.draw import Task1DrawObserver
from ec_26_e.tasks.contracts import TaskCancelledError, TaskContext
from ec_26_e.tasks.detection import close_detector
from ec_26_e.tasks.task1.actuation import MovePlanActuator
from ec_26_e.tasks.task1.contracts import (
    Detect1Processor,
    PlanActuator,
    PuzzlePlanner,
    ResizeProcessor,
    Task1Error,
    Task1FrameObserver,
)
from ec_26_e.tasks.task2.contracts import Task2Error
from ec_26_e.tasks.task2.runner import run_task2
from ec_26_e.vision import (
    Detect1Config,
    Resize,
    ResizeConfig,
)
from ec_26_e.vision import create_production_detector


CameraFactory = Callable[[CameraConfig], CameraSource]
ResizeFactory = Callable[[ResizeConfig], ResizeProcessor]
Detect1Factory = Callable[[Detect1Config], Detect1Processor]
ObserverFactory = Callable[[], Task1FrameObserver]
ActuatorFactory = Callable[[Task2Config], PlanActuator]


def _default_actuator_factory(config: Task2Config) -> MovePlanActuator:
    """使用 Task2 配置构造保留原最终磁铁提示的真实执行器。"""

    return MovePlanActuator(
        config.actuation,
        completion_indicator_factory=None,
    )


def execute_task2(
    config: Task2Config,
    context: TaskContext,
    *,
    camera_factory: CameraFactory = FFmpegCamera,
    resize_factory: ResizeFactory = Resize,
    detector_factory: Detect1Factory = create_production_detector,
    planner: PuzzlePlanner = plan_sheet_moves,
    observer_factory: ObserverFactory = Task1DrawObserver,
    actuator_factory: ActuatorFactory = _default_actuator_factory,
) -> None:
    """构造真实依赖并把随机 MovePlan 安全交给运动执行器。"""

    if not isinstance(config, Task2Config):
        raise TypeError("config 必须是 Task2Config")
    if not isinstance(context, TaskContext):
        raise TypeError("context 必须是 TaskContext")
    for name, dependency in (
        ("camera_factory", camera_factory),
        ("resize_factory", resize_factory),
        ("detector_factory", detector_factory),
        ("planner", planner),
        ("observer_factory", observer_factory),
        ("actuator_factory", actuator_factory),
    ):
        if not callable(dependency):
            raise TypeError(f"{name} 必须可调用")

    try:
        detector = detector_factory(config.detect1)
        try:
            camera = camera_factory(config.camera)
            resizer = resize_factory(config.resize)
            observer = observer_factory() if context.draw_enabled else None
            plan = run_task2(
                camera,
                resizer,
                detector,
                options=config.options,
                planner=planner,
                observer=observer,
            )
        finally:
            close_detector(detector)
    except TaskCancelledError:
        raise
    except (PlanningError, ValueError) as error:
        raise Task2Error(f"移动计划生成失败：{error}") from error
    except (CameraCaptureError, Task1Error, OSError, RuntimeError) as error:
        raise Task2Error(f"识别检测失败：{error}") from error

    actuator = actuator_factory(config)
    execute = getattr(actuator, "execute", None)
    if not callable(execute):
        raise TypeError("actuator_factory 必须返回提供 execute 方法的对象")
    try:
        execute(plan)
    except ActuationError:
        raise
