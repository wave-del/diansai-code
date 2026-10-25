"""装配 Task1 的视觉、规划和真实运动能力并运行完整生命周期。"""

from __future__ import annotations

from collections.abc import Callable
from ec_26_e.actuation import ActuationError
from ec_26_e.cam import (
    CameraCaptureError,
    CameraConfig,
    CameraSource,
    FFmpegCamera,
)
from ec_26_e.config import Task1Config
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
from ec_26_e.tasks.task1.runner import run_task1
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
ActuatorFactory = Callable[[Task1Config], PlanActuator]


def _default_actuator_factory(config: Task1Config) -> MovePlanActuator:
    """使用 Task1 强类型运动配置构造真实执行器。"""

    return MovePlanActuator(config.actuation)


def execute_task1(
    config: Task1Config,
    context: TaskContext,
    *,
    camera_factory: CameraFactory = FFmpegCamera,
    resize_factory: ResizeFactory = Resize,
    detector_factory: Detect1Factory = create_production_detector,
    planner: PuzzlePlanner = plan_sheet_moves,
    observer_factory: ObserverFactory = Task1DrawObserver,
    actuator_factory: ActuatorFactory = _default_actuator_factory,
) -> None:
    """构造真实依赖并运行识别、规划、运动和安全退出。

    MovePlan 只在当前调用栈中交给运动执行器，不写文件、不打印
    也不返回。全部工厂均可替换，以便无设备测试覆盖完整流程。
    """

    if not isinstance(config, Task1Config):
        raise TypeError("config 必须是 Task1Config")
    if not isinstance(context, TaskContext):
        raise TypeError("context 必须是 TaskContext")
    dependencies = (
        ("camera_factory", camera_factory),
        ("resize_factory", resize_factory),
        ("detector_factory", detector_factory),
        ("planner", planner),
        ("observer_factory", observer_factory),
        ("actuator_factory", actuator_factory),
    )
    for name, dependency in dependencies:
        if not callable(dependency):
            raise TypeError(f"{name} 必须可调用")

    try:
        detector = detector_factory(config.detect1)
        try:
            camera = camera_factory(config.camera)
            resizer = resize_factory(config.resize)
            observer = observer_factory() if context.draw_enabled else None
            plan = run_task1(
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
        raise Task1Error(f"移动计划生成失败：{error}") from error
    except CameraCaptureError as error:
        raise Task1Error(f"识别检测失败：{error}") from error
    except Task1Error as error:
        raise Task1Error(f"识别检测失败：{error}") from error
    except (OSError, RuntimeError) as error:
        raise Task1Error(f"识别检测失败：{error}") from error

    actuator = actuator_factory(config)
    execute = getattr(actuator, "execute", None)
    if not callable(execute):
        raise TypeError("actuator_factory 必须返回提供 execute 方法的对象")
    try:
        execute(plan)
    except ActuationError:
        raise
    return None
