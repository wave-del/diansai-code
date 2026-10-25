"""通过显式强类型注册表完成环境任务名到任务执行器的分派。"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import inspect
import re
from types import MappingProxyType
from typing import get_type_hints

from ec_26_e.cam import CameraCaptureError
from ec_26_e.config import (
    RuntimeConfig,
    Task1Config,
    Task2Config,
    load_task1_config,
    load_task2_config,
)
from ec_26_e.actuation import ActuationError
from ec_26_e.control import PlanningError
from ec_26_e.draw import DrawError
from ec_26_e.tasks.contracts import TaskCancelledError, TaskContext
from ec_26_e.tasks.task1 import Task1Error
from ec_26_e.tasks.task1.application import execute_task1
from ec_26_e.tasks.task2.application import execute_task2
from ec_26_e.tasks.task2 import Task2Error


_TASK_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")


class UnknownTaskError(ValueError):
    """表示环境选择了尚未注册的任务名称。"""


class TaskExecutionError(RuntimeError):
    """表示已注册任务发生可预期的运行或输出错误。"""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        """保存稳定错误消息及按钮调度器可用的重试语义。"""

        super().__init__(message)
        if not isinstance(retryable, bool):
            raise TypeError("retryable 必须是布尔值")
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class TaskRegistration:
    """关联任务名称、配置模型、加载器、结果模型和执行函数。"""

    name: str
    config_type: type[object]
    result_type: type[object]
    loader: Callable[..., object]
    runner: Callable[..., object]
    expected_errors: tuple[type[BaseException], ...]

    def __post_init__(self) -> None:
        """拒绝非法名称、类型、异常声明和不兼容函数签名。"""

        if not isinstance(self.name, str):
            raise TypeError("任务名称必须是字符串")
        if not _TASK_NAME_PATTERN.fullmatch(self.name):
            raise ValueError("任务名称必须使用小写字母、数字、连字符或下划线")
        if not isinstance(self.config_type, type):
            raise TypeError("config_type 必须是类型")
        if not isinstance(self.result_type, type):
            raise TypeError("result_type 必须是类型")
        if not callable(self.loader):
            raise TypeError("loader 必须可调用")
        if not callable(self.runner):
            raise TypeError("runner 必须可调用")
        if (
            not isinstance(self.expected_errors, tuple)
            or not self.expected_errors
            or any(
                not isinstance(error_type, type)
                or not issubclass(error_type, BaseException)
                for error_type in self.expected_errors
            )
        ):
            raise TypeError(
                "expected_errors 必须是非空异常类型元组"
            )
        self._validate_loader()
        self._validate_runner()

    def _validate_loader(self) -> None:
        """校验加载器可消费统一运行配置并返回注册配置模型。"""

        signature = inspect.signature(self.loader)
        try:
            signature.bind(
                self._placeholder_path(),
                env_path=None,
                environ={},
            )
        except TypeError as error:
            raise TypeError(
                "loader 必须接受 path、env_path 和 environ"
            ) from error
        try:
            hints = get_type_hints(self.loader)
        except (NameError, TypeError) as error:
            raise TypeError("loader 必须提供可解析的返回类型注解") from error
        if hints.get("return") is not self.config_type:
            raise TypeError("loader 返回类型必须与 config_type 一致")

    @staticmethod
    def _placeholder_path() -> object:
        """返回仅用于签名绑定、不触发文件访问的占位对象。"""

        return object()

    def _validate_runner(self) -> None:
        """校验执行器要求配置和公共上下文并返回注册结果模型。"""

        signature = inspect.signature(self.runner)
        positional = tuple(
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        )
        required = tuple(
            parameter
            for parameter in positional
            if parameter.default is inspect.Parameter.empty
        )
        if len(required) != 2 or len(positional) != 2:
            raise TypeError(
                "runner 必须接收配置和 TaskContext 两个必需位置参数"
            )
        try:
            hints = get_type_hints(self.runner)
        except (NameError, TypeError) as error:
            raise TypeError("runner 必须提供可解析的类型注解") from error
        if hints.get(positional[0].name) is not self.config_type:
            raise TypeError("runner 配置参数类型必须与 config_type 一致")
        if hints.get(positional[1].name) is not TaskContext:
            raise TypeError("runner 第二个参数类型必须是 TaskContext")
        if hints.get("return") is not self.result_type:
            raise TypeError("runner 返回类型必须与 result_type 一致")

    def load(self, runtime: RuntimeConfig) -> object:
        """使用冻结环境快照加载并校验本任务强类型 JSON 配置。"""

        if not isinstance(runtime, RuntimeConfig):
            raise TypeError("runtime 必须是 RuntimeConfig")
        config = self.loader(
            runtime.task_config_path,
            env_path=None,
            environ=runtime.environ,
        )
        if not isinstance(config, self.config_type):
            raise TypeError(
                f"{self.name} 加载结果必须是 {self.config_type.__name__}"
            )
        return config

    def run(
        self,
        config: object,
        context: TaskContext,
    ) -> object:
        """执行一次注册任务，并统一包装该任务声明的预期异常。"""

        if not isinstance(config, self.config_type):
            raise TypeError(
                f"{self.name} 配置必须是 {self.config_type.__name__}"
            )
        if not isinstance(context, TaskContext):
            raise TypeError("context 必须是 TaskContext")
        try:
            result = self.runner(config, context)
        except TaskCancelledError:
            raise
        except DrawError as error:
            raise TaskExecutionError(
                f"{self.name}：{error}",
                retryable=True,
            ) from error
        except self.expected_errors as error:
            raise TaskExecutionError(
                f"{self.name}：{error}",
                retryable=not isinstance(error, (ActuationError, OSError)),
            ) from error
        if not isinstance(result, self.result_type):
            raise TypeError(
                f"{self.name} 执行结果必须是 {self.result_type.__name__}"
            )
        return result


def build_task_registry(
    registrations: Iterable[TaskRegistration],
) -> Mapping[str, TaskRegistration]:
    """构造只读注册表，并拒绝重复任务名称或非法注册项。"""

    registry: dict[str, TaskRegistration] = {}
    for registration in registrations:
        if not isinstance(registration, TaskRegistration):
            raise TypeError("注册表只能包含 TaskRegistration")
        if registration.name in registry:
            raise ValueError(f"任务名称重复：{registration.name}")
        registry[registration.name] = registration
    return MappingProxyType(registry)


TASK_REGISTRY = build_task_registry(
    (
        TaskRegistration(
            name="task1",
            config_type=Task1Config,
            result_type=type(None),
            loader=load_task1_config,
            runner=execute_task1,
            expected_errors=(
                CameraCaptureError,
                Task1Error,
                PlanningError,
                ActuationError,
                OSError,
            ),
        ),
        TaskRegistration(
            name="task2",
            config_type=Task2Config,
            result_type=type(None),
            loader=load_task2_config,
            runner=execute_task2,
            expected_errors=(
                CameraCaptureError,
                Task2Error,
                PlanningError,
                ActuationError,
                OSError,
            ),
        ),
    )
)


def get_task_registration(task_name: str) -> TaskRegistration:
    """返回环境指定任务的注册项，并拒绝未知任务。"""

    if not isinstance(task_name, str):
        raise TypeError("task_name 必须是字符串")
    try:
        return TASK_REGISTRY[task_name]
    except KeyError as error:
        raise UnknownTaskError(
            f"未注册任务：{task_name}"
        ) from error
