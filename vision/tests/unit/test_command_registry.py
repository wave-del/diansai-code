"""验证任务层显式注册表的类型、分派和异常包装约束。

该文件不依赖设备或外部文件，可通过
``python -m unittest tests.unit.test_command_registry`` 独立运行。
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import unittest

from ec_26_e.config import (
    RuntimeConfig,
    Task1Config,
    Task2Config,
    load_task1_config,
    load_task2_config,
)
from ec_26_e.draw import DrawError
from ec_26_e.tasks import TaskCancelledError, TaskContext
from ec_26_e.tasks.registry import (
    TASK_REGISTRY,
    TaskExecutionError,
    TaskRegistration,
    UnknownTaskError,
    build_task_registry,
    get_task_registration,
)
from ec_26_e.tasks.task1.application import execute_task1
from ec_26_e.tasks.task2.application import execute_task2


class ExampleConfig:
    """提供注册表签名测试使用的最小配置类型。"""


class ExampleResult:
    """提供通用任务结果类型测试使用的最小对象。"""


def load_example(
    path: Path = Path("example.json"),
    *,
    env_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> ExampleConfig:
    """消费统一加载参数并返回示例配置。"""

    del path, env_path, environ
    return ExampleConfig()


def load_invalid_result(
    path: Path = Path("example.json"),
    *,
    env_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> ExampleConfig:
    """以合法注解返回错误对象以验证运行时保护。"""

    del path, env_path, environ
    return object()  # type: ignore[return-value]


def load_wrong_annotation(
    path: Path = Path("example.json"),
    *,
    env_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> object:
    """故意使用错误返回注解以验证注册拒绝。"""

    del path, env_path, environ
    return ExampleConfig()


def run_example(
    config: ExampleConfig,
    context: TaskContext,
) -> ExampleResult:
    """消费示例配置并返回合法结果。"""

    del config, context
    return ExampleResult()


def run_wrong_config(
    config: object,
    context: TaskContext,
) -> ExampleResult:
    """故意使用错误配置注解以验证注册拒绝。"""

    del config, context
    return ExampleResult()


def run_wrong_context(
    config: ExampleConfig,
    context: object,
) -> ExampleResult:
    """故意使用错误上下文注解以验证注册拒绝。"""

    del config, context
    return ExampleResult()


def run_wrong_return(
    config: ExampleConfig,
    context: TaskContext,
) -> object:
    """故意使用错误返回注解以验证注册拒绝。"""

    del context
    return config


def run_invalid_result(
    config: ExampleConfig,
    context: TaskContext,
) -> ExampleResult:
    """以合法注解返回非法对象以验证运行时保护。"""

    del context
    return config  # type: ignore[return-value]


def run_expected_error(
    config: ExampleConfig,
    context: TaskContext,
) -> ExampleResult:
    """抛出注册项声明的预期异常。"""

    del config, context
    raise RuntimeError("模拟预期失败")


def run_unexpected_error(
    config: ExampleConfig,
    context: TaskContext,
) -> ExampleResult:
    """抛出未声明异常以验证其不会被宽泛吞并。"""

    del config, context
    raise AssertionError("模拟程序错误")


def run_cancelled(
    config: ExampleConfig,
    context: TaskContext,
) -> ExampleResult:
    """模拟用户从全局输出适配器取消任务。"""

    del config, context
    raise TaskCancelledError("模拟取消")


def run_draw_error(
    config: ExampleConfig,
    context: TaskContext,
) -> ExampleResult:
    """模拟 GUI 初始化或显示失败。"""

    del config, context
    raise DrawError("模拟绘制失败")


def make_registration(
    *,
    loader: object = load_example,
    runner: object = run_example,
) -> TaskRegistration:
    """构造一个有效或按需替换函数的示例注册项。"""

    return TaskRegistration(
        name="example",
        config_type=ExampleConfig,
        result_type=ExampleResult,
        loader=loader,  # type: ignore[arg-type]
        runner=runner,  # type: ignore[arg-type]
        expected_errors=(RuntimeError,),
    )


class TaskRegistryTest(unittest.TestCase):
    """验证 Task1 注册和通用任务注册表不变量。"""

    def test_task1_registration_is_complete(self) -> None:
        """正式注册项应关联 Task1 配置、加载器和任务层执行适配器。"""

        self.assertEqual(tuple(TASK_REGISTRY), ("task1", "task2"))
        registration = TASK_REGISTRY["task1"]
        self.assertIs(registration.config_type, Task1Config)
        self.assertIs(registration.result_type, type(None))
        self.assertIs(registration.loader, load_task1_config)
        self.assertIs(registration.runner, execute_task1)
        self.assertIs(get_task_registration("task1"), registration)
        task2 = TASK_REGISTRY["task2"]
        self.assertIs(task2.config_type, Task2Config)
        self.assertIs(task2.loader, load_task2_config)
        self.assertIs(task2.runner, execute_task2)
        self.assertIs(get_task_registration("task2"), task2)

    def test_builds_read_only_registry_and_rejects_duplicates(
        self,
    ) -> None:
        """注册表应不可变，并拒绝相同任务名称重复注册。"""

        registration = make_registration()
        registry = build_task_registry((registration,))
        self.assertIs(registry["example"], registration)
        with self.assertRaises(TypeError):
            registry["other"] = registration  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "任务名称重复"):
            build_task_registry((registration, registration))

    def test_rejects_invalid_name_and_function_signatures(self) -> None:
        """注册项应拒绝非法名称、加载器和执行器类型注解。"""

        with self.assertRaises(ValueError):
            TaskRegistration(
                "Task One",
                ExampleConfig,
                ExampleResult,
                load_example,
                run_example,
                (RuntimeError,),
            )
        with self.assertRaisesRegex(TypeError, "loader 返回类型"):
            make_registration(loader=load_wrong_annotation)
        with self.assertRaisesRegex(TypeError, "配置参数类型"):
            make_registration(runner=run_wrong_config)
        with self.assertRaisesRegex(TypeError, "TaskContext"):
            make_registration(runner=run_wrong_context)
        with self.assertRaisesRegex(TypeError, "result_type"):
            make_registration(runner=run_wrong_return)

    def test_load_uses_runtime_snapshot_and_checks_actual_type(
        self,
    ) -> None:
        """加载包装应消费运行选择并验证实际配置对象类型。"""

        runtime = RuntimeConfig(
            task_name="example",
            task_config_path=Path("configs/tasks/example.json"),
            environ={"EC_TASK": "example"},
        )
        self.assertIsInstance(
            make_registration().load(runtime),
            ExampleConfig,
        )
        with self.assertRaisesRegex(TypeError, "加载结果"):
            make_registration(loader=load_invalid_result).load(runtime)

    def test_run_wraps_only_declared_errors_and_checks_result(self) -> None:
        """执行包装应区分预期失败、程序异常和非法结果。"""

        with self.assertRaisesRegex(TaskExecutionError, "模拟预期失败"):
            make_registration(runner=run_expected_error).run(
                ExampleConfig(),
                TaskContext(),
            )
        with self.assertRaisesRegex(AssertionError, "模拟程序错误"):
            make_registration(runner=run_unexpected_error).run(
                ExampleConfig(),
                TaskContext(),
            )
        with self.assertRaisesRegex(TypeError, "执行结果"):
            make_registration(runner=run_invalid_result).run(
                ExampleConfig(),
                TaskContext(),
            )

    def test_run_preserves_cancellation_and_wraps_draw_failures(
        self,
    ) -> None:
        """用户取消应直达入口，而 GUI 故障应作为预期任务失败。"""

        with self.assertRaisesRegex(TaskCancelledError, "模拟取消"):
            make_registration(runner=run_cancelled).run(
                ExampleConfig(),
                TaskContext(draw_enabled=True),
            )
        with self.assertRaisesRegex(TaskExecutionError, "模拟绘制失败"):
            make_registration(runner=run_draw_error).run(
                ExampleConfig(),
                TaskContext(draw_enabled=True),
            )

    def test_unknown_task_is_rejected_before_config_loading(self) -> None:
        """未注册环境任务名必须作为启动配置错误被拒绝。"""

        with self.assertRaisesRegex(UnknownTaskError, "未注册任务"):
            get_task_registration("missing")


if __name__ == "__main__":
    unittest.main()
