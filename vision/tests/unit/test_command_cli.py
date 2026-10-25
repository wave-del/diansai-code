"""验证唯一 Typer 入口的环境分派、帮助和退出码。

该文件通过内存运行配置和注册项替换真实 Task1，不访问相机、不写文件。
可通过 ``python -m unittest tests.unit.test_command_cli`` 独立运行。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from typer.testing import CliRunner

from ec_26_e.command.app import app
from ec_26_e.config import (
    DEFAULT_ENV_PATH,
    RuntimeConfig,
    Task1Config,
    load_task1_config,
)
from ec_26_e.tasks import TaskCancelledError, TaskContext
from ec_26_e.tasks.registry import TaskExecutionError


@dataclass
class FakeRegistration:
    """记录统一入口对任务加载和执行边界的调用。"""

    config: Task1Config
    load_error: BaseException | None = None
    run_error: BaseException | None = None

    def __post_init__(self) -> None:
        """初始化运行配置和任务配置调用记录。"""

        self.load_calls: list[RuntimeConfig] = []
        self.run_calls: list[Task1Config] = []
        self.context_calls: list[TaskContext] = []

    def load(self, runtime: RuntimeConfig) -> object:
        """记录运行选择并返回或抛出预设配置结果。"""

        self.load_calls.append(runtime)
        if self.load_error is not None:
            raise self.load_error
        return self.config

    def run(self, config: object, context: TaskContext) -> object:
        """记录任务配置并返回或抛出预设任务结果。"""

        if not isinstance(config, Task1Config):
            raise TypeError("测试注册项要求 Task1Config")
        if not isinstance(context, TaskContext):
            raise TypeError("测试注册项要求 TaskContext")
        self.run_calls.append(config)
        self.context_calls.append(context)
        if self.run_error is not None:
            raise self.run_error
        return None


class CommandCliTest(unittest.TestCase):
    """验证 app.py 保持任务无关且只接受环境文件入口参数。"""

    @classmethod
    def setUpClass(cls) -> None:
        """加载一次正式 Task1 配置供内存分派场景复用。"""

        cls.config = load_task1_config(
            environ={"MOTION_MAGNET_GPIO_LINE": "12"},
            env_path=None,
        )
        cls.runtime = RuntimeConfig(
            task_name="task1",
            task_config_path=Path("configs/tasks/task1.json"),
            environ={"EC_TASK": "task1"},
        )
        cls.runner = CliRunner()

    def setUp(self) -> None:
        """替换跨进程锁，保持CLI单元测试不访问/run。"""

        patcher = patch("ec_26_e.command.app.TaskExecutionLock")
        self.addCleanup(patcher.stop)
        self.execution_lock = patcher.start()

    def test_help_has_no_side_effect_and_no_task_subcommand(self) -> None:
        """帮助入口不得加载环境，且不再暴露 task1/config/output。"""

        loader = Mock(side_effect=AssertionError("不应加载环境"))
        with patch(
            "ec_26_e.command.app.load_runtime_config",
            loader,
        ):
            result = self.runner.invoke(app, ["--help"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("--env", result.output)
        self.assertIn("--task", result.output)
        self.assertIn("--button-service", result.output)
        self.assertNotIn("--output", result.output)
        self.assertNotIn("Commands", result.output)
        loader.assert_not_called()

    def test_no_arguments_runs_env_selected_task(self) -> None:
        """无子命令启动应读取默认 .env 并运行其选择的注册任务。"""

        registration = FakeRegistration(self.config)
        runtime_loader = Mock(return_value=self.runtime)
        registration_lookup = Mock(return_value=registration)
        with (
            patch(
                "ec_26_e.command.app.load_runtime_config",
                runtime_loader,
            ),
            patch(
                "ec_26_e.command.app.get_task_registration",
                registration_lookup,
            ),
        ):
            result = self.runner.invoke(app, [])

        self.assertEqual(result.exit_code, 0, result.output)
        runtime_loader.assert_called_once_with(
            DEFAULT_ENV_PATH,
            task_name=None,
            task_config_path=None,
        )
        registration_lookup.assert_called_once_with("task1")
        self.assertEqual(registration.load_calls, [self.runtime])
        self.assertEqual(registration.run_calls, [self.config])
        self.assertEqual(registration.context_calls, [TaskContext()])
        self.assertIn("有效配置：task=task1", result.output)
        self.assertIn("json=configs/tasks/task1.json", result.output)
        self.assertIn("draw=false", result.output)
        self.assertIn("任务完成：task=task1", result.output)
        self.assertNotIn("result=", result.output)

    def test_env_option_only_changes_environment_source(self) -> None:
        """CLI 只允许切换 .env 来源，不直接选择任务或任务参数。"""

        registration = FakeRegistration(self.config)
        runtime_loader = Mock(return_value=self.runtime)
        with (
            patch(
                "ec_26_e.command.app.load_runtime_config",
                runtime_loader,
            ),
            patch(
                "ec_26_e.command.app.get_task_registration",
                return_value=registration,
            ),
        ):
            result = self.runner.invoke(
                app,
                ["--env", "deploy/site.env"],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        runtime_loader.assert_called_once_with(
            Path("deploy/site.env"),
            task_name=None,
            task_config_path=None,
        )

    def test_explicit_task_selection_is_forwarded_without_other_overrides(self) -> None:
        """按钮子进程应能只覆盖任务名称和JSON路径。"""

        registration = FakeRegistration(self.config)
        runtime_loader = Mock(return_value=self.runtime)
        with (
            patch("ec_26_e.command.app.load_runtime_config", runtime_loader),
            patch(
                "ec_26_e.command.app.get_task_registration",
                return_value=registration,
            ),
        ):
            result = self.runner.invoke(
                app,
                [
                    "--env",
                    "generation/machine.env",
                    "--task",
                    "task1",
                    "--task-config",
                    "generation/tasks/task1.json",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        runtime_loader.assert_called_once_with(
            Path("generation/machine.env"),
            task_name="task1",
            task_config_path=Path("generation/tasks/task1.json"),
        )

    def test_draw_setting_is_passed_through_global_context(self) -> None:
        """环境绘制开关应经 TaskContext 传给任务且出现在摘要中。"""

        registration = FakeRegistration(self.config)
        runtime = RuntimeConfig(
            task_name="task1",
            task_config_path=Path("configs/tasks/task1.json"),
            environ={"EC_TASK": "task1", "EC_DRAW_ENABLED": "true"},
            draw_enabled=True,
        )
        with (
            patch(
                "ec_26_e.command.app.load_runtime_config",
                return_value=runtime,
            ),
            patch(
                "ec_26_e.command.app.get_task_registration",
                return_value=registration,
            ),
        ):
            result = self.runner.invoke(app, [])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            registration.context_calls,
            [TaskContext(draw_enabled=True)],
        )
        self.assertIn("draw=true", result.output)

    def test_selection_or_json_failures_exit_two_before_run(self) -> None:
        """环境选择及任务 JSON 错误应退出 2 且不运行任务。"""

        for error in (
            ValueError("缺少 EC_TASK"),
            FileNotFoundError("任务 JSON 不存在"),
        ):
            with self.subTest(error=type(error).__name__):
                runtime_loader = Mock(side_effect=error)
                with patch(
                    "ec_26_e.command.app.load_runtime_config",
                    runtime_loader,
                ):
                    result = self.runner.invoke(app, [])

                self.assertEqual(result.exit_code, 2)
                self.assertIn("启动配置失败", result.output)

    def test_expected_task_failure_exits_one(self) -> None:
        """注册表包装的任务预期异常应映射为退出码 1。"""

        registration = FakeRegistration(
            self.config,
            run_error=TaskExecutionError("task1：模拟失败"),
        )
        with (
            patch(
                "ec_26_e.command.app.load_runtime_config",
                return_value=self.runtime,
            ),
            patch(
                "ec_26_e.command.app.get_task_registration",
                return_value=registration,
            ),
        ):
            result = self.runner.invoke(app, [])

        self.assertEqual(result.exit_code, 1)
        self.assertIn("任务执行失败：task1：模拟失败", result.output)

    def test_retryable_task_failure_uses_temporary_exit_code(self) -> None:
        """可恢复失败应使用按钮调度器唯一识别的75退出码。"""

        registration = FakeRegistration(
            self.config,
            run_error=TaskExecutionError("暂时失败", retryable=True),
        )
        with (
            patch(
                "ec_26_e.command.app.load_runtime_config",
                return_value=self.runtime,
            ),
            patch(
                "ec_26_e.command.app.get_task_registration",
                return_value=registration,
            ),
        ):
            result = self.runner.invoke(app, [])

        self.assertEqual(result.exit_code, 75)

    def test_keyboard_interrupt_exits_130(self) -> None:
        """用户中断应使用稳定退出码并包含环境选择的任务名。"""

        registration = FakeRegistration(
            self.config,
            run_error=KeyboardInterrupt(),
        )
        with (
            patch(
                "ec_26_e.command.app.load_runtime_config",
                return_value=self.runtime,
            ),
            patch(
                "ec_26_e.command.app.get_task_registration",
                return_value=registration,
            ),
        ):
            result = self.runner.invoke(app, [])

        self.assertEqual(result.exit_code, 130)
        self.assertIn("任务已由用户中断：task1", result.output)

    def test_preview_cancellation_exits_130(self) -> None:
        """Esc 或窗口关闭触发的任务取消应映射为退出码 130。"""

        registration = FakeRegistration(
            self.config,
            run_error=TaskCancelledError("用户关闭预览"),
        )
        with (
            patch(
                "ec_26_e.command.app.load_runtime_config",
                return_value=self.runtime,
            ),
            patch(
                "ec_26_e.command.app.get_task_registration",
                return_value=registration,
            ),
        ):
            result = self.runner.invoke(app, [])

        self.assertEqual(result.exit_code, 130)
        self.assertIn("任务已由用户中止：task1", result.output)


if __name__ == "__main__":
    unittest.main()
