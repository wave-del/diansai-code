"""验证由 .env 驱动的任务选择和任务 JSON 路径解析。

该文件只使用临时环境文件，不访问设备或运行任务。可通过
``python -m unittest tests.unit.test_runtime_config`` 独立运行。
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ec_26_e.config import (
    DEFAULT_TASK_CONFIG_DIRECTORY,
    DRAW_ENABLED_ENV,
    RuntimeConfig,
    load_runtime_config,
)


class RuntimeConfigTest(unittest.TestCase):
    """验证环境任务选择的默认派生、覆盖顺序和严格校验。"""

    def test_env_selects_task_and_derives_json_path(self) -> None:
        """EC_TASK 应选择任务并默认映射到同名任务 JSON。"""

        with TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text("EC_TASK=task1\n", encoding="utf-8")

            runtime = load_runtime_config(env_path, environ={})

        self.assertEqual(runtime.task_name, "task1")
        self.assertEqual(
            runtime.task_config_path,
            DEFAULT_TASK_CONFIG_DIRECTORY / "task1.json",
        )
        self.assertEqual(runtime.environ["EC_TASK"], "task1")
        self.assertFalse(runtime.draw_enabled)

    def test_env_can_explicitly_choose_task_json(self) -> None:
        """EC_TASK_CONFIG 应允许机器部署选择对应任务 JSON 路径。"""

        with TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "EC_TASK=task1\n"
                "EC_TASK_CONFIG=configs/tasks/site-task1.json\n",
                encoding="utf-8",
            )

            runtime = load_runtime_config(env_path, environ={})

        self.assertEqual(
            runtime.task_config_path,
            Path("configs/tasks/site-task1.json"),
        )

    def test_explicit_task_and_json_override_only_selection(self) -> None:
        """按钮选择应覆盖任务字段，同时保留.env中的其他机器参数。"""

        with TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "EC_TASK=task1\n"
                "EC_TASK_CONFIG=configs/tasks/task1.json\n"
                "CAMERA_DEVICE_PATH=/dev/video7\n",
                encoding="utf-8",
            )
            runtime = load_runtime_config(
                env_path,
                environ={},
                task_name="task2",
                task_config_path=Path("frozen/tasks/task2.json"),
            )

        self.assertEqual(runtime.task_name, "task2")
        self.assertEqual(
            runtime.task_config_path,
            Path("frozen/tasks/task2.json"),
        )
        self.assertEqual(runtime.environ["CAMERA_DEVICE_PATH"], "/dev/video7")

    def test_exported_environment_wins_without_mutation(self) -> None:
        """导出环境应覆盖 .env，且调用方映射和文件保持不变。"""

        with TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            content = (
                "EC_TASK=other\n"
                "EC_TASK_CONFIG=configs/tasks/other.json\n"
            )
            env_path.write_text(content, encoding="utf-8")
            environ = {
                "EC_TASK": "task1",
                "EC_TASK_CONFIG": "configs/tasks/task1.json",
            }
            snapshot = dict(environ)

            runtime = load_runtime_config(
                env_path,
                environ=environ,
            )

            self.assertEqual(environ, snapshot)
            self.assertEqual(
                env_path.read_text(encoding="utf-8"),
                content,
            )

        self.assertEqual(runtime.task_name, "task1")
        self.assertEqual(
            runtime.task_config_path,
            Path("configs/tasks/task1.json"),
        )

    def test_contract_is_frozen_and_environment_is_read_only(self) -> None:
        """运行选择对象及其环境快照都不得被调用方修改。"""

        runtime = load_runtime_config(
            None,
            environ={"EC_TASK": "task1"},
        )

        self.assertEqual(
            tuple(field.name for field in fields(RuntimeConfig)),
            ("task_name", "task_config_path", "environ", "draw_enabled"),
        )
        self.assertFalse(hasattr(runtime, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            runtime.task_name = "other"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            runtime.environ["EC_TASK"] = "other"  # type: ignore[index]

    def test_draw_enabled_accepts_strict_boolean_spellings(self) -> None:
        """全局绘制开关应接受约定拼写并保持默认关闭。"""

        accepted = {
            "true": True,
            "1": True,
            "yes": True,
            "on": True,
            " TRUE ": True,
            "false": False,
            "0": False,
            "no": False,
            "off": False,
            " OFF ": False,
        }
        for spelling, expected in accepted.items():
            with self.subTest(spelling=spelling):
                runtime = load_runtime_config(
                    None,
                    environ={
                        "EC_TASK": "task1",
                        DRAW_ENABLED_ENV: spelling,
                    },
                )
                self.assertIs(runtime.draw_enabled, expected)

    def test_exported_draw_setting_wins_over_env_file(self) -> None:
        """进程环境中的绘制开关应覆盖 .env 且不修改输入映射。"""

        with TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "EC_TASK=task1\nEC_DRAW_ENABLED=false\n",
                encoding="utf-8",
            )
            environ = {
                "EC_TASK": "task1",
                DRAW_ENABLED_ENV: "true",
            }
            snapshot = dict(environ)
            runtime = load_runtime_config(env_path, environ=environ)

        self.assertTrue(runtime.draw_enabled)
        self.assertEqual(environ, snapshot)

    def test_invalid_draw_setting_is_rejected(self) -> None:
        """模糊或非字符串绘制开关必须在任务分派前失败。"""

        for value in ("enabled", "2", "", "真"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, DRAW_ENABLED_ENV):
                    load_runtime_config(
                        None,
                        environ={
                            "EC_TASK": "task1",
                            DRAW_ENABLED_ENV: value,
                        },
                    )
        with self.assertRaisesRegex(TypeError, DRAW_ENABLED_ENV):
            load_runtime_config(
                None,
                environ={
                    "EC_TASK": "task1",
                    DRAW_ENABLED_ENV: True,  # type: ignore[dict-item]
                },
            )

    def test_missing_or_invalid_selection_is_rejected(self) -> None:
        """缺失任务、非法名称和非法 JSON 路径必须在分派前失败。"""

        invalid_environments = (
            {},
            {"EC_TASK": ""},
            {"EC_TASK": "Task1"},
            {"EC_TASK": "task one"},
            {
                "EC_TASK": "task1",
                "EC_TASK_CONFIG": "",
            },
            {
                "EC_TASK": "task1",
                "EC_TASK_CONFIG": "configs/tasks/task1.yaml",
            },
        )
        for environ in invalid_environments:
            with self.subTest(environ=environ):
                with self.assertRaises((TypeError, ValueError)):
                    load_runtime_config(None, environ=environ)


if __name__ == "__main__":
    unittest.main()
