"""从机器环境中解析本次启动应运行的任务及其 JSON 路径。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
import re
from types import MappingProxyType

from ec_26_e.config.env import load_env_file


DEFAULT_ENV_PATH = Path(".env")
DEFAULT_TASK_CONFIG_DIRECTORY = Path("configs/tasks")
TASK_NAME_ENV = "EC_TASK"
TASK_CONFIG_PATH_ENV = "EC_TASK_CONFIG"
DRAW_ENABLED_ENV = "EC_DRAW_ENABLED"
_TASK_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")
_TRUE_VALUES = frozenset({"true", "1", "yes", "on"})
_FALSE_VALUES = frozenset({"false", "0", "no", "off"})


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """描述一次启动的任务选择、配置路径和冻结环境快照。"""

    task_name: str
    task_config_path: Path
    environ: Mapping[str, str]
    draw_enabled: bool = False

    def __post_init__(self) -> None:
        """校验任务名称、JSON 路径和环境快照的强类型结构。"""

        if not isinstance(self.task_name, str):
            raise TypeError("task_name 必须是字符串")
        if not _TASK_NAME_PATTERN.fullmatch(self.task_name):
            raise ValueError(
                "task_name 必须使用小写字母、数字、连字符或下划线"
            )
        if not isinstance(self.task_config_path, Path):
            raise TypeError("task_config_path 必须是 pathlib.Path")
        if (
            not self.task_config_path.name
            or self.task_config_path.suffix.lower() != ".json"
        ):
            raise ValueError("task_config_path 必须指向 .json 文件")
        if not isinstance(self.environ, Mapping):
            raise TypeError("environ 必须是字符串映射")
        for name, value in self.environ.items():
            if not isinstance(name, str) or not isinstance(value, str):
                raise TypeError("environ 必须只包含字符串键值")
        if not isinstance(self.draw_enabled, bool):
            raise TypeError("draw_enabled 必须是布尔值")
        object.__setattr__(
            self,
            "environ",
            MappingProxyType(dict(self.environ)),
        )


def _environment_boolean(
    environ: Mapping[str, str],
    name: str,
    default: bool,
) -> bool:
    """严格解析一个环境布尔值，并拒绝其他宽松拼写。"""

    raw_value = environ.get(name)
    if raw_value is None:
        return default
    if not isinstance(raw_value, str):
        raise TypeError(f"环境变量 {name} 必须是字符串")
    normalized = raw_value.strip().casefold()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(
        f"环境变量 {name} 必须为 true/false、1/0、yes/no 或 on/off"
    )


def load_runtime_config(
    env_path: Path | None = DEFAULT_ENV_PATH,
    *,
    environ: Mapping[str, str] | None = None,
    task_name: str | None = None,
    task_config_path: Path | None = None,
) -> RuntimeConfig:
    """合并导出环境与 ``.env``，并解析唯一任务选择。

    已导出的环境变量优先于 ``.env``。函数只操作环境副本，不会
    修改 ``os.environ`` 或调用方传入的映射。
    """

    if environ is not None and not isinstance(environ, Mapping):
        raise TypeError("environ 必须是字符串映射或 None")
    merged = dict(os.environ if environ is None else environ)
    if env_path is not None:
        if not isinstance(env_path, Path):
            raise TypeError("env_path 必须是 pathlib.Path 或 None")
        load_env_file(env_path, environ=merged, override=False)

    if task_name is None and task_config_path is not None:
        raise ValueError("显式 task_config_path 必须同时提供 task_name")
    if task_name is not None:
        if not isinstance(task_name, str):
            raise TypeError("task_name 必须是字符串或 None")
        merged[TASK_NAME_ENV] = task_name
        selected_path = (
            task_config_path
            if task_config_path is not None
            else DEFAULT_TASK_CONFIG_DIRECTORY / f"{task_name}.json"
        )
        merged[TASK_CONFIG_PATH_ENV] = str(selected_path)

    raw_task_name = merged.get(TASK_NAME_ENV)
    if raw_task_name is None:
        raise ValueError(f"环境变量 {TASK_NAME_ENV} 未配置")
    if not isinstance(raw_task_name, str):
        raise TypeError(f"环境变量 {TASK_NAME_ENV} 必须是字符串")
    task_name = raw_task_name.strip()
    if not task_name:
        raise ValueError(f"环境变量 {TASK_NAME_ENV} 不能为空")

    raw_config_path = merged.get(TASK_CONFIG_PATH_ENV)
    if raw_config_path is None:
        task_config_path = (
            DEFAULT_TASK_CONFIG_DIRECTORY / f"{task_name}.json"
        )
    else:
        if not isinstance(raw_config_path, str):
            raise TypeError(
                f"环境变量 {TASK_CONFIG_PATH_ENV} 必须是字符串"
            )
        if not raw_config_path.strip():
            raise ValueError(
                f"环境变量 {TASK_CONFIG_PATH_ENV} 不能为空"
            )
        task_config_path = Path(raw_config_path.strip())

    return RuntimeConfig(
        task_name=task_name,
        task_config_path=task_config_path,
        environ=MappingProxyType(merged),
        draw_enabled=_environment_boolean(
            merged,
            DRAW_ENABLED_ENV,
            False,
        ),
    )
