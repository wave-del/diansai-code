"""公开项目环境配置与 Task1 强类型 JSON 装配入口。"""

from ec_26_e.config.env import load_env_file
from ec_26_e.config.runtime import (
    DEFAULT_ENV_PATH,
    DEFAULT_TASK_CONFIG_DIRECTORY,
    DRAW_ENABLED_ENV,
    RuntimeConfig,
    TASK_CONFIG_PATH_ENV,
    TASK_NAME_ENV,
    load_runtime_config,
)
from ec_26_e.config.task1 import (
    DEFAULT_TASK1_CONFIG_PATH,
    DETECT1_PIECE_MODE_ENV,
    TASK1_ALLOWED_JUMP_FRAMES_ENV,
    TASK1_REVIEW_CYCLE_SECONDS_ENV,
    Task1Config,
    load_task1_config,
)
from ec_26_e.config.task2 import (
    DEFAULT_TASK2_CONFIG_PATH,
    TASK2_ALLOWED_JUMP_FRAMES_ENV,
    TASK2_DETECT1_PIECE_MODE_ENV,
    TASK2_REVIEW_CYCLE_SECONDS_ENV,
    Task2Config,
    load_task2_config,
)

__all__ = [
    "DEFAULT_ENV_PATH",
    "DEFAULT_TASK_CONFIG_DIRECTORY",
    "DEFAULT_TASK1_CONFIG_PATH",
    "DEFAULT_TASK2_CONFIG_PATH",
    "DETECT1_PIECE_MODE_ENV",
    "DRAW_ENABLED_ENV",
    "RuntimeConfig",
    "TASK_CONFIG_PATH_ENV",
    "TASK1_ALLOWED_JUMP_FRAMES_ENV",
    "TASK1_REVIEW_CYCLE_SECONDS_ENV",
    "TASK2_ALLOWED_JUMP_FRAMES_ENV",
    "TASK2_DETECT1_PIECE_MODE_ENV",
    "TASK2_REVIEW_CYCLE_SECONDS_ENV",
    "TASK_NAME_ENV",
    "Task1Config",
    "Task2Config",
    "load_env_file",
    "load_runtime_config",
    "load_task1_config",
    "load_task2_config",
]
