"""严格加载 Task2 随机拼图 JSON 与机器环境并构造冻结配置。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path

from ec_26_e.actuation import ActuationConfig
from ec_26_e.cam import CameraConfig
from ec_26_e.config.env import load_env_file
from ec_26_e.config.runtime import DEFAULT_ENV_PATH
from ec_26_e.config.task1 import (
    _DETECT1_FIELDS,
    _PLANNING_FIELDS,
    _RESIZE_FIELDS,
    _SCHEMA_VERSION,
    _TASK_FIELDS,
    _TOP_LEVEL_FIELDS,
    _actuation_from_env,
    _camera_from_env,
    _detect1_from_sources,
    _env_float,
    _env_integer,
    _json_point,
    _read_json_object,
    _reject_unknown_fields,
    _resize_from_sources,
    _section,
)
from ec_26_e.control import PlanningOptions
from ec_26_e.tasks.task2 import Task2Options
from ec_26_e.vision import Detect1Config, ResizeConfig
from ec_26_e.vision.resize.contracts import (
    ASPECT_RATIO_TOLERANCE,
    aspect_error,
)


DEFAULT_TASK2_CONFIG_PATH = Path("configs/tasks/task2.json")
TASK2_REVIEW_CYCLE_SECONDS_ENV = "TASK2_REVIEW_CYCLE_SECONDS"
TASK2_ALLOWED_JUMP_FRAMES_ENV = "TASK2_ALLOWED_JUMP_FRAMES"
TASK2_DETECT1_PIECE_MODE_ENV = "TASK2_DETECT1_PIECE_MODE"


@dataclass(frozen=True, slots=True)
class Task2Config:
    """聚合 Task2 启动前需要的全部强类型配置。"""

    camera: CameraConfig
    resize: ResizeConfig
    detect1: Detect1Config
    options: Task2Options
    actuation: ActuationConfig

    def __post_init__(self) -> None:
        """校验模块类型及 Task2 的随机拼图组合约束。"""

        expected_types = (
            ("camera", self.camera, CameraConfig),
            ("resize", self.resize, ResizeConfig),
            ("detect1", self.detect1, Detect1Config),
            ("options", self.options, Task2Options),
            ("actuation", self.actuation, ActuationConfig),
        )
        for name, value, expected_type in expected_types:
            if not isinstance(value, expected_type):
                raise TypeError(f"{name} 必须是 {expected_type.__name__}")
        if (
            self.resize.detect_width > self.camera.width
            or self.resize.detect_height > self.camera.height
        ):
            raise ValueError("工作区定位分辨率不能大于相机采集分辨率")
        camera_ratio = self.camera.width / self.camera.height
        detect_ratio = self.resize.detect_width / self.resize.detect_height
        if (
            aspect_error(detect_ratio, camera_ratio)
            > ASPECT_RATIO_TOLERANCE
        ):
            raise ValueError("工作区定位分辨率必须保持相机宽高比")
        if self.options.planning_options.mode != "random":
            raise ValueError("Task2 仅支持 random 拼图模式")
        if self.detect1.max_pieces < 2:
            raise ValueError("random 模式要求 Detect1 至少允许 2 片")


def _task2_from_sources(
    raw: dict[str, object],
    environ: Mapping[str, str],
) -> Task2Config:
    """把严格原始 JSON 与最终环境映射装配为 Task2Config。"""

    _reject_unknown_fields(raw, _TOP_LEVEL_FIELDS, "Task2 JSON")
    version = raw.get("schema_version")
    if type(version) is not int or version != _SCHEMA_VERSION:
        raise ValueError(
            f"Task2 JSON.schema_version 必须为 {_SCHEMA_VERSION}"
        )
    task_section = _section(raw, "task", _TASK_FIELDS)
    resize_section = _section(raw, "resize", _RESIZE_FIELDS)
    detect1_section = _section(raw, "detect1", _DETECT1_FIELDS)
    planning_section = _section(raw, "planning", _PLANNING_FIELDS)
    if "start_xy_a4_mm" in planning_section:
        planning_section["start_xy_a4_mm"] = _json_point(
            planning_section["start_xy_a4_mm"],
            "Task2 JSON.planning.start_xy_a4_mm",
        )

    planning_section.setdefault("mode", "random")
    planning = PlanningOptions(**planning_section)
    option_defaults = Task2Options()
    options = Task2Options(
        **task_section,
        review_cycle_seconds=_env_float(
            environ,
            TASK2_REVIEW_CYCLE_SECONDS_ENV,
            option_defaults.review_cycle_seconds,
        ),
        allowed_jump_frames=_env_integer(
            environ,
            TASK2_ALLOWED_JUMP_FRAMES_ENV,
            option_defaults.allowed_jump_frames,
        ),
        planning_options=planning,
    )
    return Task2Config(
        camera=_camera_from_env(environ),
        resize=_resize_from_sources(environ, resize_section),
        detect1=_detect1_from_sources(
            environ,
            detect1_section,
            piece_mode_env=TASK2_DETECT1_PIECE_MODE_ENV,
        ),
        options=options,
        actuation=_actuation_from_env(environ),
    )


def load_task2_config(
    path: Path = DEFAULT_TASK2_CONFIG_PATH,
    *,
    env_path: Path | None = DEFAULT_ENV_PATH,
    environ: Mapping[str, str] | None = None,
) -> Task2Config:
    """无副作用加载环境和 Task2 JSON，并返回完整强类型配置。"""

    if environ is not None and not isinstance(environ, Mapping):
        raise TypeError("environ 必须是字符串映射或 None")
    raw = _read_json_object(path)
    merged_environ = dict(os.environ if environ is None else environ)
    if env_path is not None:
        if not isinstance(env_path, Path):
            raise TypeError("env_path 必须是 pathlib.Path 或 None")
        load_env_file(env_path, environ=merged_environ, override=False)
    return _task2_from_sources(raw, merged_environ)
