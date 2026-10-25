"""管理现场标定草稿、整套校验、原子发布和历史恢复。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import fcntl
import os
from pathlib import Path
import re
import shutil

from ec_26_e.config.env import load_env_file
from ec_26_e.config.task1 import Task1Config, load_task1_config
from ec_26_e.config.task2 import Task2Config, load_task2_config


_ENV_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_SUPPORTED_TASKS = frozenset({"task1", "task2"})
_VALIDATION_MARKER = ".validated"


@dataclass(frozen=True, slots=True)
class CalibrationPaths:
    """集中描述现场标定数据目录中的稳定路径。"""

    root: Path

    @property
    def editing_directory(self) -> Path:
        """返回用户直接编辑的草稿目录。"""

        return self.root / "editing"

    @property
    def editing_env(self) -> Path:
        """返回草稿机器环境文件。"""

        return self.editing_directory / "machine.env"

    @property
    def editing_task1(self) -> Path:
        """返回 Task1 草稿 JSON。"""

        return self.editing_directory / "tasks" / "task1.json"

    @property
    def editing_task2(self) -> Path:
        """返回 Task2 草稿 JSON。"""

        return self.editing_directory / "tasks" / "task2.json"

    @property
    def generations_directory(self) -> Path:
        """返回所有已验证发布版本的目录。"""

        return self.root / "generations"

    @property
    def active_link(self) -> Path:
        """返回原子指向当前发布版本的符号链接。"""

        return self.root / "active"

    @property
    def active_env(self) -> Path:
        """返回当前生效的机器环境文件。"""

        return self.active_link / "machine.env"

    def active_task(self, task_name: str) -> Path:
        """返回当前生效任务的 JSON 路径。"""

        if task_name not in _SUPPORTED_TASKS:
            raise ValueError(f"不支持的任务：{task_name}")
        return self.active_link / "tasks" / f"{task_name}.json"

    @property
    def lock_file(self) -> Path:
        """返回串行化初始化、发布和恢复操作的锁文件。"""

        return self.root / ".configuration.lock"


@dataclass(frozen=True, slots=True)
class CalibrationValidation:
    """保存一套草稿配置的强类型校验结果。"""

    selected_task: str
    task1: Task1Config
    task2: Task2Config


def default_calibration_paths() -> CalibrationPaths:
    """返回当前用户持久化现场标定配置的默认位置。"""

    return CalibrationPaths(Path.home() / ".local/share/ec26e-calibration")


def _atomic_write(path: Path, content: str) -> None:
    """在同目录写入、同步并原子替换一个 UTF-8 文本文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _environment_values(path: Path) -> dict[str, str]:
    """读取环境文件为独立映射，避免污染当前进程环境。"""

    values: dict[str, str] = {}
    load_env_file(path, environ=values, override=True)
    return values


def _render_environment(
    template: str,
    values: Mapping[str, str],
) -> str:
    """使用现场值覆盖模板，同时保留模板注释和未知现场变量。"""

    rendered: list[str] = []
    consumed: set[str] = set()
    for line in template.splitlines():
        match = _ENV_ASSIGNMENT.match(line.strip())
        if match is None or match.group(1) not in values:
            rendered.append(line)
            continue
        name = match.group(1)
        value = values[name]
        if "\n" in value or "\r" in value:
            raise ValueError(f"环境变量 {name} 不允许包含换行")
        rendered.append(f"{name}={value}")
        consumed.add(name)
    extras = sorted(set(values) - consumed)
    if extras:
        rendered.extend(("", "# 当前设备额外保留的现场变量"))
        rendered.extend(f"{name}={values[name]}" for name in extras)
    return "\n".join(rendered).rstrip() + "\n"


def _replace_environment_values(
    content: str,
    updates: Mapping[str, str],
) -> str:
    """替换环境文本中的指定键，并在缺失时追加。"""

    lines = content.splitlines()
    remaining = dict(updates)
    result: list[str] = []
    for line in lines:
        match = _ENV_ASSIGNMENT.match(line.strip())
        if match is None or match.group(1) not in remaining:
            result.append(line)
            continue
        name = match.group(1)
        result.append(f"{name}={remaining.pop(name)}")
    if remaining:
        result.append("")
        result.extend(f"{name}={value}" for name, value in remaining.items())
    return "\n".join(result).rstrip() + "\n"


def _normalized_environment(
    paths: CalibrationPaths,
    source: Path,
) -> tuple[str, str]:
    """校验任务选择并把 JSON 选择固定到原子 active 路径。"""

    values = _environment_values(source)
    selected_task = values.get("EC_TASK", "").strip()
    if selected_task not in _SUPPORTED_TASKS:
        raise ValueError("EC_TASK 必须是 task1 或 task2")
    content = source.read_text(encoding="utf-8-sig")
    normalized = _replace_environment_values(
        content,
        {
            "EC_TASK": selected_task,
            "EC_TASK_CONFIG": str(paths.active_task(selected_task)),
        },
    )
    return normalized, selected_task


def validate_calibration(
    env_path: Path,
    task1_path: Path,
    task2_path: Path,
) -> CalibrationValidation:
    """使用正式 Task1/Task2 加载器校验一整套现场配置。"""

    values = _environment_values(env_path)
    selected_task = values.get("EC_TASK", "").strip()
    if selected_task not in _SUPPORTED_TASKS:
        raise ValueError("EC_TASK 必须是 task1 或 task2")
    task1 = load_task1_config(task1_path, env_path=env_path, environ={})
    task2 = load_task2_config(task2_path, env_path=env_path, environ={})
    return CalibrationValidation(
        selected_task=selected_task,
        task1=task1,
        task2=task2,
    )


def validate_draft(paths: CalibrationPaths) -> CalibrationValidation:
    """校验当前编辑草稿但不发布或访问设备。"""

    return validate_calibration(
        paths.editing_env,
        paths.editing_task1,
        paths.editing_task2,
    )


def _generation_name() -> str:
    """生成按时间可排序且同秒不冲突的版本名称。"""

    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _switch_active(paths: CalibrationPaths, generation: Path) -> None:
    """通过同目录符号链接替换原子切换整套生效配置。"""

    relative_target = generation.relative_to(paths.root)
    temporary = paths.root / f".active.tmp-{os.getpid()}"
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(relative_target, target_is_directory=True)
    os.replace(temporary, paths.active_link)
    root_fd = os.open(paths.root, os.O_RDONLY)
    try:
        os.fsync(root_fd)
    finally:
        os.close(root_fd)


def _copy_active_to_draft(paths: CalibrationPaths) -> None:
    """把当前已验证版本完整复制到稳定草稿路径。"""

    active = paths.active_link.resolve(strict=True)
    paths.editing_directory.mkdir(parents=True, exist_ok=True)
    (paths.editing_directory / "tasks").mkdir(parents=True, exist_ok=True)
    _atomic_write(
        paths.editing_env,
        (active / "machine.env").read_text(encoding="utf-8"),
    )
    _atomic_write(
        paths.editing_task1,
        (active / "tasks/task1.json").read_text(encoding="utf-8"),
    )
    _atomic_write(
        paths.editing_task2,
        (active / "tasks/task2.json").read_text(encoding="utf-8"),
    )


def initialize_calibration(
    paths: CalibrationPaths,
    *,
    source_env: Path,
    env_template: Path,
    source_task1: Path,
    source_task2: Path,
) -> None:
    """首次从仓库现场值和完整模板建立可编辑、可回退配置。"""

    paths.root.mkdir(parents=True, exist_ok=True)
    paths.generations_directory.mkdir(parents=True, exist_ok=True)
    with paths.lock_file.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not paths.active_link.exists():
            values = _environment_values(source_env)
            merged = _render_environment(
                env_template.read_text(encoding="utf-8-sig"),
                values,
            )
            bootstrap = paths.root / ".bootstrap.env"
            _atomic_write(bootstrap, merged)
            try:
                normalized, _ = _normalized_environment(paths, bootstrap)
            finally:
                bootstrap.unlink(missing_ok=True)
            generation = paths.generations_directory / _generation_name()
            (generation / "tasks").mkdir(parents=True, exist_ok=False)
            _atomic_write(generation / "machine.env", normalized)
            _atomic_write(
                generation / "tasks/task1.json",
                source_task1.read_text(encoding="utf-8-sig"),
            )
            _atomic_write(
                generation / "tasks/task2.json",
                source_task2.read_text(encoding="utf-8-sig"),
            )
            validate_calibration(
                generation / "machine.env",
                generation / "tasks/task1.json",
                generation / "tasks/task2.json",
            )
            _atomic_write(generation / _VALIDATION_MARKER, "ok\n")
            _switch_active(paths, generation)
        else:
            active = paths.active_link.resolve(strict=True)
            marker = active / _VALIDATION_MARKER
            if not marker.exists():
                validate_calibration(
                    active / "machine.env",
                    active / "tasks/task1.json",
                    active / "tasks/task2.json",
                )
                _atomic_write(marker, "ok\n")
        if not all(
            path.exists()
            for path in (
                paths.editing_env,
                paths.editing_task1,
                paths.editing_task2,
            )
        ):
            _copy_active_to_draft(paths)


def publish_draft(
    paths: CalibrationPaths,
    *,
    keep_generations: int = 10,
) -> CalibrationValidation:
    """校验草稿、创建完整版本并原子切换 active。"""

    if keep_generations < 2:
        raise ValueError("keep_generations 必须至少为 2")
    paths.root.mkdir(parents=True, exist_ok=True)
    with paths.lock_file.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        normalized, _ = _normalized_environment(paths, paths.editing_env)
        temporary_env = paths.root / ".candidate.env"
        _atomic_write(temporary_env, normalized)
        try:
            validation = validate_calibration(
                temporary_env,
                paths.editing_task1,
                paths.editing_task2,
            )
        finally:
            temporary_env.unlink(missing_ok=True)
        generation = paths.generations_directory / _generation_name()
        (generation / "tasks").mkdir(parents=True, exist_ok=False)
        _atomic_write(generation / "machine.env", normalized)
        _atomic_write(
            generation / "tasks/task1.json",
            paths.editing_task1.read_text(encoding="utf-8-sig"),
        )
        _atomic_write(
            generation / "tasks/task2.json",
            paths.editing_task2.read_text(encoding="utf-8-sig"),
        )
        validate_calibration(
            generation / "machine.env",
            generation / "tasks/task1.json",
            generation / "tasks/task2.json",
        )
        _atomic_write(generation / _VALIDATION_MARKER, "ok\n")
        _switch_active(paths, generation)
        _atomic_write(paths.editing_env, normalized)
        _prune_generations(paths, keep_generations)
        return validation


def _generation_directories(paths: CalibrationPaths) -> list[Path]:
    """返回按名称升序排列的完整发布版本。"""

    if not paths.generations_directory.exists():
        return []
    return sorted(
        path
        for path in paths.generations_directory.iterdir()
        if path.is_dir() and (path / _VALIDATION_MARKER).is_file()
    )


def _prune_generations(paths: CalibrationPaths, keep: int) -> None:
    """删除超过保留数量且当前未生效的最旧版本。"""

    generations = _generation_directories(paths)
    active = paths.active_link.resolve(strict=True)
    removable = [path for path in generations if path.resolve() != active]
    excess = max(0, len(generations) - keep)
    for generation in removable[:excess]:
        shutil.rmtree(generation)


def restore_previous(paths: CalibrationPaths) -> Path:
    """把 active 原子切换到最近一个非当前的已验证版本。"""

    with paths.lock_file.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        active = paths.active_link.resolve(strict=True)
        candidates = [
            path
            for path in _generation_directories(paths)
            if path.resolve() != active
        ]
        if not candidates:
            raise RuntimeError("没有可恢复的上一版配置")
        target = candidates[-1]
        validate_calibration(
            target / "machine.env",
            target / "tasks/task1.json",
            target / "tasks/task2.json",
        )
        _switch_active(paths, target)
        _copy_active_to_draft(paths)
        return target


def reset_draft(paths: CalibrationPaths) -> None:
    """丢弃未发布修改并从当前 active 重建草稿。"""

    with paths.lock_file.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        _copy_active_to_draft(paths)


def active_generation(paths: CalibrationPaths) -> Path:
    """返回当前生效版本的真实目录。"""

    return paths.active_link.resolve(strict=True)
