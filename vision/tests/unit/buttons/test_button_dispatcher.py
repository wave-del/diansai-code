"""
验证实体按键消抖、GPIO映射、配置冻结和两分钟重试状态机。

该文件使用临时文件、假按键源和假子进程，不访问真实GPIO、相机或运动设备。
可通过pytest独立运行。
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from ec_26_e.buttons import ButtonBinding, ButtonDebouncer, ButtonServiceConfig
from ec_26_e.tasks.button_dispatcher import ButtonDispatcher


class FakeSource:
    """提供不访问GPIO的固定按键源。"""

    def open(self) -> None:
        """模拟成功打开。"""

    def read_pressed(self) -> dict[str, bool]:
        """返回两个按键均松开。"""

        return {"task1": False, "task2": False}

    def close(self) -> None:
        """模拟幂等释放。"""


class FakeProcess:
    """保存可控制退出码的任务子进程替身。"""

    def __init__(self, pid: int = 4321) -> None:
        """初始化运行中状态。"""

        self.pid = pid
        self.return_code: int | None = None

    def poll(self) -> int | None:
        """返回当前退出码。"""

        return self.return_code


def make_config(root: Path) -> ButtonServiceConfig:
    """创建包含冻结配置版本和临时锁路径的服务配置。"""

    generation = root / "generations" / "v1"
    (generation / "tasks").mkdir(parents=True)
    (generation / "machine.env").write_text(
        "EC_TASK=task2\nEC_DRAW_ENABLED=false\n",
        encoding="utf-8",
    )
    (generation / "tasks" / "task1.json").write_text("{}\n", encoding="utf-8")
    (generation / "tasks" / "task2.json").write_text("{}\n", encoding="utf-8")
    (root / "active").symlink_to(Path("generations/v1"))
    return ButtonServiceConfig(
        bindings=(
            ButtonBinding(35, Path("/dev/gpiochip0"), 38, "task1"),
            ButtonBinding(37, Path("/dev/gpiochip1"), 36, "task2"),
        ),
        active_link=root / "active",
        status_path=root / "status.json",
        lock_path=root / "task.lock",
        retry_window_seconds=120.0,
        retry_delay_seconds=2.0,
    )


def test_a7z_physical_pin_mapping_uses_two_gpiochips(tmp_path: Path) -> None:
    """验证物理35和37使用内核实际报告的chip及line。"""

    config = make_config(tmp_path)

    assert config.bindings == (
        ButtonBinding(35, Path("/dev/gpiochip0"), 38, "task1"),
        ButtonBinding(37, Path("/dev/gpiochip1"), 36, "task2"),
    )
    assert all(binding.physical_pin != 40 for binding in config.bindings)


def test_debouncer_requires_continuous_stability() -> None:
    """验证抖动会重新计时且只报告稳定状态变化。"""

    tracker = ButtonDebouncer(0.03)

    assert not tracker.update(False, 0.00)
    assert not tracker.update(False, 0.02)
    assert tracker.update(False, 0.03)
    assert not tracker.update(True, 0.04)
    assert not tracker.update(False, 0.05)
    assert not tracker.update(True, 0.06)
    assert not tracker.update(True, 0.08)
    assert tracker.update(True, 0.09)
    assert tracker.stable_pressed is True


def test_dispatcher_freezes_generation_and_only_overrides_task(tmp_path: Path) -> None:
    """验证任务命令固定同一代env/JSON并传入显式任务选择。"""

    config = make_config(tmp_path)
    process = FakeProcess()
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        """记录任务命令并返回运行中的子进程。"""

        calls.append((command, kwargs))
        return process

    dispatcher = ButtonDispatcher(
        config,
        source=FakeSource(),  # type: ignore[arg-type]
        popen=fake_popen,  # type: ignore[arg-type]
    )
    dispatcher._start_selected_task("task1", 10.0)

    command, kwargs = calls[0]
    generation = config.active_link.resolve()
    assert command[command.index("--env") + 1] == str(generation / "machine.env")
    assert command[command.index("--task") + 1] == "task1"
    assert command[command.index("--task-config") + 1] == str(
        generation / "tasks" / "task1.json"
    )
    assert "--execution-lock-held" in command
    assert kwargs["start_new_session"] is True
    environment = kwargs["env"]
    assert isinstance(environment, dict)
    assert "EC_TASK" not in environment
    dispatcher._finish_active("completed", "测试结束")


def test_retryable_failure_restarts_inside_window(tmp_path: Path) -> None:
    """验证75退出码在120秒内按固定延时启动下一次尝试。"""

    config = make_config(tmp_path)
    processes = [FakeProcess(1001), FakeProcess(1002)]
    calls = 0

    def fake_popen(_command: list[str], **_kwargs: object) -> FakeProcess:
        """依次返回两个假子进程。"""

        nonlocal calls
        process = processes[calls]
        calls += 1
        return process

    dispatcher = ButtonDispatcher(
        config,
        source=FakeSource(),  # type: ignore[arg-type]
        popen=fake_popen,  # type: ignore[arg-type]
    )
    dispatcher._start_selected_task("task2", 20.0)
    processes[0].return_code = 75
    dispatcher._service_active_task(21.0)
    assert dispatcher._state == "retry_wait"
    dispatcher._service_active_task(22.9)
    assert calls == 1
    dispatcher._service_active_task(23.0)
    assert calls == 2
    dispatcher._finish_active("completed", "测试结束")


def test_nonretryable_failure_releases_lock_immediately(tmp_path: Path) -> None:
    """验证运动等不可重试失败不会进入自动重试。"""

    config = make_config(tmp_path)
    process = FakeProcess()
    dispatcher = ButtonDispatcher(
        config,
        source=FakeSource(),  # type: ignore[arg-type]
        popen=lambda *_args, **_kwargs: process,  # type: ignore[arg-type]
    )
    dispatcher._start_selected_task("task1", 0.0)
    process.return_code = 1
    dispatcher._service_active_task(1.0)

    assert dispatcher._active_task is None
    assert dispatcher._state == "failed"
    assert dispatcher._execution_lock is None


def test_running_timeout_requests_safe_sigint(tmp_path: Path) -> None:
    """验证120秒到期先发送SIGINT而不是直接强杀。"""

    config = make_config(tmp_path)
    process = FakeProcess()
    dispatcher = ButtonDispatcher(
        config,
        source=FakeSource(),  # type: ignore[arg-type]
        popen=lambda *_args, **_kwargs: process,  # type: ignore[arg-type]
    )
    dispatcher._start_selected_task("task1", 0.0)

    with patch("ec_26_e.tasks.button_dispatcher.os.killpg") as killpg:
        dispatcher._service_active_task(120.0)

    killpg.assert_called_once_with(process.pid, 2)
    assert dispatcher._state == "stopping"
    process.return_code = 130
    dispatcher._service_active_task(121.0)
    assert dispatcher._active_task is None


def test_status_file_is_valid_json(tmp_path: Path) -> None:
    """验证实时状态采用可原子替换的JSON结构。"""

    config = make_config(tmp_path)
    dispatcher = ButtonDispatcher(
        config,
        source=FakeSource(),  # type: ignore[arg-type]
    )
    dispatcher._set_status("idle", "等待按键")

    status = json.loads(config.status_path.read_text(encoding="utf-8"))
    assert status["state"] == "idle"
    assert status["active_task"] is None
