"""在真实A7Z设备上监听物理35、37脚，按下时打印物理引脚编号。

该脚本依赖 ``/dev/gpiochip0``、``/dev/gpiochip1`` 和用户级systemd。
独立运行：``poetry run python tests/external/gpio/test_buttons.py``。
默认临时停止正式按键服务，并在退出时恢复服务。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
import subprocess
import sys
import time

import typer


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ec_26_e.buttons import (
    ButtonBinding,
    ButtonDebouncer,
    PeripheryButtonSource,
    create_a7z_button_bindings,
)


SERVICE_NAME = "ec26e-button-dispatcher.service"
STATUS_PATH = Path("/run/user/1000/ec26e/buttons-status.json")

app = typer.Typer(
    add_completion=False,
    help="测试A7Z物理35和37脚；按下时打印对应物理引脚编号。",
)


def _run_systemctl(action: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """调用当前用户的systemd管理正式按键服务。"""

    return subprocess.run(
        ["systemctl", "--user", action, SERVICE_NAME],
        check=check,
        capture_output=True,
        text=True,
    )


def _service_is_active() -> bool:
    """判断正式按键服务当前是否正在占用GPIO。"""

    result = _run_systemctl("is-active", check=False)
    if result.returncode == 0:
        return True
    if result.returncode == 3:
        return False
    detail = result.stderr.strip() or result.stdout.strip()
    raise RuntimeError(f"无法查询按键服务状态：{detail}")


def _refuse_if_task_active(status_path: Path = STATUS_PATH) -> None:
    """状态快照显示正式任务执行中时拒绝停止服务。"""

    if not status_path.is_file():
        return
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"无法确认正式任务是否空闲：{error}") from error
    active_task = payload.get("active_task")
    if active_task is not None:
        raise RuntimeError(f"{active_task}正在执行，拒绝接管GPIO")


def monitor_button_presses(
    bindings: tuple[ButtonBinding, ...],
    *,
    debounce_seconds: float = 0.03,
    poll_interval_seconds: float = 0.01,
    source: PeripheryButtonSource | None = None,
    emit: Callable[[str], None] = print,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """等待全部按键松开，然后为每次稳定按下打印物理引脚编号。"""

    if debounce_seconds <= 0 or poll_interval_seconds <= 0:
        raise ValueError("消抖和轮询间隔必须大于0")
    button_source = source or PeripheryButtonSource(bindings)
    binding_by_task = {binding.task_name: binding for binding in bindings}
    trackers = {
        binding.task_name: ButtonDebouncer(debounce_seconds)
        for binding in bindings
    }
    armed = False
    button_source.open()
    try:
        emit("等待物理35、37脚全部松开……")
        while True:
            raw_states: Mapping[str, bool] = button_source.read_pressed()
            changes: list[str] = []
            for task_name, tracker in trackers.items():
                if tracker.update(raw_states[task_name], monotonic()):
                    changes.append(task_name)

            if not armed:
                if all(tracker.stable_pressed is False for tracker in trackers.values()):
                    armed = True
                    emit("开始监听；按 Ctrl+C 退出。")
            else:
                for task_name in changes:
                    if trackers[task_name].stable_pressed is True:
                        emit(str(binding_by_task[task_name].physical_pin))
            sleep(poll_interval_seconds)
    finally:
        button_source.close()


@app.command()
def main(
    manage_service: bool = typer.Option(
        True,
        "--manage-service/--no-manage-service",
        help="运行前停止正式按键服务，退出时恢复原状态。",
    ),
    debounce_seconds: float = typer.Option(
        0.03,
        "--debounce",
        min=0.001,
        help="确认按键电平稳定所需秒数。",
    ),
    poll_interval_seconds: float = typer.Option(
        0.01,
        "--poll-interval",
        min=0.001,
        help="GPIO轮询间隔秒数。",
    ),
) -> None:
    """安全接管两个GPIO并持续打印按下的物理引脚编号。"""

    service_was_active = False
    try:
        if manage_service:
            service_was_active = _service_is_active()
            if service_was_active:
                _refuse_if_task_active()
                typer.echo("临时停止正式按键服务……", err=True)
                _run_systemctl("stop")
        monitor_button_presses(
            create_a7z_button_bindings(),
            debounce_seconds=debounce_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
    except KeyboardInterrupt:
        typer.echo("\n按钮测试结束。", err=True)
    except (OSError, RuntimeError, ValueError) as error:
        typer.echo(f"按钮测试失败：{error}", err=True)
        raise typer.Exit(code=1) from error
    finally:
        if service_was_active:
            typer.echo("恢复正式按键服务……", err=True)
            try:
                _run_systemctl("start")
            except subprocess.CalledProcessError as error:
                detail = error.stderr.strip() or error.stdout.strip()
                typer.echo(f"恢复按键服务失败：{detail}", err=True)


if __name__ == "__main__":
    app()
