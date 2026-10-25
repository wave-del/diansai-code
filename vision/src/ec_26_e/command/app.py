"""提供只依赖环境选择和任务注册表的唯一 Typer 启动入口。"""

from __future__ import annotations

from pathlib import Path

import typer

from ec_26_e.buttons import ButtonServiceConfig, create_a7z_button_bindings
from ec_26_e.config import (
    DEFAULT_ENV_PATH,
    RuntimeConfig,
    load_runtime_config,
)
from ec_26_e.tasks.registry import (
    TaskExecutionError,
    get_task_registration,
)
from ec_26_e.tasks import TaskCancelledError, TaskContext
from ec_26_e.tasks.button_dispatcher import run_button_dispatcher
from ec_26_e.tasks.execution_lock import (
    DEFAULT_EXECUTION_LOCK_PATH,
    DEFAULT_EXECUTION_RUNTIME_DIRECTORY,
    TaskBusyError,
    TaskExecutionLock,
)


RETRYABLE_TASK_EXIT_CODE = 75
DEFAULT_ACTIVE_CONFIG_LINK = Path.home() / ".local/share/ec26e-calibration/active"
DEFAULT_BUTTON_STATUS_PATH = (
    DEFAULT_EXECUTION_RUNTIME_DIRECTORY / "buttons-status.json"
)


app = typer.Typer(
    name="ec-26-e",
    help="根据 .env 选择并运行一个已注册任务。",
    add_completion=False,
)


def _effective_config_summary(
    runtime: RuntimeConfig,
    config: object,
    env_path: Path,
) -> str:
    """生成任务无关且不包含完整环境变量的最终配置摘要。"""

    return (
        "有效配置："
        f"task={runtime.task_name}，"
        f"env={env_path}，"
        f"json={runtime.task_config_path}，"
        f"draw={str(runtime.draw_enabled).lower()}，"
        f"model={type(config).__name__}，"
        f"values={config!r}"
    )


@app.command()
def main(
    env_path: Path = typer.Option(
        DEFAULT_ENV_PATH,
        "--env",
        help="机器部署 .env 路径，其中 EC_TASK 选择要运行的任务。",
    ),
    task_name: str | None = typer.Option(
        None,
        "--task",
        help="显式选择本次任务；只覆盖.env中的EC_TASK。",
    ),
    task_config_path: Path | None = typer.Option(
        None,
        "--task-config",
        help="配合--task显式选择本次JSON；其他参数仍读取.env。",
    ),
    execution_lock_held: bool = typer.Option(
        False,
        "--execution-lock-held",
        hidden=True,
        help="仅供已持有全局任务锁的按钮调度器调用。",
    ),
    button_service: bool = typer.Option(
        False,
        "--button-service",
        help="常驻监听实体按键并串行分派Task1或Task2。",
    ),
    active_config_link: Path = typer.Option(
        DEFAULT_ACTIVE_CONFIG_LINK,
        "--active-config",
        help="按钮触发时需要原子冻结的active配置链接。",
    ),
    task1_button_chip: Path = typer.Option(
        Path("/dev/gpiochip0"),
        "--task1-button-chip",
        help="Task1按键所在GPIO字符设备。",
    ),
    task1_button_line: int = typer.Option(
        38,
        "--task1-button-line",
        min=0,
        help="Task1按键GPIO line；A7Z物理35脚为38。",
    ),
    task2_button_chip: Path = typer.Option(
        Path("/dev/gpiochip1"),
        "--task2-button-chip",
        help="Task2按键所在GPIO字符设备。",
    ),
    task2_button_line: int = typer.Option(
        36,
        "--task2-button-line",
        min=0,
        help="Task2按键GPIO line；A7Z物理37脚为36。",
    ),
    button_retry_window_seconds: float = typer.Option(
        120.0,
        "--button-retry-window",
        min=1.0,
        help="单次按键允许可重试任务持续尝试的总秒数。",
    ),
    button_status_path: Path = typer.Option(
        DEFAULT_BUTTON_STATUS_PATH,
        "--button-status",
        help="实时按键和任务状态JSON路径。",
    ),
) -> None:
    """加载环境选择和任务 JSON，然后分派并运行一次任务。"""

    if button_service:
        if task_name is not None or task_config_path is not None:
            typer.echo("按钮服务不能同时指定单次--task/--task-config", err=True)
            raise typer.Exit(code=2)
        service_config = ButtonServiceConfig(
            bindings=create_a7z_button_bindings(
                task1_chip_path=task1_button_chip,
                task1_line_offset=task1_button_line,
                task2_chip_path=task2_button_chip,
                task2_line_offset=task2_button_line,
            ),
            active_link=active_config_link,
            status_path=button_status_path,
            lock_path=DEFAULT_EXECUTION_LOCK_PATH,
            retry_window_seconds=button_retry_window_seconds,
        )
        try:
            run_button_dispatcher(service_config)
        except (OSError, RuntimeError, ValueError) as error:
            typer.echo(f"按键服务失败：{error}", err=True)
            raise typer.Exit(code=1) from error
        return

    try:
        runtime = load_runtime_config(
            env_path,
            task_name=task_name,
            task_config_path=task_config_path,
        )
        registration = get_task_registration(runtime.task_name)
        config = registration.load(runtime)
    except (OSError, TypeError, ValueError) as error:
        typer.echo(f"启动配置失败：{error}", err=True)
        raise typer.Exit(code=2) from error

    typer.echo(
        _effective_config_summary(runtime, config, env_path),
        err=True,
    )
    execution_lock = None if execution_lock_held else TaskExecutionLock()
    try:
        if execution_lock is not None:
            execution_lock.acquire()
        registration.run(config, TaskContext(draw_enabled=runtime.draw_enabled))
    except TaskBusyError as error:
        typer.echo(f"任务未启动：{error}", err=True)
        raise typer.Exit(code=3) from error
    except KeyboardInterrupt as error:
        typer.echo(
            f"任务已由用户中断：{runtime.task_name}",
            err=True,
        )
        raise typer.Exit(code=130) from error
    except TaskCancelledError as error:
        typer.echo(
            f"任务已由用户中止：{runtime.task_name}",
            err=True,
        )
        raise typer.Exit(code=130) from error
    except TaskExecutionError as error:
        typer.echo(f"任务执行失败：{error}", err=True)
        raise typer.Exit(
            code=RETRYABLE_TASK_EXIT_CODE if error.retryable else 1
        ) from error
    finally:
        if execution_lock is not None:
            execution_lock.release()

    typer.echo(f"任务完成：task={runtime.task_name}")
