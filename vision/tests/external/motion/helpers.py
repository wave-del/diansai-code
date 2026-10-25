"""提供 Servo42C 外部诊断脚本共享的配置、摘要和时间戳输出。

该模块只定义辅助函数，不会在导入时打开串口或驱动电机。
"""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from queue import Queue
from threading import Thread

import typer

from ec_26_e.motion import (
    MotionCommand,
    MotionExecution,
    MotionSerialConfig,
    SerialWriteEvent,
)


def create_serial_config(
    device_path: Path,
    baudrate: int,
    read_timeout_seconds: float,
    write_timeout_seconds: float,
    reply_timeout_seconds: float,
) -> MotionSerialConfig:
    """把已收集的 CLI 值转换为访问设备前校验的串口配置。"""

    return MotionSerialConfig(
        device_path=device_path,
        baudrate=baudrate,
        read_timeout_seconds=read_timeout_seconds,
        write_timeout_seconds=write_timeout_seconds,
        reply_timeout_seconds=reply_timeout_seconds,
    )


def dry_execution(
    commands: tuple[MotionCommand, ...],
) -> MotionExecution:
    """构造不含回包和耗时的 dry-run 执行结果。"""

    return MotionExecution(
        commands=commands,
        replies=(),
        elapsed_seconds=0.0,
    )


def execution_summary(
    *,
    operation: str,
    execute: bool,
    config: MotionSerialConfig,
    executions: tuple[MotionExecution, ...],
    requested: dict[str, object],
    no_op: bool = False,
) -> dict[str, object]:
    """生成两个外部诊断脚本共用的 JSON 兼容摘要。"""

    return {
        "operation": operation,
        "dry_run": not execute,
        "executed": execute and not no_op,
        "no_op": no_op,
        "serial": {
            "device_path": str(config.device_path),
            "baudrate": config.baudrate,
            "read_timeout_seconds": config.read_timeout_seconds,
            "write_timeout_seconds": config.write_timeout_seconds,
            "reply_timeout_seconds": config.reply_timeout_seconds,
        },
        "requested": requested,
        "executions": [
            execution.to_dict() for execution in executions
        ],
    }


def emit_summary(summary: dict[str, object]) -> None:
    """以 UTF-8 JSON 打印一次外部诊断摘要。"""

    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))


def emit_serial_write_event(event: SerialWriteEvent) -> None:
    """把一次真实串口写入的纳秒时间戳输出到终端 stderr。"""

    typer.echo(
        "[MOTION_TX] "
        + json.dumps(
            event.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        err=True,
    )


class SerialWriteEventReporter:
    """在后台线程逐条打印串口写入事件，避免阻塞发送链路。"""

    def __init__(self) -> None:
        """创建尚未启动的无界事件队列和输出线程。"""

        self._queue: Queue[SerialWriteEvent | None] = Queue()
        self._thread = Thread(
            target=self._run,
            name="motion-serial-write-reporter",
            daemon=False,
        )
        self._started = False
        self._closed = False

    def __enter__(self) -> SerialWriteEventReporter:
        """启动后台输出线程并返回发布器。"""

        if self._started:
            raise RuntimeError("串口时间戳输出器不能重复启动")
        self._started = True
        self._thread.start()
        return self

    def publish(self, event: SerialWriteEvent) -> None:
        """快速入队一条不可变写入事件。"""

        if not isinstance(event, SerialWriteEvent):
            raise TypeError("event 必须是 SerialWriteEvent")
        if not self._started or self._closed:
            raise RuntimeError("串口时间戳输出器未运行")
        self._queue.put_nowait(event)

    def _run(self) -> None:
        """消费事件并输出；终端故障不得影响串口控制线程。"""

        while True:
            event = self._queue.get()
            try:
                if event is None:
                    return
                try:
                    emit_serial_write_event(event)
                except Exception:
                    pass
            finally:
                self._queue.task_done()

    def close(self) -> None:
        """幂等停止线程，并等待此前事件全部打印完成。"""

        if not self._started or self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join()

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object,
    ) -> None:
        """退出上下文时刷完全部时间戳并关闭输出线程。"""

        self.close()


def run_with_cli_errors(callback: Callable[[], dict[str, object]]) -> None:
    """执行外部诊断入口，统一打印失败并映射稳定退出码。"""

    try:
        summary = callback()
    except KeyboardInterrupt as error:
        typer.echo("运动诊断已由用户中断", err=True)
        raise typer.Exit(code=130) from error
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        typer.echo(f"运动诊断失败：{error}", err=True)
        raise typer.Exit(code=1) from error
    emit_summary(summary)
