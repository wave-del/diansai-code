"""常驻维护实体按键状态，并串行分派带两分钟重试边界的任务。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from ec_26_e.buttons import (
    ButtonDebouncer,
    ButtonServiceConfig,
    PeripheryButtonSource,
)
from ec_26_e.tasks.execution_lock import TaskBusyError, TaskExecutionLock


RETRYABLE_TASK_EXIT_CODE = 75
_CHILD_ENVIRONMENT_ALLOWLIST = frozenset(
    {
        "DBUS_SESSION_BUS_ADDRESS",
        "DISPLAY",
        "HOME",
        "LANG",
        "LC_ALL",
        "PATH",
        "PYTHONIOENCODING",
        "PYTHONUNBUFFERED",
        "TZ",
        "USER",
        "XAUTHORITY",
        "XDG_RUNTIME_DIR",
    }
)


def _utc_timestamp() -> str:
    """返回便于日志和状态文件比较的UTC时间。"""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _notify_systemd(message: str) -> None:
    """在systemd通知套接字存在时发送READY、WATCHDOG或STATUS。"""

    address = os.environ.get("NOTIFY_SOCKET")
    if not address:
        return
    if address.startswith("@"):
        address = "\0" + address[1:]
    notifier = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        try:
            notifier.connect(address)
            notifier.sendall(message.encode("utf-8"))
        except OSError:
            # 通知失败不能影响GPIO监听和正在运行的任务。
            return
    finally:
        notifier.close()


class ButtonDispatcher:
    """实现消抖、冲突拒绝、任务重试、超时和安全子进程清理。"""

    def __init__(
        self,
        config: ButtonServiceConfig,
        *,
        source: PeripheryButtonSource | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    ) -> None:
        """保存可替换依赖并初始化常驻状态机。"""

        if not isinstance(config, ButtonServiceConfig):
            raise TypeError("config 必须是 ButtonServiceConfig")
        self.config = config
        self.source = source or PeripheryButtonSource(config.bindings)
        self._monotonic = monotonic
        self._sleep = sleep
        self._popen = popen
        self._trackers = {
            binding.task_name: ButtonDebouncer(config.debounce_seconds)
            for binding in config.bindings
        }
        self._stop_requested = False
        self._armed = False
        self._release_since: float | None = None
        self._pending_tasks: set[str] = set()
        self._selection_deadline: float | None = None
        self._active_task: str | None = None
        self._active_generation: Path | None = None
        self._task_deadline: float | None = None
        self._next_retry_at: float | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._terminating = False
        self._cleanup_deadline: float | None = None
        self._attempts = 0
        self._execution_lock: TaskExecutionLock | None = None
        self._state = "starting"
        self._message = "服务正在初始化"
        self._last_watchdog = 0.0

    def request_stop(self, _signum: int | None = None, _frame: object = None) -> None:
        """记录服务停止请求；实际清理仍由主循环串行执行。"""

        self._stop_requested = True

    def _button_states(self) -> dict[str, str]:
        """返回适合状态文件展示的消抖后按键状态。"""

        return {
            task_name: (
                "unknown"
                if tracker.stable_pressed is None
                else "pressed" if tracker.stable_pressed else "released"
            )
            for task_name, tracker in self._trackers.items()
        }

    def _write_status(self) -> None:
        """原子写入当前按键、任务和重试状态。"""

        payload = {
            "updated_at": _utc_timestamp(),
            "state": self._state,
            "message": self._message,
            "armed": self._armed,
            "buttons": self._button_states(),
            "active_task": self._active_task,
            "active_generation": (
                None
                if self._active_generation is None
                else str(self._active_generation)
            ),
            "attempts": self._attempts,
            "child_pid": None if self._process is None else self._process.pid,
        }
        self.config.status_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.config.status_path.with_name(
            f".{self.config.status_path.name}.tmp-{os.getpid()}"
        )
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.config.status_path)

    def _set_status(self, state: str, message: str) -> None:
        """更新状态、写入原子快照并同步到journald。"""

        changed = state != self._state or message != self._message
        self._state = state
        self._message = message
        if changed:
            self._write_status()
            print(f"[{_utc_timestamp()}] {state}: {message}", flush=True)
            _notify_systemd(f"STATUS={state}: {message}")

    def _watchdog(self, now: float) -> None:
        """按固定周期报告主循环仍在运行。"""

        if now - self._last_watchdog >= 5.0:
            _notify_systemd("WATCHDOG=1")
            self._last_watchdog = now

    def _all_released(self) -> bool:
        """判断全部按键是否已形成稳定松开状态。"""

        return all(
            tracker.stable_pressed is False
            for tracker in self._trackers.values()
        )

    def _update_rearm(self, now: float) -> None:
        """要求全部按键稳定松开一段时间后才重新受理。"""

        if self._armed or self._active_task is not None or self._pending_tasks:
            self._release_since = None
            return
        if not self._all_released():
            self._release_since = None
            self._set_status("waiting_release", "等待全部按键稳定松开")
            return
        if self._release_since is None:
            self._release_since = now
            return
        if now - self._release_since >= self.config.release_guard_seconds:
            self._armed = True
            self._release_since = None
            self._set_status("idle", "按键已就绪，等待任务请求")

    def _consume_button_samples(
        self,
        raw_states: Mapping[str, bool],
        now: float,
    ) -> None:
        """更新全部消抖器，并在空闲时收集冲突窗口内的按压。"""

        stable_state_changed = False
        for task_name, tracker in self._trackers.items():
            changed = tracker.update(raw_states[task_name], now)
            stable_state_changed = stable_state_changed or changed
            if not changed or tracker.stable_pressed is not True:
                continue
            if self._active_task is not None:
                self._set_status(
                    "running",
                    f"{self._active_task}运行中，忽略{task_name}按键",
                )
                continue
            if self._armed or self._pending_tasks:
                self._pending_tasks.add(task_name)
                self._armed = False
                if self._selection_deadline is None:
                    self._selection_deadline = (
                        now + self.config.conflict_window_seconds
                    )
                self._set_status("selecting", "正在确认按键任务")
        if stable_state_changed:
            self._write_status()

    def _resolve_generation(self, task_name: str) -> tuple[Path, Path, Path]:
        """冻结active符号链接，并返回同一版本的env与任务JSON。"""

        generation = self.config.active_link.resolve(strict=True)
        env_path = generation / "machine.env"
        task_path = generation / "tasks" / f"{task_name}.json"
        if not env_path.is_file():
            raise FileNotFoundError(f"机器配置不存在：{env_path}")
        if not task_path.is_file():
            raise FileNotFoundError(f"任务配置不存在：{task_path}")
        return generation, env_path, task_path

    @staticmethod
    def _child_environment() -> dict[str, str]:
        """只保留图形和基础进程变量，避免外部环境覆盖machine.env。"""

        environment = {
            name: value
            for name, value in os.environ.items()
            if name in _CHILD_ENVIRONMENT_ALLOWLIST
        }
        environment.setdefault("PYTHONUNBUFFERED", "1")
        return environment

    def _spawn_attempt(self) -> None:
        """使用冻结配置启动一次独立任务子进程。"""

        if (
            self._active_task is None
            or self._active_generation is None
            or self._task_deadline is None
        ):
            raise RuntimeError("任务重试上下文尚未初始化")
        env_path = self._active_generation / "machine.env"
        task_path = (
            self._active_generation
            / "tasks"
            / f"{self._active_task}.json"
        )
        command = [
            sys.executable,
            "-m",
            "ec_26_e",
            "--env",
            str(env_path),
            "--task",
            self._active_task,
            "--task-config",
            str(task_path),
            "--execution-lock-held",
        ]
        self._attempts += 1
        self._process = self._popen(
            command,
            cwd=Path(__file__).resolve().parents[3],
            env=self._child_environment(),
            start_new_session=True,
        )
        self._next_retry_at = None
        self._set_status(
            "running",
            f"执行{self._active_task}，第{self._attempts}次尝试",
        )

    def _start_selected_task(self, task_name: str, now: float) -> None:
        """申请全局锁、冻结配置并开始两分钟尝试窗口。"""

        task_lock = TaskExecutionLock(self.config.lock_path)
        try:
            task_lock.acquire()
        except TaskBusyError:
            self._set_status("busy_rejected", f"硬件忙，拒绝{task_name}")
            return
        try:
            generation, _env_path, _task_path = self._resolve_generation(task_name)
        except (OSError, RuntimeError) as error:
            task_lock.release()
            self._set_status("config_error", f"拒绝{task_name}：{error}")
            return
        self._execution_lock = task_lock
        self._active_task = task_name
        self._active_generation = generation
        self._task_deadline = now + self.config.retry_window_seconds
        self._attempts = 0
        try:
            self._spawn_attempt()
        except OSError as error:
            self._next_retry_at = now + self.config.retry_delay_seconds
            self._set_status("retry_wait", f"启动失败，准备重试：{error}")

    def _finish_active(self, state: str, message: str) -> None:
        """释放本次任务锁和重试上下文，随后等待按键松开。"""

        if self._execution_lock is not None:
            self._execution_lock.release()
        self._execution_lock = None
        self._process = None
        self._active_task = None
        self._active_generation = None
        self._task_deadline = None
        self._next_retry_at = None
        self._terminating = False
        self._cleanup_deadline = None
        self._attempts = 0
        self._armed = False
        self._release_since = None
        self._set_status(state, message)

    def _request_child_stop(self, now: float, reason: str) -> None:
        """向子进程组发送SIGINT并给现有安全清理逻辑留出时间。"""

        if self._process is None or self._terminating:
            return
        try:
            os.killpg(self._process.pid, signal.SIGINT)
        except ProcessLookupError:
            return
        self._terminating = True
        self._cleanup_deadline = now + self.config.cleanup_timeout_seconds
        self._set_status("stopping", reason)

    def _service_active_task(self, now: float) -> None:
        """非阻塞推进子进程退出、重试、总超时和强制兜底。"""

        if self._active_task is None or self._task_deadline is None:
            return
        if self._process is None:
            if self._stop_requested:
                self._finish_active("stopped", "服务停止，任务未再重试")
                return
            if now >= self._task_deadline:
                self._finish_active("timeout", "两分钟任务窗口结束")
                return
            if self._next_retry_at is not None and now >= self._next_retry_at:
                try:
                    self._spawn_attempt()
                except OSError as error:
                    self._next_retry_at = now + self.config.retry_delay_seconds
                    self._set_status("retry_wait", f"启动失败，准备重试：{error}")
            return

        return_code = self._process.poll()
        if return_code is None:
            if (
                self._terminating
                and self._cleanup_deadline is not None
                and now >= self._cleanup_deadline
            ):
                try:
                    os.killpg(self._process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self._set_status("forced_stop", "安全清理超时，已强制结束进程")
            elif self._stop_requested:
                self._request_child_stop(now, "服务停止，正在安全结束任务")
            elif now >= self._task_deadline:
                self._request_child_stop(now, "两分钟到期，正在安全结束任务")
            return

        self._process = None
        if self._terminating:
            self._finish_active("timeout", "任务已停止并释放资源")
        elif return_code == 0:
            self._finish_active("completed", "任务完成并释放资源")
        elif (
            return_code == RETRYABLE_TASK_EXIT_CODE
            and now < self._task_deadline
            and not self._stop_requested
        ):
            self._next_retry_at = now + self.config.retry_delay_seconds
            self._set_status(
                "retry_wait",
                f"任务可重试失败，{self.config.retry_delay_seconds:.1f}秒后重试",
            )
        else:
            self._finish_active(
                "failed",
                f"任务不可重试退出，退出码={return_code}",
            )

    def _service_selection(self, now: float) -> None:
        """在冲突窗口结束后拒绝多键或启动唯一任务。"""

        if self._selection_deadline is None or now < self._selection_deadline:
            return
        selected = sorted(self._pending_tasks)
        self._pending_tasks.clear()
        self._selection_deadline = None
        if len(selected) != 1:
            self._set_status("conflict_rejected", "检测到多键冲突，未执行任务")
            return
        self._start_selected_task(selected[0], now)

    def run(self) -> None:
        """持续监听按键，直到收到服务停止信号并完成安全清理。"""

        old_sigterm = signal.signal(signal.SIGTERM, self.request_stop)
        old_sigint = signal.signal(signal.SIGINT, self.request_stop)
        source_open = False
        try:
            self.source.open()
            source_open = True
            self._set_status("waiting_release", "GPIO已打开，等待全部按键松开")
            _notify_systemd("READY=1")
            while True:
                now = self._monotonic()
                self._watchdog(now)
                raw_states = self.source.read_pressed()
                self._consume_button_samples(raw_states, now)
                self._service_selection(now)
                self._service_active_task(now)
                self._update_rearm(now)
                if self._stop_requested and self._active_task is None:
                    break
                self._sleep(self.config.poll_interval_seconds)
        finally:
            if self._process is not None:
                now = self._monotonic()
                self._request_child_stop(now, "服务退出，正在安全结束任务")
                deadline = now + self.config.cleanup_timeout_seconds
                while self._process.poll() is None and self._monotonic() < deadline:
                    self._sleep(0.05)
                if self._process.poll() is None:
                    try:
                        os.killpg(self._process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self._process.wait(timeout=2.0)
            if self._execution_lock is not None:
                self._execution_lock.release()
                self._execution_lock = None
            if source_open:
                self.source.close()
            self._set_status("stopped", "按键服务已停止并释放GPIO")
            signal.signal(signal.SIGTERM, old_sigterm)
            signal.signal(signal.SIGINT, old_sigint)


def run_button_dispatcher(config: ButtonServiceConfig) -> None:
    """使用真实GPIO和系统进程启动常驻按键调度器。"""

    ButtonDispatcher(config).run()
