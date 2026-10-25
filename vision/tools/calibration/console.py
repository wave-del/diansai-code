"""提供适合 1280×720 触摸屏的现场标定文件控制台。"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
import os
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox

import typer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ec_26_e.config.calibration import (
    CalibrationPaths,
    active_generation,
    default_calibration_paths,
    initialize_calibration,
    publish_draft,
    reset_draft,
    restore_previous,
    validate_draft,
)


DEFAULT_SOURCE_ENV = PROJECT_ROOT / ".env"
DEFAULT_ENV_TEMPLATE = PROJECT_ROOT / ".env.example"
DEFAULT_TASK1 = PROJECT_ROOT / "configs/tasks/task1.json"
DEFAULT_TASK2 = PROJECT_ROOT / "configs/tasks/task2.json"
VISION_SCRIPT = PROJECT_ROOT / "tests/external/cam/record_detected_video.py"
EDITOR_CANDIDATES = (
    (Path("/usr/bin/kate"), ("--new",)),
    (Path("/usr/bin/kwrite"), ("--new",)),
    (Path("/usr/bin/xdg-open"), ()),
)
EXTERNAL_GUI_ENVIRONMENT_KEYS = (
    "QT_QPA_PLATFORM_PLUGIN_PATH",
    "QT_QPA_FONTDIR",
    "QT_PLUGIN_PATH",
)


class PreviewTask(str, Enum):
    """列出桌面快捷入口允许预览的正式任务。"""

    task1 = "task1"
    task2 = "task2"


def initialize_default_workspace(paths: CalibrationPaths) -> None:
    """使用当前仓库现场值初始化持久化标定工作区。"""

    initialize_calibration(
        paths,
        source_env=DEFAULT_SOURCE_ENV,
        env_template=DEFAULT_ENV_TEMPLATE,
        source_task1=DEFAULT_TASK1,
        source_task2=DEFAULT_TASK2,
    )


def _editor_command(
    path: Path,
    candidates: tuple[tuple[Path, tuple[str, ...]], ...] = EDITOR_CANDIDATES,
) -> list[str]:
    """选择本机可用且会创建独立窗口的图形编辑器命令。"""

    for executable, arguments in candidates:
        if executable.is_file():
            return [str(executable), *arguments, str(path)]
    raise RuntimeError("未找到 Kate、KWrite 或 xdg-open")


def _external_gui_environment(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """移除 OpenCV 注入的 Qt 路径，避免系统 KDE 程序加载冲突插件。"""

    result = dict(os.environ if environ is None else environ)
    for name in EXTERNAL_GUI_ENVIRONMENT_KEYS:
        result.pop(name, None)
    return result


def preview_command(paths: CalibrationPaths, task: PreviewTask) -> list[str]:
    """构造不录像且显式绑定 active 任务 JSON 的预览命令。"""

    return [
        sys.executable,
        str(VISION_SCRIPT),
        "--env",
        str(paths.active_env),
        "--task",
        task.value,
        "--task-config",
        str(paths.active_task(task.value)),
        "--no-record",
        "--preview",
        "--fullscreen",
    ]


def run_quick_preview(paths: CalibrationPaths, task: PreviewTask) -> int:
    """初始化配置后在前台运行指定任务的快速视觉预览。"""

    initialize_default_workspace(paths)
    completed = subprocess.run(
        preview_command(paths, task),
        cwd=PROJECT_ROOT,
        env=dict(os.environ),
        check=False,
    )
    return completed.returncode


class CalibrationConsole:
    """组织文本编辑、校验发布、恢复和视觉预览操作。"""

    def __init__(self, root: tk.Tk, paths: CalibrationPaths) -> None:
        """构建大按钮现场标定控制台。"""

        self.root = root
        self.paths = paths
        self.preview_process: subprocess.Popen[bytes] | None = None
        self.editor_process: subprocess.Popen[bytes] | None = None
        self.status = tk.StringVar()
        self._configure_window()
        self._build_layout()
        self.refresh_status("现场标定工作区已就绪。")

    def _configure_window(self) -> None:
        """配置适合当前 1280×720 显示器的全屏窗口。"""

        self.root.title("EC26E 现场标定")
        self.root.configure(background="#16202a")
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", lambda _event: self.root.destroy())

    def _button(
        self,
        parent: tk.Widget,
        text: str,
        command: object,
        *,
        background: str = "#2f6f9f",
    ) -> tk.Button:
        """创建统一尺寸和配色的触摸按钮。"""

        return tk.Button(
            parent,
            text=text,
            command=command,
            font=("Noto Sans CJK SC", 18, "bold"),
            foreground="white",
            background=background,
            activebackground="#438bc0",
            activeforeground="white",
            relief=tk.FLAT,
            borderwidth=0,
            padx=16,
            pady=16,
        )

    def _build_layout(self) -> None:
        """构建标题、编辑、发布、预览和状态区域。"""

        title = tk.Label(
            self.root,
            text="EC26E 现场标定",
            font=("Noto Sans CJK SC", 28, "bold"),
            foreground="white",
            background="#16202a",
            pady=14,
        )
        title.pack(fill=tk.X)

        body = tk.Frame(self.root, background="#16202a")
        body.pack(fill=tk.BOTH, expand=True, padx=26, pady=6)
        for column in range(3):
            body.grid_columnconfigure(column, weight=1, uniform="buttons")
        for row in range(3):
            body.grid_rowconfigure(row, weight=1)

        buttons = (
            ("编辑机器 .env", self.open_environment, "#2f6f9f"),
            ("编辑 Task1 JSON", self.open_task1, "#2f6f9f"),
            ("编辑 Task2 JSON", self.open_task2, "#2f6f9f"),
            ("检查草稿", self.validate, "#526d82"),
            ("检查并应用", self.publish, "#23855b"),
            ("放弃未应用修改", self.discard, "#8b6535"),
            ("预览 Task1（不录像）", self.preview_task1, "#6d4ba5"),
            ("预览 Task2（不录像）", self.preview_task2, "#6d4ba5"),
            ("恢复上一版", self.restore, "#a34b4b"),
        )
        for index, (text, command, color) in enumerate(buttons):
            button = self._button(body, text, command, background=color)
            button.grid(
                row=index // 3,
                column=index % 3,
                sticky="nsew",
                padx=8,
                pady=8,
            )

        footer = tk.Frame(self.root, background="#0f171f")
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        status_label = tk.Label(
            footer,
            textvariable=self.status,
            justify=tk.LEFT,
            anchor="w",
            font=("Noto Sans CJK SC", 13),
            foreground="#d6e4ef",
            background="#0f171f",
            padx=18,
            pady=12,
        )
        status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        exit_button = self._button(
            footer,
            "退出",
            self.root.destroy,
            background="#5c6570",
        )
        exit_button.configure(font=("Noto Sans CJK SC", 14, "bold"), pady=8)
        exit_button.pack(side=tk.RIGHT, padx=12, pady=8)

    def refresh_status(self, message: str) -> None:
        """显示操作结果和当前 active 版本。"""

        try:
            generation = active_generation(self.paths).name
        except (OSError, RuntimeError):
            generation = "不可用"
        self.status.set(
            f"{message}\n当前版本：{generation}    草稿：{self.paths.editing_directory}"
        )

    def _open(self, path: Path) -> None:
        """隐藏全屏控制台，并用前台独立窗口打开指定草稿文件。"""

        if self.editor_process is not None and self.editor_process.poll() is None:
            messagebox.showwarning("编辑器已打开", "请先关闭当前编辑窗口。")
            return
        if not path.is_file():
            messagebox.showerror("打开失败", f"草稿文件不存在：\n{path}")
            return
        self.root.withdraw()
        try:
            self.editor_process = subprocess.Popen(
                _editor_command(path),
                cwd=PROJECT_ROOT,
                env=_external_gui_environment(),
            )
        except (OSError, RuntimeError) as error:
            self._restore_console()
            messagebox.showerror("打开失败", str(error))
            return
        self.refresh_status(f"正在编辑：{path}")
        self.root.after(250, self._poll_editor)

    def _poll_editor(self) -> None:
        """轮询独立编辑器，关闭后自动恢复全屏控制台。"""

        if self.editor_process is not None and self.editor_process.poll() is None:
            self.root.after(250, self._poll_editor)
            return
        return_code = (
            self.editor_process.returncode if self.editor_process is not None else None
        )
        self.editor_process = None
        self._restore_console()
        if return_code not in (None, 0):
            messagebox.showerror("编辑器异常退出", f"编辑器退出码：{return_code}")
            return
        self.refresh_status("编辑器已关闭；保存后可检查并应用配置。")

    def _restore_console(self) -> None:
        """恢复全屏控制台并主动获取输入焦点。"""

        self.root.deiconify()
        self.root.attributes("-fullscreen", True)
        self.root.lift()
        self.root.focus_force()

    def open_environment(self) -> None:
        """打开包含全部机器现场参数的环境草稿。"""

        self._open(self.paths.editing_env)

    def open_task1(self) -> None:
        """打开 Task1 完整 JSON 草稿。"""

        self._open(self.paths.editing_task1)

    def open_task2(self) -> None:
        """打开 Task2 完整 JSON 草稿。"""

        self._open(self.paths.editing_task2)

    def validate(self) -> None:
        """校验草稿中的环境和两个任务配置。"""

        try:
            result = validate_draft(self.paths)
        except (OSError, TypeError, ValueError) as error:
            self.refresh_status(f"草稿校验失败：{error}")
            messagebox.showerror("草稿校验失败", str(error))
            return
        self.refresh_status(f"草稿校验通过；当前选择 {result.selected_task}。")
        messagebox.showinfo("校验通过", "machine.env、Task1、Task2 全部有效。")

    def publish(self) -> None:
        """确认后校验并原子发布整套草稿。"""

        if not messagebox.askyesno(
            "应用配置",
            "确认已经在编辑器中保存？\n将校验并原子应用整套配置。",
        ):
            return
        try:
            result = publish_draft(self.paths)
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            self.refresh_status(f"应用失败，原配置保持不变：{error}")
            messagebox.showerror("应用失败", str(error))
            return
        self.refresh_status(f"配置已应用；当前选择 {result.selected_task}。")
        messagebox.showinfo("应用成功", "新配置已生效，旧版本已保留。")

    def discard(self) -> None:
        """确认后丢弃草稿修改并恢复当前生效内容。"""

        if not messagebox.askyesno(
            "放弃修改",
            "将丢弃尚未应用的草稿修改，是否继续？",
        ):
            return
        try:
            reset_draft(self.paths)
        except OSError as error:
            messagebox.showerror("恢复草稿失败", str(error))
            return
        self.refresh_status("草稿已恢复为当前生效配置。")

    def restore(self) -> None:
        """确认后恢复最近一个非当前的已验证版本。"""

        if not messagebox.askyesno(
            "恢复上一版",
            "将恢复上一版已验证配置，并覆盖当前草稿，是否继续？",
        ):
            return
        try:
            target = restore_previous(self.paths)
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            messagebox.showerror("恢复失败", str(error))
            return
        self.refresh_status(f"已恢复版本：{target.name}")
        messagebox.showinfo("恢复成功", f"当前版本：{target.name}")

    def _preview(self, task: PreviewTask) -> None:
        """启动一个非录像预览，并拒绝同一控制台重复启动。"""

        if self.preview_process is not None:
            if self.preview_process.poll() is None:
                messagebox.showwarning("预览已运行", "请先在预览窗口按 Esc 退出。")
                return
            self.preview_process = None
        try:
            self.preview_process = subprocess.Popen(
                preview_command(self.paths, task),
                cwd=PROJECT_ROOT,
                env=dict(os.environ),
            )
        except OSError as error:
            messagebox.showerror("预览启动失败", str(error))
            return
        self.refresh_status(f"已启动 {task.value} 快速预览；Esc 返回。")

    def preview_task1(self) -> None:
        """启动使用 active Task1 配置的快速预览。"""

        self._preview(PreviewTask.task1)

    def preview_task2(self) -> None:
        """启动使用 active Task2 配置的快速预览。"""

        self._preview(PreviewTask.task2)


app = typer.Typer(add_completion=False, help="启动 EC26E 现场标定控制台。")


@app.command()
def main(
    data_root: Path = typer.Option(
        default_calibration_paths().root,
        "--data-root",
        help="现场标定草稿、发布版本和 active 链接的持久化目录。",
    ),
    preview_task: PreviewTask | None = typer.Option(
        None,
        "--preview-task",
        help="不打开控制台，直接启动指定任务的不录像视觉预览。",
    ),
) -> None:
    """初始化标定数据，并打开控制台或直接运行快速预览。"""

    paths = CalibrationPaths(data_root)
    try:
        initialize_default_workspace(paths)
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        typer.echo(f"现场标定初始化失败：{error}", err=True)
        raise typer.Exit(code=2) from error
    if preview_task is not None:
        raise typer.Exit(code=run_quick_preview(paths, preview_task))
    root = tk.Tk()
    CalibrationConsole(root, paths)
    root.mainloop()


if __name__ == "__main__":
    app()
