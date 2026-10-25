"""
验证现场标定控制台的编辑器命令构造。

该文件不依赖相机、显示器或其他外部环境，可通过 pytest 独立运行。
"""

from pathlib import Path

import pytest

from tools.calibration.console import _editor_command, _external_gui_environment


def test_editor_command_forces_independent_kate_window(tmp_path: Path) -> None:
    """验证 Kate 使用独立窗口参数，避免启动请求静默退出。"""

    executable = tmp_path / "kate"
    executable.touch()
    target = tmp_path / "machine.env"

    command = _editor_command(target, ((executable, ("--new",)),))

    assert command == [str(executable), "--new", str(target)]


def test_editor_command_falls_back_to_next_available_editor(tmp_path: Path) -> None:
    """验证首选编辑器不存在时会使用后备编辑器。"""

    missing = tmp_path / "missing-kate"
    fallback = tmp_path / "xdg-open"
    fallback.touch()
    target = tmp_path / "task1.json"

    command = _editor_command(
        target,
        ((missing, ("--new",)), (fallback, ())),
    )

    assert command == [str(fallback), str(target)]


def test_editor_command_rejects_missing_editors(tmp_path: Path) -> None:
    """验证没有可用图形编辑器时返回明确错误。"""

    with pytest.raises(RuntimeError, match="未找到 Kate"):
        _editor_command(tmp_path / "machine.env", ())


def test_external_gui_environment_removes_opencv_qt_paths() -> None:
    """验证系统编辑器不会继承 OpenCV 私有 Qt 插件目录。"""

    source = {
        "DISPLAY": ":0",
        "QT_QPA_PLATFORM_PLUGIN_PATH": "/venv/site-packages/cv2/qt/plugins",
        "QT_QPA_FONTDIR": "/venv/site-packages/cv2/qt/fonts",
        "QT_PLUGIN_PATH": "/venv/site-packages/cv2/qt/plugins",
        "EC_TASK": "task1",
    }

    result = _external_gui_environment(source)

    assert result == {"DISPLAY": ":0", "EC_TASK": "task1"}
    assert source["QT_QPA_PLATFORM_PLUGIN_PATH"].endswith("cv2/qt/plugins")
