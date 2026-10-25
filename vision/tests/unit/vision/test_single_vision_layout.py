"""验证仓库只保留统一 vision，且不存在旧检测器的源码依赖。

该文件只读取仓库 Python 源码，不依赖 NPU 或摄像头，可由默认单元测试运行。
"""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _python_sources() -> tuple[Path, ...]:
    """返回生产源码和测试中的全部 Python 文件。"""

    return tuple((PROJECT_ROOT / "src").rglob("*.py")) + tuple(
        (PROJECT_ROOT / "tests").rglob("*.py")
    )


def test_numbered_and_legacy_vision_directories_are_absent() -> None:
    """多代视觉实现目录必须从文件系统物理删除。"""

    assert not (PROJECT_ROOT / "src/ec_26_e/vision2").exists()
    assert not (PROJECT_ROOT / "src/ec_26_e/vision/detect1").exists()
    assert not (PROJECT_ROOT / "tests/unit/vision2").exists()
    assert not (PROJECT_ROOT / "tests/external/vision2").exists()


def test_source_tree_has_no_legacy_import_or_detector_call() -> None:
    """源码和测试不得重新引入旧包、旧检测器或旧工厂。"""

    forbidden_modules = (
        "ec_26_e." + "vision2",
        "ec_26_e.vision." + "detect1",
    )
    forbidden_names = {
        "Detect" + "1",
        "StableDetect" + "1",
        "create_sequence_" + "detector",
    }
    failures: list[str] = []
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules = (node.module or "",)
            else:
                modules = ()
            if any(
                module == forbidden
                or module.startswith(forbidden + ".")
                for module in modules
                for forbidden in forbidden_modules
            ):
                failures.append(f"{path}:{node.lineno} 旧模块导入")
            if isinstance(node, ast.Name) and node.id in forbidden_names:
                failures.append(
                    f"{path}:{node.lineno} 旧符号 {node.id}"
                )
    assert failures == []
