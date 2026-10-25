"""使用python-periphery申请带内部上拉的多gpiochip按键输入。"""

from __future__ import annotations

from collections.abc import Mapping

from ec_26_e.buttons.contracts import ButtonBinding


class PeripheryButtonSource:
    """持有多个GPIO输入句柄并返回按下语义的电平快照。"""

    def __init__(self, bindings: tuple[ButtonBinding, ...]) -> None:
        """保存按键绑定；构造阶段不申请GPIO。"""

        if not isinstance(bindings, tuple) or not bindings:
            raise ValueError("bindings 必须是非空元组")
        self.bindings = bindings
        self._handles: dict[str, object] = {}

    def open(self) -> None:
        """以输入、内部上拉和低电平按下方式申请全部GPIO。"""

        if self._handles:
            raise RuntimeError("按键GPIO已经打开")
        try:
            from periphery import GPIO
        except ImportError as error:
            raise RuntimeError("未安装 python-periphery") from error
        opened: dict[str, object] = {}
        try:
            for binding in self.bindings:
                opened[binding.task_name] = GPIO(
                    str(binding.chip_path),
                    binding.line_offset,
                    "in",
                    bias="pull_up",
                )
        except Exception as error:
            for handle in opened.values():
                handle.close()
            raise RuntimeError(f"申请按键GPIO失败：{error}") from error
        self._handles = opened

    def read_pressed(self) -> Mapping[str, bool]:
        """读取全部按键，并把低电平转换为pressed=True。"""

        if not self._handles:
            raise RuntimeError("按键GPIO尚未打开")
        try:
            return {
                task_name: not bool(handle.read())
                for task_name, handle in self._handles.items()
            }
        except Exception as error:
            raise RuntimeError(f"读取按键GPIO失败：{error}") from error

    def close(self) -> None:
        """释放所有GPIO；重复关闭安全。"""

        handles = self._handles
        self._handles = {}
        errors: list[str] = []
        for task_name, handle in handles.items():
            try:
                handle.close()
            except Exception as error:
                errors.append(f"{task_name}：{error}")
        if errors:
            raise RuntimeError("释放按键GPIO失败：" + "；".join(errors))
