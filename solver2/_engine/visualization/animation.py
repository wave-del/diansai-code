"""可选动画预留接口。"""

from pathlib import Path


def render_animation(*_args, output: Path, **_kwargs) -> Path:
    raise NotImplementedError("GIF 动画是预留接口，不属于当前强制验收范围")

