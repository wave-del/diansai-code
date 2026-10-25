"""公开蓝色 A4 工作区定位和透视校正接口。"""

from ec_26_e.vision.resize.contracts import ResizeConfig, ResizeResult
from ec_26_e.vision.resize.resizer import Resize

__all__ = ["Resize", "ResizeConfig", "ResizeResult"]
