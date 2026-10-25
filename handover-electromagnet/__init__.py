"""PL6 / Pin 13 电磁铁独立交付接口。"""

from .electromagnet import (
    __version__,
    ElectromagnetConnectionError,
    ElectromagnetController,
    ElectromagnetError,
    ElectromagnetStateError,
)

__all__ = [
    "__version__",
    "ElectromagnetConnectionError",
    "ElectromagnetController",
    "ElectromagnetError",
    "ElectromagnetStateError",
]
