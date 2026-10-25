"""把工程内 reference-vision 与 handover-puzzle 加入 sys.path，供本包复用。"""

from __future__ import annotations

import sys
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_DIR = _PACKAGE_DIR.parent.parent

_PATHS = (
    _REPO_DIR / "vision" / "src",
    _REPO_DIR / "handover-puzzle",
)


def ensure() -> None:
    for path in _PATHS:
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
