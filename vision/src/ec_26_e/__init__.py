"""Vision/task package; geometry is owned by the sibling solver2 package."""
from pathlib import Path
import sys

# Source checkouts may be launched from vision/ with only vision/src on the
# path. Resolve the shared package relative to this checkout, never a machine
# specific directory, and leave installed solver2 packages to normal imports.
_project_root = Path(__file__).resolve().parents[3]
if (_project_root / "solver2" / "__init__.py").is_file():
    if str(_project_root) in sys.path:
        sys.path.remove(str(_project_root))
    sys.path.insert(0, str(_project_root))
"""提供 EC 26 E 项目的可复用能力模块。"""
