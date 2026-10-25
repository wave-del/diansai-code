"""组织 EC 26 E 项目的分层测试与模块调试脚本。"""
from pathlib import Path
import sys

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
