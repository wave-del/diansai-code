#!/usr/bin/env python3
"""example 入口：启动带全程实时预览的视觉质心抓取测试。"""
from __future__ import annotations

import os
from pathlib import Path
import sys


if __name__ == "__main__":
    source = Path(__file__).resolve().parents[1] / "handover-puzzle" / "test_vision_centroid_pick_rotate_place.py"
    # exec 替换当前进程，确保视觉脚本使用 multiprocessing spawn 时，子进程
    # 可正确重新导入真实脚本模块，而不是这个薄入口文件。
    os.execv(sys.executable, [sys.executable, str(source), *sys.argv[1:]])
