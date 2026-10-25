#!/usr/bin/env python3
"""启动同事 ec_26_e 的实时检测显示：原始画面 + A4 工作区 + 拼图。

- 直接调用同事 record_detection_video()，ResizeConfig 显式传
  本机位参考角（从现场录像标定），不依赖同事的固定机位标定；
- Q6A 由推理器显式加载项目 model-lib/q6a，不修改环境或创建旧模型链接。

用法：
  .venv/bin/python tests/a4_robust/run_colleague_live.py [--no-save] [--max-frames N]
按 Esc（窗口焦点）或 Ctrl+C 结束。
"""

from __future__ import annotations

import sys
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_DIR = _PACKAGE_DIR.parent.parent
MODEL_LIB = _REPO_DIR / "model-lib"
# 视觉包（原独立仓库 ec_26_e）已并入本仓库的 vision/ 目录
COLLEAGUE_REPO = _REPO_DIR / "vision"

# Q6A 默认模型路径由 VisionConfig 统一解析，不创建 A733 兼容 symlink。

for candidate in (COLLEAGUE_REPO / "src", COLLEAGUE_REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

# 本机位参考角（由现场录像 camera_20260731_222707.avi 标定）
OUR_REFERENCE_CORNERS = (
    (0.2551, 0.1031),
    (0.8419, 0.0669),
    (0.8466, 0.7911),
    (0.2786, 0.8134),
)

import json  # noqa: E402

from ec_26_e.cam import CameraConfig  # noqa: E402
from ec_26_e.draw import OpenCVPreview  # noqa: E402
from ec_26_e.vision import Detect1Config, ResizeConfig  # noqa: E402
from tests.external.cam.record_detected_video import (  # noqa: E402
    record_detection_video,
)
from tests.external.cam.helpers import query_camera_info  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", type=Path, default=None)
    parser.add_argument("--preview/--no-preview", dest="preview", default=True)
    parser.add_argument("--fullscreen/--windowed", dest="fullscreen", default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--device", type=Path, default=Path("/dev/video0"))
    args = parser.parse_args(argv)

    # 显式构造 ResizeConfig：参考角用本机位标定值（不依赖同事固定机位标定）
    resize_config = ResizeConfig(
        detect_width=640,
        detect_height=360,
        target_width=764,
        target_height=540,
        min_workspace_ratio=0.25,
        reference_corners_relative=OUR_REFERENCE_CORNERS,
        search_band_ratio=0.14,
    )
    camera_config = CameraConfig(
        device_path=args.device,
        width=1280,
        height=720,
        fps=30,
        input_format="mjpeg",
    )
    camera_info = query_camera_info(args.device)
    preview = (
        OpenCVPreview(
            display_name=":0",
            fullscreen=args.fullscreen,
            window_name="EC 26 E - Detection Preview",
            window_size=None if args.fullscreen else (1280, 720),
        )
        if args.preview
        else None
    )
    print(f"参考角（本机位标定）：{OUR_REFERENCE_CORNERS}")
    summary = record_detection_video(
        camera_config,
        resize_config,
        Detect1Config(),
        camera_info,
        args.output_prefix,
        overwrite=args.overwrite,
        preview=preview,
    )
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
