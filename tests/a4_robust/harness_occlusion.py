#!/usr/bin/env python3
"""遮挡合成测试：对录像帧施加角点/边遮挡，对比 原管线 vs 遮挡鲁棒管线。

测量口径：
- 最终四角 = Resize（精拟合后）输出的 source_corners_px，地面真值取干净帧
  同一管线的最终四角；
- 碎片检测成功 = Detect1 返回的 pieces 数量；
- 报告各遮挡场景的成功率与角误差（px）。
只处理本地录像/图片，不连接摄像头与任何硬件。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from . import _paths
except ImportError:
    import _paths
_paths.ensure()

import test_a4_detection as a4d

try:
    from .pipeline import detect_robust
except ImportError:
    from pipeline import detect_robust

DEFAULT_VIDEO = Path(__file__).resolve().parent.parent.parent / "example" / "outputs" / "camera" / "camera_20260731_222707.avi"


def _occlude_corner(frame: np.ndarray, corner: np.ndarray, radius: int, color: tuple[int, int, int]) -> np.ndarray:
    occluded = frame.copy()
    cv2.circle(occluded, (int(corner[0]), int(corner[1])), radius, color, -1)
    return occluded


def _occlude_edge(
    frame: np.ndarray,
    corner_a: np.ndarray,
    corner_b: np.ndarray,
    thickness: int,
    color: tuple[int, int, int],
) -> np.ndarray:
    occluded = frame.copy()
    cv2.line(
        occluded,
        (int(corner_a[0]), int(corner_a[1])),
        (int(corner_b[0]), int(corner_b[1])),
        color,
        thickness,
    )
    return occluded


def _corner_error(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    return float(np.linalg.norm(predicted - ground_truth, axis=1).max())


def _build_scenarios(frame: np.ndarray, corners: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """生成遮挡场景：(名称, 遮挡帧)。角顺序 TL,TR,BR,BL。"""
    black, white = (0, 0, 0), (235, 235, 235)
    scenarios: list[tuple[str, np.ndarray]] = [
        ("黑圆盖TL角 r=35", _occlude_corner(frame, corners[0], 35, black)),
        ("黑圆盖TL角 r=70", _occlude_corner(frame, corners[0], 70, black)),
        ("黑圆盖TR角 r=70", _occlude_corner(frame, corners[1], 70, black)),
        ("黑圆盖BR角 r=110", _occlude_corner(frame, corners[2], 110, black)),
        ("亮块盖TL角 r=60", _occlude_corner(frame, corners[0], 60, white)),
        ("黑条盖顶边 t=70", _occlude_edge(frame, corners[0], corners[1], 70, black)),
        ("黑条盖右边 t=80", _occlude_edge(frame, corners[1], corners[2], 80, black)),
        ("黑条盖底边 t=70", _occlude_edge(frame, corners[2], corners[3], 70, black)),
        ("黑条盖左边 t=60", _occlude_edge(frame, corners[3], corners[0], 60, black)),
        ("双角遮挡 TR+BL r=50", None),  # 占位，下面手动构造
    ]
    double = frame.copy()
    cv2.circle(double, tuple(corners[1].astype(int)), 50, black, -1)
    cv2.circle(double, tuple(corners[3].astype(int)), 50, black, -1)
    scenarios[-1] = ("双角遮挡 TR+BL r=50", double)
    # 手形遮挡：不规则暗色团块盖住一条边（直线拟合残差高，应被拒绝）
    hand = frame.copy()
    edge_mid = (corners[1] + corners[2]) / 2.0
    direction = (corners[2] - corners[1]) / np.linalg.norm(corners[2] - corners[1])
    normal = np.asarray([-direction[1], direction[0]])
    for offset in (-60, -30, 0, 30, 60):
        center = edge_mid + normal * offset
        radius = 45 - abs(offset) // 4
        cv2.circle(hand, tuple(center.astype(int)), max(20, radius), black, -1)
    scenarios.append(("手形遮挡右边缘", hand))
    return scenarios


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=str(DEFAULT_VIDEO), help="输入 AVI/MP4")
    parser.add_argument("--params", choices=sorted(a4d.PARAM_PRESETS), default="new", help="参数预设")
    parser.add_argument("--frames", type=int, default=8, help="均匀采样帧数")
    args = parser.parse_args(argv)

    if not Path(args.video).exists():
        print(f"⛔ 找不到录像：{args.video}", file=sys.stderr)
        return 2
    params = dict(a4d.PARAM_PRESETS[args.params])
    a4_args = argparse.Namespace(
        hue_min=int(params["hue_min"]), hue_max=int(params["hue_max"]),
        sat_min=int(params["sat_min"]), value_min=int(params["value_min"]),
        min_a4_area=0.25,
    )

    capture = cv2.VideoCapture(args.video)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(i * (total - 1) / max(1, args.frames - 1)) for i in range(args.frames)]

    # 汇总统计
    original_ok = 0
    robust_ok = 0
    corner_errors: list[float] = []
    pieces_original: list[int] = []
    pieces_robust: list[int] = []

    print(f"参数预设：{args.params}（HSV={params['hue_min']}/{params['hue_max']}/{params['sat_min']}/{params['value_min']}）")
    print("场景                                            原管线         鲁棒管线         角误差(px)")
    print("-" * 100)
    for frame_index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        # 干净帧真值
        try:
            clean = detect_robust(frame, params, a4_args)
        except Exception as exc:
            print(f"frame {frame_index}: 干净帧检测失败：{exc}")
            continue
        gt = np.asarray(clean.resize_result.source_corners_px, dtype=np.float32)
        scenarios = _build_scenarios(frame, clean.corners)
        for name, occluded in scenarios:
            # 原管线
            try:
                resize_class, resize_config, detector = a4d._build_reference_detectors(params)
                rr, dd, _, _, _ = a4d._detect(
                    occluded, resize_class, resize_config, detector,
                    argparse.Namespace(
                        hue_min=int(params["hue_min"]), hue_max=int(params["hue_max"]),
                        sat_min=int(params["sat_min"]), value_min=int(params["value_min"]),
                        min_a4_area=0.25, max_refine_shift=35.0,
                    ),
                )
                original = f"OK p={len(dd.pieces)}"
                original_ok += 1
                pieces_original.append(len(dd.pieces))
            except Exception as exc:
                original = f"FAIL {str(exc)[:24]}"
            # 鲁棒管线
            try:
                result = detect_robust(occluded, params, a4_args)
                final = np.asarray(result.resize_result.source_corners_px, dtype=np.float32)
                err = _corner_error(final, gt)
                robust = f"OK p={len(result.detect_result.pieces)}"
                robust_ok += 1
                pieces_robust.append(len(result.detect_result.pieces))
            except Exception as exc:
                robust = f"FAIL {str(exc)[:24]}"
                err = float("nan")
            corner_errors.append(err)
            print(f"frame {frame_index:3d} {name:22s}  {original:18s}  {robust:18s}  {err:6.1f}")
    capture.release()

    print("-" * 100)
    total_scenes = len(indices) * len(_build_scenarios(np.zeros((720, 1280, 3), np.uint8), np.zeros((4, 2), np.float32)))
    print(f"合计 {len(indices)} 帧 × 场景数：")
    print(f"  原管线成功  {original_ok}/{total_scenes}")
    print(f"  鲁棒管线成功 {robust_ok}/{total_scenes}")
    valid_errors = [e for e in corner_errors if np.isfinite(e)]
    if valid_errors:
        print(f"  鲁棒管线角误差：中位={np.median(valid_errors):.1f}px  最大={np.max(valid_errors):.1f}px")
    if pieces_robust:
        print(f"  鲁棒管线碎片检出：平均={np.mean(pieces_robust):.2f} 个/成功帧")
    if pieces_original:
        print(f"  原管线碎片检出：平均={np.mean(pieces_original):.2f} 个/成功帧")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
