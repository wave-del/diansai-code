"""在冻结反光正负样本上扫描 YOLO score 阈值并选择生产默认值。

独立运行：
``poetry run python tests/external/vision/sweep_thresholds.py --help``
"""

from __future__ import annotations

from pathlib import Path
import json

import cv2
import typer

from ec_26_e.vision import NpuYolo11SegInferencer, Resize, VisionConfig

app = typer.Typer(add_completion=False, help="扫描 score 阈值并以反光负样本零误报优先选值。")


def _parse_thresholds(value: str) -> list[float]:
    """解析逗号分隔阈值并保持升序去重。"""

    try:
        thresholds = sorted({float(item.strip()) for item in value.split(",")})
    except ValueError as error:
        raise typer.BadParameter("thresholds 必须是逗号分隔实数") from error
    if not thresholds or any(not 0.0 <= item <= 1.0 for item in thresholds):
        raise typer.BadParameter("thresholds 必须位于 [0, 1]")
    return thresholds


@app.command()
def run(
    video: Path = typer.Option(
        Path("outputs/cam/detection_20260730_180643_595024_raw.mp4"),
        "--video",
        exists=True,
        dir_okay=False,
    ),
    reference_json: Path = typer.Option(
        Path("tests/external/vision/180643_pseudo_truth.json"),
        "--reference-json",
        exists=True,
        dir_okay=False,
    ),
    thresholds: str = typer.Option("0.05,0.10,0.15,0.20,0.25,0.35,0.50"),
    output: Path = typer.Option(Path("outputs/vision/threshold_sweep.json")),
) -> None:
    """逐阈值运行冻结帧，并报告 precision/recall/亮斑误报率。"""

    threshold_values = _parse_thresholds(thresholds)
    payload = json.loads(reference_json.read_text(encoding="utf-8"))
    references = [item for item in payload.get("frames", []) if "instance_count" in item or item.get("is_glare_negative")]
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise typer.BadParameter(f"无法打开视频：{video}")
    resizer = Resize()
    images = []
    for reference in references:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(reference["frame_index"]))
        ok, frame = capture.read()
        if not ok:
            continue
        try:
            images.append((reference, resizer.resize(frame).image_bgr))
        except ValueError:
            continue
    capture.release()

    rows = []
    for threshold in threshold_values:
        true_positive = false_positive = false_negative = negative_fp = negatives = 0
        inferencer = NpuYolo11SegInferencer(
            VisionConfig(score_threshold=threshold)
        )
        try:
            for reference, image in images:
                count = len(inferencer.infer(image))
                expected = int(reference.get("instance_count", 0))
                if bool(reference.get("is_glare_negative", False)):
                    negatives += 1
                    negative_fp += int(count > 0)
                else:
                    true_positive += min(count, expected)
                    false_positive += max(0, count - expected)
                    false_negative += max(0, expected - count)
        finally:
            inferencer.close()
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        rows.append({
            "score_threshold": threshold,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(2.0 * precision * recall / max(precision + recall, 1e-9), 4),
            "glare_false_positive_frames": negative_fp,
            "glare_negative_frames": negatives,
            "glare_false_positive_rate": round(negative_fp / max(negatives, 1), 4),
        })
    zero_false_positive = [row for row in rows if row["glare_false_positive_frames"] == 0]
    if zero_false_positive:
        best_recall = max(row["recall"] for row in zero_false_positive)
        selected = min(
            (row for row in zero_false_positive if row["recall"] == best_recall),
            key=lambda row: row["score_threshold"],
        )
        selection_reason = "反光负样本零误报前提下，最高召回率对应的最低 score"
    else:
        selected = max(rows, key=lambda row: row["f1"])
        selection_reason = "无零误报兼顾点，使用最佳 F1；模型需要补充 hard-negative 训练"
    report = {
        "video": str(video),
        "reference_json": str(reference_json),
        "evaluated_frames": len(images),
        "rows": rows,
        "selected": selected,
        "selection_reason": selection_reason,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
