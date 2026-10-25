# vision 使用说明

Q6A 后端为 `qnn.py`，`npu.py` 保留旧类名导出。默认资源来自项目根目录
`model-lib/q6a/`；仅显式构造推理器才初始化 HTP。当前支持本地图片验证，
相机、连续帧与运动未验收，详见项目 `host/Q6A-NPU-DEPLOYMENT.md`。

`vision` 是 Task1、Task2 唯一使用的 YOLO11n-seg 语义门控与传统视觉精修模块。主任务通过 `create_production_detector(config.detect1)` 构造 9 帧窗口、6 帧确认的稳定器；不存在纯传统检测器或运行时回退。

职责边界是：YOLO 只决定是否为纸片、实例数量、核心和局部搜索区；最终边界必须来自原图的颜色/梯度边缘、稳健直线拟合及相邻直线求交。若传统视觉找不到真实像素支持，该实例不输出，不会用 YOLO 粗 mask 轮廓代替。

```python
from ec_26_e.vision import create_production_detector

with create_production_detector(config.detect1) as detector:
    stable = detector.detect(resize_result.image_bgr)
```

- `VisionDetector.detect(a4_image_bgr) -> Detect1Result`：无状态单帧接口，输入必须是现有 `Resize` 输出的横向 A4 图。
- `StableVisionDetector.detect(a4_image_bgr) -> Detect1Result`：状态接口，默认保留 9 帧并在连续 6 帧确认后输出；未稳定时返回同契约空结果。
- `last_diagnostics`：包含原生 NPU 分阶段计时、粗/精 mask、`boundary_source`、逐边 `edge_sources`、边支持率、法向残差和交点不确定度等，不进入业务契约。
- 默认阈值为 score `0.15`、NMS `0.45`、mask `0.50`。score 来自冻结样本扫描：`0.05/0.10` 会把手部误识别成纸片，`0.15` 是保持正样本召回且反光/手部负样本零误报的最低测试值。
- 最短物理边硬下限为 `6 mm`。小于该值的边会被合并；拟合交点后会再次检查，避免亚像素求交重新产生过短边。

独立诊断：

```bash
poetry run python tests/external/vision/analyze_video.py --help
poetry run python tests/external/vision/sweep_thresholds.py --help
poetry run python tests/external/vision/record_camera.py --help
poetry run python tests/external/vision/main_pipeline_dry_run.py --help
```

实时摄像头测试默认打开 `/dev/video0`，在屏幕显示 A4 校正后的 YOLO
粗掩膜、单帧精修边和七帧稳定结果，同时保存同步的原始与标注 H.264：

```bash
poetry run python tests/external/vision/record_camera.py
```

按 Esc 或 Ctrl+C 结束。文件默认写入
`outputs/vision/live_<时间戳>_raw.mp4`、
`outputs/vision/live_<时间戳>_annotated.mp4`，同名前缀的 JSON 保存耗时和
丢帧统计。无桌面环境可增加 `--no-preview`；只验证显示而不录像可增加
`--no-save`。

目前 `max_pieces=4` 与项目已确认上限一致；含 5 个或更多实例的画面会按置信度截断，不能用于评价完整计数召回。
