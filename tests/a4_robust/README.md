# a4_robust：遮挡鲁棒的 A4 矩形检测（试验包）

不修改工程内任何现有文件；全部新代码都在本目录，随时可整体删除。
复用 `reference-vision/ec_26_e` 的 Resize/Detect1 与
`handover-puzzle/test_a4_detection.py` 的参数预设。

## 解决的问题

原管线（`_blue_a4_seed` + Resize）要求蓝纸**完整可见**：
- 遮挡一个角 → 多边形顶点数不为 4 / 面积不足 → "未找到完整四边形蓝纸候选"；
- 遮挡一条边 → Resize 边缘支撑不足 → 整条链路失败；
- 手影/补光变化让蓝色绝对电平跨过 HSV 门槛 → 检测时好时坏。

本包的目标（对应你提的方案）：
1. **3-4 条实测边 + 已知宽高比（206.5×292）补全第 4 角**；
2. **逐角可靠性标记**（实测/估计），估计角交给放宽校验的 ResizePartial；
3. 宽高比先验区分"真实纸边组合"与"遮挡物边界冒充纸边"。

## 文件

| 文件 | 作用 |
|---|---|
| `seed.py` | 遮挡鲁棒蓝纸定位：完整四边形快速路径 + 边拟合回退（分段→聚类→矩形结构/宽高比筛选→3 边补全） |
| `resize_partial.py` | `ResizePartial(Resize)`：估计角相邻边放宽支撑/搜索带/边界/外观校验 |
| `pipeline.py` | `detect_robust(frame, params, a4_args)` 全流程胶水 |
| `run_live.py` | 实时摄像头诊断（只读），橙色圈 = 估计角 |
| `harness_occlusion.py` | 离线遮挡合成测试（录像帧 + 黑圆/亮块/黑条/手形遮挡） |

## 用法

```bash
# 实时诊断（只读摄像头；--params current|new 同主工程）
.venv/bin/python tests/a4_robust/run_live.py --params new

# 离线遮挡合成测试（用 example/outputs 下的现场录像）
.venv/bin/python tests/a4_robust/harness_occlusion.py --frames 8
```

## 实测结果（camera_20260731_222707.avi，new 参数）

| 场景 | 原管线 | 鲁棒管线 | 最终角误差 |
|---|---|---|---|
| 黑圆盖角（r=35~110）/ 亮块盖角 | 0% 成功 | 100% 成功 | 0.1~2.4 px |
| 双角遮挡 | 失败 | 成功 | ~1 px |
| 手形遮挡一条边（真实场景） | 失败 | 成功 | 0.1~0.8 px |
| 黑色直条盖整边（合成最坏情形） | 部分成功 | 成功 | 31~43 px（隐藏边固有误差） |
| 干净帧 | 成功 | 成功 | ~0 px |

碎片检测成功率：鲁棒管线平均 1.09 片/成功帧，无回归。

## 已知限制

1. **直线遮挡物盖住整条边**：单帧 mask 几何上存在歧义（"右边被切"与
   "左边被切"两个解释自洽）。宽高比 + 最小面积优先已能选对组合，
   但隐藏边位置只能靠补全，误差 ≈ 遮挡深度（实测 31~43px）。
   真实的手/物体边界直线拟合残差高会被直接拒绝，不受此限。
2. **遮挡过半**（蓝 mask 面积 <15%）无法定位 —— 物理上已不可见。
3. 估计角处的校正图包含遮挡物，该处碎片物理上也不可见，属正常。

## 与主工程的衔接（已集成）

- **YOLO 组合管线**（[yolo_pipeline.py](yolo_pipeline.py)）：
  `YoloRobustDetector` = 每帧 a4_robust seed（参考角，机位无关）
  → 同事 Resize（含 Canny+Hough 兜底）→ 同事 YOLO11n-seg（NPU）
  → 可选七帧稳定器。依赖同事仓库 `/home/radxa/repositories/ec_26_e`
  与 `model-lib/` 四件套（模型 + libyolo11_seg_npu.so + 两个 VIPLite
  运行时库）；`LD_LIBRARY_PATH` 由脚本 re-exec 自包含。
- **取放脚本**（handover-puzzle/test_vision_centroid_pick_rotate_place.py）：
  `--yolo`（默认）/ `--traditional` 选择检测引擎；yolo 模式下
  每片抓走后自动 `reset()` 稳定器轨迹。
  注意：yolo 与传统两套 `ec_26_e` 包同名，不能在同一个进程混用
  （本脚本已按引擎条件导入，互不干扰）。

```bash
# YOLO 引擎（默认）：遮挡鲁棒 seed + 同事 Resize + YOLO + 七帧稳定
.venv/bin/python handover-puzzle/test_vision_centroid_pick_rotate_place.py --params new

# 传统引擎回退
.venv/bin/python handover-puzzle/test_vision_centroid_pick_rotate_place.py --traditional --params new
```

- 如需消除黑条类极限场景，可加**时间先验**（上一帧角点位置），
  纸面静止时两帧间的歧义可被消除。
