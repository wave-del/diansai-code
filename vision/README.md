# EC 26 E

> Q6A 当前使用 QNN HTP 后端，资源位于项目 `model-lib/q6a/`。
> 本地图片入口及限制见 [Q6A NPU 部署说明](../host/Q6A-NPU-DEPLOYMENT.md)。
> 几何求解统一使用同级 [`solver2`](../solver2/README.md)，旧 control 算法已移除；交接脚本会同时打包两个目录。
> 以下旧交接中的 VIPLite 模型和 A7Z 硬件部署方法不适用于 Q6A。

## 交接与快速开始

面向后续维护者的完整交接说明见 [docs/项目交接手册.md](docs/项目交接手册.md)：其中
包含目录职责、安装与验证步骤、两项任务的运行流程、硬件安全边界、标定/
部署方法以及常见故障排查。需要重新制作可分发压缩包时，运行：

```bash
bash tools/create_handover_archive.sh
```

脚本会把代码、配置、模型、原生推理库、样例与文档一并打包，同时排除 Git
历史、Python 缓存、测试缓存、临时产物和本机 IDE 设置。

硬件复刻请先阅读 [docs/硬件复刻指南.md](docs/硬件复刻指南.md)。其中区分了
仓库已确认的接线/参数和需要对原机实物确认的内容，切勿直接凭软件 GPIO 编号接线。

`vision` 是唯一视觉模块：`Resize` 负责蓝色 A4 工作区四边拟合与
透视校正，生产检测器依次执行 YOLO 实例判断、传统视觉真实边精修和
七帧直线稳定。

```python
from ec_26_e.vision import Resize, create_production_detector

resize_result = Resize().resize(image_bgr)
with create_production_detector(config.detect1) as detector:
    detect_result = detector.detect(resize_result.image_bgr)
```

`Detect1Result` 的每个纸片同时包含校正图像像素坐标和 `[0, 1]` A4 相对
坐标，不包含原始相机图坐标。运行无外部依赖的视觉测试：

```bash
poetry run python -m unittest \
  tests.unit.vision.test_resize \
  tests.unit.vision.test_detector \
  tests.unit.vision.test_result_contract
```

查看真实图片调试参数（该操作不会访问图片或写入文件）：

```bash
poetry run python tests/external/vision/analyze_video.py --help
```

实际调试时通过脚本参数覆盖配置，不读取 `.env`：

```bash
poetry run python tests/external/vision/analyze_video.py \
  --video outputs/cam/detection_20260730_180643_595024_raw.mp4 \
  --output-prefix outputs/vision/offline
```

工作区定位失败的帧会写入同尺寸的 `workspace_failed` 占位画面，因此
不会改变输出视频尺寸或中断其余帧处理。

## 统一任务配置与运行

项目只提供一个正式 Typer 启动入口。具体运行哪个任务由 `.env` 的
`EC_TASK` 选择，而相机等机器部署参数也保存在 `.env`：

```dotenv
EC_TASK=task1
EC_TASK_CONFIG=configs/tasks/task1.json
EC_DRAW_ENABLED=false
```

`EC_TASK_CONFIG` 可省略，此时自动读取
`configs/tasks/<EC_TASK>.json`。任务 JSON 不负责选择任务，只保存所选
任务自身的行为参数。当前 `task1.json` 使用严格的
`schema_version: 3`，其中 `planning.mode` 表示 Task1 内部使用 fixed
或 random 拼图算法，不是项目任务选择字段。

安装项目后直接运行：

```bash
ec-26-e
```

也可使用模块入口：

```bash
python -m ec_26_e
```

只有在需要切换环境文件时才传 CLI 参数：

```bash
ec-26-e --env path/to/.env
```

入口依次执行“加载环境选择 → 查询任务注册表 → 加载对应强类型 JSON
→ 运行任务”。已导出的环境变量优先于 `.env`，任务参数来自 JSON，
缺省字段回落到代码默认值。新增任务时应增加任务模块、注册项和对应
JSON，而不修改 `command/app.py` 的分派逻辑。

## 现场标定桌面入口

本机桌面提供 `EC26E 现场标定`、`EC26E 预览 Task1` 和
`EC26E 预览 Task2` 三个入口。现场标定控制台不实现重复的参数表单，
而是用 Kate 打开包含全部机器参数的 `.env` 草稿和两个完整任务 JSON；
保存后必须点击“检查并应用”，只有 Task1、Task2 都通过正式强类型加载器
校验时才会生效。

现场配置持久化在：

```text
~/.local/share/ec26e-calibration/
├── editing/       # Kate 直接编辑的草稿
├── generations/   # 最近 10 个已验证版本
└── active         # 原子指向当前整套生效配置
```

发布时同时写入 `.env`、`task1.json` 和 `task2.json` 的新版本目录，再原子
切换 `active`，因此校验失败或中途异常不会留下半套新配置。“恢复上一版”
会切回最近的已验证版本并同步覆盖草稿。快速预览固定读取 `active`，不会
读取尚未应用的草稿，也不会启动双路录像或创建 MP4 文件。

`EC_DRAW_ENABLED` 是所有任务共享的测试观察开关，严格接受
`true/false`、`1/0`、`yes/no` 或 `on/off`。默认 `false` 时不会创建
窗口、复制帧或执行绘制；测试时设为 `true` 会在原始相机画面上显示 A4
工作区和逆映射后的视觉检测结果。板端预览沿用相机诊断工具验证过的
X11 `:0` 显示会话，并通过系统 `xdpyinfo` 在启动 OpenCV/Qt 前检查
连接。按 Esc 或关闭窗口会中止当前任务并使用退出码 `130`，不会生成
新的任务结果文件。该能力只用于实时显示，不会录制视频，也不参与稳定
审查或控制计算。

Task1 在内存中生成并预检 `MovePlan`，随后直接执行完整运动生命周期；
计划不会写入文件、打印或返回。全部移动周期完成时进入 `FINISH`，之后
立即失能 XYZ、关闭磁铁并释放串口和 GPIO，安全清理完成后退出。

- `0`：任务移动完成且安全退出；
- `1`：识别、规划、运动或安全清理发生预期失败；
- `2`：环境任务选择、注册项或任务 JSON 配置无效；
- `130`：用户中断。

运行前会在标准错误输出任务名称、环境文件、任务 JSON 和最终强类型
配置摘要。`--help` 只显示帮助，不读取配置、不启动相机，也不创建
输出文件。

## F32C 旋转轴

旋转轴使用已验收的 F32C 参考控制器，默认通过 Cubie A7Z UART3
`/dev/ttyAS3` 通信；它与 XY 的 `/dev/ttyAS4` 完全独立。`.env` 可配置
`FOC_SERIAL_PORT`、`FOC_ADDRESS`、`FOC_RPM` 等字段，完整默认值见
`.env.example`。业务代码只调用 `motion` 的封装，不直接改写 F32C 协议：

```python
import os
from pathlib import Path

from ec_26_e.config import load_env_file
from ec_26_e.motion import FocRotationAxis, FocRotationConfig

load_env_file(Path(".env"))
config = FocRotationConfig.from_environ(os.environ)
with FocRotationAxis(config) as axis:
    axis.move_clockwise_from_zero(270.0)
    axis.return_counterclockwise_to_zero(270.0)
```

定向接口仅接受 `[0, 360]`，不计算最短路径：从零位固定顺时针到目标，
再按相同角度逆时针回零；`360°` 表示真实完整一圈。独立诊断所需的任意
协议方向整圈仍可使用 `axis.rotate_one_turn(direction=1)` 或
`direction=-1`。封装的进入和退出会执行安全使能、失能与 UART 释放。

完成验收后，可运行外部诊断脚本一次定位或回零：

```bash
# 从机械零位顺时针转到 90°（会真实使能并运动）
poetry run python tests/external/motion/control_foc_rotation.py --angle 90

# 已在顺时针 90° 目标时，逆时针沿原路径回机械零点
poetry run python tests/external/motion/control_foc_rotation.py --angle 90 --zero
```

`--angle` 是必需参数；`--zero` 表示执行对应目标的逆时针回零。可使用
`--port`、`--address`、`--rpm` 临时覆盖 `.env` 中的 FOC 配置。

## 相机采集

`cam` 模块通过 FFmpeg/V4L2 采集相机，并在后台持续排空图像管道。模块
只保留最新一帧，视觉处理较慢时不会积压历史帧。当前 USB 相机在 YUYV
1920×1080 下最高仅支持 5 fps；当前检测管线使用 MJPEG
1280×720、30 fps，再由 FFmpeg 解码为 BGR24。

```python
from ec_26_e.cam import FFmpegCamera

with FFmpegCamera() as camera:
    frame = camera.read_latest()
    image_bgr = frame.image_bgr
```

等待比上一帧更新的画面时传入序号；序号跳跃表示中间帧已被最新帧覆盖：

```python
next_frame = camera.read_latest(after_sequence=frame.sequence_id)
```

运行无设备依赖的 cam 单元测试：

```bash
poetry run python -m unittest tests.unit.cam.test_camera
```

使用真实相机录制 10 秒 H.264 MP4（默认写入忽略版本控制的
`outputs/cam/`）：

```bash
poetry run python tests/external/cam/record_camera.py --duration 10
```

两个摄像头诊断脚本都会读取项目根目录的 `.env`。检测录像的推荐配置为：

```dotenv
CAMERA_DEVICE_PATH=/dev/video0
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720

WORKSPACE_DETECT_WIDTH=640
WORKSPACE_DETECT_HEIGHT=360

A4_RESIZE_WIDTH=764
A4_RESIZE_HEIGHT=540
```

`CAMERA_DEVICE_PATH` 指定 V4L2 设备路径；多摄像头环境可配置为
`/dev/v4l/by-id/` 下的稳定设备链接，单次运行可用 `--device` 覆盖。
工作区定位分辨率必须保持相机宽高比且不能放大输入；A4 输出必须是
横向 `297:210`，允许 0.5% 的整数取整误差。`.env.example` 同时列出
`509×360`、`636×450`、`764×540`、`840×594` 和 `960×679` 五档
完整搭配。显式 CLI 参数优先于环境变量，已导出的环境变量优先于
`.env`。

录像器使用墙钟时间戳保持请求的播放时长。软件编码跟不上相机时，
FFmpeg 会复制最近画面补齐 30 fps，命令输出会同时报告实际送入编码器
的帧率和被最新帧覆盖的数量。

查看全部录像诊断参数不会访问相机或创建文件：

```bash
poetry run python tests/external/cam/record_camera.py --help
```

实时读取摄像头信息，将最新帧送入视觉检测，并同步保存原始视频与
检测绘制视频。脚本通过正式任务注册表读取 `.env` 中的 `EC_TASK`、
`EC_TASK_CONFIG`，并复用对应任务的 `camera`、`resize` 和 `detect1`
强类型配置，保证默认预览与正式任务视觉参数一致。脚本默认连接本机
X11 显示会话 `:0`，全屏实时显示同一张检测绘制帧：

```bash
poetry run python tests/external/cam/record_detected_video.py
```

只有临时诊断确实需要偏离正式配置时，才传入 `--device`、`--width`、
`--piece-mode` 等显式覆盖参数；使用其他环境文件时传入
`--env path/to/.env`。

该脚本不设置录制时长。按 `Esc` 或关闭显示窗口后停止采集，写完异步
录像器中的最后一帧并等待两路 MP4 完成封装；无屏运行使用
`--no-preview`，通过 `Ctrl+C` 安全结束。`Q` 不会停止录像。窗口模式
使用 `--windowed`，其他 X11 会话可通过 `--display` 覆盖。

同一次运行只生成一个会话时间戳，输出文件成对保存：

```text
outputs/cam/detection_{timestamp}_raw.mp4
outputs/cam/detection_{timestamp}_annotated.mp4
```

两路视频由同一个 FFmpeg 输入时间基准生成，保持相同帧 ID、帧数、
尺寸和标称帧率。标注视频在原图上绘制蓝色 A4 四角，并仅在录像脚本
内把 Detect1 结果逆映射回原图。单帧失败时分别绘制
`workspace_failed` 或 `detect1_failed`；最终摘要分别统计两个阶段的
失败、成功检测和编码覆盖数量。

## Servo42C 串口运动诊断

`src/ec_26_e/motion/` 提供独立的 Servo42C 串口协议、串口传输、纯
Python XY 状态机和 Z 轴控制器。Task1 通过上层无回包生命周期适配器
调用这些底层能力；Python
XY 状态机逐项复刻 `tmp-workspace/MKS/driver/xy_axes.c` 的状态、时序和
报文流程，不会调用或链接 C 代码。串口固定使用 115200、8N1、无流控；
设备和超时可在 `.env` 中通过 `MOTION_SERIAL_*` 配置，也可由命令行覆盖。

三个外部脚本默认会打开串口并发送报文。E3 复刻脚本会直接使能 E3、上移
90°，等待 Enter 后返回并失能：

```bash
# 默认真实复刻当前 E3 上移 90°、按 Enter 后返回的行为
poetry run python tests/external/motion/replay_current_behavior.py

# X/Y 使用绝对目标坐标和 0.01 mm 精度；默认按当前 C 配置不等待回包
poetry run python tests/external/motion/control_axes.py move --x 10 --y 5

# Z 仍是独立的相对角度控制
poetry run python tests/external/motion/control_axes.py move --z 30

# 连续 X 流程复用同一 Python XY 状态机，依次到达 15/30/45/60 mm
poetry run python tests/external/motion/repeat_x_move.py

# 普通使能、失能和停止入口只允许 Z，禁止绕过 XY 状态机
poetry run python tests/external/motion/control_axes.py enable --axes z
poetry run python tests/external/motion/control_axes.py disable --axes z
poetry run python tests/external/motion/control_axes.py stop --axes z
```

如需只生成报文而不访问设备，显式传入 `--dry-run`。通用脚本的全局
开关需要放在子命令之前：

```bash
poetry run python tests/external/motion/control_axes.py --dry-run move --x 10 --y 5
poetry run python tests/external/motion/replay_current_behavior.py --dry-run
```

Z 回包拒绝、状态异常、超时或用户中断时，外部脚本只报告错误并关闭
串口，不会自动发送停止或失能命令。XY 的故障、紧急停止、位置有效性、
回原点请求和停止失能顺序由 Python 复刻状态机自身管理。

真实发送时，每条完整报文都会在终端 stderr 输出一行 `[MOTION_TX]`
单行 JSON，其中包含 UTC 纳秒时间、单调时钟开始/完成时间、写入耗时及
相邻报文开始时间差。`delta_from_previous_start_us` 可直接查看 X0/X1
软件写入调用的时间差；它不等同于 UART TX 引脚上的物理发送时间。

XY 坐标完全沿用当前 C 接口：`X=0..230 mm`、`Y=0..360 mm` 的绝对
目标坐标，初始化时假定 `(0,0)`，移动完成后保持使能；X 增大向左，Y
增大向上。Z 正角度向上。模块单元测试不依赖真实串口：

```bash
poetry run python -m unittest discover -s tests/unit/motion -p "test_*.py"
```
# 几何算法入口

本目录保留视觉识别、任务编排和运动适配。拼图几何算法统一由上一级
[`solver2`](../solver2/README.md) 提供，原 `src/ec_26_e/control` 已删除。
部署源代码时保留 `diansai-code/vision` 与 `diansai-code/solver2` 的相邻结构；
从本目录启动正式任务时，`ec_26_e` 会解析同仓库根目录的 `solver2`。

