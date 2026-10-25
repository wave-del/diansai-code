# 拼图装置（EC 26 E）

> **当前部署目标已改为 Dragon Q6A。** 默认无外设模式，实体按键和蜂鸣器已从主程序取消。
> 使用 `./host/start_host.sh` 启动，详见 [Q6A 使用说明](host/Q6A-MIGRATION.md) 与
> [验证记录](host/Q6A-VALIDATION.md)。板载 NPU 验证单独保存在同级 `../npu_q6a_test/`。
> Q6A HTP 本地图片后端已接入，见 [NPU 部署与验证](host/Q6A-NPU-DEPLOYMENT.md)。
> 拼图算法统一维护于 [`solver2/`](solver2/README.md)，网页和正式任务使用同一套边链几何引擎；`vision` 内的重复算法已删除。见 [算法更新记录](vision/docs/算法更新记录.md)。
> 相机与运动尚未验收；下文保留原 A7Z 交接资料，其中硬件映射与旧启动方式不适用于 Q6A。

全国大学生电子设计竞赛 E 题「拼图装置」的实现。用一台市售**写字机**改装的机械平台，
在蓝纸（A4）工作区上完成拼图碎片的**识别 → 求解 → 抓取 → 旋转 → 放置**，全程自动运行。

平台没有可靠的限位开关，也没有位置编码器：所有运动都是**开环步进**，原点由人工放置后
用 `G92` 建立。这一点决定了整套软件的安全设计（见下文「坐标约定」与「安全边界」）。

## 系统架构

上位机（Linux）负责视觉与决策，下位机负责实时运动。上位机**不生成步进脉冲**——
它把毫米坐标以 G-code 文本发给 ESP32，由 ESP32 上的 GRBL 固件做轨迹规划。

```
┌──────────────────────────────────────────────────────────┐
│ Radxa Cubie A7Z（Linux 上位机）                            │
│                                                          │
│  host/             PySide6 触摸界面・任务调度・完成声光提示  │
│  handover-puzzle/  视觉采集→稳定→解算→取放流程编排          │
│  solver/           几何求解器（独立验证台，不碰硬件）        │
│  solver2/          网页和正式任务共用的拼图几何算法          │
│  vision/           YOLO11-seg NPU 推理 + A4 透视校正        │
└──────┬────────────────┬───────────────────┬───────────────┘
       │ USB 串口        │ UART3             │ GPIO
       │ 115200 8N1     │ 115200 8N1        │
       ▼                ▼                   ▼
  ESP32 GRBL 1.3a   F32C 无刷 FOC      电磁铁 / 按键 / 蜂鸣器
  XY + Z 步进       拼图块旋转定位       gpiochip1 / gpiochip0
  （G-code 文本）   （自定义二进制帧）
```

**三条控制链路互不相同**，排查问题时不要混淆：

| 链路 | 传输 | 协议 | 代码入口 |
|---|---|---|---|
| 写字机 XY/Z | USB 串口（CH340），115200 8N1 | GRBL G-code 文本，`ok` 流控 | [handover-writer/grbl_controller.py](handover-writer/grbl_controller.py) |
| F32C 旋转 | A7Z 板载 UART3 `/dev/ttyAS3` | 自定义帧 `0x7A…0x7B` + BCC 校验 | [handover-foc/f32c_controller.py](handover-foc/f32c_controller.py) |
| 电磁铁/按键/蜂鸣器 | 直连 GPIO | 电平 | [handover-electromagnet/electromagnet.py](handover-electromagnet/electromagnet.py) |

> ESP32 上跑的是开源固件 **Grbl_ESP32 1.3a**（四轴双 Y），仓库里**没有** ESP32 侧的代码——
> 它是现成的 GRBL 变体。上位机只负责按 GRBL 协议发指令、解析状态行。
> 若你也在找"ESP32 的代码"，它不在这个仓库里。

## 目录结构

| 目录 | 职责 |
|---|---|
| `host/` | 上位机触摸界面、实体按键、相机预览、安全停靠、2 分钟计时 |
| `handover-puzzle/` | 拼图主流程：视觉采集、五帧稳定、求解适配、取放执行 |
| `handover-writer/` | 写字机 GRBL 控制（可独立复制的交付包） |
| `handover-foc/` | F32C 无刷电机角度控制（仅标准库，无第三方依赖） |
| `handover-electromagnet/` | 电磁铁 GPIO 控制 |
| `handover-button/` `handover-buzzer/` | 实体按键与蜂鸣器 |
| `solver/` | 拼图解算器实验台（PySide6 GUI，不访问 GPIO/相机/机械臂） |
| [`solver2/`](solver2/README.md) | 网页与正式任务共用的唯一拼图几何算法，含固定四片、随机 2～4 片、留缝规划及测试 |
| `vision/` | 视觉模块：A4 透视校正、YOLO 实例分割、边精修（原 `ec-26-e` 工程） |
| `code/` `example/` `tests/` | 方向标定、运动冒烟测试与联调脚本 |

`handover-*` 各目录都是**可独立复制**的交付包，`handover-puzzle/` 内自带一份
控制器副本，因此目录间存在有意为之的重复——不要为了"去重"把它们合并。

## 硬件与接口

| 部件 | 规格 | 接口 |
|---|---|---|
| 上位机 | Radxa Cubie A7Z | — |
| 摄像头 | USB UVC，MJPG 1280×720@30 | `/dev/v4l/by-id/…` |
| 写字机主控 | ESP32 + Grbl_ESP32 1.3a，四轴双 Y | CH340 USB 串口，115200 8N1 |
| XY 步进 | 100 steps/mm，最大 20000 mm/min | GRBL X/Y |
| Z 抬落笔 | 步进，50 steps/mm，落笔 Z=2.0 mm | GRBL Z |
| 旋转电机 | F32C 无刷 FOC | UART3 `/dev/ttyAS3`，115200 8N1 |
| 电磁铁 | 高电平吸合 | `/dev/gpiochip1` line 6 |
| 实体按键 | 低电平有效，上拉 | `/dev/gpiochip0` line 36 / 40 / 39 |
| 蜂鸣器 | 低电平触发 | `/dev/gpiochip1` line 37 |

## 坐标约定（必读）

**业务坐标原点在蓝纸右上角**：`X+` 向下，`Y+` 向左，单位 mm。
所有绝对 XY 运动都必须**经过 `WriterController`**，它只做一次偏置与限幅：

```text
X_cmd = clamp(X_input - 23.5, 0, 190)
Y_cmd = clamp(Y_input + 16.45, 0, 300)
```

视觉、算法与交互层**禁止**再次偏置、取反或限幅，也禁止绕过控制器直接发绝对 G-code。
底层还会做一次物理轴对调与方向反转（`swap_xy` / `reverse_x` / `reverse_y`），
这属于硬件标定，同样不要在业务层重复处理。

## 快速开始

```bash
# 1) 建虚拟环境（在仓库根目录）
python3 -m venv .venv
.venv/bin/pip install -r handover-puzzle/requirements.txt

# 2) 硬件侧：先把机构**人工**放到写字机机械限位角，Z 自然下垂，再上电

# 3) 启动上位机
host/start_host.sh
```

首次上电需点击「建立原点并前往安全停靠点」建立本次上电坐标，之后才能执行任务。
`host/ui_settings.json` 记录本机相机/串口路径与现场调参值，**不纳入版本控制**——
首次运行前从 `host/ui_settings.json.example` 复制一份再改。

**不接硬件先跑视觉与解算**（推荐首次上手）：

```bash
cd handover-puzzle
../.venv/bin/python run_task1_tangram.py --task-profile random-white --plan-only
```

`--plan-only` 完整执行相机采集、稳定识别、解算与计划预检，但绝不打开串口、
F32C 或电磁铁。上位机里对应「仅检测与解算（硬件隔离）」开关，默认开启。

## 安全边界

- **开环系统**：返回坐标是 GRBL 的步数计数，不是实测位置。撞限位或丢步后必须重新
  人工归位并调用 `home()`。
- **无自动回零**：没有可靠的限位开关，`home()` 不执行 `$H`，只发送 `$X`、`$1=255`
  与 `G92 X0 Y0 Z0`，把人工放置的位置定义为本次上电原点。
- **运动结束后步进保持使能**（`$1=255`），机构受扰动或手推时保持位置。
- **`disconnect()` 不是急停**：已经进入固件 planner 的运动会继续执行。
- 现场可恢复的越界参数（取放坐标、Z 抬起高度等）会被**限幅并打印告警**后继续；
  仅对非有限值等不可恢复输入报错。

## 已知限制

- **视觉模型不随仓库分发。** `model-lib/` 与 `vision/model-bin/` 被排除在版本控制外，
  其中包含训练好的 `puzzle_seg_yolo11n_640_pcq_a733.nb`（YOLO11-seg，A733 NPU）。
  代码完整，但要真正运行视觉链路，需要自行训练或向作者获取该模型。
- **厂商 NPU 运行库需自备。** `libNBGlinker.so` / `libVIPhal.so` 属 SoC 厂商 SDK 组件，
  不在此仓库内，请从 Radxa/VeriSilicon 对应 SDK 获取。
- `vision/native/yolo11_seg_npu/` 是前后处理原生库的**源码**，需按其中的 CMake 工程自行编译。
- **`vision/deploy/` 下的部署脚本含目标机绝对路径**（`/home/radxa/...`，即 A7Z 板子的默认
  用户目录）与 poetry 虚拟环境路径。它们按独立仓库的部署方式编写，合并到本仓库后需按
  实际克隆位置与虚拟环境调整；`.desktop` / `.service` 因需要绝对路径，无法自动适配。
- 电赛题目原文与设计报告（`doc/`）未包含在本仓库中。

## 模块文档

- 上位机：[host/README.md](host/README.md)
- 取放流程与坐标映射：[handover-puzzle/README.md](handover-puzzle/README.md)
- 写字机控制与 API 全表：[handover-writer/README.md](handover-writer/README.md)
- F32C 电机：[handover-foc/README.md](handover-foc/README.md)
- 解算器实验台：[solver/README.md](solver/README.md)
- 视觉与硬件复刻：[vision/docs/项目交接手册.md](vision/docs/项目交接手册.md)、
  [vision/docs/硬件复刻指南.md](vision/docs/硬件复刻指南.md)

## 参与

本项目由团队成员共同完成，提交历史中保留了原作者的署名。
