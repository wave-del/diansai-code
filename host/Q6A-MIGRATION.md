# Dragon Q6A 使用说明

本项目目前只面向 Q6A，已在 Ubuntu 24.04 / Python 3.12 / aarch64 上验证。
默认不接外设运行；主程序已取消实体按键 1/2/3 和蜂鸣器，保留屏幕任务按钮。

## 文件分层

沿用原有目录职责，以下路径均相对项目根目录：

- `handover-puzzle/`：共享板级适配 `q6a_platform.py` 与设备配置 `q6a_config.json`。
- `host/`：启动和安装脚本、依赖清单、Q6A 说明与验证记录；本地 Qt 库放在 `.runtime/`，日志放在 `outputs/`。
- `tests/`：Q6A 专项测试及原有测试。
- `vision/native/qnn_htp/`、`vision/src/ec_26_e/vision/qnn.py`：常驻 QNN 桥接和现有视觉接口适配。
- `model-lib/q6a/`：本项目独立的 Context、运行库与校验清单。
- 项目同级 `../npu_q6a_test/`：独立 NPU 验证，保持原位置。

## 安装与启动

在项目根目录运行（本机缺少 ensurepip，使用系统 pip 安装到虚拟环境）：

```bash
./host/setup_q6a.sh
./host/start_host.sh
# 与默认模式相同
./host/start_host.sh --offline
```

`host/requirements.txt` 是直接依赖，`host/requirements-dev.txt` 增加测试依赖，
`host/requirements.lock.txt` 记录本次验证的确切版本。
安装脚本不需要 sudo；若系统缺少 Qt 所需的 `libxcb-cursor0`，只下载官方 Ubuntu
软件包并解包到`host/.runtime/`，由启动脚本为当前进程加载，不修改系统库。
图形界面需要桌面会话；无显示服务的 SSH 终端可设置 `QT_QPA_PLATFORM=offscreen` 做启动检查。

默认模式不打开摄像头、任何串口或 GPIO，不初始化 NPU，也不会循环重连。
屏幕上的任务按钮、停靠及设备刷新均不可用；参数仍可查看和调整。
旧设置中的 `plan_only=false` 不能解除无外设模式。

命令行任务入口同样默认隔离外设：

```bash
.venv/bin/python handover-puzzle/run_task1_tangram.py --offline
```

此命令只报告平台状态，不执行视觉采集或生成真实拼图计划。
`--offline` 与 `--hardware` 互斥；执行、原点复用和停靠必须显式选择 `--hardware`。

## 后续接入设备

`handover-puzzle/q6a_config.json` 的 `camera_path`、`writer_port`、`motor_port` 初始均为空。
优先配置实际设备的稳定路径；界面选择的相机和写字机路径会保存到
`host/ui_settings.json`，Q6A 配置中的非空路径优先。

F32C 必须先选择并启用合适的 UART 或配置 USB 串口。本次没有修改设备树，
没有启用 UART，系统控制台 `/dev/ttyMSM0` 及其符号链接会被拒绝。
摄像头发现通过 V4L2 能力检查排除编解码器和元数据接口；Q6A 的
`/dev/video0`、`/dev/video1` 当前是 qcom-iris 编解码器。

电磁铁保留物理 13 脚，当前为 `/dev/gpiochip4` line 0，内核线名 `PIN_13`，
高电平吸合、低电平释放。GPIO 控制器在 open() 时校验线名。
各交付包保留独立副本，未合并；按键和蜂鸣器目录仅作历史参考。

```bash
./host/start_host.sh --hardware
```

硬件模式仍不启动相机任务；屏幕任务按钮保持禁用。配置完整并主动取消
“仅检测与解算”后，停靠入口沿用原有人工建立原点流程；缺少 F32C 配置会在
连接执行器前报错。当前没有接外设，机械操作尚未在 Q6A 上实测。
原有坐标变换、速度、旋转协议、停靠点和旋转标定均保留，不代表已重新标定。

## 视觉与 NPU

包内的 `.nb`、VIPLite 库和原生库属于 A733，不能用于 Q6A。
现有视觉接口已接入 QNN HTP，支持本地图片推理、A4 校正与几何精修。
使用方法和验收边界见 [Q6A NPU 部署说明](Q6A-NPU-DEPLOYMENT.md)。
Context、四个 Qualcomm 运行库和桥接库均位于本项目 `model-lib/q6a/`，
运行不依赖 SDK 或 NPU 测试目录。缺少文件或校验失败会报错，不作 CPU 回退。

NPU 验证全部位于项目的同级目录 `../npu_q6a_test/`（相对项目根目录），包含独立环境、官方示例、
运行库、脚本及输出。项目启动不读取或依赖该目录，也不自动执行 NPU 自检。
独立验证、原始 PT/ONNX 和诊断报告仍保留在该目录。本地图片后端可用
不等于相机、几何轮廓和机构执行已验收。

## 验证

```bash
.venv/bin/python -m unittest tests.test_q6a_runtime tests.test_qnn_backend
(cd handover-puzzle && ../.venv/bin/python -m unittest \
  test_hardware_discovery test_task1_adapter test_task1_yolo_flow test_f32c_arrival_policy)
QT_QPA_PLATFORM=offscreen .venv/bin/python solver/tests/run_regression.py
```

详细结果与旧测试的已知差异见 [Q6A-VALIDATION.md](Q6A-VALIDATION.md)。
`vision/deploy/` 内旧 A7Z 部署脚本和历史联调指南不适用于当前 Q6A；本次未安装自启动服务。
