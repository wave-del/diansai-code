# Q6A 本地图片 NPU 部署

**网页测试目录调整：** 所有网页服务、页面及新增求解/渲染代码已迁入
`../npu_q6a_test/web_preview/`。两枚四片按钮只计算与显示，视觉为 NPU，
原几何算法仍为 CPU；几何 NPU 加速尚未实现，运动禁用。
[新目录说明](../../npu_q6a_test/web_preview/README.md) · [本轮验收](../../npu_q6a_test/web_preview/docs/REPORT.md)。

**摄像头只读网页预览已由用户明确授权启动。** 使用
`../npu_q6a_test/web_preview/app/q6a_camera_preview.py`，显示原始画面、HTP 轮廓叠加与蓝纸校正图；
网页及算法接线已整理到独立测试目录；固定/随机四片只计算与显示，运动、串口、GPIO 禁用，默认 GUI 不变。
这只是现场观察，不把静态精度未通过项改为通过。
地址为 `http://192.168.1.32:8080/`；完整启动/退出说明见 [网页预览](../../npu_q6a_test/web_preview/README.md)。
以下“相机未开启”为此前静态验收阶段记录。

**最新视觉专项（2026-10-24 17:55，北京时间）：** 仅视觉，求解不参与验收。
保留全部模型/数值参数，新增定位拒绝阶段诊断及受物理约束的精修回溯。
16 张严格 A4 均完整输出 63 片，连续边界计量下 61/63 在 4 mm 观察预算内。
定位仍 16/28，9/23 的轮廓/标注和 24 等 A4 问题尚未解决，完整视觉验收未通过。
110 项相关测试及 5 项独立计量测试通过；最终源码 500 次真实 HTP 无不一致，
完整静态链路平均约 384 ms、2.60 FPS。相机与运动未开启。
见 [本轮报告](/mnt/nvme/initial-code/npu_q6a_test/qnn_conversion/vision-stage-20261024/REPORT.md)。

**当前范围：仅视觉。** 图片不保证组成完整矩形，因此历史求解 0/16 不作为视觉失败依据。
本阶段不执行 `--solve`，只验收检测、掩膜、A4、轮廓/角点、性能与稳定性。
仍有 A4 定位与几何/标注待核实，相机与运动暂未启用。
见 [视觉范围和工作顺序](/mnt/nvme/initial-code/npu_q6a_test/qnn_conversion/geometry-acceptance-20261024/continuation-20261024/VISION-SCOPE.md)。

**当前验收状态（2026-10-24 17:26，北京时间）：** 用户授权重启后 HTP 通信恢复。
索引 12 的背景污染和局部恢复随机波动已修复，最终常驻真实 HTP 500 次通过，无 CPU 回退。
28 张全图 float/HTP 均匹配全部 102 GT，16 张严格 A4 均输出完整 63 片，
4 mm 离线观察预算内均为 60/63，两后端最大角点差约 0.40 mm。
106 项相关测试、独立测量测试及四个求解回归通过。
完整静态路径平均约 377 ms、2.65 FPS，不是相机实时 FPS。
真实图片生产求解各 0/16 达到既有矩形条件，完整部署仍未通过，相机与运动继续禁用。
最新证据和命令见
[最终静态复验报告](/mnt/nvme/initial-code/npu_q6a_test/qnn_conversion/geometry-acceptance-20261024/continuation-20261024/FINAL-REPORT.md)。

**恢复前历史（2026-10-24）：** A4 和精修已有局部修复，原 95 项相关测试及新增 8 项测试通过。
严格 A4 路径为 16/28，浮点参考 63/63 实例完整；采用代码尺寸 292×207 mm，
4 mm 仅作为可调整的离线观察预算。真实图片求解与完整几何验收尚未通过。
当前 FastRPC/HTP 初始化失败，官方 ResNet 也失败，后端恢复前以下 NPU 命令可能在初始化退出。
相机与运动保持禁用。
见 [最新逐图验收报告](/mnt/nvme/initial-code/npu_q6a_test/qnn_conversion/geometry-acceptance-20261024/ACCEPTANCE-REPORT.md)。

2026-10-24：已接入真实 QNN HTP 常驻后端；默认界面继续无外设运行。
原有顶层目录和各独立交付目录保留。没有重新训练、修改原 PT/ONNX、
安装服务、修改设备树或操作摄像头、串口、GPIO、执行器。

## 运行

在项目根目录执行，使用本项目 `.venv`，无需启动独立 NPU 测试环境：

```bash
# 全图只检查模型实例，适合蓝纸没有完整入镜的图片
.venv/bin/python handover-puzzle/q6a_image.py --image /path/to/photo.jpg --full-frame

# 使用原项目正式蓝纸候选、A4 校正和几何精修；默认沿用左侧来源区
.venv/bin/python handover-puzzle/q6a_image.py --image /path/to/photo.jpg

# 已校正为横向 A4 的图片，直接送入现有视觉接口
.venv/bin/python handover-puzzle/q6a_image.py --image /path/to/a4.png --rectified

# 在上一步基础上运行随机几何求解，只保存候选
.venv/bin/python handover-puzzle/q6a_image.py --image /path/to/a4.png --rectified --solve
```

输出默认位于 `host/outputs/q6a-images/时间戳/`，包括 `report.json`、
`result.png`，全图模式另存实例掩膜。`--output` 可指定目录；`--source-region all`
只用于离线查看所有纸片，不改变正式任务的左侧来源区规则。
求解需要 2～4 片，任何模型实例未通过精修时拒绝求解。
生成的 `solver/solution.json` 明确标记 `offline_only=true`、
`machine_executable=false`，单图结果没有连续帧稳定或运动验收。
退出 0 表示所请求的本地处理完成，不表示候选是正确拼合或可机械执行。

默认界面仍通过 `./host/start_host.sh` 启动，不初始化 NPU、不打开外设、
不重连。任务按钮保持禁用，显示“本地图片后端已接入、相机及运动任务待验收”。

## 文件分层与运行方式

- `vision/src/ec_26_e/vision/qnn.py`：BGR→RGB、640 letterbox、四输出解码、NMS、掩膜恢复。
- `vision/src/ec_26_e/vision/npu.py`：保留 `NpuYolo11SegInferencer` 旧接口。
- `vision/native/qnn_htp/`：使用官方 QAIRT 2.37.1 头文件编译的 C 桥接源码与构建说明。
- `model-lib/q6a/`：约 18 MB，包含修复后的 Context、桥接库、HTP/Stub/Skel/System 及 SHA256 清单。
- `handover-puzzle/q6a_image.py`：只读本地图片并输出视觉/求解结果。
- `../npu_q6a_test/`：模型原件、SDK、实验环境与诊断；项目运行不依赖此目录。

Context 使用本机实测 `v68 / soc_model=93 / VTCM=2 MB`；模型加载一次，
后续在内存中反复执行。原生 UINT8 四输出仅反量化一次，解码使用已导出
xywh 和概率，不再次执行旧 A733 的 DFL 或 sigmoid。后处理沿用 score=0.15、
NMS=0.45、mask=0.5 及现有 0.94 掩膜包含去重；不先截断为四片。
同分候选按 anchor 顺序决定，保证量化分数打平时结果确定。

## 本轮几何修复前的后端验证记录

以下 21/28、80/81 和图执行耗时来自先前阶段；最新严格路径、真实图片求解和
当前 HTP 可用性以上方验收报告为准。新路径要求实际观测四条边，拒绝参考角点回退。

- 28 张全图：主项目环境检测 102 个，对 102 个 GT 全部匹配，额外/漏检为 0。
  GT 匹配采用 mask IoU≥0.5，并非 COCO mAP；平均 IoU 约 0.917，最低 0.751。
- 固定原始输入的 112 个输出与官方运行器逐值一致，没有 CPU 回退。
  项目环境实测图执行平均约 7.5 ms，不包含图像预处理、掩膜和几何精修。
- OpenCV 5.0 与参考环境 4.10 的预处理最多相差 1 个 UINT8 灰度级，
  已分别评估，主环境仍为 102/102。没有向项目环境安装 Torch 或 ONNX Runtime。
- 正式单帧 A4 路径：21/28 张定位通过；7 张拒绝处理。
  通过定位的 81 个 GT，浮点与 HTP 均检出 81 个；现有几何精修均只输出 80 个。
  `WIN_20260729_21_39_47_Pro.jpg` 对应索引 23，
  失败是无法形成满足既有物理角边及 IoU 约束的多边形，浮点也会失败。
- 掩膜精修后平均 GT IoU 约 0.945；HTP 有 3 个多边形顶点数与人工标签不同，
  浮点有 1 个。不能仅凭检测数量和平均 IoU 开放现场任务。
- 同顶点数且按掩膜配对的 HTP 轮廓顶点误差平均约 0.94 mm，P95 约 2.33 mm。
  这些毫米数假定蓝纸为 292×207 mm，未确认实拍纸张物理尺寸，不能作为机构标定值。

相机、真实连续帧稳定和运动仍未验收；`accuracy_accepted`、
`deployment_accepted`、`motion_accepted` 保持 false。本地 NPU 后端可用与
现场完整系统通过验收是分别记录的状态。

专项测试见 `tests/test_qnn_backend.py` 与 `tests/test_q6a_runtime.py`；
完整证据及重跑入口见独立目录的 `board-stage-20261024-092050/DEPLOYMENT-REPORT.md`。
