# 拼图装置联动控制交付

本目录是一份可独立复制的 Python 交付，运行时不依赖同级
`handover-writer`、`handover-foc` 或 `handover-electromagnet`。它把以下
三个已调通部件组合成一次完整取放动作：

- 写字机：XY 平面搬运和 Z 轴升降；
- F32C 无刷电机：拼图件的机械绝对角度；
- PL6 / Pin 13 电磁铁：高电平吸合，低电平释放。

## 强制蓝纸坐标映射（所有写字机绝对运动必经）

业务坐标原点是蓝纸右上角，`X+` 向下、`Y+` 向左，单位均为 mm。无论视觉、
交互测试、取放流程还是未来算法调用，`WriterController.move_to()`、
`fast_move_to()` 和 `move_path()` 都会且只会执行一次：

```text
X_cmd = clamp(X_input - 23.5, 0, 190)
Y_cmd = clamp(Y_input + 16.45, 0, 300)
```

调用方必须直接传蓝纸坐标，**禁止**重复偏置/限幅，也禁止绕过控制器直接发送
绝对 XY G-code。上电后先让机构位于写字机自身机械原点，再调用 `home()` 或
`establish_origin()`；两者只建立软件原点，不主动运动。正常任务结束时调用
`move_to(0,0)` 回蓝纸业务原点；`return_to_machine_origin()` 仅供上电/维护时
显式回写字机自身机械原点使用。

面向算法的正数 Z 接口为 `set_lift()` / `move_xyz()`：`0` 为自然下垂，正数
为抬起；Z 速度固定 `1500 mm/min`。当前实机验证范围为 `0..15 mm`。

正式拼图流程的 XY 默认进给速度为 `4500 mm/min`（75 mm/s），适用于
`PuzzleConfig`、`WriterController` 和本目录的取放/视觉测试脚本；各接口仍可
显式传入 `xy_feed` 或 `--xy-feed` 覆盖。首次在新的机构负载下使用该速度时，
应先进行重复短距离往返并观察是否丢步；Z 轴不随此设置变化，仍固定
`1500 mm/min`。

底层取放接口也可以单独使用，不要求视觉在线；完整 Task1 入口则会读取
摄像头，并在所有运动硬件尚未连接时先完成视觉、求解和限位预检。

## 鲁棒视觉 Task1 的完整嫁接

`run_task1_tangram.py` 默认复用现场已经实测通过的整条检测链路：

```text
a4_robust 遮挡鲁棒 seed
  -> 同事 Resize（Canny/Hough 兜底）
  -> A733 NPU YOLO11n-seg 门控 + 像素边缘精修
  -> 五帧 StableVisionDetector
  -> 同事版完整拼图求解器
  -> 本机坐标/限位适配
  -> PuzzleController
```

视觉实现直接来自 `tests/a4_robust/yolo_pipeline.py`，模型和本机 NPU 动态库
直接来自 `model-lib/`，规划直接使用 `/home/radxa/repositories/ec_26_e` 的新
求解器，没有重写检测算法。旧 `reference-vision` 链路仍可通过
`--vision traditional` 明确选择。两套同名 Python 包在进程入口处分流，
不会混合导入。YOLO 分支默认读取同事工程最终的 `configs/tasks/task1.json`
（包括现场使用的规划参数），traditional 分支才读取旧参考配置。参考工程
原有 Servo42C/MKS 执行器不会被加载或使用。

首次部署时复制环境示例并填写真实相机路径：

```bash
cp task1.tangram.env.example task1.tangram.env
```

默认仅做视觉、规划、坐标换算和写字机限位预检，不会连接任何运动硬件：

```bash
.venv/bin/python \
  run_task1_tangram.py --env task1.tangram.env
```

只要求解器找到拼法，就会在 `outputs/task1_plans/时间戳/` 保存
`solution_after.png`、`solution.json`、输入轮廓和诊断日志。即使方案随后被
严格执行门拒绝，图片仍会保留，便于先查看拼合形状再决定下一步。

需要一边观察检测框和五帧稳定状态，一边生成计划时加 `--draw`；默认全屏，
调试时可再加 `--windowed`：

```bash
.venv/bin/python \
  run_task1_tangram.py --env task1.tangram.env --draw
```

输出中的 `pickup_xy_mm`、`place_xy_mm` 是传给 `PuzzleController` 的蓝纸
右上角业务坐标。参考标准 A4 的 `(x,y)` 会按实测蓝纸 `206.5×292mm` 等比
轴向换算。计划任一点落入写字机的静默限幅区都会在连接硬件前失败。

真实执行必须先标定“视觉顺时针”相对 F32C 机械角度的正方向。使用转盘或
夹具上的标记（**不要放未吸住的拼图件**）运行交互式标定；它只连接 F32C，
按 `+30°` 小角度探测、询问实际俯视方向、再按 `-30°` 回到起点：

```bash
.venv/bin/python \
  calibrate_f32c_rotation.py
```

结果会保存为 `task1.tangram.rotation.json`。真实任务自动读取此文件；也可
临时使用 `--rotation-sign 1`（同向）或 `--rotation-sign -1`（反向）覆盖。
未完成标定时，即使视觉与规划成功，带旋转计划也不会执行。确认计划、机构
已位于写字机自身原点、Z 自然下垂且路径安全后运行：

```bash
.venv/bin/python \
  run_task1_tangram.py --env task1.tangram.env \
  --execute
```

求解结果若已通过题目的名义几何门，但仅因严格鲁棒误差预算被拒绝，测试时可
显式人工放行。首次实机只执行第 1 片，Z 使用与 `example/TangramAPI` 相同的
“正数抬高”语义：搬运抬高 10mm，在纸面上方 2mm 吸取和释放，结束保持抬高
10mm：

```bash
.venv/bin/python \
  run_task1_tangram.py --env task1.tangram.env \
  --allow-competition-compliant --max-moves 1 \
  --travel-lift-mm 10 --pickup-lift-mm 2 \
  --place-lift-mm 2 --rest-lift-mm 10 \
  --execute
```

该开关不会放行几何竞赛门失败、机器几何不可执行、候选无效或有歧义的方案。
去掉 `--max-moves 1` 才会连续执行完整计划。

程序仍会在执行前要求输入 `RUN`。一次任务只连接一次硬件，避免重开 CH340
串口导致 GRBL 工作坐标复位。离线适配测试：

```bash
.venv/bin/python test_task1_adapter.py
.venv/bin/python test_task1_yolo_flow.py
```

## 独立 `test_transfer.py` 保守硬件测试

本节描述旧的单点联动测试脚本，不是上面的完整 Task1 默认 Z 参数；完整
Task1 当前使用 `10/2/2/10 mm` 的正数抬高配置。

默认流程正是：

1. XY 位于右下角、Z 自然下垂到下方机械限位时，建立本次
   `X0/Y0/Z0`；
2. 在Z轴完成标定前，临时只命令抬高到 `Z=-2 mm`；
3. 移动到 `X=50 mm, Y=50 mm`（即 5 cm、5 cm）；
4. Z 回到自然下垂位置 `0 mm`，电磁铁吸合并临时抬高到 `-2 mm`；
5. 移动到 `X=100 mm, Y=100 mm`（即 10 cm、10 cm）；
6. Z 回到自然下垂位置 `0 mm`，电磁铁释放并临时抬高到 `-2 mm`；
7. 在 Z 抬高状态下让 XY 回到 `(0, 0)`；
8. XY 到原点后，把 Z 恢复到初始自然落下状态 `0 mm`；
9. 关闭时无刷电机失能，写字机因 `$1=255` 继续保持使能。

进入项目虚拟环境后运行：

```bash
.venv/bin/python \
  handover-puzzle/test_transfer.py
```

脚本优先寻找稳定的写字机 USB by-id 路径，找不到时再寻找
`/dev/ttyUSB*`。也可以明确指定：

```bash
.venv/bin/python \
  handover-puzzle/test_transfer.py \
  --writer-port /dev/serial/by-id/usb-1a86_USB_Serial-if00-port0 \
  --motor-port /dev/ttyAS3
```

启动后脚本先完整打印运动计划。确认机构已经完成右下角归位、Z 已自然
下垂并停在下方机械限位、运动范围内没有手和障碍物，再输入 `RUN`。

### 同时测试拼图旋转

默认搬运不改变无刷电机当前角度，但仍会连接、安全使能并在退出时失能。
需要抓起后转到机械绝对 `90°`，可运行：

```bash
.venv/bin/python \
  handover-puzzle/test_transfer.py \
  --rotation-angle 90
```

加上 `--return-rotation-zero`，会在放件并抬高后让无刷电机回到驱动板中
保存的机械零点。`0°` 和 `360°` 是同一个机械方向，因此接口接受的绝对
角度范围为 `[0, 360)`。

### 调整 Z 高度

默认值沿用写字机交付中的配置：

- `--safe-z -2`：未完成标定前的临时保守命令，单位 mm；
- `--pickup-z 0`：自然下垂的取件位置，单位 mm；
- `--place-z 0`：自然下垂的放件位置，单位 mm；
- `--final-z 0`：XY 回原点后的最终自然落下位置，单位 mm；
- `--z-feed 1500`：Z 轴实测可靠工作速度，单位 mm/min。

本机方向是 `Z+` 向下、接近拼图件，`Z-` 向上。按本装置的机械结构，
自然下垂并停在下方机械限位的位置定义为 `Z0`；默认从取放 `Z=0mm`
临时移动到 `Z=-2mm`。高速20mm已经从真实下方零点撞到上方机械限位，
所以在速度/距离关系完成标定前，联动层会拒绝超过5mm的Z命令跨度。

现场已经确认高速20mm和60mm都会撞上方机械限位，不能继续把GRBL的Z
“毫米”直接当成末端真实垂直毫米。
不同拼图厚度、磁铁安装高度或台面高度必须重新实测参数，不能在未知机械
间隙时直接增大 Z 值。例如：

```bash
.../test_transfer.py --safe-z -2 --pickup-z 0 --place-z 0 \
  --final-z 0 --z-feed 1500
```

## 坐标与上电归位语义

- 原点：工作区域最右下角；
- `X+`：向左；
- `Y+`：向上；
- `Z+`：向下、接近拼图件；
- XY/Z 单位：毫米 `mm`；
- XY/Z 速度：毫米每分钟 `mm/min`；
- 无刷角度：度 `°`；
- 无刷速度：转每分钟 `RPM`。

装置的外部上电流程可以自动把机构送到右下角，但本交付中的
`establish_origin()` 仍然必须调用。它只发送 `$X`、`$1=255` 和
`G92 X0 Y0 Z0`，把已经到位的当前位置定义成软件原点；它不会执行 `$H`
或主动寻找机械限位。若机构实际上尚未到右下角，不能调用它后继续运动。

## 业务 API

最小调用方式见 `example_usage.py`：

```python
from puzzle_controller import PuzzleConfig, PuzzleController, TransferRequest

controller = PuzzleController(PuzzleConfig())
with controller:
    controller.connect(
        writer_port="/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0",
        motor_port="/dev/ttyAS3",
    )
    controller.establish_origin()
    result = controller.transfer_piece(
        TransferRequest(
            pickup_x=50.0,
            pickup_y=50.0,
            place_x=100.0,
            place_y=100.0,
            rotation_angle_deg=90.0,  # 可选
        )
    )
```

`transfer_piece()` 是同步阻塞接口：只有电磁铁已释放、Z 曾安全抬高、
XY 已回到原点并且 Z 已恢复到 `rest_z` 后才返回。调用方应在同一线程
顺序使用同一实例。

未来接入视觉时，视觉层负责：

1. 像素坐标标定成从右下角原点出发的毫米坐标；
2. 判断坐标是否在真实工作区域和无碰撞区域；
3. 提供拼图件目标角度；
4. 逐件创建 `TransferRequest`。

视觉层不应直接发 G-code、UART 帧或 GPIO 电平。

## 安装和离线自检

Python 需要 3.10 或更高版本。写字机串口使用 PySide6：

```bash
python -m pip install -r requirements.txt
python self_check.py
```

`self_check.py` 不接真实硬件，验证：

- 完整取放和旋转命令顺序；
- 省略角度时不产生旋转运动；
- 中途失败后尝试抬高、释放和回原点；
- 非法负坐标在产生运动前被拒绝。

它不能替代真实机械测试。

## 相机棋盘格标定

`camera_calibration.py` 只读取 USB 摄像头，绝不会连接写字机、无刷或电磁铁。
当前现场棋盘格已确认是 `6×4` 个内部角点（7×5 个黑白方格）。先用尺子测量
一个方格的真实边长，再运行：

```bash
.venv/bin/python \
  handover-puzzle/camera_calibration.py \
  --square-mm 25
```

`25` 必须替换成你的实测毫米数。画面中棋盘格角点变绿时按 Space 保存样本；
在画面中心、四角、四边和几张轻微倾斜的姿态各采一张，总计建议 20～30 张。
按 `C` 计算，JSON 标定参数会保存到 `calibration/camera_1280x720.json`；该文件
仅适用于 1280×720 采集分辨率。

## Z 速度、距离与保持使能诊断

不要再用完整搬运流程猜测 Z 的真实行程。使用专门脚本：

```bash
.venv/bin/python \
  handover-puzzle/test_z_calibration.py \
  --port /dev/serial/by-id/usb-1a86_USB_Serial-if00-port0
```

脚本分两阶段：

1. 同样命令抬高 2mm，分别使用 300 和 1500mm/min；
2. 固定 1500mm/min，逐级测试 1、2、3、5mm。

每轮都要求用尺子输入真实抬高毫米数。最后会判断高速、低速是否存在显著
差异，检查命令距离与真实距离是否线性，并显示每组数据推算的候选 `$102`。
脚本只计算和打印结果，绝不会自动修改 `$102`。

为获得可重复的真实机械零点，每轮之间脚本会使用本机已实测的 `$1=25`
临时释放，让 Z 自然下垂到下方限位，再恢复 `$1=255` 和 `G92 Z0`。
这段时间 X/Y 也可能失能，因此绝对不要触碰 X/Y。只有明确输入 `DOWN`
确认真实落到底、再输入 `MOVE`，脚本才会发送一条Z运动命令。

脚本硬性拒绝大于5mm的测试距离。现场已确认高速20mm和60mm会撞上方
机械限位；撞限位后听到的反复敲击通常是同一条开环命令剩余脉冲继续输出，
不代表脚本反复发送多个坐标命令。

脚本不再让主线程阻塞于 `input()`；`Ctrl+C` 会触发GRBL实时暂停字符 `!`
并退出，不会在信号清理阶段再次等待串口应答。

### Z 保持使能的含义

实机读取到 `$1=255`，说明固件已经配置为“运动后持续保持”。它不一定
表示控制板刚上电、Z 尚未产生任何运动时就主动激活 Z 驱动。

现场曾在校准流程的释放确认阶段中断，导致临时 `$1=25` 被保存在EEPROM，
下次上电后所有步进电机都可以轻松推动。现已实机写回并验证为 `$1=255`。
校准脚本现在通过 `finally` 保证取消或Ctrl+C也会恢复255；正式联动在
连接写字机后也会先显式恢复全轴保持设置。

`test_z_calibration.py` 建立 Z0 后会以实测有效的 `1500mm/min` 执行一次
`-0.2/0mm` 小动作，明确激活 Z 驱动，然后分别询问：

- 激活动作后 Z 是否锁住；
- 串口关闭前是否锁住；
- 串口关闭后是否仍锁住。

如果运动后仍不锁，需要检查 Z 驱动使能引脚或固件轴使能逻辑；如果只在
关闭串口后释放，需要检查串口 DTR/控制板复位行为。Python 程序尚未运行
时，无法保证刚上电硬件的状态；真正要求通电瞬间锁住 Z，需要固件或硬件
启动逻辑支持。

“保持使能”不等于机械抱闸：带负载的步进 Z 轴可被较大手力反驱，反驱后
GRBL 的开环坐标会失真。只有在不施力时仍会自行下坠，才应判定为没有足够
保持力矩；此时需检查 Z 驱动的 EN 接线与保持电流，必要时增加配重、减速或
机械制动，不能靠 Python/G-code 补偿。

正常拼图联动已经在最终落到 `rest_z=0` 后再次显式发送 `$1=255`，确保
程序运行期间以及正常退出前使用正确的持续保持配置。

### 单点 Z 保持测试（抬起 8mm）

如需只判断 Z 在抬起后能否保持，运行：

```bash
python test_z_hold_8mm.py --port /dev/serial/by-id/usb-1a86_USB_Serial-if00-port0
```

脚本不会移动 XY。它要求先让 Z 自然落到下方机械限位，再以实测有效的
`1500mm/min` 从 `Z0` 抬到 `Z=-8mm`，并保持串口打开供现场观察。完成后输入
`down`、确认 `LOWER` 回到 Z0，再输入 `quit`；不要在 Z 抬起时直接退出。

### 吸取、同步旋转与放件测试

`test_pick_rotate_place.py` 是本项目当前的基础真实联动测试，默认执行：

1. 人工右下角 / Z 自然下垂位置建立 `X0/Y0/Z0`；
2. 无刷先回驱动板已保存的机械零点；
3. 在 `(50, 50) mm`（5cm、5cm）落下并吸合铁片；
4. 抬起 Z 8mm；
5. XY 前往 `(100, 100) mm`（10cm、10cm）的**同时**，无刷从当前机械角度
   相对旋转 `+180°`；
6. 在 `(100, 100) mm` 落下并松开铁片，再抬起 Z、仅让 XY 回到 `(0,0)`。

运行：

```bash
python test_pick_rotate_place.py \
  --writer-port /dev/serial/by-id/usb-1a86_USB_Serial-if00-port0 \
  --motor-port /dev/ttyAS3
```

最终 Z 保持抬起位置，不会回到 `Z0`。`+180°` 是相对于无刷归零后角度的实际
半圈旋转，不是“转到绝对 180°”；实际顺/逆时针取决于安装方向。任何一次硬件
运行前均应确认铁片、工作区和人员安全。正常完成后无刷驱动板默认继续保持
使能及最后位置；只有显式传入 `--disable-motor-on-exit` 才在退出前失能。

### 实时视觉抓取测试

`test_vision_pick_place.py` 复用 `reference-vision/ec_26_e` 的蓝色 A4
透视校正和白色纸片检测。它全屏显示原始摄像头、参考工程精确拟合的 A4 四角、
校正 A4、纸片轮廓、检测置信度、抓取点以及已经转换好的写字机 X/Y 坐标；在
终端输入 `RUN` 只会冻结一片置信度合格的拼图，多个或零个候选时绝不运动。冻结画面和坐标经终端 `RUN`
二次确认后，才复用已验证的抓取/途中相对旋转 180°/放件流程。

```bash
python test_vision_pick_place.py \
  --camera /dev/v4l/by-id/usb-Generic_USB_Camera3_200901010001-video-index0 \
  --writer-port /dev/serial/by-id/usb-1a86_USB_Serial-if00-port0 \
  --motor-port /dev/ttyAS3
```

业务输入坐标固定为：蓝纸右上角 = A4 业务坐标 `(0,0)`，`X+` 向下、`Y+` 向左；
这与写字机自身机械原点不同。现场蓝纸实测短边为 `206.5 mm`、长边为
`292.0 mm`（不是标准 A4）。参考视觉检测的
横向相对坐标 `(u,v)` 使用固定轴向关系
`X=206.5*v`、`Y=292*(1-u)` 转为毫米；这不是新的相机标定。仅看检测效果、不
连接任何运动硬件时，加 `--preview-only`。截图保存在 `outputs/vision/`。

### 视觉质心抓取与 90° 旋转测试

`test_vision_centroid_pick_rotate_place.py` 在终端输入一次 `RUN` 后冻结唯一白色拼图的几何质心，再将
该毫米坐标交给已验证的取放流程：质心吸取、搬运途中相对旋转 90°、在
`(10,10) mm` 放件、抬起 Z 并使 XY 回到原点。预览阶段不连接任何硬件；只有
终端再次输入 `RUN` 才会开始真实运动。

```bash
.venv/bin/python \
  handover-puzzle/test_vision_centroid_pick_rotate_place.py
```

只核对视觉质心、不连接硬件时，增加 `--preview-only`。

批量模式的重要限制：CH340 打开串口时会产生 DTR 脉冲，可能复位 GRBL 并清除
工作坐标。因此批量视觉脚本在第一轮后保持同一个硬件工作进程和同一个写字机
串口，后续轮次绝不重新打开串口、绝不重发 `G92`。不得把批量流程改成“每片新建
进程/重新连接串口”；否则第二片可能出现坐标漂移。

## 文件说明

- `puzzle_controller.py`：公开联动接口和安全恢复；
- `test_transfer.py`：命令行真实硬件联动测试；
- `test_z_calibration.py`：Z 速度/距离对照和保持使能诊断；
- `test_z_hold_8mm.py`：抬起 8mm 后的 Z 静态保持测试；
- `test_pick_rotate_place.py`：5cm 取件、途中同步旋转 180°、10cm 放件测试；
- `test_vision_centroid_pick_rotate_place.py`：视觉质心抓取、相对旋转 90°、放到 `(10,10) mm`；
- `test_vision_pick_place.py`：全屏实时视觉检测、确认后按视觉坐标抓取测试；
- `example_usage.py`：业务代码调用示例；
- `self_check.py`：无需硬件的流程自检；
- `TEST_REPORT.md`：本版本离线及真实硬件验证记录；
- `writer_controller.py` / `grbl_controller.py`：写字机底层驱动；
- `f32c_controller.py`：F32C 无刷电机底层驱动；
- `electromagnet.py`：PL6 电磁铁底层驱动；
- `requirements.txt`：唯一第三方 Python 依赖。

## 安全边界

当前硬件没有向本流程提供“已吸到拼图件”传感器，因此程序只能确认 GPIO
已经输出高/低电平，不能证明拼图件实际吸住或已经脱落。

当前版本也没有视觉碰撞检查、实体限位确认和硬件急停。可以通过
`--workspace-x-max`、`--workspace-y-max` 设置可选的软件坐标上界，但
最终安全工作区仍需按真实机构标定。`Ctrl+C` 会触发软件恢复（尝试抬高、
释放并回原点），但串口或控制板失联时软件恢复不等同于硬件急停。

PL6 只能连接电磁铁控制器的逻辑输入，不能直接驱动电磁铁线圈；控制器输入
必须兼容 3.3 V。建议硬件输入端设置下拉，确保主机重启或 GPIO 尚未接管时
保持释放。
