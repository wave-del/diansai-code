# 写字机控制独立交付包

本目录可以单独复制给视觉同事使用，不依赖仓库中的 `reference/`、`code/`
或虚拟环境。实机固件为 `Grbl_ESP32 1.3a`，四轴双 Y，串口 115200。

> 当前拼图装置强制坐标约定（优先于本文较早的通用写字机描述）：人工将电磁铁
> 机构位于写字机自身机械原点后调用 `home()`；业务 `X+` 向下、`Y+` 向左。每次
> `move_to` / `fast_move_to` / `move_path` 都自动执行
> `X_cmd=clamp(X-23.5,0,190)`、`Y_cmd=clamp(Y+16.45,0,300)`，单位 mm。
> 业务、视觉和 AI **不得再次做偏置或限幅**，也不得绕过 `WriterController`
> 直接发送绝对 G-code。

> `move_to(0,0)` 是蓝纸业务坐标零点，仍会应用偏置，也是正常任务结束时的返回
> 位置；`return_to_machine_origin()` 仅用于上电/维护时显式回写字机自身原点。

Z 的算法侧接口是 `set_lift(lift_mm)` 和 `move_xyz(...)`：`lift_mm` 正数为
抬起、0 为自然下垂，固定使用实测可靠的 `1500 mm/min`。当前实机验证范围为
0..15 mm。

## 1. 文件说明

| 文件 | 用途 |
|---|---|
| `writer_controller.py` | 视觉业务使用的同步高层 API |
| `grbl_controller.py` | Qt 串口、GRBL 协议、坐标变换和状态解析 |
| `example_usage.py` | 最小接入示例 |
| `coordinate_console.py` | 在终端交互输入用户 X/Y/Z 坐标 |
| `test_queue.py` | XY、队列、错误传播自动化验收 |
| `test_pen.py` | 可调 Z 高度抬笔/落笔验收 |
| `test_absolute.py` | 用户绝对坐标往返验收 |
| `calibrate_direction.py` | 人工确认 X/Y 用户坐标方向 |
| `test_hold.py` | 可选的 `$1` 电机保持人工测试 |
| `firmware_queue_probe.py` | 原始串口及 40 条盲发压力探针 |
| `requirements.txt` | 已验证的 Python 依赖 |

## 2. 安装

推荐 Python 3.11：

```bash
python3 -m venv .venv
.venv/bin/pip install -r handover/requirements.txt
```

Linux 用户需有串口权限；本机设备为 `/dev/ttyUSB0`。

## 3. 每次上电的固定流程

本机没有可靠的自动回零传感器。每次断电、控制板复位或重新开始任务时：

1. 人工把 XY 机构放到约定的**最右下角机械位置**；
2. 人工把笔放到约定的**抬笔基准高度**；
3. 给设备上电并连接串口；
4. 调用一次 `controller.home()`。

这里的 `home()` **不会执行 `$H`，也不会让机器自行寻找限位**。它执行解锁、
`$1=255` 和 `G92 X0 Y0 Z0`：

- 把人工放置的位置定义为本次运行的用户原点；
- 设置运动结束后步进电机持续使能，机构受到轻微扰动或手推时保持位置。

按本次交付约定，接口不强制拦截“未调用 `home()` 就运动”，因为现场流程会保证
人工归位。调用方仍必须遵守以上顺序。

## 4. 坐标约定

- 原点：人工确定的最右下角；
- 用户 X+：从右下角向左进入工作区；
- 用户 Y+：从右下角向上进入工作区；
- Z0：人工确定的抬笔基准高度；
- 默认落笔：Z2.0mm。

`move_to()` 和 `move_path()` 自动完成物理 X/Y 对调和方向反转。不要在视觉侧再次
交换或取反。

这是开环步进系统。返回坐标是 GRBL 的步数坐标，不是编码器测得的真实位置；撞限位
或机械丢步后必须重新人工归位并调用 `home()`。

## 5. 最小接入

同目录运行：

```python
from writer_controller import WriterController

ctrl = WriterController()
ctrl.connect(port="/dev/ttyUSB0")

# 人工归位后，每次上电调用一次
ctrl.home()

ctrl.configure_pen(up_z=0.0, down_z=2.0, speed=3000)
ctrl.pen_up()

# x/y 是用户绝对坐标，单位 mm；feedrate 单位 mm/min
ctrl.move_to(x=20.0, y=10.0, feedrate=2500)
ctrl.move_path(
    points=[(30.0, 10.0), (30.0, 20.0)],
    feedrate=2500,
)

ctrl.pen_down()
ctrl.pen_up()
ctrl.move_to(x=0.0, y=0.0)
ctrl.disconnect()
```

若把整个目录作为 Python 包放在项目中，也可以：

```python
from handover import WriterController
```

完整示例见 `example_usage.py`。

交付示例统一使用关键字参数，建议视觉业务也保持这种写法，避免把“位移”和“速度”
两个同为数字的参数写反。代码中也可以随时查看，例如：

```python
help(WriterController.pen_jog)
```

## 6. API

### 单位与符号总表

| 名称 | 含义 | 单位/取值 |
|---|---|---|
| `x`, `y` | 用户绝对 XY 坐标 | mm |
| `dx`, `dy` | 相对当前位置的 XY 位移 | mm |
| `z` | 相对于人工抬笔基准 Z0 的绝对笔高 | mm |
| `dz` | 相对当前笔高的 Z 位移 | mm |
| `feedrate` | 轨迹进给速度，不是位移 | mm/min |
| `baud` | 串口传输速率 | bit/s |
| `points` | 顺序执行的用户绝对坐标点 | `[(x_mm, y_mm), ...]` |

方向规定：

- `x > 0` 向左，`x < 0` 向右；
- `y > 0` 向上，`y < 0` 向下；
- `dz > 0` 笔向下、接近纸面，`dz < 0` 笔向上；
- `feedrate=200` 表示每分钟 200mm，即约 3.33mm/s，不是移动 200mm。

### 连接与原点

| API | 参数 | 返回/阻塞行为 |
|---|---|---|
| `connect(port, baud=115200)` | `port`：串口名；`baud`：bit/s | 启动握手和状态读取成功后返回 |
| `home()` | 无 | 执行 `$X`、`$1=255`、`G92 X0 Y0 Z0`；不会移动机器 |
| `disconnect()` | 无 | 关闭串口；不是急停 |
| `is_connected()` | 无 | `bool` |

### XY

| API | 参数及物理单位 | 返回/阻塞行为 |
|---|---|---|
| `move_to(x, y, feedrate=2000)` | `x/y`：绝对位置 mm；`feedrate`：mm/min | 到达目标并真实 `Idle` 后返回 |
| `move_path(points, feedrate=2000)` | `points`：绝对 `(x,y)` mm 序列；速度 mm/min | 整条连续路径完成后返回 |
| `jog(dx, dy, feedrate=2500)` | `dx/dy`：相对位移 mm；速度 mm/min | 相对运动完成后返回 |
| `fast_move_to(x, y)` | `x/y`：绝对位置 mm | G0 完成后返回；速度由固件决定 |
| `position` | 属性，无参数 | 返回 `(x_mm, y_mm)` |
| `is_busy` | 属性，无参数 | 返回 `bool` |
| `machine_state` | 属性，无参数 | 返回 `Idle / Run / Hold / Alarm ...` |

例如：

```python
# 以 2500mm/min 移动到用户绝对坐标 X=20mm、Y=10mm。
ctrl.move_to(x=20.0, y=10.0, feedrate=2500)

# 从当前位置向左相对移动 2mm、向上相对移动 1mm，速度 600mm/min。
ctrl.jog(dx=2.0, dy=1.0, feedrate=600)
```

### Z 与笔高度

```python
ctrl.configure_pen(
    up_z=0.0,       # 抬笔绝对位置：Z=0.0mm
    down_z=2.0,     # 落笔绝对位置：Z=2.0mm
    speed=3000,     # 默认笔轴速度：3000mm/min
)

ctrl.pen_up()                              # 移动到配置的 Z=0.0mm
ctrl.pen_down()                            # 移动到配置的 Z=2.0mm
ctrl.pen_down(z=1.5, feedrate=500)         # 本次移动到 Z=1.5mm，速度 500mm/min
ctrl.pen_move_to(z=0.8, feedrate=300)      # 移动到绝对 Z=0.8mm
ctrl.pen_jog(dz=0.1, feedrate=200)         # 从当前位置向下 0.1mm，速度 200mm/min
ctrl.pen_jog(dz=-0.1, feedrate=200)        # 从当前位置向上 0.1mm，速度 200mm/min
z_mm = ctrl.pen_position                   # 当前绝对笔高，单位 mm
```

`configure_pen()` 只修改本次 Python 进程的参数，不写 GRBL EEPROM。

Z API 明细：

| API | 参数及物理单位 | 行为 |
|---|---|---|
| `configure_pen(up_z, down_z, speed)` | 高度 mm；速度 mm/min | 只保存参数，不移动 |
| `pen_up(z=None, feedrate=None)` | 可选绝对高度 mm、速度 mm/min | 省略时使用已配置抬笔参数 |
| `pen_down(z=None, feedrate=None)` | 可选绝对高度 mm、速度 mm/min | 省略时使用已配置落笔参数 |
| `pen_move_to(z, feedrate=None)` | `z`：绝对 mm；速度 mm/min | 移动到任意绝对笔高 |
| `pen_jog(dz, feedrate=300)` | `dz`：相对 mm；速度 mm/min | 正数向下，负数向上 |
| `pen_position` | 属性，无参数 | 返回当前绝对 Z，单位 mm |
| `pen_up_height` | 属性，无参数 | 返回配置的抬笔 Z，单位 mm |
| `pen_down_height` | 属性，无参数 | 返回配置的落笔 Z，单位 mm |
| `pen_speed` | 属性，无参数 | 返回配置的笔轴速度，单位 mm/min |

### 原始 G-code

| API | 行为 |
|---|---|
| `send_gcode(cmd)` | 原始固件坐标；等待应答和 Idle |
| `send_gcode_nowait(cmd)` | 加入主机流队列，每收到一条 `ok` 再发下一条 |
| `wait_idle()` | 等全部应答和新的 `Idle` |

业务坐标优先使用 `move_to()`/`move_path()`；原始 G-code 不应用用户坐标变换。

### 电机空闲保持

```python
ctrl.set_motor_hold(enable=True)   # 写入 $1=255：停止后仍保持使能
ctrl.set_motor_hold(enable=False)  # 写入 $1=25：停止 25ms 后失能
```

`enable` 是布尔值，不是时间；本方法会修改 GRBL EEPROM。`home()` 已自动设置
`$1=255`，所以正常业务不需要再调用。只有维护时明确希望运动结束后可以手推机构，
才调用 `set_motor_hold(enable=False)`。

## 7. 队列机制结论

不需要等一个坐标完整运动结束后再发送下一个。生产接口采用：

1. 当前 G-code 收到 `ok`；
2. 立即发送下一条，不等待当前运动结束；
3. 全部命令 `ok` 收齐后，请求新的状态；
4. 收到 `Idle` 才向业务层返回。

2026-07-31 实机结果：

- 20 段路径：20/20 个 `ok`，终点 X20.000mm；
- 30 段 XY 折线：全部完成，终点坐标偏差 0.000mm；
- 一次盲发 40 条、920 字节：40/40 个 `ok`，终点 X40.000mm；
- `G999` 返回 `error:20`，高层正确抛异常，随后正常命令可继续；
- 连续重连 3 次均成功。

盲发成功不代表任意长度都能无流控发送；生产代码坚持使用 `ok` 流控。

## 8. 验收命令

### 交互输入 X/Y/Z 坐标

```bash
.venv/bin/python handover/coordinate_console.py --port /dev/ttyUSB0
```

人工归位并按 Enter 后，可直接输入：

```text
20 10 1.5
```

三个数字依次表示：

- 用户绝对 X=20mm；
- 用户绝对 Y=10mm；
- 笔轴绝对 Z=1.5mm。

控制台会按“抬笔 → 移动 XY → 移动 Z”的安全顺序执行，然后显示最终坐标。
还支持 `pos`、`up`、`down`、`z 0.5`、`jogz 0.1`、`origin`、`help` 和
`quit`。运行 `help` 会再次显示每个命令的单位与方向。

### 不连接硬件的安装自检

```bash
.venv/bin/python handover/self_check.py
```

### XY 和队列自动化验收

确认当前位置有 X30mm、Y10mm 余量：

```bash
.venv/bin/python handover/test_queue.py --port /dev/ttyUSB0 --auto
```

结束后自动回测试原点，不移动 Z。

### Z 轴验收

先把笔人工放在抬笔基准高度：

```bash
.venv/bin/python handover/test_pen.py \
  --port /dev/ttyUSB0 --up 0 --down 2 --speed 3000
```

若只想小行程验证，可用 `--down 0.5 --cycles 1`。

### 固件 40 条盲发探针

```bash
.venv/bin/python handover/firmware_queue_probe.py \
  --port /dev/ttyUSB0 --blind
```

## 9. 本次明确不做的保护

根据当前项目约定：

- 不强制阻止未 `home()` 的移动；
- 不在控制层设置 XY/Z 软件边界；
- 不提供业务急停/暂停按钮；
- 不实现跨线程调用。

调用规则是：**同一线程、同步调用**。如果未来视觉程序必须从工作线程控制设备，
再增加一个专用控制线程和消息队列，不要让多个线程同时操作同一个控制器。

## 10. 异常处理

所有同步运动都可能抛出：

- `RuntimeError`：未连接、GRBL `error/ALARM`、串口断线；
- `TimeoutError`：启动、命令应答或运动完成超时；
- `ValueError`：无效的笔高度或速度参数。

业务入口建议使用 `try/finally`，确保最后调用 `disconnect()`。关闭串口不是急停；
已经进入固件 planner 的运动可能继续执行。
