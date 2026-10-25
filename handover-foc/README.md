# F32C 无刷电机与电磁铁 Python 独立交付包

本目录可单独复制使用。它通过 Radxa Cubie A7Z 的本机 UART3，直接控制
WHEELTEC F32C 无刷云台电机驱动板，不依赖 `reference-foc/`、MSPM0 固件、
PySide 或 pyserial；同时通过PL6 / Pin 13控制高电平有效的电磁铁控制器。

交付版本：**1.1.0**。已于 **2026-07-31** 在真实 Cubie A7Z、UART3 和
地址为2的F32C上完成连续绝对定位、机械回零、启动保持及反馈丢包恢复验证。

## 快速开始

先做不运动的离线自检：

```bash
python3 self_check.py
```

再做真实硬件通信与保持检查。该步骤会以1 RPM安全使能、立即保持当前位置，
读取反馈后自动失能，不发送业务角度目标：

```bash
python3 hardware_probe.py --port /dev/ttyAS3 --address 2
```

最后启动角度测试台。程序会自动安全使能，随后直接输入 `0～360`：

```bash
python3 test_motor.py --port /dev/ttyAS3 --rpm 15
```

```text
angle> 30
angle> 90
angle> zero
angle> q
```

电磁铁测试台启动时默认输出低电平：

```bash
python3 test_electromagnet.py
```

```text
magnet> on
magnet> off
magnet> pulse 0.5
magnet> q
```

## 算法侧旋转接口

`F32CMotor.move_to(angle_deg)` 是机械绝对角度，范围 `0 <= angle < 360`，自动
按最短路径转动。`return_to_mechanical_zero()` 回到驱动板保存的机械零点。
新增 `rotate_relative(delta_deg)` 支持 `-180..+180°` 的相对角度，尤其会保留
`+180°` 与 `-180°` 的方向差异；实际顺/逆时针由电机安装方向决定。

## 1. 交付内容

| 文件 | 用途 |
|---|---|
| `f32c_controller.py` | F32C 协议、Linux UART 和同步高层 API |
| `electromagnet.py` | PL6 / Pin 13电磁铁同步控制API |
| `__init__.py` | 将目录改成合法包名后使用的导出入口 |
| `example_usage.py` | 视觉/业务程序最小接入示例 |
| `motor_console.py` | 显式使能/失能的高级交互控制台 |
| `hardware_probe.py` | 安全使能保持、查询反馈和漂移检查 |
| `test_motor.py` | 启动自动使能、直接输入0～360°的真实硬件测试台 |
| `test_electromagnet.py` | 电磁铁吸合、松开、定时脉冲测试台 |
| `self_check.py` | 不连接硬件的协议和安装自检 |
| `requirements.txt` | 依赖说明；当前无第三方依赖 |
| `CHANGELOG.md` | 版本及真实硬件验证记录 |

运行环境：Linux、Python 3.10 或更高版本。

## 2. 串口和接线

本机固定使用：

```text
/dev/ttyAS3
115200 bit/s, 8 data bits, no parity, 1 stop bit, no flow control
F32C address = 2
```

Cubie A7Z 已启用 `sun60iw2p1-uart3.dtbo`：

| Cubie A7Z | 连接到 F32C |
|---|---|
| 排针 Pin 5，UART3 TX | F32C RX |
| 排针 Pin 3，UART3 RX | F32C TX |
| 任一确认过的 GND | F32C GND |

电气要求：

- UART 是 **3.3V TTL**，不是 RS-232 或 RS-485；
- Cubie A7Z 和 F32C 必须共地；
- 电机使用独立的 8～15V 电源，建议稳定 12V；
- 不要从开发板 UART 或 USB 给电机供电；
- 接线和断线应在断电状态完成；
- Python 直连时，必须断开其他控制器到 F32C RX 的 TX 线，避免两个 TX
  同时驱动一根线。

摄像头使用 `/dev/video*`，写字机使用 USB CH340，F32C 使用 SoC
`/dev/ttyAS3`，三者不会因编号发生混淆。

电磁铁控制接线：

| Cubie A7Z | 电磁铁控制器 |
|---|---|
| 物理Pin 13，PL6 | 控制信号输入 |
| 任一确认过的GND | 控制器信号GND |

本机映射已确认：`PL6 / Pin 13 = /dev/gpiochip1 line 6`，内核名称为
`PIN_13`。高电平吸合，低电平松开。

注意：

- Pin 13只能连接电磁铁**控制器的逻辑输入**，不能直接驱动电磁铁线圈；
- 必须确认控制器输入兼容3.3V，并与Cubie A7Z共地；
- 推荐控制器输入端有硬件下拉电阻。软件会在关闭GPIO句柄前输出低电平，
  但句柄释放后引脚不再由本进程持有，最终方向和电平不能依赖软件状态；
  断电、崩溃或句柄释放后的安全松开仍应由硬件下拉保证。

## 3. 零点和角度语义

F32C 同时存在两个坐标系：

1. **上电多圈坐标**：驱动板每次上电时的位置为 0°，掉电后重新建立；
2. **持久机械坐标**：此前用 `0x0A` 保存到驱动板，掉电后仍保留。

本交付 API 的公开角度统一使用第二种，也就是已经保存的机械零点：

- `move_to(0)`：到持久机械零点；
- `move_to(90)`：到机械绝对 90°；
- 普通目标满足 `0 <= angle < 360`；
- 普通绝对定位总是选择最短路径；
- `360°` 与 `0°` 是相同机械方向，因此普通接口不接收 `360`；
- 需要真实完整旋转一圈时，调用 `rotate_one_turn()`。

`return_to_mechanical_zero()` 不是简单发送多圈目标 0°。实现会读取当前
持久机械角度和上电多圈角度，计算最短有符号位移，再生成正确的内部多圈目标。

本交付包**不提供设置/覆盖机械零点的公共接口**，防止同事误删已经标定好的零点。

当前硬件实测，持久机械角度与协议多圈角度的正方向相反，因此构造函数默认
`mechanical_direction=-1`。业务代码仍只使用直观的机械绝对角度，不需要自行
取反。若未来更换驱动器配置并确认两个坐标同向，才应显式改为 `1`。

## 4. 最小 API 示例

在本目录同级运行：

```python
from f32c_controller import F32CMotor

motor = F32CMotor(
    port="/dev/ttyAS3",
    address=2,
    max_rpm=30,  # 高层安全上限；参考工程实际使用15 RPM
)

try:
    # 只打开串口；不会使能，也不会运动。
    motor.connect()

    # 低速使能后立刻保持当前位置，清除残留目标，再设置15 RPM。
    motor.enable(rpm=15)

    # 以15 RPM沿最短路径到持久机械90°，确认真实反馈到位后返回。
    motor.move_to(
        angle_deg=90.0,  # 单位：度（°）
        rpm=15,          # 单位：转/分钟（RPM）
        wait=True,
    )

    # 沿最短路径回驱动板内保存的持久机械零点。
    motor.return_to_mechanical_zero(wait=True)
finally:
    # 默认先失能，再关闭 UART3。
    motor.close(disable=True)
```

如将目录改名为合法的 Python 包名，也可以通过 `__init__.py` 导入。当前目录名
含连字符，最直接的方式是在目录内运行业务程序并从 `f32c_controller` 导入。

## 5. API 说明

| API | 参数和单位 | 行为 |
|---|---|---|
| `connect(verify=False)` | 可选只读验证 | 打开 UART；默认不查询、不使能、不运动 |
| `enable(rpm=15)` | `rpm`：1～`max_rpm`整数 | 以1 RPM使能并保持当前位置、清除残留目标，再设置请求速度 |
| `disable()` | 无 | 立即发送失能命令 |
| `set_speed(rpm)` | 1～`max_rpm`整数 | 修改后续位置速度 |
| `get_angle()` | 无 | 返回持久机械单圈角度 `[0, 360)`，单位° |
| `get_total_angle()` | 无 | 返回本次上电有符号多圈角度，单位° |
| `get_status()` | 无 | 返回端口、地址、使能、速度和两种角度字典 |
| `move_to(angle_deg, rpm=None, wait=True)` | 机械绝对角度°；速度RPM | 最短路径定位，默认阻塞到位 |
| `return_to_mechanical_zero(...)` | 可选RPM、等待和超时 | 最短路径回持久机械零点 |
| `rotate_one_turn(direction=1, ...)` | `1`=协议正向，`-1`=协议负向 | 从当前位置完整旋转一圈 |
| `close(disable=True)` | `disable`：布尔值 | 默认先失能再关闭UART |

实际顺时针/逆时针取决于安装方向，因此整圈接口使用“协议正/负方向”描述。

重要构造参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `port` | `/dev/ttyAS3` | Linux UART设备 |
| `address` | `2` | F32C串口地址 |
| `max_rpm` | `30` | 高层API允许的速度上限，单位RPM |
| `mechanical_direction` | `-1` | 机械角度相对多圈角度的方向；本机实测反向 |
| `feedback_timeout` | `0.3` | 单次反馈等待秒数 |
| `feedback_attempts` | `3` | 单次查询最多尝试次数 |

到位判定默认要求：

- 总角度误差不超过 1°；
- 连续三次反馈满足条件；
- 随后再读取持久机械角度做二次确认。

## 6. 异常与安全行为

接口可能抛出：

| 异常 | 含义 |
|---|---|
| `ValueError` | 角度、速度、地址或超时参数非法 |
| `F32CConnectionError` | 串口不存在、被占用、断线或读写失败 |
| `F32CProtocolError` | 帧头、地址、帧尾、BCC、长度或机械角度不合法 |
| `F32CTimeoutError` | 没有反馈或运动未在期限内到位 |
| `F32CError` | 例如未使能就请求运动 |

安全策略：

- `connect()` 不查询、不使能、不运动；因为实测失能状态可能不返回反馈；
- `enable()` 会先预设1 RPM，使能后立刻发送当前位置保持目标以清除残留目标；
- 协议可编码至1000 RPM，但高层默认 `max_rpm=30`；提高上限必须在构造函数
  中显式声明；
- 运动过程发生通信异常、反馈异常或到位超时时，尽力发送失能命令；
- 同一 UART 使用内核独占和文件锁，防止本交付程序重复打开；
- `close()` 默认失能；
- 没有“设置机械零点”命令；
- 控制命令间隔不小于 2ms；
- 反馈解析支持分包、粘包、噪声重同步和 BCC 校验。
- 单次角度查询默认最多尝试3次，偶发丢失一帧不会中止正常运动。

失能不等同于机械急停。如果设备涉及负载坠落、夹伤或线缆缠绕，仍需独立的硬件
断电和机械防护。

## 7. 电磁铁API

最小示例：

```python
from electromagnet import ElectromagnetController

magnet = ElectromagnetController()
try:
    magnet.open()       # 原子地初始化为低电平/松开
    magnet.engage()     # 高电平/吸合
    magnet.release()    # 低电平/松开
    magnet.pulse(0.5)   # 吸合0.5秒，finally中自动松开
finally:
    magnet.close()      # 再次尝试低电平，然后释放GPIO
```

| API | 行为 |
|---|---|
| `open()` | 独占`gpiochip1 line 6`并初始化为低电平 |
| `engage()` | 高电平，电磁铁吸合 |
| `release()` | 低电平，电磁铁松开 |
| `set_engaged(bool)` | 按逻辑状态设置并读取GPIO复核 |
| `is_engaged` | 读取当前逻辑吸合状态 |
| `pulse(seconds)` | 吸合指定秒数，最终自动松开 |
| `close(release=True)` | 默认先低电平，再释放GPIO句柄 |

实现直接使用Linux GPIO字符设备，不需要Python `gpiod`包，也不会启动后台
`gpioset`进程。同一时刻只能有一个进程申请该GPIO。

## 8. 验收流程

### 8.1 离线自检

不打开 UART、不发送任何命令：

```bash
python3 handover-foc/self_check.py
```

### 8.2 硬件通信与保持探测

当前F32C在失能状态可能不返回反馈。探测脚本会预设1 RPM、安全使能并立即
保持当前位置，连续读取反馈、检查漂移，最后自动失能；不会发送业务角度目标：

```bash
python3 handover-foc/hardware_probe.py \
  --port /dev/ttyAS3 \
  --address 2
```

### 8.3 F32C交互控制台

首次接上真实电机时，可以运行角度测试台：

```bash
python3 handover-foc/test_motor.py \
  --port /dev/ttyAS3 \
  --address 2 \
  --rpm 15
```

脚本启动后会直接打开UART3、使能电机并进入多圈T型位置模式。看到
`angle>` 提示符后直接输入 `0～360` 范围内的机械绝对角度，例如
`30`、`90`、`180`。`360°` 与机械 `0°` 是同一方向；需要真实完整
旋转一圈时输入 `turn 1` 或 `turn -1`。脚本不会设置或覆盖机械零点。

首次验收通过后，再使用交互控制台：

```bash
python3 handover-foc/motor_console.py \
  --port /dev/ttyAS3 \
  --address 2 \
  --rpm 15
```

连接后先执行：

```text
enable 15
status
move 10
zero
disable
quit
```

首次运动建议从 5～10°开始，确认协议正方向、机械安装、线缆余量和持久机械零点
均符合预期后，再测试更大角度或 `turn 1`。

### 8.4 电磁铁测试台

```bash
python3 handover-foc/test_electromagnet.py
```

启动后输出低电平。输入`on`吸合、`off`松开、`pulse 0.5`短时吸合，
输入`q`或按Ctrl+C时恢复低电平并退出。

## 9. 线程和进程规则

- 一个电机只创建一个 `F32CMotor`；
- 同一业务线程顺序调用同步 API；
- 不要让其他程序同时打开 `/dev/ttyAS3`；
- 若未来必须异步控制，应由一个专用电机线程独占对象，其他线程通过消息队列发任务。
- 一个电磁铁只创建一个`ElectromagnetController`，并在整个控制周期保持对象打开。

## 10. 与参考工程的区别

`reference-foc/wit-rotate` 是 MSPM0G3507 裸机示例。其当前 `empty.c` 上电后会
自动使能并转到上电多圈 360°，PA12 也只回上电零点，不回持久机械零点。

本 Python 交付包不会复用这些启动行为，而是直接控制 F32C：

- 启动不运动；
- 公开坐标统一基于持久机械零点；
- 支持任意时刻正确回持久机械零点；
- 所有业务运动都有反馈、到位和超时处理。

## 11. 已验证结果与交付边界

真实硬件验证结果：

| 项目 | 结果 |
|---|---|
| 安全使能后保持 | 多圈漂移0.0°，机械漂移0.0° |
| 机械114.9° → 124.9° | 反馈124.7° |
| 返回124.9° → 114.9° | 反馈约起点，正常 |
| 连续120° → 80° → 100° | 反馈119.1°、80.8°、99.0° |
| 回持久机械零点 | 反馈0.7° |
| 单帧反馈丢失 | 自动重试恢复 |
| 退出 | 发送失能并释放UART3 |
| PL6启动状态 | 低电平读取复核通过 |
| PL6高电平脉冲 | 0.3秒写入成功，随后恢复低电平 |
| PL6退出 | GPIO句柄释放，line 6未占用且输入读取为0 |

边界说明：

- 本交付不设置机械零点，默认驱动板中已经正确保存；
- 没有软件角度行程限制、机械限位检测或独立急停；
- `360°`在测试台中等同于机械`0°`；真实整圈使用`turn 1/-1`；
- 同一时刻只能有一个进程占用`/dev/ttyAS3`；
- `move_to()`是同步接口，默认确认真实到位后才返回；
- 程序运行期间电机保持使能；正常退出或异常运动失败时会尝试失能；
- 整圈旋转前必须确认负载、线缆和机构允许完整旋转。
- 电磁铁API不限制持续吸合时长；允许持续通电前必须确认线圈和控制器的温升能力；
- GPIO软件退出前会输出低电平，但异常掉电后的松开状态必须由控制器输入端硬件下拉保证。
