# PL6 电磁铁 Python 独立交付包

本目录可单独复制使用，不依赖项目中的其他目录。它使用Cubie A7Z的
PL6 / 物理Pin 13控制高电平有效的电磁铁控制器。

交付版本：**1.0.0**。运行环境：Linux、Python 3.10或更高版本。
仅使用Python标准库，不需要安装Python `gpiod`。

## 快速测试

先运行不访问硬件的离线自检：

```bash
python3 self_check.py
```

再启动真实电磁铁测试台：

```bash
python3 test_electromagnet.py
```

进入`magnet>`后输入：

```text
on          高电平，保持吸合
off         低电平，松开
pulse 0.5   高电平吸合0.5秒，然后自动松开
status      查看当前状态
q           恢复低电平并退出
```

完整路径运行示例：

```bash
.venv/bin/python \
handover-electromagnet/test_electromagnet.py
```

## 交付内容

| 文件 | 用途 |
|---|---|
| `electromagnet.py` | Linux GPIO和电磁铁同步控制API |
| `test_electromagnet.py` | 命令行交互测试台 |
| `example_usage.py` | 业务程序最小接入示例 |
| `self_check.py` | 不访问硬件的伪GPIO状态机自检 |
| `__init__.py` | Python包导出入口 |
| `requirements.txt` | 依赖说明 |
| `CHANGELOG.md` | 版本与验收记录 |

## 硬件映射与接线

本机已经确认：

```text
物理排针：Pin 13
SoC引脚：PL6
Linux设备：/dev/gpiochip1
line offset：6
内核名称：PIN_13
有效电平：高电平吸合，低电平松开
```

| Cubie A7Z | 电磁铁控制器 |
|---|---|
| Pin 13 / PL6 | 控制器逻辑输入 |
| 任一确认过的GND | 控制器信号GND |

安全要求：

- Pin 13只能连接控制器的逻辑输入，不能直接连接电磁铁线圈；
- 控制器逻辑输入必须兼容3.3V；
- Cubie A7Z和电磁铁控制器必须共地；
- 推荐控制器输入端配置硬件下拉电阻；
- 软件关闭句柄前会输出低电平，但崩溃、掉电或句柄释放后的最终安全状态
  必须由硬件下拉保证；
- 持续吸合前必须确认电磁铁线圈和控制器允许长期通电及对应温升。

当前用户必须能够读写`/dev/gpiochip1`。本机用户已经属于`gpio`组。

## Python API

```python
from electromagnet import ElectromagnetController

magnet = ElectromagnetController(
    chip="/dev/gpiochip1",
    line_offset=6,
)

try:
    magnet.open()       # 原子地初始化为低电平/松开
    magnet.engage()     # 高电平/吸合
    magnet.release()    # 低电平/松开
    magnet.pulse(0.5)   # 吸合0.5秒，最终自动松开
finally:
    magnet.close()      # 再次尝试低电平并释放GPIO
```

| API | 参数和单位 | 行为 |
|---|---|---|
| `open()` | 无 | 独占PL6并初始化为低电平 |
| `engage()` | 无 | 高电平，电磁铁吸合 |
| `release()` | 无 | 低电平，电磁铁松开 |
| `set_engaged(engaged)` | 布尔值 | 设置逻辑吸合状态并读取复核 |
| `is_engaged` | 属性 | 返回当前逻辑吸合状态 |
| `pulse(duration_seconds)` | 秒 | 吸合指定时间，`finally`中松开 |
| `close(release=True)` | 布尔值 | 默认先低电平，再释放GPIO句柄 |

也可以使用上下文管理器：

```python
from electromagnet import ElectromagnetController

with ElectromagnetController() as magnet:
    magnet.engage()
    # 执行业务动作
    magnet.release()
```

## 行为与异常

- `open()`使用默认低电平原子申请GPIO，避免启动瞬间误吸合；
- 每次设置后读取GPIO状态复核；
- 同一时刻只能有一个进程申请line 6；
- 对象打开期间持续持有GPIO，输出状态不会依赖后台进程；
- `pulse()`无论正常还是发生Python异常，都会尝试松开；
- `test_electromagnet.py`在正常退出、EOF和Ctrl+C时都会尝试低电平。

异常类型：

| 异常 | 含义 |
|---|---|
| `ValueError` | line offset、布尔参数或脉冲时长非法 |
| `ElectromagnetConnectionError` | GPIO不存在、无权限、被占用或映射不符 |
| `ElectromagnetStateError` | GPIO写入或读取复核失败 |
| `ElectromagnetError` | 电磁铁错误基类 |

## 真实硬件验收

2026-07-31在本机完成：

| 项目 | 结果 |
|---|---|
| GPIO映射 | `gpiochip1 line 6 = PIN_13` |
| 启动状态 | 低电平读取复核通过 |
| 高电平脉冲 | 0.3秒写入成功 |
| 脉冲结束 | 恢复低电平 |
| 句柄释放 | line 6恢复未占用，输入读取为0 |
| 离线自检 | 低→高→低→脉冲→低状态序列通过 |

软件能够确认GPIO电平切换。实际电磁铁是否产生吸合动作，需要操作者现场通过
声音或机构动作确认。
