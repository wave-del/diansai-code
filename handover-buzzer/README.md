# handover-buzzer — 蜂鸣器

低电平触发有源蜂鸣器的驱动与上层封装，供手递（handover）装置做操作
反馈提示音（按钮按下、任务完成等）。

## 硬件接线

| 引脚 | 信号 | gpiochip | 线号 | 说明 |
|------|------|----------|------|------|
| 32   | PM5  | gpiochip1（R_PIO） | 37 | 蜂鸣器控制脚，**低电平触发** |

- 蜂鸣器控制脚接引脚 32（PM5）；电源端（VCC/GND）按模块要求接 3.3V/5V 与 GND
- **PM5 与 3.3V 之间接 10kΩ 上拉电阻**（已接）

### 为什么必须加上拉电阻

PM5 位于 R_PIO 控制器（`7025000.pinctrl`），上电复位后引脚不受软件控制，
且固件未对其做任何 pinmux 配置。低电平触发的蜂鸣器在上电瞬间若引脚电平
不受控，就会"一直滴滴叫"。

软件方案（脚本运行时配上拉）只能覆盖"脚本运行期间"，覆盖不了上电 →
内核 boot → 脚本启动这段窗口；设备树 overlay 虽能覆盖内核 boot 之后，
但工作量大。**硬件上拉电阻是唯一从上电第一个微秒起就保证高电平（静音）
的方案**，故采用。

## 依赖

```bash
.venv/bin/pip install python-periphery
```

## 测试脚本

```bash
.venv/bin/python gpio_buzzer_test.py
```

测试内容：
1. 释放状态下引脚应为高电平（静音）——验证上拉电阻
2. 长响 1 秒
3. 短响 3 声
4. 完成提示 5 声短响

## 上层封装接口（Buzzer）

供上层模块（handover-puzzle / handover-writer 等）直接 import 使用，
沿用仓库 `sys.path.insert` 惯例：

```python
import sys
sys.path.insert(0, "<仓库根>/handover-buzzer")   # 按实际克隆位置替换
from buzzer import Buzzer

with Buzzer() as buzzer:      # 退出时自动恢复静音并释放 GPIO
    buzzer.beep()             # 短响一声 (0.1s)
    buzzer.beep(0.5, 3)       # 0.5s/声，间隔 0.2s，共 3 声
    buzzer.on()               # 持续响
    buzzer.off()              # 静音
```

### API

| 方法 | 说明 |
|------|------|
| `Buzzer(chip=CHIP, line=LINE)` | 打开 GPIO（direction `"high"`：输出方向且打开瞬间即高电平静音，无启动杂音） |
| `on()` | 持续响（输出低电平） |
| `off()` | 静音（输出高电平） |
| `beep(duration=0.1, times=1, interval=0.2)` | 按节奏响若干次（**阻塞式**） |
| `close()` | 恢复静音并释放 GPIO |
| `with Buzzer() as b:` | 上下文管理器，退出自动释放 |

### 注意事项

- `beep()` 是阻塞式的，会占用当前线程直到响完；需要非阻塞场景时用
  `on()`/`off()` 自行配合定时器控制。
- 引脚在程序退出/释放后会回到上拉电阻决定的高电平（静音），无需额外处理。
- 同一时刻只有一个进程能占用该 GPIO（内核会拒绝重复打开并报错），
  上层模块之间注意互斥。
