# handover-button — 按钮 GPIO 测试

用 [python-periphery](https://github.com/vsergeev/python-periphery) 读取三个按键，
用于手递（handover）装置的按钮功能测试。

## 接线（40 pin 排针）

| 物理引脚 | 信号 | gpiochip0 线号 | 说明 |
|---------|------|----------------|------|
| 36      | PB4  | 36             | 按钮1 |
| 38      | PB8  | 40             | 按钮2 |
| 40      | PB7  | 39             | 按钮3 |

按钮另一端接 GND（按下时拉低）。默认悬空读高电平，按下读低电平。

内部上拉由脚本在每次运行时自动配置（`bias="pull_up"`），无需改设备树；
上拉随脚本运行重新生效，按下时按键拉 GND 仍能稳定读到低电平。

> 线号说明：官方示例中 PB7 对应 `/dev/gpiochip0` 线 39，
> GPIOB 引脚连续排列，故 PB4=36、PB8=40。

## 安装依赖

```bash
.venv/bin/pip install python-periphery
```

## 运行

```bash
.venv/bin/python gpio_button_test.py
```

启动时打印三个引脚的初始电平；按下/松开时分别打印 `[1]` / `[2]` / `[3]`
（含物理引脚与信号名）以区分按钮。Ctrl+C 退出。

软件消抖：电平变化后在 20ms 窗口内每 2ms 连续采样，全部一致才确认，
避免机械抖动导致的重复打印。
