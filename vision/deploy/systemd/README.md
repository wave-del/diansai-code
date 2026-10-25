# EC26E实体按键服务

当前A7Z设备通过内核GPIO line名称确认以下映射：

| 任务 | 物理引脚 | GPIO字符设备 | line | 有效电平 |
|---|---:|---|---:|---|
| Task1 | 35 | `/dev/gpiochip0` | 38 | 按下接GND，低有效 |
| Task2 | 37 | `/dev/gpiochip1` | 36 | 按下接GND，低有效 |

物理40脚不申请、不监听。服务启动时为两个输入启用内部上拉。

本机`radxa`已启用systemd lingering和SDDM自动登录，因此用户服务会随开机启动，
不需要sudo，也不依赖手工登录。

安装：

```bash
/home/radxa/repositories/ec_26_e/deploy/systemd/user/install_button_dispatcher.sh
```

状态和日志：

```bash
systemctl --user status ec26e-button-dispatcher.service
journalctl --user -u ec26e-button-dispatcher.service -f
cat /run/user/1000/ec26e/buttons-status.json
```

服务启动或重启后必须先看到两个按键稳定松开，才进入`idle`。任务运行时
继续维护按键状态但忽略新请求，不排队、不抢占。可重试失败每2秒重试一次，
总窗口为120秒；成功、不可重试错误或窗口到期后释放任务锁，并重新等待按键。

按钮只负责请求任务，不能替代直接切断驱动使能或电源的硬件急停回路。
