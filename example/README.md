# Tangram 算法调用 API（最终交接入口）

算法、视觉或其他 AI 只应使用 `tangram_api.py` 的 `TangramAPI`，不要直接发送
G-code、UART 二进制帧或 GPIO 电平。本目录可作为同事阅读和调用硬件的唯一入口：

| 文件 | 用途 |
|---|---|
| `tangram_api.py` | 面向业务代码的唯一 Python API；封装写字机、FOC、电磁铁 |
| `tangram_console.py` | 真实硬件交互测试台；命令行验证所有 API |
| `test_vision_pick_rotate_place.py` | 视觉质心抓取、途中旋转 90°、放到 `(10,10)` 的真实测试入口 |
| `record_camera_10s.py` | 只读取 USB 摄像头、默认录制 10 秒的 MJPEG AVI 视频 |
| `README.md` | 坐标、单位、限制、测试步骤和调用示例 |

## 坐标与硬限制

- 输入 XY 原点是蓝纸右上角；`X+` 向下、`Y+` 向左，单位都是 mm。
- 所有 XY 自动执行且仅执行一次：`X_cmd=clamp(X-23.5,0,190)`、
 `Y_cmd=clamp(Y+16.45,0,300)`。调用方不得再偏置或限幅。
- 超出范围的命令**不会发送原始超大坐标**，而是夹到 `(190,300)` 等软件边界
  坐标。例如输入 `(500,500)` 实际下发 `(190,300)`。这不是限位开关或 GRBL
  硬限位；前提是该软件边界本身位于机械安全工作区内。
- 因此任务完成后“蓝纸 `(0,0)`”的底层状态可能显示为内部 `Y=16.45 mm`；
  这是正确的偏置结果，不代表没有回到纸原点。只有显式维护接口
  `return_machine_origin()` 才会去写字机自身机械 `(0,0)`。
- 每次上电后，先让机构位于**写字机自身机械原点**，再调用
  `establish_origin()`。蓝纸右上角不是机器原点；它由下述 XY 偏置映射得到。
- Z 使用正数抬起语义：`0` 是自然下垂，`15` 是抬起 15 mm；本机验证范围为
  0..15 mm，Z 速度固定 1500 mm/min。

## API 速查

| API | 参数与单位 | 行为 |
|---|---|---|
| `connect(motor_rpm=15)` | RPM | 连接三套硬件，FOC 使能；不运动 |
| `establish_origin()` | 无 | 机构在写字机自身原点时建立机器 `(0,0)`；不运动 |
| `move_xyz(x,y,lift,xy_feed_mm_min=2500)` | XY / 抬高：mm；XY 速度：mm/min | 先安全抬高、XY 移动、最后到目标 Z；XY 速度可调 |
| `return_machine_origin(xy_feed_mm_min=2500)` | XY 速度：mm/min | 仅上电/维护时回写字机自身机械原点；刻意绕过蓝纸偏置 |
| `set_lift(lift)` | `0..15` mm | 正数抬起，0 落到自然下垂；Z 固定 1500 mm/min |
| `foc_zero(rpm=15)` | RPM | 回驱动板保存的机械零点，最短路径 |
| `foc_angle(angle,rpm=15)` | 绝对 `-360..360°` | 先按一圈归一化，再从当前位置最短路径定位；`-50` 等同绝对 `310` |
| `foc_rotate(delta,rpm=15)` | 相对 `-180..180°` | 严格按正/负方向相对旋转；`+180` 和 `-180` 不同 |
| `magnet_on()` / `magnet_off()` | 无 | PL6 高电平吸合 / 低电平松开 |
| `close()` | 无 | 松开电磁铁；XY、FOC 保持使能及最终位置 |

默认 XY 速度是 `2500 mm/min`，约 `41.7 mm/s`。仍可在每次调用中传入
`xy_feed_mm_min` 覆盖它；首次使用新速度时，请用重复往返动作确认平稳性和
是否丢步。Z 速度不是可调参数，必须保持 `1500 mm/min`。

## 最小示例

```python
from tangram_api import TangramAPI

with TangramAPI() as arm:
    # 机构已回写字机自身原点后：
    arm.establish_origin()
    arm.move_xyz(50, 50, 15, xy_feed_mm_min=2500) # XY速度可调
    arm.set_lift(0)                                # 落到 Z0
    arm.magnet_on()
    arm.set_lift(8)
    arm.foc_zero()                                 # 驱动板机械零点
    arm.foc_angle(-90)                             # 绝对 270°，自动就近
    arm.foc_rotate(-90)                            # 相对转，支持 -180..+180
    arm.move_xyz(10, 10, 0, xy_feed_mm_min=1000)
    arm.magnet_off()
    arm.move_xyz(0, 0, 8)                         # 正常任务回蓝纸坐标 (0,0)
```

正常退出会松开电磁铁；写字机和 FOC 保持使能及当前位置，避免外力扰动。

## 交互测试

```bash
.venv/bin/python \
  example/tangram_console.py
```

输入 `help` 查看所有 XYZ、FOC 回零/绝对/相对旋转、电磁铁命令。任何 XY 运动前
必须先执行 `origin` 并输入确认词。

## 视觉抓取测试（全程实时预览）

蓝纸上只放一片白色拼图后，直接运行：

```bash
.venv/bin/python \
  example/test_vision_pick_rotate_place.py
```

操作完全在终端完成，屏幕持续显示完整原始相机画面与透视校正后的完整蓝纸：

视觉抓取候选固定为**透视校正蓝纸画面的左半边**（黄色竖线左侧，标记为
`PICKUP`）；右半边标记为 `IGNORE`，即使检测到已拼好的白色图形，也不会计入
候选、不可能被冻结或抓取。这与本装置“左侧随机碎片、右侧目标拼图区”的现场
布局一致。

1. 确认画面中只有一个绿色轮廓的有效白色拼图，红点为几何质心；
2. 输入第一次 `RUN`：冻结质心，预览仍持续刷新；
3. **第一轮**开始前确认机构在写字机自身原点、使 Z 自然下垂，并确认路径无人手；
4. 输入第二次 `RUN`：开始抓取、搬运途中相对旋转 `+90°`、放到 `(10,10)`；
5. 真实机械运行期间预览继续刷新，动作完成后 XY 回到**蓝纸坐标 `(0,0)`**，
   而不是写字机自身机械原点；随后脚本自动重新进入预览，等待放入下一片并再次
   输入两次 `RUN`。第二轮起脚本复用第一轮建立的机器参考，不会再次 G92，
   更重要的是：整个批量脚本只打开一次写字机 USB 串口，避免 CH340 的 DTR 脉冲
   重启 GRBL 并丢失工作坐标。因此不需要也不应把机构移回机械原点。此时不要触碰机构；窗口的
   `q/Esc` 不会中断运动。需要紧急中止时在终端按 `Ctrl+C`，程序会向机械子进程
   转发中断以执行既有的抬 Z/松开电磁铁恢复流程。

## 录制摄像头视频（默认 10 秒）

```bash
.venv/bin/python \
  example/record_camera_10s.py
```

视频默认保存到 `example/outputs/camera/`，文件名包含录制时间。脚本只打开
摄像头，不连接写字机、FOC 或电磁铁。需要其他时长时，例如 30 秒：

```bash
.venv/bin/python \
  example/record_camera_10s.py --seconds 30
```

推荐首轮确认顺序：

```text
origin
xyz 0 0 15 2500
xyz 50 50 15 3000
z 0
z 15
foc -50
rotate 90
zero
magnet on
magnet off
```

## 当前现场验收结论

- 写字机蓝纸坐标偏置/限幅已通过命令行多点往返确认；该映射由
  `WriterController` 统一执行。
- XYZ、Z 抬高保持、FOC 回机械零点及绝对定位、电磁铁吸合/松开均已在真实设备
  分阶段验证。
- `foc_angle()` 的负角度语义是绝对坐标别名；不要把它与 `foc_rotate()` 的相对
  角度混用。
- 写字机和 Z 是开环步进系统；发生撞限位、被手动反驱或怀疑丢步后，必须重新让
  机构回写字机自身原点并执行 `establish_origin()`。
