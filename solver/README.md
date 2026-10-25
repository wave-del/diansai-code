# 拼图解算器实验台

这是与 `Tangram-26` 完全独立的 PySide6 验证工具：不访问 GPIO、相机或机械臂。

启动：

```bash
# 在仓库根目录（含 solver/ 的那一层）执行
python -m venv .venv && .venv/bin/pip install -r solver/requirements.txt
.venv/bin/python solver/app.py
```

使用流程：设置矩形尺寸，选择切割方式并切分（最多四块），随机散落，点击“开始解算”。
左侧保留完整矩形和真值；中间显示输入给解算器的散落碎片；右侧显示 Top-K 解算候选。

切割方式：

- `直线拖拽`：在目标碎片内部按住鼠标拖拽；红色虚线表示实际执行的无限直线切缝。
- `三点折线`：依次点击原始矩形外的起点、目标碎片内部的折点、原始矩形外的终点，生成一条有限的两段折线切缝。橙色实线为已确认段，蓝色虚线为待确认段，红色虚线为真正落在目标碎片内的部分。`Esc` 或右键取消，退格撤回上一个点。

两种方式都会校验题目约束：最多 4 块、每块最多 5 边、每条边至少约 20 mm，且不能产生自交多边形。

案例会保存为 JSON，日志和结果报告保存在 `solver/reports/`。扑克牌图片只用于生成测试碎片和真值，解算核心不会得到完整牌面模板。

几何核心回归测试：

```bash
.venv/bin/python solver/tests/run_regression.py
```

自动压力测试（直线、T 型与折线切割；默认 12 组）：

```bash
.venv/bin/python solver/tests/run_stress.py
```

加入较慢的四块规则网格测试，或模拟检测顶点抖动：

```bash
.venv/bin/python solver/tests/run_stress.py --count 16 --four-piece --jitter 0.3 --timeout 70
```
