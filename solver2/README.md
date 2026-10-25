# 第二套拼图算法：边链几何求解器

本目录是本工程拼图几何算法的**唯一维护位置**，来源于先前从
`fpga-code` 迁入、随后在 Q6A 项目中修改的版本。支持固定四片和随机 2～4 片，
包含模板拟合、反向边链搜索、联合位姿优化、留缝规划与最终结果检查。
输入是毫米坐标多边形，输出是拼合结果与移动方案；算法默认在 CPU 上运行。

2026-10-25 已同步到网页第三轮修复版（`round-3-fix-1`），覆盖旧的搜索顺序、
角点候选、凹多边形拼缝重建及联合留缝实现。原 `vision/src/ec_26_e/control`
重复实现已删除；摄像头网页、网页实验台、正式任务和求解适配层统一导入本包。

本版包含按边界局部方向识别凹口拼缝、逐段计算间隙，以及带几何硬约束的留缝修复。
网页版对应源码已完成 2967 次固定输入回归；独立副本的验证结果见
[`docs/VALIDATION.md`](docs/VALIDATION.md)，统一调用与删除旧目录后的验证见
[`docs/CENTRALIZATION.md`](docs/CENTRALIZATION.md)。

本包使用 `solver2` 包名，内部依赖都已放入本目录，可随同目录复制到其他
Python 项目。所有调用方统一从本包导入 `PuzzlePiece`、`MovePlan`、
`PlanningOptions`、求解函数和异常类型。修改本目录会作用于网页后续新启动的求解任务。

后续维护只修改本目录，并更新校验和与验证记录，不再向 `vision` 复制算法。
`vision` 保留视觉识别、任务编排及运动适配；从其源代码目录启动时会定位同仓库的本包。

## 目录

| 路径 | 用途 |
|---|---|
| `__init__.py` | 公开 Python 接口 |
| `contracts.py` | 纸片、配置、移动方案与错误类型 |
| `planner.py` | `plan_sheet_moves` 求解入口 |
| `_engine/solver/fixed_template_solver.py` | 固定四片模板求解 |
| `_engine/solver/search.py` | 随机 2～4 片边链搜索 |
| `_engine/solver/solve.py` | 求解流程、候选筛选及最终检查 |
| `_engine/planning/` | 留缝、吸取点、移动顺序和坐标变换计算 |
| `_engine/appearance/` | 外观候选消歧；实际图像评分由调用方注入 |
| `_engine/generator/` | 测试碎片生成 |
| `_engine/visualization/`、`_engine/evaluation/` | 结果绘图和离线评估 |
| `acceleration.py`、`candidate_acceleration.py` | 可注入的数值计算后端接口 |
| `tests/` | 回归测试、凹口与搜索顺序修复测试及输入样例 |
| `docs/SOURCE.json` | 当前唯一实现的版本、文件清单及 SHA256 |
| `docs/EXTRACTION-SOURCE.json` | 删除旧实现前的提取来源和校验历史 |

## 使用

以下命令均在 `/mnt/nvme/initial-code/diansai-code` 下执行，使用已有的 `.venv`。
另建环境时需要 Python 3.10 或更新版本，并安装 `solver2/requirements.txt`。

```bash
cd /mnt/nvme/initial-code/diansai-code
.venv/bin/python -m solver2 --help
.venv/bin/python -m solver2 solve \
  --input solver2/tests/fixtures/fixed4_reference.json \
  --output solver2/outputs/fixed4
```

命令行输出保留原算法的诊断格式，结果见输出目录的 `solution.json`。
调用方须检查结果中的成功状态和执行检查字段。

公开 Python 接口：

```python
from solver2 import PlanningOptions, PuzzlePiece, plan_sheet_moves

pieces = (
    PuzzlePiece("left", ((10., 10.), (60., 10.), (60., 70.), (10., 70.))),
    PuzzlePiece("right", ((100., 20.), (150., 20.), (150., 80.), (100., 80.))),
)
plan = plan_sheet_moves(pieces, PlanningOptions(mode="random"))
result = plan.to_dict()
```

坐标单位为 mm，工作区沿用原算法的 **207 × 292 mm**，原点左上、X 向右、Y 向下。
`mode="fixed"` 要求输入恰好四片且符合既有模板。随机公开入口默认不设时间上限；
无解或未通过最终检查时抛出 `PlanningError`。

## 验证

复用原工程已安装的 pytest：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest solver2/tests/unit/control -q
```

原测试中有两个已知固定模板断言与现场模板不一致：外框宽度断言为 100 mm，
实际约 111.831 mm；最小间隙断言至少 5 mm，实际约 4.149 mm。
本次同步保留现场模板及这两个既有断言，验证结果记录于 `docs/VALIDATION.md`。
