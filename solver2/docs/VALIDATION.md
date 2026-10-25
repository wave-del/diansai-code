# solver2 最新算法同步验证

当前已统一到 `solver2` 唯一实现，旧 `vision/src/ec_26_e/control` 已删除。
最新调用链迁移与删除后验证见 [CENTRALIZATION.md](CENTRALIZATION.md)。
下文记录删除旧目录前的同步与独立复制验证历史，原验证数据仍保留。

日期：2026-10-25（北京时间）。本次覆盖旧实现，同步到网页第三轮修复版
`round-3-fix-1`；使用 `diansai-code/.venv` 离线验证。

## 同步范围

- 核对全部 102 个算法 Python 文件。除 `ec_26_e.control` → `solver2`
  包名及 CLI 显示名称外，文本与当前网页算法一致。
- 实际覆盖四个旧文件：`search.py`、`corner_hypotheses.py`、
  `random_safety.py`、`random_clearance.py`。包含前期搜索顺序、候选缓存优化，
  以及本轮凹口拼缝重建、逐段间隙和硬约束留缝修复。
- 同步三个修复测试模块及三个输入样例，均通过本包导入；测试辅助入口仅依赖 `solver2`。
- `SOURCE.json` 的 128 个来源映射及两个生成文件均已核对 SHA256；
  包内全部 Python 文件语法检查通过。CLI 总帮助和 `solve --help` 正常。

## 独立复制重放

将整个 `solver2` 复制到临时目录，使用 Python `-I` 隔离进程，清除
`PYTHONPATH`，并禁止导入 `ec_26_e`、`puzzle_solver` 和
`random_white_solver_adapter`。实际加载的 71 个 `solver2` 模块均来自临时副本。

使用第三轮保存的原始观测坐标，按网页相同几何配置（白片、候选目标 1、
最短边参数 10mm、无搜索时间上限）重放原 46 次失败：**46／46 成功**。
独立检查完整碎片、刚体闭合、留缝前填充率至少 95%、零重叠、
全部真实接口的 3～6mm 间隙及工作区范围。同时验证公开
`plan_sheet_moves` 接口成功求解两片案例。

逐例结果见 [ISOLATED-REPLAY.json](ISOLATED-REPLAY.json)，
汇总见 [UPDATE-VALIDATION.json](UPDATE-VALIDATION.json)。

网页原算法的 **2967／2967** 完整回归记录位于：

```text
/mnt/nvme/initial-code/npu_q6a_test/web-random-test/round-3-fix-1/
```

该 2967 次记录属于网页原算法；本次独立副本额外执行的是 46 例隔离重放，
并未将源码一致性核对描述为再次运行了 2967 次。

## 单元与回归测试

在 `diansai-code` 根目录执行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -B -m pytest solver2/tests/unit/control -q
```

结果：**110 项通过，9 个子测试通过，2 项既有固定模板测试失败**。
新增的凹口、逐段间隙、硬约束修复及搜索顺序测试全部通过。

两个失败项均为 `test_planner.py` 中原已记录的固定模板断言差异：

| 测试 | 原断言 | 实际值 |
|---|---|---|
| `test_fixed_plan_is_complete_and_on_lower_half` | 外框宽度 100 mm | 111.83105462640475 mm |
| `test_fixed_plan_uses_gapped_non_overlapping_final_layout` | 最小间隙至少 5 mm | 4.148604454382078 mm |

本次另对当前 `vision/tests/unit/control/test_planner.py` 的同名两项测试进行了
对照，仍在相同数值上失败；没有新增回归失败。同步保留当前现场模板及原断言。

完整日志及离线重放输出：

```text
/mnt/nvme/initial-code/npu_q6a_test/web-random-test/solver2-update-20261025/
```
