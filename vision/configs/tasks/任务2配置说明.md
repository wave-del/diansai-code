# Task2 配置说明

`task2.json` 使用严格标准 JSON 和 `schema_version: 3`。Task2 复用 Task1
的相机、A4 校正、稳定审查及运动生命周期配置；机器参数仍配置在 `.env`。

## Task2 行为

- `detect1.piece_mode` 正式值为 `1`，只识别包含白色核心的纸片。
- `detect1.short_edge_merge_mm` 与 `detect1.minimum_physical_edge_mm` 均为
  `20.0 mm`；角点最小转角、深度和候选 IoU 分别为 `12°`、`4.0 mm`
  和 `0.85`，并由 Task2 自己的 JSON 独立配置。
- `planning.mode` 必须为 `random`，不使用固定拼图模板。
- 周期内稳定候选为 2～4 片时全部规划；多于 4 片时按跳变数、有效观测数、
  几何抖动、检测置信度和轨迹 ID 的顺序选取前 4 片。
- 稳定审查未通过但五秒内已有基础审查通过的帧时，直接使用最后一帧继续规划；找不到完整候选时不发送运动指令。候选置信度不再阻断运动指令。
- `minimum_execution_confidence` 保留在任务 JSON 中以兼容公共配置，正式执行固定为 `0.0`，由机器坐标、路径和执行器安全校验负责阻断危险动作。

## 任务环境覆盖

`TASK2_REVIEW_CYCLE_SECONDS` 与 `TASK2_ALLOWED_JUMP_FRAMES` 分别覆盖审查
周期和允许跳变帧数；`TASK2_DETECT1_PIECE_MODE` 可用于诊断时临时覆盖 JSON
中的视觉模式。导出的环境变量优先于 `.env`，`.env` 优先于任务 JSON。

其余字段的含义、范围与 Task1 同名字段一致。通过 `EC_TASK=task2` 启动时，
未设置 `EC_TASK_CONFIG` 将自动加载本文件。
