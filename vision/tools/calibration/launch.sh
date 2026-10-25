#!/usr/bin/env bash
# 启动 EC26E 现场标定控制台或指定任务的快速预览。

set -euo pipefail

PROJECT_DIRECTORY=/home/radxa/repositories/ec_26_e
POETRY_EXECUTABLE=/home/radxa/.local/bin/poetry
PYTHON_EXECUTABLE=/home/radxa/.cache/pypoetry/virtualenvs/ec-26-e-Y3NiaORr-py3.10/bin/python

cd "$PROJECT_DIRECTORY"
if [[ -x "$PYTHON_EXECUTABLE" ]]; then
  exec "$PYTHON_EXECUTABLE" tools/calibration/console.py "$@"
fi
exec "$POETRY_EXECUTABLE" run python tools/calibration/console.py "$@"
