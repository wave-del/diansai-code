#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
else
    echo "缺少项目虚拟环境；请参阅 host/Q6A-MIGRATION.md 安装依赖。" >&2
    exit 2
fi
# Local Qt X11 dependency only; this directory contains no A733 runtime.
QT_LOCAL_LIB="$ROOT/host/.runtime/usr/lib/aarch64-linux-gnu"
if [[ -d "$QT_LOCAL_LIB" ]]; then
    export LD_LIBRARY_PATH="$QT_LOCAL_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
exec "$PYTHON" "$ROOT/host/tangram_host.py" "$@"
