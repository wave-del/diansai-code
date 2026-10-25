#!/usr/bin/env bash
# Install into this project only; no sudo, drivers, services or device access.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 -m venv --without-pip "$PROJECT_ROOT/.venv"
python3 -m pip --python "$PROJECT_ROOT/.venv/bin/python" install pip
"$PROJECT_ROOT/.venv/bin/python" -m pip install -r "$PROJECT_ROOT/host/requirements.lock.txt"
if ! ldconfig -p | grep 'libxcb-cursor.so.0' >/dev/null; then
    mkdir -p "$PROJECT_ROOT/host/.runtime/downloads"
    (
        cd "$PROJECT_ROOT/host/.runtime/downloads"
        apt-get download libxcb-cursor0
        for archive in libxcb-cursor0_*.deb; do
            dpkg-deb -x "$archive" "$PROJECT_ROOT/host/.runtime"
            sha256sum "$archive"
            dpkg-deb -f "$archive" Package Version Architecture
        done > "$PROJECT_ROOT/host/.runtime/package-info.txt"
    )
fi
echo "Q6A 本地环境已准备完成。启动：$PROJECT_ROOT/host/start_host.sh"
