#!/usr/bin/env bash
# 安装、校验并启用当前用户的EC26E实体按键服务。

set -euo pipefail

PROJECT_DIRECTORY=/home/radxa/repositories/ec_26_e
UNIT_SOURCE="$PROJECT_DIRECTORY/deploy/systemd/user/ec26e-button-dispatcher.service"
UNIT_DIRECTORY=/home/radxa/.config/systemd/user
UNIT_TARGET="$UNIT_DIRECTORY/ec26e-button-dispatcher.service"

test -c /dev/gpiochip0
test -c /dev/gpiochip1
test -f /home/radxa/.local/share/ec26e-calibration/active/machine.env
test -f /home/radxa/.local/share/ec26e-calibration/active/tasks/task1.json
test -f /home/radxa/.local/share/ec26e-calibration/active/tasks/task2.json

install -d -m 755 "$UNIT_DIRECTORY"
install -m 644 "$UNIT_SOURCE" "$UNIT_TARGET"
systemd-analyze --user verify "$UNIT_TARGET"
systemctl --user daemon-reload
systemctl --user enable --now ec26e-button-dispatcher.service
