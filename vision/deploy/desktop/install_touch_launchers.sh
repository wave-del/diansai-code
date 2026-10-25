#!/usr/bin/env bash
# 安装 EC26E 应用入口、登录自启入口和 Plasma 原生桌面触摸按钮。

set -euo pipefail

PROJECT_DIRECTORY=/home/radxa/repositories/ec_26_e
DESKTOP_SOURCE_DIRECTORY="$PROJECT_DIRECTORY/deploy/desktop"
APPLICATION_DIRECTORY=/home/radxa/.local/share/applications
AUTOSTART_DIRECTORY=/home/radxa/.config/autostart
TOUCH_APPLICATION_URL=file:///home/radxa/.local/share/applications/ec26e-calibration.desktop

install -d -m 755 "$APPLICATION_DIRECTORY" "$AUTOSTART_DIRECTORY"
install -m 755 \
    "$DESKTOP_SOURCE_DIRECTORY/ec26e-calibration.desktop" \
    "$APPLICATION_DIRECTORY/ec26e-calibration.desktop"
install -m 755 \
    "$DESKTOP_SOURCE_DIRECTORY/ec26e-preview-task1.desktop" \
    "$APPLICATION_DIRECTORY/ec26e-preview-task1.desktop"
install -m 755 \
    "$DESKTOP_SOURCE_DIRECTORY/ec26e-preview-task2.desktop" \
    "$APPLICATION_DIRECTORY/ec26e-preview-task2.desktop"
install -m 644 \
    "$DESKTOP_SOURCE_DIRECTORY/ec26e-touch-autostart.desktop" \
    "$AUTOSTART_DIRECTORY/ec26e-touch-autostart.desktop"

# 图标小部件原生处理触摸点击，避开 Plasma 5.20 文件夹视图对触摸双击的限制。
if [[ -S /run/user/1000/bus ]] && command -v qdbus >/dev/null 2>&1; then
    export DISPLAY="${DISPLAY:-:0}"
    export XAUTHORITY="${XAUTHORITY:-/home/radxa/.Xauthority}"
    export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/1000/bus}"
    read -r -d '' PLASMA_SCRIPT <<EOF || true
var targetUrl = "$TOUCH_APPLICATION_URL";
var allDesktops = desktops();
if (allDesktops.length > 0) {
    var desktop = allDesktops[0];
    var found = false;
    var existing = desktop.widgets();
    for (var index = 0; index < existing.length; index++) {
        if (existing[index].type === "org.kde.plasma.icon") {
            existing[index].currentConfigGroup = ["General"];
            if (existing[index].readConfig("url", "") === targetUrl) {
                found = true;
            }
        }
    }
    if (!found) {
        var launcher = desktop.addWidget("org.kde.plasma.icon");
        launcher.currentConfigGroup = ["General"];
        launcher.writeConfig("url", targetUrl);
        launcher.reloadConfig();
    }
}
EOF
    qdbus org.kde.plasmashell /PlasmaShell \
        org.kde.PlasmaShell.evaluateScript "$PLASMA_SCRIPT"
fi

command -v update-desktop-database >/dev/null 2>&1 \
    && update-desktop-database "$APPLICATION_DIRECTORY" >/dev/null 2>&1 \
    || true
