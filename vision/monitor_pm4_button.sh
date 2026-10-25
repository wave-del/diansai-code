#!/usr/bin/env bash
# 监听 Cubie A7Z 的 PM4（40-pin 排针 37 脚）。
# 约定：按键按下时将 PM4 接到 GND（低电平），松开时为高电平。

set -euo pipefail

GPIO_CHIP=/dev/gpiochip1
GPIO_OFFSET=36  # PM4 = M 组第 4 脚；gpiochip1 上 M 组从 offset 32 开始
POLL_INTERVAL=0.01  # 秒
DEBOUNCE_SAMPLES=5  # 连续 5 次采样一致（约 50 ms）才确认状态变化

read_level() {
  gpioget --bias=as-is "$GPIO_CHIP" "$GPIO_OFFSET"
}

stable_level=$(read_level)
candidate_level=$stable_level
candidate_count=0

echo "Listening on PM4 (physical pin 37). Press Ctrl-C to stop." >&2

while true; do
  level=$(read_level)

  if [[ $level == "$stable_level" ]]; then
    candidate_level=$stable_level
    candidate_count=0
  elif [[ $level == "$candidate_level" ]]; then
    ((candidate_count += 1))
  else
    candidate_level=$level
    candidate_count=1
  fi

  if (( candidate_count >= DEBOUNCE_SAMPLES )); then
    stable_level=$candidate_level
    candidate_count=0
    # 常见接法是内部/外部上拉、按下接地：0 即按下。
    if [[ $stable_level == 0 ]]; then
      printf 'down\n'
    fi
  fi

  sleep "$POLL_INTERVAL"
done
