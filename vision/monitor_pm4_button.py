#!/usr/bin/env python3
"""监听 Cubie A7Z 的 PM4（40-pin 排针 37 脚）按键输入。"""

import time

from periphery import GPIO


GPIO_CHIP = "/dev/gpiochip1"
GPIO_OFFSET = 36  # PM4：M 组从 gpiochip1 的 offset 32 开始，因此 PM4 = 32 + 4
POLL_INTERVAL_SECONDS = 0.01
DEBOUNCE_SECONDS = 0.05


def read_pm4_button() -> None:
    """按键接地时视为按下；确认消抖后打印一次 down。"""
    gpio_in = None

    try:
        gpio_in = GPIO(GPIO_CHIP, GPIO_OFFSET, "in")
        stable_value = gpio_in.read()
        candidate_value = stable_value
        candidate_since = time.monotonic()

        print("Listening on PM4 (physical pin 37). Press Ctrl-C to exit.")

        while True:
            value = gpio_in.read()
            now = time.monotonic()

            if value != candidate_value:
                candidate_value = value
                candidate_since = now
            elif (
                candidate_value != stable_value
                and now - candidate_since >= DEBOUNCE_SECONDS
            ):
                stable_value = candidate_value
                # 常见按键接法：松开为高电平，按下时接到 GND（低电平）。
                if not stable_value:
                    print("down", flush=True)

            time.sleep(POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nProgram exited")
    except Exception as error:
        print(f"Error occurred: {error}")
    finally:
        if gpio_in is not None:
            gpio_in.close()


if __name__ == "__main__":
    read_pm4_button()
