"""日志初始化。"""

import logging
from pathlib import Path


def configure_logging(log_file: Path | None = None, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("solver2")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    # FileHandler 在 Windows 上会保持文件锁；独立 API 调用会反复创建
    # 临时输出目录，因此替换 handler 前必须显式关闭旧句柄。
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

