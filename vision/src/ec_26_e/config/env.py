"""集中加载项目级 ``.env`` 部署配置。"""

from __future__ import annotations

from collections.abc import MutableMapping
import os
from pathlib import Path
import re


_ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parse_env_value(raw_value: str, *, line_number: int) -> str:
    """解析单个 ``.env`` 值，并拒绝未闭合的引号。"""

    value = raw_value.strip()
    if not value:
        return ""
    if value[0] in {"'", '"'}:
        quote = value[0]
        if len(value) < 2 or value[-1] != quote:
            raise ValueError(f".env 第 {line_number} 行的引号未闭合")
        return value[1:-1]
    # 非引号值仅把空白后的井号视为注释，
    # 保留值自身包含的井号。
    return re.split(r"\s+#", value, maxsplit=1)[0].rstrip()


def load_env_file(
    path: Path,
    *,
    environ: MutableMapping[str, str] | None = None,
    override: bool = False,
) -> dict[str, str]:
    """读取基础 ``KEY=VALUE`` 配置，并默认保留已导出的环境变量。

    文件不存在时返回空字典。返回值只含本次实际写入的配置，
    便于测试和启动诊断确认来源。
    """

    if not isinstance(path, Path):
        raise TypeError("path 必须是 pathlib.Path")
    target_environ = os.environ if environ is None else environ
    if not path.exists():
        return {}
    if not path.is_file():
        raise ValueError(f".env 路径不是文件：{path}")

    loaded: dict[str, str] = {}
    for line_number, source_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        line = source_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, raw_value = line.partition("=")
        name = name.strip()
        if not separator or not _ENV_NAME_PATTERN.fullmatch(name):
            raise ValueError(f".env 第 {line_number} 行格式无效")
        value = _parse_env_value(raw_value, line_number=line_number)
        if override or name not in target_environ:
            target_environ[name] = value
            loaded[name] = value
    return loaded
