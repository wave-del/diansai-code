#!/usr/bin/env bash
# 生成可交接的源码包，包含同级 solver2；不会写入源码目录。
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project_name="$(basename "$project_root")"
today="$(date +%Y%m%d)"
output_path="${1:-$project_root/../${project_name}-handover-${today}.zip}"

case "$output_path" in
  /*) ;;
  *) output_path="$PWD/$output_path" ;;
esac

mkdir -p "$(dirname "$output_path")"
python3 - "$project_root" "$output_path" <<'PY'
from __future__ import annotations

from pathlib import Path
import sys
import zipfile

root = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2]).resolve()
solver_root = root.parent / "solver2"
if not (solver_root / "__init__.py").is_file():
    raise SystemExit(f"Missing canonical solver package: {solver_root}")
excluded_directory_names = {".git", ".venv", ".pytest_cache", ".vscode", "__pycache__"}
excluded_paths = {"Testing/Temporary", "outputs"}

if destination.exists():
    destination.unlink()

with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
    for package_root in (root, solver_root):
        for path in package_root.rglob("*"):
            relative = path.relative_to(package_root)
            if (
                any(part in excluded_directory_names for part in relative.parts)
                or any(str(relative).startswith(prefix) for prefix in excluded_paths)
                or path.suffix == ".pyc"
                or path == destination
                or not path.is_file()
            ):
                continue
            archive.write(path, Path(package_root.name) / relative)

with zipfile.ZipFile(destination) as archive:
    bad_file = archive.testzip()
if bad_file:
    raise SystemExit(f"ZIP integrity check failed: {bad_file}")
PY

printf '交接包已生成并校验：%s\n' "$output_path"
