#!/usr/bin/env bash
set -euo pipefail
# Rebuild only: SDK headers are not needed to run the shipped backend.
QNN_HEADERS_ROOT="${1:?usage: bash build.sh /path/to/sdk [output.so]}"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QNN_BRIDGE_OUTPUT="${2:-$SOURCE_DIR/build/libq6a_qnn.so}"
mkdir -p "$(dirname "$QNN_BRIDGE_OUTPUT")"
cc -shared -fPIC -O2 -Wall -Wextra -Werror \
  -I "$QNN_HEADERS_ROOT/include/QNN" "$SOURCE_DIR/q6a_bridge.c" \
  -o "$QNN_BRIDGE_OUTPUT" -ldl -lm
