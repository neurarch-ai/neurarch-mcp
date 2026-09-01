#!/usr/bin/env sh
# Trace scripts/e2e_model.py and push the result through the MCP server.
# Usage (from anywhere): sh python/neurarch-trace/scripts/e2e.sh [python]
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${1:-python3}
OUT=${TMPDIR:-/tmp}/mini-resnet.neurarch.json
PYTHONPATH="$HERE/.." "$PY" -m neurarch_trace "$HERE/e2e_model.py:MiniResNet" --input 1,3,32,32 -o "$OUT"
node "$HERE/e2e.mjs" "$OUT"
