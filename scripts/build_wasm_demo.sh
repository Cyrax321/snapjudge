#!/usr/bin/env bash
# Build the snapjudge WASM browser demo: the checkpoint is embedded into the
# WASM virtual filesystem, so the module runs fully self-contained (no network,
# no filesystem, no server).
#
# Usage:  scripts/build_wasm_demo.sh [checkpoint-dir]
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="${1:-build/tiny-ckpt}"

if ! command -v em++ >/dev/null 2>&1; then
  echo "em++ not found. Install emsdk and source emsdk_env.sh first." >&2
  exit 1
fi

mkdir -p build-wasm
em++ tests/capi/wasm_demo.c -I include -I third_party \
  build-wasm/libsnapjudge.a \
  --embed-file "$CKPT@/checkpoint" \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s MODULARIZE=1 -s EXPORT_NAME=createSnapjudge \
  -o build-wasm/snapjudge-demo.js

echo "Built: build-wasm/snapjudge-demo.js (+ snapjudge-demo.wasm)"
echo "Run: node -e \"require('./build-wasm/snapjudge-demo.js')().then(m => m._main())\""
