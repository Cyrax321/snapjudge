#!/usr/bin/env bash
# Build snapjudge to WebAssembly with Emscripten (the "runs in the browser" path).
#
# Prereqs: emsdk installed and `emcmake`/`emcc` on PATH
#   https://emscripten.org/docs/getting_started/downloads.html
#
# Produces build-wasm/snapjudge.wasm (plus .js glue). The pure-C++ GEMM fallback
# replaces BLAS, and PCRE2/CURL are disabled — no native deps, so the whole
# engine compiles to a single WASM module.
#
# Usage:  scripts/build_wasm.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v emcmake >/dev/null 2>&1; then
  echo "emcmake not found. Install emsdk and add it to PATH first." >&2
  exit 1
fi

rm -rf build-wasm
emcmake cmake -S . -B build-wasm \
  -DCMAKE_BUILD_TYPE=Release \
  -DSNAPJUDGE_NO_BLAS=ON \
  -DSNAPJUDGE_PCRE2=OFF \
  -DSNAPJUDGE_BUILD_TESTS=OFF
cmake --build build-wasm -j

echo "Built: build-wasm/libsnapjudge.a"
echo "To produce a browser-ready module, compile a small entry point, e.g.:"
echo "  emcc tests/capi/test_capi.c -I include -I third_party \\"
echo "    build-wasm/libsnapjudge.a -o snapjudge.html \\"
echo "    -s ALLOW_MEMORY_GROWTH=1 -s EXIT_RUNTIME=1"
