# snapjudge Go bindings (smoke test)

Proves the C ABI is callable from Go via cgo with zero C++ in the caller.

Build (from repo root):

    go run tests/capi/go/ 2>/dev/null || true   # placeholder, see below

The real build needs the compiled static lib first:

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j
    # then a go.mod with cgo directives pointing at build/libsnapjudge.a

This file is a minimal self-contained example (no go.mod needed if run with
`go run` in a module that vendors capi.h + the static lib).
