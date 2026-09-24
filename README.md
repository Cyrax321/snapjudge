# snapjudge

**snapjudge** is a fast, non-autoregressive System-1 decision engine written in
self-contained C++17, with calibrated probabilities.

```
state ──▶ typed questions (choice / score / noul) ──▶ one forward pass
          ──▶ calibrated answers + confidence
```

You give the model a state (text, email, ticket, or JSON) and typed questions;
snapjudge answers every question in a single bidirectional encoder forward
pass with mathematically calibrated probabilities. It never generates text —
so nothing to parse and nothing to hallucinate.

## Why C++?

The entire engine — tokenizer, transformer encoder, typed decision head,
routing, HTTP serving, training — is C++17 with zero ML runtime dependencies
(no Python, no torch, no ONNX Runtime). It links against system BLAS
(Apple Accelerate / OpenBLAS) and PCRE2. That shape fits environments where a
Python/ML runtime is too heavy or unavailable: edge binaries, embedded
services, FFI into Go/Rust/Swift, hardened sandboxes.

## Features

- **Typed decisions in one forward pass** — choice (finite labels), score
  (ordinal rubric), noul (calibrated yes/no probability)
- **Calibrated confidence** — the training loss rewards honest probabilities
  (a proper scoring rule); post-hoc temperature fitting brings ECE down
- **Routing across checkpoints** — script- and language-aware routing to the
  right checkpoint (english / multilingual / typed-decisions) in microseconds
- **Training in C++** — full backprop through the decision head, AdamW,
  gradient-checked by finite differences (`ctest`)
- **Evals and figures** — reliability diagrams, ROC, PR, confusion matrices,
  spread plots, LaTeX metric tables
- **Serving** — Jev-compatible `POST /v1/systemone` HTTP server and an MCP
  stdio server (`snapjudge-serve`, `snapjudge-mcp`)
- **HF-hub loading** — checkpoints resolve from the HF hub or local dirs,
  cached in an HF-compatible layout

## Quick start

```bash
# build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build        # 10 suites green out of the box

# route a request (no checkpoint needed)
./build/snapjudge "I was charged twice, please refund"

# predict with a checkpoint
./build/snapjudge "Refactor this service" --predict --model snapjudge/snapjudge
```

### As a library

```cpp
#include <snapjudge/router.hpp>
#include <snapjudge/presets.hpp>

snapjudge::Router router;
auto res = router.predict(
    nlohmann::ordered_json{{"text", "I was charged twice, please refund"}},
    snapjudge::guard_questions());
for (auto& [qid, ans] : res["answers"].items()) {
  // ans: type / probabilities / confidence / action, ...
}
```

## API surface

| area | header |
|---|---|
| agent runtime | `snapjudge/agent.hpp` |
| router (per-request checkpoint selection) | `snapjudge/router.hpp` |
| typed-decision presets | `snapjudge/presets.hpp` |
| language detection | `snapjudge/lang.hpp` |
| email cleaning | `snapjudge/email.hpp` |
| choice shortlisting | `snapjudge/shortlist.hpp` |
| integration classes (guardrail/triage/eval) | `snapjudge/client.hpp` |
| HTTP server | `snapjudge/serve.hpp` |
| MCP stdio server | `snapjudge/mcp.hpp` |
| training + temperature fitting | `snapjudge/train.hpp` |
| metrics (ECE, MCE, NLL, Brier, AUROC, Spearman …) | `snapjudge/metrics.hpp` |
| HF hub resolution | `snapjudge/hub.hpp` |

## Training

```bash
# 1. export data (any JSONL in the typed-decisions schema; see TRAINING.md)
# 2. train
./build/snapjudge-train --base <ckpt> --data data/train.jsonl \
    --val data/val.jsonl --out out/ft --epochs 1

# 3. evaluate + figures
./build/snapjudge-eval --ckpt out/ft --data data/val.jsonl --json out/report.json
./build/snapjudge-plots --report out/report.json --out out/figures

# 4. publish
export HF_TOKEN=hf_...
./build/snapjudge-train-push --ckpt out/ft --repo you/snapjudge-ft
```

Details and the loss definition: [TRAINING.md](TRAINING.md).

## Serving

```bash
SNAPJUDGE_MODELS=english,multilingual ./build/snapjudge-serve
# POST /v1/systemone — TypeSafe Jev-compatible wire format
```

```bash
./build/snapjudge-mcp    # MCP over stdio: snapjudge_status/route/predict/preset
```

## Layout

```
snapjudge/
  include/snapjudge/   public headers
  src/                 one file per module
  cuda/                optional CUDA fast path (SNAPJUDGE_CUDA=ON)
  cli/ serve_main/ mcp_main/ train_main/ eval_main/ plots_main/  binaries
  tests/               doctest suites + fixtures
  bench/               latency tooling
  third_party/         nlohmann/json, doctest, cpp-httplib, lodepng
  data/                example training data (typed-decisions schema)
```

## Testing

Ten doctest suites pin behavior with committed fixtures — including a
finite-difference gradcheck of every trainable tensor on a synthetic tiny
checkpoint (built at configure time) and an end-to-end training → checkpoint →
reload roundtrip.

```bash
ctest --test-dir build --output-on-failure
```

## Build options

| option | default | meaning |
|---|---|---|
| `SNAPJUDGE_CUDA` | OFF | CUDA fused-kernel fast path |
| `SNAPJUDGE_BUILD_TESTS` | ON | doctest suites |
| `SNAPJUDGE_PCRE2` | ON | system PCRE2 (else fetch/build it) |

## Environment

`SNAPJUDGE_HOST/PORT/DEVICE/PRELOAD/MODELS/THREADS/AUTO_TASK/API_KEY/LOG_LEVEL`
for serving; `SNAPJUDGE_THREADS` caps CPU intra-op threads anywhere.

## License

Apache-2.0
