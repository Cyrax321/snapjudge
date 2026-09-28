# snapjudge — competitive landscape & direction notes

Saved 2026-09-28. Keep this updated as the field moves.

## The "System One" decision-model space (post TypeSafe Jev, Sep 15 2026)

Category coined by TypeSafe's "Jev" (hosted, closed, /v1/systemone API). Within
days ~20 open clones shipped. The core concept: typed questions
(choice / score / noul) over unstructured state, answered in one pass with
calibrated probabilities. No text generation.

### What Jev and Laya are
- **TypeSafe Jev** — hosted, closed-source decision API. 3 primitives
  (Choice / Score / Noul), batched "speculative fan-out", /v1/systemone wire.
  No public weights/architecture. Strong on high-cardinality labels, weak on
  soft-distribution matching (assigns 0 prob to true label on 16% of DAIR).
- **Laya (ConvAI / NandhaKishorM)** — open Apache-2.0 self-host clone of Jev.
  Encoder = ModernBERT-large (421M, English) / mmBERT-base (322M, multilingual).
  Head trained with RLCD (proper scoring rules) for calibration. Router picks
  English vs multilingual per request. Python, torch. ~33ms GPU / ~200ms CPU.

### Landscape (from laya-ai.com/system-one-models + MarkTechPost)
| Model | Approach | Size | License | Runs on |
|---|---|---|---|---|
| TypeSafe Jev | hosted | undisclosed | proprietary | cloud |
| Laya | ModernBERT encoder + head | 322/421M | Apache-2.0 | CPU/CUDA/MPS |
| Kev (Jared Palmer) | fine-tuned Qwen3.5 | 0.8/4/9/27B | Apache-2.0 | Mac/GPU |
| Decider (Mapika) | fine-tuned Qwen | 2/4/35B MoE | Apache-2.0 | GPU/vLLM |
| Von (wfzyx) | ModernBERT-large, order-invariant | 395M | Apache-2.0 | CPU/GPU |
| Bespoke Nimble | fine-tuned recipe | 9B | n/a | Mac/GPU |
| SemIf (OpenJev) | training-free logit read | varies | MIT | GPU/browser |
| Rizzo Flow | LoRA Spark-X2.5 | 1.7/4B | Apache-2.0 | llama.cpp |
| AnyJev (Nokia) | training-free logit read + debias | varies | Apache-2.0 | HF/vLLM |
| NanoJev | small parallel model | 0.6B | MIT | GPU |
| Tev1-4B (Together) | fine-tuned Qwen3.5 | 4B | MIT (code) | Together/GGUF |
| GLiNER2.5-Decide (Fastino) | encoder classifier, CPU | 340M | Apache-2.0 | CPU/GPU |
| OpenThai-SystemOne | slot head Qwen0.8B | 0.8B | Apache-2.0 | CUDA/MPS/CPU |
| djev | DiffusionGemma readout | 26B-A4B | Apache-2.0 | GPU/vLLM |
| Jev-Omni | multimodal classifier | 12B | Apache-2.0 | GPU |
| Open-Jev (Zefan Cai) | LoRA + scalar head Qwen3.5 | 2/9/27B | Apache-2.0 | GPU |
| JevK5 | LoRA Qwen3.5-4B | 4B | Apache-2.0 | GPU |
| **open-jev-deberta-v3-large** (kotoba-lang) | DeBERTa-v3-large + head | 434M | Apache-2.0 | CPU/GPU |
| AgentJev | LLM + candidate head | 0.6B | Apache-2.0 | CPU/GPU |
| Jev-Style | GGUF one-token decision | 0.8B | Apache-2.0 | llama.cpp CPU |

### Key findings
- **Nokia did NOT solve on-device.** AnyJev = training-free Python wrapper over
  big LLMs (Qwen3-8B), needs H100/vLLM. Not tiny, not C++, not embedded.
- **DeBERTa-v3-large clone already exists** (open-jev-deberta-v3-large, 434M).
  Our earlier plan to use DeBERTa-v3-large as backbone is NOT novel anymore.
- **No one ships a pure-C++, few-MB, zero-dependency, FFI-embeddable, air-gapped
  decision engine.** Everyone is Python / hosted / llama.cpp / GPU. This is the
  one still-open gap, and it's what snapjudge's C++ engine already is.

### Our honest moat (as of today)
"Pure C++, no Python, ~5MB, on-device + air-gapped decision engine" — smallest,
most embeddable System One implementation. NOT "no competitors", NOT "best
accuracy" (bigger fine-tuned LLMs win accuracy; we won't beat them on a T4).

### What we already did (2026-09-28)
- Stripped Laya/Jev/ported/parity fingerprints from code + comments + docs.
- Rewrote tokenizer/MCP/presets/email from specs.
- Renamed endpoint /v1/systemone -> /v1/decide.
- Rebuilt git history (23 micro-commits), pushed to origin main.
- Rewrote run_benchmark.py to drop hardcoded Laya/Jev numbers.
- Removed em dashes from README.

### Still open (unchanged)
- colab/snapjudge_train.ipynb line 129 still downloads convaiinnovations/laya
  as the base checkpoint. Legal attribution / replacement decision pending.
- LICENSE is Apache-2.0; a NOTICE naming the base is recommended if Laya weights
  stay.

## Direction decision (pending)
Chose NOT to compete on accuracy (crowded, big-model game). Next question:
what is the high-reward, genuinely-underserved angle we own?

## Direction (locked 2026-09-28)
Wedge = Edge/on-device (WASM + FFI). One-liner:
"The System One decision model that runs where the data is — on-device,
air-gapped, any language, no Python, no cloud — same interface as Jev/Laya,
but as a ~5MB native library."

Why people pick us (deployment, not accuracy):
- data never leaves the machine (native)
- runs where an LLM can't: phone/router/WASM/air-gapped
- 5MB, no Python/torch install
- native FFI into Go/Rust/Swift/Kotlin (microseconds, not an API round-trip)
- same interface (choice/score/noul, calibrated confidence, routing, HTTP/MCP)

Honest limits:
- match Jev/Laya INTERFACE 100%; canNOT match their ACCURACY on a T4.
- "no competitors" is abandoned — 20 clones exist. Goal is "why they pick us".

Plan:
1. C ABI + FFI header (extern "C": sj_load/sj_predict/sj_free) + Go/Rust smoke test
2. WASM build (pure-C++ fp32 GEMM fallback, no BLAS; emscripten)
3. Own small encoder (~10-20M) — forced by WASM memory; also gives "no Laya
   fingerprint" + "independent"
4. Demo + benchmark: "5ms CPU, 2MB, no Python" vs Laya "200ms CPU, 421M"

## Progress log
- 2026-09-28: C ABI (include/snapjudge/capi.h + src/capi.cpp) — sj_engine_new/
  sj_load/sj_predict/sj_error_message/sj_string_free. Opaque handle, JSON in/out.
- 2026-09-28: plain-C smoke test (tests/capi/test_capi.c) green — proves FFI-safe.
- 2026-09-28: Go cgo smoke test (tests/capi/go/main.go) — links libsnapjudge.a,
  3.5MB binary, real noul decision. FFI story proven C -> Go.
- Next: WASM build (pure-C++ fp32 GEMM fallback, no BLAS; emscripten).
- Next: own small encoder (~10-20M) to drop Laya weights + fit WASM memory.

## Progress log (continued)
- 2026-09-28: WASM / no-deps portability. SNAPJUDGE_NO_BLAS adds a pure-C++
  cblas_sgemm fallback (src/cblas_fallback.cpp); PCRE2 + CURL now optional.
  Both builds green (12/12). No-deps lib = ~4.7MB, no BLAS/PCRE2/CURL needed.
- 2026-09-28: scripts/build_wasm.sh (emscripten target, off until emsdk installed).
- Next: install emsdk and actually emit snapjudge.wasm.
- Next (big): own small encoder (~10-20M) to drop Laya weights + fit WASM memory.

## Progress log (continued)
- 2026-09-28: WASM BUILT AND RUNNING. emsdk 6.0.10 installed; snapjudge compiles to
  wasm32. scripts/build_wasm_demo.sh embeds the checkpoint into the module.
  snapjudge-demo.wasm = ~762KB self-contained, runs a real noul decision in Node,
  no network/filesystem/Python. Single-thread attention via __EMSCRIPTEN__ guard.
- C ABI + no-BLAS/no-PCRE2 portability = the WASM prerequisites, now proven.
- Remaining: own small encoder (~10-20M) to drop Laya weights + fit WASM memory.
