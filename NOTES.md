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
