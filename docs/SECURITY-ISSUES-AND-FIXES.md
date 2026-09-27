# SnapJudge security issues and fix plan

Scope: Code and security loopholes only. Defensive harden, no exploit PoCs.
Branch: feat/snapjudge/security-harden. Base: main.
Surveyors: SnapJudge Build + Maya (CoS). Date (IST): 2026-09-24.

## Status after owner-side implementation

| ID | P | Where | Issue | Status |
|---|---|---|---|---|
| SEC-01 | P1 | train_main/push.cpp | Token in https://user:token@ URL + system/popen shell | **Fixed** — libcurl header auth + posix_spawnp argv + GIT_ASKPASS shim in 0700 dir, repo-name regex, redaction |
| SEC-02 | P1 | src/hub.cpp | rfilename used as cache path with no sanitize | **Fixed** — safe_rel_path + canonical containment inside blob/snap roots |
| SEC-03 | P1 | src/serve.cpp | Default 0.0.0.0 bind, optional auth | **Fixed** — default 127.0.0.1; non-loopback refuses to start without SNAPJUDGE_API_KEY |
| SEC-04 | P2 | src/hub.cpp | Unvalidated HF_ENDPOINT + open redirects | **Fixed** — allowlist (huggingface.co, https, cdn-lfs), redirect cap 2, https-only redirects, explicit insecure override for air-gapped |
| SEC-05 | P2 | src/serve.cpp | Body/question size limits absent | **Fixed** — 4 MiB body cap, 256-question cap |
| SEC-06 | P2 | src/serve.cpp | No in-process TLS | **Documented** — reverse-proxy TLS section added to README (Production serve) |
| SEC-07 | P3 | src/email.cpp | PCRE2 ReDoS class | **Deferred** — bounded, output already size-capped |
| SEC-08 | P3 | repo | No committed secrets | Pass (clean) |

## Acceptance summary

- SEC-01: `ps` / logs show no token; invalid `--repo` (traversal, spaces, shell chars) rejected before any network. argv-spawn only.
- SEC-02: unit tests reject traversal/abs/control paths and accept normal HF paths (tests/test_security.cpp). Cache writes proven to stay under the cache root via path_within.
- SEC-03: no-env default now binds 127.0.0.1. `SNAPJUDGE_HOST=0.0.0.0` without an API key exits with a clear error. Key set: missing/invalid Bearer gives 401.
- SEC-04: HF_ENDPOINT=http://127.0.0.1 rejected without explicit insecure override. Redirects: https-only, max 2.
- SEC-05: oversized body rejected 413; >256 questions rejected 400.
- SEC-07: deferred. SEC-08: clean.

Full problem statements and recommendations from the survey: in chat history
and this file's companion notes.
