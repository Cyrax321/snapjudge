#ifndef SNAPJUDGE_CAPI_H
#define SNAPJUDGE_CAPI_H

// snapjudge C API — a stable, C-ABI boundary over the snapjudge engine.
//
// Everything crosses the boundary as C types: opaque handles, NUL-terminated
// UTF-8 strings, and malloc'd strings returned to the caller (freed with
// sj_string_free). This is what makes snapjudge embeddable from Go, Rust,
// Swift, Kotlin, Python ctypes, or a WASM module without touching C++.
//
// Lifecycle:
//   sj_engine* e = sj_engine_new();
//   sj_load(e, "main", "/path/to/checkpoint");
//   char* out = sj_predict(e, state_json, questions_json, "");
//   ... use out ... sj_string_free(out);
//   sj_engine_free(e);
//
// Every function that can fail returns NULL / -1 and records a message
// readable with sj_error_message(e). The error is cleared on the next
// successful call that sets one.

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(SNAPJUDGE_SHARED)
#  ifdef SNAPJUDGE_BUILDING_DLL
#    define SJ_API __declspec(dllexport)
#  else
#    define SJ_API __declspec(dllimport)
#  endif
#else
#  define SJ_API
#endif

typedef struct sj_engine sj_engine;

// Version of the C ABI, encoded as (major * 10000 + minor * 100 + patch).
SJ_API int sj_version(void);

// Create / destroy an engine. `options_json` may be NULL or a JSON object with
// optional keys: "device" ("cpu"/"cuda"), "token" (HF access token),
// "max_loaded" (int), "default_checkpoint" (string).
SJ_API sj_engine* sj_engine_new(const char* options_json);
SJ_API void sj_engine_free(sj_engine* e);

// Load a checkpoint under `name` (e.g. "english", "multilingual",
// "typed-decisions", or any user-chosen key). `model_id_or_path` is a local
// directory or an HF repo id. Returns 0 on success, -1 on failure (see
// sj_error_message).
SJ_API int sj_load(sj_engine* e, const char* name, const char* model_id_or_path);

// Answer typed questions over a state. `state_json` and `questions_json` are
// JSON documents (objects); `model` is a checkpoint name or ""/NULL for the
// first-loaded (default) checkpoint. Returns a malloc'd JSON result string, or
// NULL on failure.
SJ_API char* sj_predict(sj_engine* e, const char* state_json,
                        const char* questions_json, const char* model);

// Last error message (UTF-8, static storage valid until the next call), or
// NULL when no error is recorded.
SJ_API const char* sj_error_message(sj_engine* e);

// Free a string returned by sj_predict / sj_route. Safe to pass NULL.
SJ_API void sj_string_free(char* s);

#ifdef __cplusplus
}
#endif

#endif  // SNAPJUDGE_CAPI_H
