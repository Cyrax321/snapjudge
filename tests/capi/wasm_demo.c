// snapjudge WASM demo entry point. Loads an embedded checkpoint (bundled into
// the WASM virtual filesystem at /checkpoint) and runs one typed prediction.
//
// Built by scripts/build_wasm_demo.sh with --embed-file, so the checkpoint is
// baked into the module — no network, no filesystem, no server. This is the
// "runs in the browser, on-device" proof.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "snapjudge/capi.h"

int main(void) {
  sj_engine* e = sj_engine_new(NULL);
  if (!e) {
    fprintf(stderr, "engine_new failed\n");
    return 1;
  }

  if (sj_load(e, "main", "/checkpoint") != 0) {
    fprintf(stderr, "load failed: %s\n", sj_error_message(e));
    sj_engine_free(e);
    return 1;
  }

  const char* state = "{\"text\":\"I was charged twice, please refund\"}";
  const char* questions =
      "{\"q\":{\"type\":\"noul\",\"instructions\":\"Does the customer want money back?\"}}";

  char* out = sj_predict(e, state, questions, "");
  if (!out) {
    fprintf(stderr, "predict failed: %s\n", sj_error_message(e));
    sj_engine_free(e);
    return 1;
  }

  printf("WASM decision: %s\n", out);
  sj_string_free(out);
  sj_engine_free(e);
  return 0;
}
