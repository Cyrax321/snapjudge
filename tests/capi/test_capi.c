// snapjudge C API smoke test, written in plain C (not C++) to prove the ABI
// is callable from any language. Loads a checkpoint, runs one typed prediction,
// and checks the result is well-formed JSON with an "answers" object.
//
// Build: the CMake test target links the snapjudge static lib and this C file.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "snapjudge/capi.h"

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;

  if (sj_version() <= 0) {
    fprintf(stderr, "FAIL: bad version\n");
    return 1;
  }

  const char* ckpt = getenv("SNAPJUDGE_TINY_CKPT");
  if (!ckpt) ckpt = "build/tiny-ckpt";

  sj_engine* e = sj_engine_new(NULL);
  if (!e) {
    fprintf(stderr, "FAIL: engine_new\n");
    return 1;
  }

  // The tiny synthetic checkpoint has no tokenizer_dir layout issue; loading
  // the local directory exercises the full Agent load path.
  if (sj_load(e, "main", ckpt) != 0) {
    fprintf(stderr, "FAIL: sj_load -> %s\n", sj_error_message(e) ? sj_error_message(e) : "?");
    sj_engine_free(e);
    return 1;
  }

  const char* state = "{\"text\":\"I was charged twice, please refund\"}";
  const char* questions =
      "{\"q\":{\"type\":\"noul\",\"instructions\":\"Does the customer want money back?\"}}";

  char* out = sj_predict(e, state, questions, "");
  if (!out) {
    fprintf(stderr, "FAIL: sj_predict -> %s\n", sj_error_message(e) ? sj_error_message(e) : "?");
    sj_engine_free(e);
    return 1;
  }

  // The result must contain an "answers" key and the "q" answer with a noul.
  if (strstr(out, "\"answers\"") == NULL) {
    fprintf(stderr, "FAIL: no answers key in result: %s\n", out);
    sj_string_free(out);
    sj_engine_free(e);
    return 1;
  }

  printf("C API OK\n%s\n", out);
  sj_string_free(out);
  sj_engine_free(e);
  return 0;
}
