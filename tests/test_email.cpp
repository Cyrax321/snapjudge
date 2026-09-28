// snapjudge email/presets regression: pinned to committed fixtures.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <fstream>
#include <string>

#include "nlohmann/json.hpp"
#include "snapjudge/email.hpp"
#include "snapjudge/presets.hpp"

using nlohmann::ordered_json;
using namespace snapjudge;

static std::string golden_dir() {
  if (const char* g = std::getenv("SNAPJUDGE_GOLDEN_DIR")) return g;
  return "tests/golden";
}

TEST_CASE("email cleaning pinned to committed fixtures") {
#ifdef SNAPJUDGE_HAVE_PCRE2
  std::ifstream f(golden_dir() + "/email.json");
  REQUIRE(f.good());
  ordered_json g;
  f >> g;
  int i = 0;
  for (const auto& c : g["clean_cases"]) {
    ++i;
    std::string got = clean_email_body(c["in"].get<std::string>());
    std::string want = c["out"].get<std::string>();
    if (got != want)
      fprintf(stderr, "clean case %d:\n got  %s\n want %s\n", i, got.c_str(), want.c_str());
    CHECK(got == want);
    ordered_json st = email_state("Re: Invoice #1042", c["in"].get<std::string>(),
                                  ordered_json::object(), "a@b.com");
    CHECK(st == c["state"]);
  }
#else
  // No PCRE2: clean_email_body runs in no-op mode. Verify it is still callable
  // and returns the input unchanged (minus whitespace normalization).
  CHECK(clean_email_body("plain body") == "plain body");
  ordered_json st = email_state("s", "b", ordered_json::object(), "a@b.com");
  CHECK(st["subject"] == "s");
  CHECK(st["body"] == "b");
#endif
}

TEST_CASE("presets are structurally complete") {
  CHECK(triage_questions().size() == 5);
  CHECK(email_questions().size() == 5);
  CHECK(guard_questions().size() == 5);
  CHECK(moderation_questions().size() == 5);
  CHECK(router_questions().size() == 4);
  {  // store first — iterators from temporary containers cannot compare
    ordered_json tq = triage_questions();
    for (auto it = tq.begin(); it != tq.end(); ++it) {
      const ordered_json& q = it.value();
      CHECK(q.contains("type"));
      CHECK(q.contains("instructions"));
    }
  }
}
