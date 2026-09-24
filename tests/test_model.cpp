// snapjudge model regression: full forward on the tiny synthetic checkpoint,
// pinned to the committed fixture (self-generated after external parity was
// proven during development).
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "snapjudge/agent.hpp"

using nlohmann::ordered_json;
using snapjudge::Agent;

static std::string golden_dir() {
  if (const char* g = std::getenv("SNAPJUDGE_GOLDEN_DIR")) return g;
  return "tests/golden";
}
static std::string tiny() {
#ifdef SNAPJUDGE_TINY_CKPT
  return SNAPJUDGE_TINY_CKPT;
#else
  return "build/tiny-ckpt";
#endif
}

TEST_CASE("model forward on tiny checkpoint matches fixture") {
  std::ifstream gf(golden_dir() + "/model_tiny.json");
  REQUIRE(gf.good());
  ordered_json g;
  gf >> g;

  Agent agent(tiny(), "cpu");
  ordered_json state = "the invoice was charged twice please fix";
  ordered_json qs = ordered_json{
      {"urgent", {{"type", "noul"}, {"instructions", "Is this time-sensitive?"}}},
      {"route", {{"type", "choice"}, {"instructions", "who handles this?"},
                 {"criteria", ordered_json{{"billing", "money"},
                                           {"tech", "broken"},
                                           {"other", "rest"}}}}}};
  ordered_json res = agent.system_one(state, qs);
  CHECK(res == g["system_one"]);
}
