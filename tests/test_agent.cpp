// snapjudge agent regression: predict_batch semantics, empty questions,
// validation errors — on the tiny synthetic checkpoint.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "snapjudge/agent.hpp"

using nlohmann::ordered_json;
using snapjudge::Agent;

static std::string tiny() {
#ifdef SNAPJUDGE_TINY_CKPT
  return SNAPJUDGE_TINY_CKPT;
#else
  return "build/tiny-ckpt";
#endif
}

TEST_CASE("agent: empty questions -> empty answers, zero tokens") {
  Agent agent(tiny(), "cpu");
  ordered_json res = agent.system_one("hello", ordered_json::object());
  CHECK(res["answers"].empty());
  CHECK(res["usage"]["input_tokens"] == 0);
}

TEST_CASE("agent: unknown question type is rejected with a named error") {
  Agent agent(tiny(), "cpu");
  ordered_json qs = {{"bad", {{"type", "nope"}, {"instructions", "?"}}}};
  try {
    agent.system_one("x", qs);
    CHECK(false);
  } catch (const std::invalid_argument& e) {
    CHECK(std::string(e.what()).find("bad") != std::string::npos);
    CHECK(std::string(e.what()).find("unknown type") != std::string::npos);
  }
}

TEST_CASE("agent: malformed criteria shapes are rejected") {
  Agent agent(tiny(), "cpu");
  // choice without criteria
  try {
    agent.system_one("x", ordered_json{{"a", {{"type","choice"},{"instructions","i"}}}});
    CHECK(false);
  } catch (const std::invalid_argument&) {}
  // score with dict criteria
  try {
    agent.system_one("x", ordered_json{{"a", {{"type","score"},{"instructions","i"},{"criteria",ordered_json::object()}}}});
    CHECK(false);
  } catch (const std::invalid_argument&) {}
}

TEST_CASE("agent: predict_batch chunks and aligns with states") {
  Agent agent(tiny(), "cpu");
  ordered_json state = "the invoice was charged twice please fix";
  ordered_json qs = {
      {"urgent", {{"type", "noul"}, {"instructions", "Is this time-sensitive?"}}},
      {"route", {{"type", "choice"}, {"instructions", "who?"},
                 {"criteria", ordered_json{{"billing", "m"}, {"tech", "t"}, {"other", "o"}}}}}};
  auto single = agent.system_one(state, qs);
  auto batch = agent.predict_batch({state, state, state}, qs, 2);
  CHECK(batch.size() == 3);
  for (const auto& r : batch) CHECK(r == single);
}
