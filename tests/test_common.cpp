// snapjudge common regression: rendered strings / clamps pinned to fixtures.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "snapjudge/common.hpp"

using nlohmann::ordered_json;
using namespace snapjudge;

static std::string golden_dir() {
  if (const char* g = std::getenv("SNAPJUDGE_GOLDEN_DIR")) return g;
  return "tests/golden";
}

TEST_CASE("render_criterion / render_options / serialize_state pinned") {
  std::ifstream f(golden_dir() + "/common.json");
  REQUIRE(f.good());
  ordered_json g;
  f >> g;

  SUBCASE("render_criterion") {
    for (const auto& c : g["render_criterion"])
      CHECK(render_criterion(c["in"]) == c["out"].get<std::string>());
  }
  SUBCASE("render_options") {
    for (const auto& c : g["render_options"]) {
      InternalQ q;
      q.t = c["q"]["t"].get<std::string>();
      q.ins = c["q"]["ins"].get<std::string>();
      q.crit = c["q"]["crit"];
      CHECK(render_options(q) == c["out"].get<std::vector<std::string>>());
    }
  }
  SUBCASE("serialize_state") {
    for (const auto& c : g["serialize_state"])
      CHECK(serialize_state(c["in"]) == c["out"].get<std::string>());
  }
  SUBCASE("clamp_temperature") {
    for (const auto& c : g["clamp_temperature"]) {
      const ordered_json& in = c["in"];
      double got;
      if (in.is_null()) got = clamp_temperature(static_cast<const Json&>(in));
      else if (in.is_string()) {
        if (in.get<std::string>() == "__NAN__") got = clamp_temperature(std::nan(""));
        else got = clamp_temperature(static_cast<const Json&>(in));
      } else got = clamp_temperature(in.get<double>());
      CHECK(got == doctest::Approx(c["out"].get<double>()));
    }
  }
  SUBCASE("temp_bucket") {
    for (const auto& c : g["temp_bucket"])
      CHECK(temp_bucket(c["in"][0], c["in"][1]) == c["out"].get<std::string>());
  }
}

TEST_CASE("proper_reward / ece_score / confidence_from_probs basics") {
  CHECK(confidence_from_probs(nullptr, 1) == 1.0);
  double half[2] = {0.5, 0.5};
  CHECK(confidence_from_probs(half, 2) == doctest::Approx(0.0));
  double sure[2] = {0.999, 1e-3};
  CHECK(confidence_from_probs(sure, 2) > 0.95);
  CHECK(ece_score({}, {}) != ece_score({}, {}));  // nan != nan path
}
