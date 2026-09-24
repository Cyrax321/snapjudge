// snapjudge lang regression: script/language detection pinned to fixtures
// (self-generated; upstream parity was validated during development).
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>
#include <fstream>
#include <string>

#include "nlohmann/json.hpp"
#include "snapjudge/lang.hpp"

using nlohmann::ordered_json;
using namespace snapjudge;

static std::string golden_dir() {
  if (const char* g = std::getenv("SNAPJUDGE_GOLDEN_DIR")) return g;
  return "tests/golden";
}

TEST_CASE("lang: detection pinned to committed fixtures") {
  std::ifstream f(golden_dir() + "/lang.json");
  REQUIRE(f.good());
  ordered_json g;
  f >> g;

  int i = 0;
  for (const auto& c : g["cases"]) {
    ++i;
    ordered_json st = c["state"];
    std::string label = "case " + std::to_string(i);
    CHECK_MESSAGE(state_text(st) == c["state_text"].get<std::string>(), label.c_str());
    CHECK_MESSAGE(detect_script(state_text(st)) == c["detect_script"].get<std::string>(), label.c_str());
    auto prof = script_profile(state_text(st));
    const auto& want = c["script_profile"];
    CHECK_MESSAGE(prof.size() == want.size(), label.c_str());
    for (auto it = want.begin(); it != want.end(); ++it) {
      REQUIRE_MESSAGE(prof.count(it.key()), (label + ": missing script " + it.key()).c_str());
      CHECK_MESSAGE(std::fabs(prof[it.key()] - it.value().get<double>()) < 1e-9, (label + ": " + it.key()).c_str());
    }
    CHECK_MESSAGE(analyse(st) == c["analyse"], label.c_str());
    CHECK_MESSAGE(is_english(st) == c["is_english"].get<bool>(), label.c_str());
  }
}

TEST_CASE("lang: hardcoded spot checks") {
  CHECK(detect_script("hello world") == "latin");
  CHECK(detect_script("我的账户") == "han");
  CHECK(detect_script("12345") == "unknown");
  CHECK(is_english("the quick brown fox and the lazy dog") == true);
  CHECK(is_english("mein konto und das geld") == false);
}
