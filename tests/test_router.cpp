// snapjudge router + shortlist regression: pinned to committed fixtures
// (self-generated; upstream parity was validated during development).
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <fstream>
#include <string>
#include <unordered_map>

#include "nlohmann/json.hpp"
#include "snapjudge/router.hpp"
#include "snapjudge/shortlist.hpp"

using nlohmann::ordered_json;
using namespace snapjudge;

static std::string golden_dir() {
  if (const char* g = std::getenv("SNAPJUDGE_GOLDEN_DIR")) return g;
  return "tests/golden";
}

TEST_CASE("router route() pinned to committed fixtures") {
  std::ifstream f(golden_dir() + "/router.json");
  REQUIRE(f.good());
  ordered_json g;
  f >> g;
  const auto& cases = g["route_cases"];
  // inputs mirror tests/gen_fixtures.cpp Spec list
  struct Spec { std::string text, model, task, lang, guess; };
  std::vector<Spec> specs = {
      {"I was charged twice, please refund"}, {"Mein Konto wurde zweimal belastet"},
      {"my account is charged twice"}, {"ami nota chai"}, {"Quero cancelar"},
      {"Order 1042"}, {"12345!!!"}, {"Hello", "", "", "de", ""},
      {"Hello", "", "", "en-US", ""}, {"Hello", "", "", "en_US.UTF-8", ""},
      {"Hello", "ml"}, {"Hello", "typed"}, {"hi", "", "typed_decisions"},
      {"hi", "", "", "", "de"}, {"hi", "", "", "", "en"},
  };
  REQUIRE(cases.size() == specs.size());
  Router r;
  ordered_json q = {{"q", {{"type", "noul"}, {"instructions", "x?"}}}};
  for (size_t i = 0; i < specs.size(); ++i) {
    auto got = r.route(ordered_json{{"text", specs[i].text}}, q, specs[i].model,
                       specs[i].task, specs[i].lang, specs[i].guess);
    const auto& want = cases[i];
    if (got != want)
      fprintf(stderr, "route %zu:\n got  %s\n want %s\n", i,
              got.dump(-1, ' ', false).c_str(), want.dump(-1, ' ', false).c_str());
    CHECK(got["model"] == want["model"]);
    CHECK(got["reason"] == want["reason"]);
    CHECK(got["repo"] == want["repo"]);
  }
}

TEST_CASE("normalise_name aliases and errors") {
  CHECK(normalise_name("en") == "english");
  CHECK(normalise_name("EN") == "english");
  CHECK(normalise_name("ml") == "multilingual");
  CHECK(normalise_name("typed") == "typed-decisions");
  CHECK(normalise_name("decisions") == "typed-decisions");
  CHECK(normalise_name("english") == "english");
  CHECK(normalise_name("typed-decisions") == "typed-decisions");
  try { normalise_name("nope"); CHECK(false); } catch (const std::invalid_argument&) {}
  try { normalise_name("french"); CHECK(false); } catch (const std::invalid_argument&) {}
}

TEST_CASE("typed-decision workflow signature matching") {
  ordered_json q1 = {{"action", ordered_json::object()},
                     {"needs_review", ordered_json::object()},
                     {"outcome", ordered_json::object()},
                     {"risk", ordered_json::object()},
                     {"urgency", ordered_json::object()}};
  CHECK(match_typed_decisions_workflow(q1) == "agent_trace_observability");
  ordered_json q2 = {{"action", ordered_json::object()},
                     {"needs_review", ordered_json::object()},
                     {"outcome", ordered_json::object()},
                     {"risk", ordered_json::object()}};
  CHECK(match_typed_decisions_workflow(q2).empty());
  ordered_json q3 = {{"urgency", ordered_json::object()}};
  CHECK(match_typed_decisions_workflow(q3).empty());
}

TEST_CASE("shortlist_choice: cosine ranking, stable ties, passthrough") {
  // deterministic fake embeddings: label index decides similarity
  std::unordered_map<std::string, std::vector<double>> embs;
  embs["query"] = {1.0, 0.0};
  embs["close"] = {0.9, 0.1};
  embs["far"] = {0.0, 1.0};
  embs["middle"] = {0.7, 0.7};
  embs["farthest"] = {-1.0, 0.0};
  EmbedFn embed = [&](const std::vector<std::string>& ts) {
    std::vector<std::vector<double>> out;
    for (const auto& t : ts) out.push_back(embs.at(t));
    return out;
  };
  ordered_json criteria = {{"close", nullptr}, {"far", nullptr},
                           {"middle", nullptr}, {"farthest", nullptr}};
  auto got = shortlist_choice("query", criteria, embed, 2);
  CHECK((got == std::vector<std::string>{"close", "middle"}));
  auto all = shortlist_choice("query", criteria, embed, 10);
  CHECK(all.size() == 4);
}

TEST_CASE("predict_shortlist_above: threshold gates high-cardinality choice") {
  std::unordered_map<std::string, std::vector<double>> embs;
  embs["query"] = {1.0, 0.0};
  embs["a"] = {0.9, 0.1};
  embs["b"] = {0.7, 0.3};
  embs["c"] = {0.1, 0.9};
  embs["d"] = {-0.5, 0.5};
  EmbedFn embed = [&](const std::vector<std::string>& ts) {
    std::vector<std::vector<double>> out;
    for (const auto& t : ts) out.push_back(embs.at(t));
    return out;
  };
  ordered_json big_criteria = {{"a", nullptr}, {"b", nullptr},
                               {"c", nullptr}, {"d", nullptr}};
  ordered_json questions = {
      {"big", {{"type", "choice"}, {"instructions", ""},
               {"criteria", big_criteria}}},
      {"small", {{"type", "choice"}, {"instructions", ""},
                 {"criteria", ordered_json{{"x", nullptr}, {"y", nullptr}}}}}};

  auto runner = [](const ordered_json& s, const ordered_json& q) {
    ordered_json answers = ordered_json::object();
    for (auto it = q.begin(); it != q.end(); ++it) {
      const auto& c = it.value()["criteria"];
      std::string first = c.begin().key();
      answers[it.key()] = {{"type", "choice"}, {"choice", first}};
    }
    return ordered_json{{"answers", answers}};
  };

  // threshold 2: only the 4-option question is shortlisted to k=2
  auto r = predict_shortlist_above(runner, "query", questions, embed, 2, 2);
  CHECK(r.contains("shortlist"));
  CHECK(r["shortlist"].contains("big"));
  CHECK(!r["shortlist"].contains("small"));
  CHECK(r["shortlist"]["big"]["labels"].size() == 2);

  // threshold 10: nothing exceeds it -> no shortlist key at all
  auto r2 = predict_shortlist_above(runner, "query", questions, embed, 2, 10);
  CHECK(!r2.contains("shortlist"));
  CHECK(r2["answers"]["big"]["choice"] == "a");
  CHECK(r2["answers"]["small"]["choice"] == "x");
}
