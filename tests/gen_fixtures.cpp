// snapjudge fixture generator (dev tool, not part of ctest).
//
// Runs snapjudge's own implementations over fixed batteries and writes the
// outputs to tests/golden/ as regression fixtures. The tests then pin those
// outputs. The initial fixtures were verified against the upstream reference
// implementation during development; from that point on they lock OUR outputs.
//
// Run from the repo root:  ./build/gen_fixtures [golden_dir]
#include <fstream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "snapjudge/email.hpp"
#include "snapjudge/lang.hpp"
#include "snapjudge/presets.hpp"
#include "snapjudge/router.hpp"
#include "snapjudge/shortlist.hpp"
#include "snapjudge/tokenizer.hpp"
#include "snapjudge/common.hpp"
#include "snapjudge/model.hpp"
#include "snapjudge/agent.hpp"

using nlohmann::ordered_json;
using namespace snapjudge;

static std::vector<std::string> texts;
static std::string GOLDEN;

static void write(const std::string& name, const ordered_json& j) {
  std::ofstream(GOLDEN + "/" + name) << j.dump(2, ' ', false);
  std::fprintf(stderr, "wrote %s/%s\n", GOLDEN.c_str(), name.c_str());
}

int main(int argc, char** argv) {
  GOLDEN = argc > 1 ? argv[1] : "tests/golden";
  std::string tiny = std::getenv("SNAPJUDGE_TINY_CKPT") ? std::getenv("SNAPJUDGE_TINY_CKPT")
                                                        : "build/tiny-ckpt";

  // input list shared by the tokenizer/lang fixtures
  {
    std::ifstream f(GOLDEN + "/texts.json");
    ordered_json t;
    f >> t;
    texts = t["texts"].get<std::vector<std::string>>();
  }

  // ---- tokenizer ----
  {
    auto tok = Tokenizer::from_dir(tiny + "/tokenizer");
    ordered_json cases = ordered_json::array();
    for (const auto& s : texts) {
      auto ids = tok->encode(s);
      cases.push_back({{"text", s}, {"ids", ids}});
    }
    write("tokenizer_tiny.json",
          {{"special", {{"cls", tok->cls_id}, {"sep", tok->sep_id}, {"pad", tok->pad_id},
                        {"mask", tok->mask_id}, {"mask_token", tok->mask_token}}},
           {"cases", cases}});
  }

  // ---- lang ----
  {
    ordered_json cases = ordered_json::array();
    for (const auto& s : texts) {
      ordered_json st = s;
      cases.push_back({{"state", st},
                       {"state_text", state_text(st)},
                       {"detect_script", detect_script(state_text(st))},
                       {"script_profile", script_profile(state_text(st))},
                       {"analyse", analyse(st)},
                       {"is_english", is_english(st)}});
    }
    write("lang.json", {{"cases", cases}});
  }

  // ---- common ----
  {
    ordered_json rc = ordered_json::array();
    std::vector<ordered_json> crits = {
        "plain text",
        ordered_json{{"desc", "x"}, {"n", 3}},
        ordered_json::array({"a", 1, true}),
        0, false, "unicode naive",
    };
    auto render_of = [](InternalQ q) {
      ordered_json out = {{"t", q.t}, {"ins", q.ins}, {"crit", q.crit}};
      return ordered_json{{"q", out}, {"out", render_options(q)}};
    };
    InternalQ q1; q1.t = "choice"; q1.ins = "";
    q1.crit = ordered_json{{"a", nullptr}, {"b", "bee"}, {"c", ordered_json{{"x", 1}}}};
    InternalQ q2; q2.t = "score"; q2.ins = "";
    q2.crit = ordered_json{"zero", "one", ordered_json{{"d", 2}}};
    InternalQ q3; q3.t = "noul"; q3.ins = "";
    q3.crit = ordered_json::object();
    ordered_json ro = {render_of(q1), render_of(q2), render_of(q3)};
    ordered_json crits_unused; (void)crits_unused;

    ordered_json crits_j = ordered_json::array({
        {{"in", "plain text"}, {"out", render_criterion("plain text")}},
        {{"in", ordered_json{{"desc", "x"}, {"n", 3}}},
         {"out", render_criterion(ordered_json{{"desc", "x"}, {"n", 3}})}},
        {{"in", ordered_json::array({"a", 1, true})},
         {"out", render_criterion(ordered_json::array({"a", 1, true}))}},
        {{"in", 0}, {"out", render_criterion(0)}},
        {{"in", false}, {"out", render_criterion(false)}},
        {{"in", "unicode naive"}, {"out", render_criterion("unicode naive")}},
    });
    ordered_json ser = ordered_json::array({
        {{"in", "just text"}, {"out", serialize_state("just text")}},
        {{"in", ordered_json{{"a", 1}, {"b", {"x", "y"}}}},
         {"out", serialize_state(ordered_json{{"a", 1}, {"b", {"x", "y"}}})}},
    });
    ordered_json clamp_cases = ordered_json::array({
        {{"in", 3.0}, {"out", clamp_temperature(3.0)}},
        {{"in", 0.1}, {"out", clamp_temperature(0.1)}},
        {{"in", 7.0}, {"out", clamp_temperature(7.0)}},
        {{"in", "__NAN__"}, {"out", clamp_temperature(std::nan(""))}},
        {{"in", nullptr}, {"out", clamp_temperature(static_cast<const Json&>(nullptr))}},
        {{"in", "2.5"}, {"out", clamp_temperature(static_cast<const Json&>("2.5"))}},
        {{"in", "oops"}, {"out", clamp_temperature(static_cast<const Json&>("oops"))}},
        {{"in", -0.0}, {"out", clamp_temperature(-0.0)}},
    });
    ordered_json buckets = ordered_json::array({
        {{"in", {0, 2}}, {"out", temp_bucket(0, 2)}},
        {{"in", {0, 11}}, {"out", temp_bucket(0, 11)}},
        {{"in", {1, 5}}, {"out", temp_bucket(1, 5)}},
        {{"in", {2, 3}}, {"out", temp_bucket(2, 3)}},
    });
    write("common.json",
          {{"render_criterion", crits_j}, {"render_options", ro},
           {"serialize_state", ser}, {"clamp_temperature", clamp_cases},
           {"temp_bucket", buckets}});
  }

  // ---- email ----
  {
    std::vector<std::string> bodies = {
        "I was charged twice on my card ending 4242. Please refund.",
        "Hi team, my invoice is wrong again.\n\nThanks,\nSarah\n\nOn Tue, Sep 3, 2024 at 2:15 PM Support <support@acme.com> wrote:\n> We have resolved your issue.\n> Please check.",
        "Please help, the app crashes on launch.\n\nBest,\nKim",
        "Hello, the link is broken.\n\nSent from my iPhone",
        "Boa tarde, preciso da nota fiscal.\n\nAtenciosamente,\nMaria",
    };
    ordered_json cases = ordered_json::array();
    for (const auto& b : bodies) {
      cases.push_back({{"in", b}, {"out", clean_email_body(b)},
                       {"state", email_state("Re: Invoice #1042", b,
                                             ordered_json::object(), "a@b.com")}});
    }
    write("email.json", {{"clean_cases", cases}});
  }

  // ---- router ----
  {
    Router r;
    struct Spec { std::string text, model, task, lang, guess; };
    std::vector<Spec> specs = {
        {"I was charged twice, please refund"}, {"Mein Konto wurde zweimal belastet"},
        {"my account is charged twice"}, {"ami nota chai"}, {"Quero cancelar"},
        {"Order 1042"}, {"12345!!!"}, {"Hello", "", "", "de", ""},
        {"Hello", "", "", "en-US", ""}, {"Hello", "", "", "en_US.UTF-8", ""},
        {"Hello", "ml"}, {"Hello", "typed"}, {"hi", "", "typed_decisions"},
        {"hi", "", "", "", "de"}, {"hi", "", "", "", "en"},
    };
    ordered_json cases = ordered_json::array();
    ordered_json q = {{"q", {{"type", "noul"}, {"instructions", "x?"}}}};
    for (const auto& s : specs)
      cases.push_back(r.route(ordered_json{{"text", s.text}}, q, s.model, s.task,
                              s.lang, s.guess));
    write("router.json", {{"route_cases", cases}});
  }

  // ---- model + agent on the tiny checkpoint ----
  {
    Agent agent(tiny, "cpu");
    ordered_json qs = ordered_json{
        {"urgent", {{"type", "noul"}, {"instructions", "Is this time-sensitive?"}}},
        {"route", {{"type", "choice"}, {"instructions", "who handles this?"},
                   {"criteria", ordered_json{{"billing", "money"},
                                             {"tech", "broken"},
                                             {"other", "rest"}}}}}};
    ordered_json state = "the invoice was charged twice please fix";
    ordered_json res = agent.system_one(state, qs);
    write("model_tiny.json", {{"system_one", res}});
  }

  std::fprintf(stderr, "fixtures written to %s\n", GOLDEN.c_str());
  return 0;
}
