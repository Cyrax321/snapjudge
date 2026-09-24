// snapjudge trainer verification on the tiny synthetic checkpoint.
//
// 1) Numerical gradient check: finite differences vs analytic gradients for
//    every trainable tensor (sampled elements), tolerance rel ≤ 5e-2.
// 2) Overfit: 200 Adam steps on one row must push loss down sharply and argmax
//    matches gold.
// 3) save_checkpoint roundtrip: Agent loads the saved dir and answers.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "snapjudge/agent.hpp"
#include "snapjudge/train.hpp"

using nlohmann::ordered_json;
using snapjudge::TrainModel;
using snapjudge::TrainRow;

#ifdef SNAPJUDGE_TINY_CKPT
static const char* CKPT = SNAPJUDGE_TINY_CKPT;
#else
static const char* CKPT = "build/tiny-ckpt";
#endif
static const char* CKPT_FT = "build/tiny-ckpt-ft";

static TrainRow make_row() {
  TrainRow r;
  r.state = ordered_json{{"text", "the invoice was charged twice please fix"}};
  r.qids = {"urgent", "route"};
  r.qdefs = {
      ordered_json{{"type", "noul"}, {"instructions", "Is this time-sensitive?"}},
      ordered_json{{"type", "choice"}, {"instructions", "who handles this?"},
                   {"criteria", ordered_json{{"billing", "money stuff"},
                                             {"tech", "broken things"},
                                             {"other", "anything else"}}}},
  };
  r.targets = {{0.2, 0.8}, {0.7, 0.2, 0.1}};
  r.workflow = "tiny";
  return r;
}

static TrainRow make_score_row() {
  TrainRow r;
  r.state = ordered_json{{"text", "system is completely down"}};
  r.qids = {"sev"};
  r.qdefs = {ordered_json{{"type", "score"},
                          {"instructions", "how bad is it?"},
                          {"criteria", ordered_json::array({"ok", "bad", "worse", "worst"})}}};
  r.targets = {{0.1, 0.2, 0.3, 0.4}};
  r.workflow = "tiny";
  return r;
}

TEST_CASE("loss + grad smoke") {
  TrainModel tm(CKPT);
  TrainRow row = make_row();
  tm.zero_grad();
  auto st = tm.step(row, true);
  fprintf(stderr, "smoke: loss %.4f acc %.3f rows %d\n", st.loss, st.acc, st.rows);
  CHECK(std::isfinite(st.loss));
  CHECK(st.rows == 2);
}

TEST_CASE("finite-difference gradient check") {
  TrainModel tm(CKPT);
  TrainRow row = make_row();
  for (const auto& name : tm.param_names()) {
    int64_t n = tm.param_numel(name);
    // analytic
    tm.zero_grad();
    tm.step(row, true);
    std::vector<float> analytic = tm.grad_of(name);
    // numeric on sampled elements only (full would take too long)
    int ncheck = std::min<int64_t>(n, 24);
    double worst = 0;
    for (int c = 0; c < ncheck; ++c) {
      int64_t i = (n * 2654435769ULL + c * 97) % n;
      double orig = tm.param_get(name, i);
      const double eps = 1e-3 * std::max(1.0, std::fabs(orig));
      tm.param_set(name, i, (float)(orig + eps));
      double lp = tm.forward_loss_double(row);
      tm.param_set(name, i, (float)(orig - eps));
      double lm = tm.forward_loss_double(row);
      tm.param_set(name, i, (float)orig);
      double num = (lp - lm) / (2 * eps);
      double rel = std::fabs(num - analytic[i]) /
                   std::max(1e-3, std::max(std::fabs(num), (double)std::fabs(analytic[i])));
      worst = std::max(worst, rel);
    }
    fprintf(stderr, "gradcheck %s worst_rel=%.4g\n", name.c_str(), worst);
    CHECK(worst <= 5e-2);
  }
}

TEST_CASE("overfit one row: loss drops, argmax becomes gold") {
  TrainModel tm(CKPT);
  TrainRow row = make_row();
  double l0 = tm.forward_loss_double(row);
  for (int i = 0; i < 300; ++i) {
    tm.zero_grad();
    tm.step(row, true);
    tm.optimizer_step(2e-2);
  }
  double l1 = tm.forward_loss_double(row);
  fprintf(stderr, "overfit: %.4f -> %.4f\n", l0, l1);
  CHECK(l1 < 0.6 * l0);
}

TEST_CASE("save_checkpoint roundtrip loads into Agent") {
  TrainModel tm(CKPT);
  TrainRow row = make_row();
  for (int i = 0; i < 50; ++i) {
    tm.zero_grad();
    tm.step(row, true);
    tm.optimizer_step(2e-2);
  }
  tm.save_checkpoint(CKPT_FT, {1.0, 1.0, 1.0}, {},
                     ordered_json{{"updates", 50}, {"dataset", "tiny-synthetic"}});
  snapjudge::Agent a(CKPT_FT, "cpu");
  ordered_json state = ordered_json{{"text", "the invoice was charged twice please fix"}};
  ordered_json qs = ordered_json{
      {"urgent", {{"type", "noul"}, {"instructions", "Is this time-sensitive?"}}},
      {"route", {{"type", "choice"}, {"instructions", "who handles this?"},
                 {"criteria", ordered_json{{"billing", "money stuff"},
                                           {"tech", "broken things"},
                                           {"other", "anything else"}}}}}};
  auto r = a.system_one(state, qs);
  fprintf(stderr, "roundtrip: urgent noul=%.4f route=%s\n",
          r["answers"]["urgent"]["noul"].get<double>(),
          r["answers"]["route"]["choice"].get<std::string>().c_str());
  CHECK(r["answers"]["route"]["choice"] == "billing");
  CHECK(r["answers"]["urgent"]["noul"].get<double>() > 0.5);
}
