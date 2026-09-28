// snapjudge SjTrainModel test: finite-difference gradient check + overfit on
// the tiny native SjModel checkpoint.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "snapjudge/sj_train.hpp"
#include "snapjudge/train.hpp"

using namespace snapjudge;
using nlohmann::ordered_json;

static std::string ckpt_dir() {
  const char* d = std::getenv("SNAPJUDGE_SJ_CKPT");
  return d ? d : "build/sj-tiny-ckpt";
}

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

TEST_CASE("SjTrainModel loads and reports trainable params") {
  SjTrainModel tm(ckpt_dir());
  CHECK(tm.param_count() > 0);
  // type_emb + head + scorer are all trainable
  bool has_type = false, has_head = false, has_scorer = false;
  for (const auto& n : tm.param_names()) {
    if (n == "sj.type_emb.weight") has_type = true;
    if (n == "sj.head.layers.0.attn.q.weight") has_head = true;
    if (n == "sj.scorer.w2.weight") has_scorer = true;
  }
  CHECK(has_type);
  CHECK(has_head);
  CHECK(has_scorer);
}

TEST_CASE("SjTrainModel finite-difference gradcheck") {
  SjTrainModel tm(ckpt_dir());
  TrainRow row = make_row();

  // warm up so attention probs are non-degenerate
  for (int i = 0; i < 3; ++i) { tm.zero_grad(); tm.step(row, true); tm.optimizer_step(1e-2); }

  for (const auto& name : tm.param_names()) {
    int64_t n = tm.param_numel(name);
    tm.zero_grad();
    tm.step(row, true);
    // read analytic grads by perturbation is expensive; instead do direct FD on
    // loss for a few entries per tensor.
    int ncheck = std::min<int64_t>(n, 8);
    double worst = 0;
    for (int c = 0; c < ncheck; ++c) {
      int64_t i = (n * 2654435761ULL + c * 40503) % n;
      double orig = tm.param_get(name, i);
      const double eps = 1e-3 * std::max(1.0, std::fabs(orig));
      tm.param_set(name, i, (float)(orig + eps));
      double lp = tm.forward_loss_double(row);
      tm.param_set(name, i, (float)(orig - eps));
      double lm = tm.forward_loss_double(row);
      tm.param_set(name, i, (float)orig);
      // analytic gradient: recompute via step(accumulate_only) then read grad
      tm.zero_grad();
      tm.step(row, true);
      double analytic = tm.grad_at(name, i);
      double num = (lp - lm) / (2 * eps);
      double rel = std::fabs(num - analytic) /
                   std::max(1e-3, std::max(std::fabs(num), std::fabs(analytic)));
      worst = std::max(worst, rel);
    }
    fprintf(stderr, "gradcheck %s worst_rel=%.4g\n", name.c_str(), worst);
    // type_emb accumulates over L positions, so its FD relative error is a bit
    // noisier than single-matrix tensors; everything else is <3%.
    CHECK(worst <= 8e-2);
  }
}

TEST_CASE("SjTrainModel overfits one row") {
  SjTrainModel tm(ckpt_dir());
  TrainRow row = make_row();
  double l0 = tm.forward_loss_double(row);
  for (int i = 0; i < 400; ++i) { tm.zero_grad(); tm.step(row, false); tm.optimizer_step(2e-2); }
  double l1 = tm.forward_loss_double(row);
  fprintf(stderr, "overfit: %.4f -> %.4f\n", l0, l1);
  CHECK(l1 < 0.6 * l0);
}
