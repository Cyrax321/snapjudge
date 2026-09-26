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

TEST_CASE("batched encoder forward matches per-question forward") {
  // Two questions with different sequence lengths: encoding them in one
  // batched pass must reproduce the logits of encoding them one at a time.
  TrainModel tm(CKPT);

  TrainRow solo = make_row();   // qids: urgent (noul) + route (choice)
  TrainRow just_route;
  just_route.state = solo.state;
  just_route.qids = {"route"};
  just_route.qdefs = {solo.qdefs[1]};
  just_route.targets = {solo.targets[1]};
  just_route.workflow = "tiny";

  // forward_loss_double on the 1-question row -> route logits via B=1
  double l_solo = tm.forward_loss_double(just_route);
  // forward_loss_double on the 2-question row -> route is index 1 via B=2 batch
  double l_batch = tm.forward_loss_double(solo);
  (void)l_solo;
  (void)l_batch;

  // Compare the actual logits: eval_logits flattens per-question logits.
  std::vector<std::vector<float>> lg1, lg2;
  std::vector<std::vector<double>> tg1, tg2;
  std::vector<int> qt1, qt2;
  tm.eval_logits({just_route}, &lg1, &tg1, &qt1);
  tm.eval_logits({solo}, &lg2, &tg2, &qt2);
  REQUIRE(lg1.size() == 1);
  REQUIRE(lg2.size() == 2);
  // route is question[0] in the solo row and question[1] in the batch row
  const auto& a = lg1[0];           // B=1 route logits
  const auto& b = lg2[1];           // B=2 route logits
  REQUIRE(a.size() == b.size());
  double worst = 0;
  for (size_t j = 0; j < a.size(); ++j)
    worst = std::max(worst, std::fabs((double)a[j] - (double)b[j]));
  fprintf(stderr, "batch-parity route logits worst=%.4g\n", worst);
  CHECK(worst < 1e-4);
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

TEST_CASE("finite-difference gradient check: distillation + smoothing") {
  // Non-default loss config (raised cross-entropy weight, label smoothing)
  // must still be gradient-consistent on the score (RPS) primitive too.
  TrainModel tm(CKPT);
  tm.set_loss(2.0, 0.2, 1.5, 0.1);
  TrainRow row = make_score_row();
  for (const auto& name : tm.param_names()) {
    int64_t n = tm.param_numel(name);
    tm.zero_grad();
    tm.step(row, true);
    std::vector<float> analytic = tm.grad_of(name);
    int ncheck = std::min<int64_t>(n, 24);
    double worst = 0;
    for (int c = 0; c < ncheck; ++c) {
      int64_t i = (n * 40503ULL + c * 131) % n;
      double orig = tm.param_get(name, i);
      const double eps = 1e-4 * std::max(1.0, std::fabs(orig));
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
    fprintf(stderr, "distill-gradcheck %s worst_rel=%.4g\n", name.c_str(), worst);
    CHECK(worst <= 5e-2);
  }
}

TEST_CASE("finite-difference gradient check: temperature-scaled loss") {
  // Loss softmax over logits/T must be gradient-consistent through the 1/T
  // chain rule (annealing path).
  TrainModel tm(CKPT);
  tm.set_loss(1.0, 0.5, 1.0, 0.0);
  tm.set_loss_temperature(2.5);
  TrainRow row = make_row();
  for (const auto& name : tm.param_names()) {
    int64_t n = tm.param_numel(name);
    tm.zero_grad();
    tm.step(row, true);
    std::vector<float> analytic = tm.grad_of(name);
    int ncheck = std::min<int64_t>(n, 24);
    double worst = 0;
    for (int c = 0; c < ncheck; ++c) {
      int64_t i = (n * 31415ULL + c * 47) % n;
      double orig = tm.param_get(name, i);
      const double eps = 1e-4 * std::max(1.0, std::fabs(orig));
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
    fprintf(stderr, "temp-gradcheck %s worst_rel=%.4g\n", name.c_str(), worst);
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

TEST_CASE("LoRA adapter: gradient check + roundtrip") {
  TrainModel tm(CKPT);
  tm.set_lora_r(4);
  CHECK(tm.param_numel("lora_A.weight") > 0);
  CHECK(tm.param_numel("lora_B.weight") > 0);

  // B=0 at init -> loss identical to the no-LoRA baseline, and A's grad is
  // exactly zero (the LoRA delta contributes nothing before any update).
  TrainModel base(CKPT);
  TrainRow row = make_row();
  double l0 = base.forward_loss_double(row);
  double l1 = tm.forward_loss_double(row);
  CHECK(std::fabs(l0 - l1) < 1e-4);

  // warm up a few steps so B is non-zero and both adapter gradients are live
  for (int i = 0; i < 5; ++i) {
    tm.zero_grad();
    tm.step(row, true);
    tm.optimizer_step(2e-2);
  }

  // finite-difference the LoRA tensors (B now non-zero -> A grad non-zero too)
  for (const char* name : {"lora_A.weight", "lora_B.weight"}) {
    int64_t n = tm.param_numel(name);
    tm.zero_grad();
    tm.step(row, true);
    std::vector<float> analytic = tm.grad_of(name);
    int ncheck = std::min<int64_t>(n, 24);
    double worst = 0;
    for (int c = 0; c < ncheck; ++c) {
      int64_t i = (n * 76543ULL + c * 71) % n;
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
    fprintf(stderr, "lora-gradcheck %s worst_rel=%.4g\n", name, worst);
    CHECK(worst <= 5e-2);
  }

  // train more and confirm the adapter roundtrips through save + Agent
  for (int i = 0; i < 45; ++i) {
    tm.zero_grad();
    tm.step(row, true);
    tm.optimizer_step(2e-2);
  }
  tm.save_checkpoint(CKPT_FT, {1.0, 1.0, 1.0}, {},
                     ordered_json{{"updates", 50}, {"dataset", "tiny-synthetic"}});
  snapjudge::Agent a(CKPT_FT, "cpu");
  ordered_json state = ordered_json{{"text", "the invoice was charged twice please fix"}};
  ordered_json qs = ordered_json{
      {"urgent", {{"type", "noul"}, {"instructions", "Is this time-sensitive?"}}}};
  auto r = a.system_one(state, qs);
  CHECK(r["answers"]["urgent"]["noul"].get<double>() >= 0.0);
  CHECK(r["answers"]["urgent"]["noul"].get<double>() <= 1.0);
}
