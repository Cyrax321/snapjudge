// snapjudge SjPretrain (MLM) test: finite-difference gradcheck + overfit on the
// tiny native SjModel checkpoint.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "snapjudge/sj_pretrain.hpp"

using namespace snapjudge;

static std::string ckpt_dir() {
  const char* d = std::getenv("SNAPJUDGE_SJ_CKPT");
  return d ? d : "build/sj-tiny-ckpt";
}

// A short sequence of ids (cls=1, sep=2, mask=4, pad=3; vocab 512).
static std::vector<int64_t> seq() {
  return {1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 2};
}

TEST_CASE("SjPretrain loads encoder params") {
  SjPretrain p(ckpt_dir());
  bool has_word = false, has_q = false, has_mlm = false;
  for (const auto& n : p.param_names()) {
    if (n == "sj.embeddings.word.weight") has_word = true;
    if (n == "sj.encoder.layers.0.attn.q.weight") has_q = true;
    if (n == "sj.mlm.weight") has_mlm = true;
  }
  CHECK(has_word);
  CHECK(has_q);
  CHECK(has_mlm);
}

TEST_CASE("SjPretrain finite-difference gradcheck") {
  SjPretrain p(ckpt_dir());
  auto ids = seq();
  // warm up
  for (int i = 0; i < 3; ++i) { p.zero_grad(); p.step(ids, true); p.optimizer_step(1e-3); }

  // check a representative subset (every tensor is expensive; full check would
  // be slow on CPU, so sample the largest + smallest tensors).
  std::vector<std::string> to_check = {
      "sj.embeddings.word.weight", "sj.embeddings.norm.weight",
      "sj.encoder.layers.0.attn.q.weight", "sj.encoder.layers.0.attn.o.bias",
      "sj.encoder.layers.1.ffn.w1.weight", "sj.encoder.final_norm.weight",
      "sj.mlm.weight", "sj.mlm.bias",
  };
  for (const auto& name : to_check) {
    int64_t n = p.param_numel(name);
    int ncheck = std::min<int64_t>(n, 6);
    double worst = 0;
    for (int c = 0; c < ncheck; ++c) {
      int64_t i = (n * 2654435761ULL + c * 40503) % n;
      double orig = p.param_get(name, i);
      const double eps = 1e-3 * std::max(1.0, std::fabs(orig));
      p.param_set(name, i, (float)(orig + eps));
      double lp = p.loss_double(ids);
      p.param_set(name, i, (float)(orig - eps));
      double lm = p.loss_double(ids);
      p.param_set(name, i, (float)orig);
      p.zero_grad();
      p.step(ids, true);
      double analytic = p.grad_at(name, i);
      double num = (lp - lm) / (2 * eps);
      double rel = std::fabs(num - analytic) /
                   std::max(1e-3, std::max(std::fabs(num), std::fabs(analytic)));
      worst = std::max(worst, rel);
    }
    fprintf(stderr, "pretrain-gradcheck %s worst_rel=%.4g\n", name.c_str(), worst);
    CHECK(worst <= 8e-2);
  }
}

TEST_CASE("SjPretrain overfits a repeated sequence") {
  SjPretrain p(ckpt_dir());
  auto ids = seq();
  double l0 = p.loss_double(ids);
  for (int i = 0; i < 300; ++i) { p.zero_grad(); p.step(ids, true); p.optimizer_step(2e-2); }
  double l1 = p.loss_double(ids);
  fprintf(stderr, "mlm overfit: %.4f -> %.4f\n", l0, l1);
  CHECK(l1 < l0);
}
