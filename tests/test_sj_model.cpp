// snapjudge SjModel (native model) test: load the tiny synthetic checkpoint and
// run a real encoder + decision-head forward. Pins the own-model path.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <fstream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "snapjudge/safetensors.hpp"
#include "snapjudge/sj_model.hpp"

using namespace snapjudge;
using nlohmann::json;

static std::string ckpt_dir() {
  const char* d = std::getenv("SNAPJUDGE_SJ_CKPT");
  return d ? d : "build/sj-tiny-ckpt";
}

TEST_CASE("SjModel loads and runs a forward") {
  std::ifstream cfgf(ckpt_dir() + "/sj_config.json");
  REQUIRE(cfgf.good());
  json cfg;
  cfgf >> cfg;

  SafeTensors w = SafeTensors::load(ckpt_dir() + "/model.safetensors");
  auto m = SjModel::load(cfg, w, "sj-tiny");
  REQUIRE(m != nullptr);

  const SjConfig& c = m->cfg();
  CHECK(c.hidden_size == 16);
  CHECK(c.num_heads == 2);
  CHECK(c.num_layers == 2);

  // A tiny batch: 1 row, 6 tokens. cls=1, sep=2, mask=4, pad=3.
  std::vector<std::vector<int64_t>> ids = {{1, 5, 6, 4, 7, 2}};
  std::vector<std::vector<int64_t>> mask = {{1, 1, 1, 1, 1, 1}};

  // encode() must return [B, L, D] with finite values.
  auto h = m->encode(ids, mask);
  REQUIRE(h.size() == 1);
  REQUIRE(h[0].size() == 6);
  REQUIRE(h[0][0].size() == c.hidden_size);
  CHECK(std::isfinite(h[0][0][0]));

  // forward(): one marker at position 3, one question, qtype choice (0).
  std::vector<std::vector<int64_t>> mpos = {{3}};
  std::vector<std::vector<uint8_t>> mmask = {{1}};
  std::vector<int64_t> qtype = {0};
  std::vector<std::vector<float>> logits, act;
  m->forward(ids, mask, mpos, mmask, qtype, logits, act);
  REQUIRE(logits.size() == 1);
  REQUIRE(logits[0].size() == 1);
  CHECK(std::isfinite(logits[0][0]));
  CHECK(act.size() == 1);
  CHECK(act[0].size() == 2);
}

TEST_CASE("SjModel rejects missing tensors with a clear error") {
  std::ifstream cfgf(ckpt_dir() + "/sj_config.json");
  json cfg;
  cfgf >> cfg;
  SafeTensors w = SafeTensors::load(ckpt_dir() + "/model.safetensors");
  // Corrupt config: wrong hidden size so shapes mismatch.
  json bad = cfg;
  bad["hidden_size"] = 32;
  CHECK_THROWS_AS(SjModel::load(bad, w, "sj-tiny"), std::runtime_error);
}
