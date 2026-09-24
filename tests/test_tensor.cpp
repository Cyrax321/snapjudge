// snapjudge tensor parity: C++ ops vs numpy fp32 goldens.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "snapjudge/safetensors.hpp"
#include "snapjudge/tensor.hpp"

using nlohmann::json;
using namespace snapjudge;

static json G;

static std::string golden_dir() {
  if (const char* g = std::getenv("SNAPJUDGE_GOLDEN_DIR")) return g;
  return "tests/golden";
}

static Tensor from_json(const json& j) {
  // accepts nested lists of floats
  std::vector<int64_t> shape;
  json cur = j;
  while (cur.is_array() && cur.size() && cur[0].is_array()) {
    shape.push_back(static_cast<int64_t>(cur.size()));
    cur = cur[0];
  }
  shape.push_back(static_cast<int64_t>(cur.size()));
  Tensor t(shape);
  int64_t idx = 0;
  std::function<void(const json&)> fill = [&](const json& n) {
    if (n.is_array())
      for (const auto& e : n) fill(e);
    else t.data[static_cast<size_t>(idx++)] = n.get<float>();
  };
  fill(j);
  return t;
}

static void check_close(const Tensor& got, const json& want, float tol, const char* name) {
  Tensor wt = from_json(want);
  CHECK(got.shape == wt.shape);
  double max_err = 0;
  for (int64_t i = 0; i < got.numel(); ++i) {
    double d = std::fabs(static_cast<double>(got.data[i]) - wt.data[i]);
    max_err = std::max(max_err, d);
  }
  fprintf(stderr, "%s: max_err=%.3g\n", name, max_err);
  CHECK(max_err <= tol);
}

TEST_CASE("tensor ops vs numpy golden") {
  std::ifstream f(golden_dir() + "/tensor_ops.json");
  REQUIRE_MESSAGE(f.good(), "tensor_ops.json missing (run python/make_tensor_golden.py)");
  f >> G;
  Tensor a = from_json(G["gemm"]["a"]);
  Tensor w = from_json(G["gemm"]["w"]);
  std::vector<float> bias;
  for (float v : G["gemm"]["bias"].get<std::vector<float>>()) bias.push_back(v);

  check_close(gemm_nt(a, w, bias.data(), 0), G["gemm"]["out"], 1e-5f, "gemm");
  check_close(gemm_nt(a, w, bias.data(), 1), G["gemm_gelu"]["out"], 1e-5f, "gemm_gelu");
  check_close(gemm_nt(a, w, bias.data(), 2), G["gemm_relu"]["out"], 1e-5f, "gemm_relu");
  check_close(gemm_geglu(a, from_json(G["geglu"]["wi"])), G["geglu"]["out"], 1e-5f, "geglu");
  check_close(layer_norm(from_json(G["layer_norm"]["x"]),
                         from_json(G["layer_norm"]["w"]).data_ptr(),
                         from_json(G["layer_norm"]["b"]).data_ptr(), 1e-5f),
              G["layer_norm"]["out"], 1e-4f, "layer_norm");
  check_close(layer_norm(from_json(G["layer_norm"]["x"]),
                         from_json(G["layer_norm"]["w"]).data_ptr(), nullptr, 1e-5f),
              G["layer_norm"]["out_nobias"], 1e-4f, "layer_norm_nobias");
  {
    Tensor z = from_json(G["softmax"]["z"]);
    softmax_rows(z);
    check_close(z, G["softmax"]["out"], 1e-6f, "softmax");
  }
  {
    Tensor q = from_json(G["attn_full"]["q"]);
    Tensor k = from_json(G["attn_full"]["k"]);
    Tensor v = from_json(G["attn_full"]["v"]);
    std::vector<int64_t> lens = G["attn_full"]["lens"].get<std::vector<int64_t>>();
    check_close(attention(q, k, v, lens, 0), G["attn_full"]["out"], 1e-5f, "attn_full");
    check_close(attention(q, k, v, lens, 2), G["attn_window"]["out"], 1e-5f, "attn_window");
  }
  {
    Tensor qkv = from_json(G["rope"]["qkv"]);
    int64_t L = G["rope"]["L"];
    int64_t H = G["rope"]["H"];
    int64_t Dh = G["rope"]["Dh"];
    Tensor c = from_json(G["rope"]["cos"]);
    Tensor s = from_json(G["rope"]["sin"]);
    rope_inplace(qkv, c.data_ptr(), s.data_ptr(), L, H, Dh);
    check_close(qkv, G["rope"]["out"], 1e-5f, "rope");
  }
}

TEST_CASE("safetensors: round trip fp16/bf16/fp32") {
  // Build a tiny safetensors file in memory and read it back.
  std::string path = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") +
                     "/snapjudge_test.safetensors";
  {
    FILE* f = fopen(path.c_str(), "wb");
    REQUIRE(f != nullptr);
    json hdr = {
        {"x_f32", {{"dtype", "F32"}, {"shape", {2, 2}}, {"data_offsets", {0, 16}}}},
        {"x_f16", {{"dtype", "F16"}, {"shape", {2}}, {"data_offsets", {16, 20}}}},
        {"x_bf16", {{"dtype", "BF16"}, {"shape", {2}}, {"data_offsets", {20, 24}}}},
    };
    std::string h = hdr.dump();
    uint64_t hl = h.size();
    fwrite(&hl, 8, 1, f);
    fwrite(h.data(), 1, h.size(), f);
    float f32[4] = {1.5f, -2.25f, 0.0f, 1e30f};
    fwrite(f32, 4, 4, f);
    // 1.0 fp16 = 0x3C00, -4.5 fp16 = 0xC480
    uint16_t f16[2] = {0x3C00, 0xC480};
    fwrite(f16, 2, 2, f);
    // 1.0 bf16 = 0x3F80, -4.5 bf16 = 0xC090
    uint16_t bf16[2] = {0x3F80, 0xC090};
    fwrite(bf16, 2, 2, f);
    fclose(f);
  }
  auto st = SafeTensors::load(path);
  CHECK(st.has("x_f32"));
  CHECK(st.has("x_f16"));
  CHECK_FALSE(st.has("nope"));
  CHECK(st.shape_of("x_f32") == std::vector<int64_t>{2, 2});
  {
    auto d = st.data_f32("x_f32");
    CHECK(d[0] == 1.5f);
    CHECK(d[1] == -2.25f);
    CHECK(d[3] == 1e30f);
  }
  {
    auto d = st.data_f32("x_f16");
    CHECK(d[0] == doctest::Approx(1.0f));
    CHECK(d[1] == doctest::Approx(-4.5f));
  }
  {
    auto d = st.data_f32("x_bf16");
    CHECK(d[0] == doctest::Approx(1.0f));
    CHECK(d[1] == doctest::Approx(-4.5f));
  }
  std::remove(path.c_str());
}

TEST_CASE("safetensors: tiny checkpoint loads and dtypes are right") {
#ifdef SNAPJUDGE_TINY_CKPT
  std::string p = SNAPJUDGE_TINY_CKPT;
#else
  std::string p = "build/tiny-ckpt";
#endif
  std::string path = p + "/model.safetensors";
  std::ifstream f(path);
  if (!f.good()) { MESSAGE("tiny checkpoint fixture not built; skipping"); return; }
  auto st = SafeTensors::load(path);
  CHECK(st.has("encoder.layers.0.attn.Wqkv.weight"));
  CHECK(st.shape_of("encoder.layers.0.attn.Wqkv.weight") == std::vector<int64_t>{48, 16});
  CHECK(st.shape_of("type_emb.weight") == std::vector<int64_t>{3, 16});
  auto d = st.data_f32("temperature");
  CHECK(st.numel_of("temperature") == 3);
  CHECK(d[0] > 0.0f);
}
