#include "snapjudge/sj_pretrain.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <stdexcept>

#include "nlohmann/json.hpp"
#include "snapjudge/backward.hpp"
#include "snapjudge/safetensors.hpp"
#include "snapjudge/safewrite.hpp"
#include "snapjudge/sj_model.hpp"

namespace snapjudge {

namespace fs = std::filesystem;
using nlohmann::json;

extern "C" {
void cblas_sgemm(int Order, int TransA, int TransB, int M, int N, int K,
                 float alpha, const float* A, int lda, const float* B, int ldb,
                 float beta, float* C, int ldc);
}

namespace {
constexpr int RM = 101, NO_T = 111, TR = 112;

struct PStore {
  std::map<std::string, std::vector<float>> w, m1, m2, g;
  std::vector<std::string> names;
  void add(const std::string& n, std::vector<float> v) {
    w[n] = std::move(v);
    names.push_back(n);
    m1[n].resize(w[n].size(), 0.f);
    m2[n].resize(w[n].size(), 0.f);
    g[n].resize(w[n].size(), 0.f);
  }
  float* at(const std::string& n) { return w.at(n).data(); }
  const float* at(const std::string& n) const { return w.at(n).data(); }
  int64_t numel(const std::string& n) const { return (int64_t)w.at(n).size(); }
  void zero_grad() { for (auto& n : names) std::fill(g[n].begin(), g[n].end(), 0.f); }
};

// y = x @ W^T + b (+ act). Caches x and pre-activation.
struct GemmC { std::vector<float> A, pre; };
void gemm_c(const float* A, int64_t M, int64_t K, const float* W, int64_t N,
            const float* bias, int act, GemmC& c, std::vector<float>& out) {
  c.A.assign(A, A + M * K);
  out.assign((size_t)(M * N), 0.f);
  cblas_sgemm(RM, NO_T, TR, (int)M, (int)N, (int)K, 1.0f, A, (int)K, W, (int)K, 0.0f,
              out.data(), (int)N);
  c.pre = out;
  if (bias) for (int64_t i = 0; i < M; ++i) for (int64_t j = 0; j < N; ++j)
    c.pre[i * N + j] += bias[j];
  out = c.pre;
  if (act == 1) for (auto& v : out) v = 0.5f * v * (1.0f + std::erf(v * 0.70710678118654757f));
  else if (act == 2) for (auto& v : out) v = std::max(v, 0.f);
}

struct LnC { std::vector<float> x, mean, var, inv; };
void ln_c(const float* x, int64_t M, int64_t D, const float* w, const float* b, float eps,
          LnC& c, std::vector<float>& out) {
  c.x.assign(x, x + M * D);
  c.mean.assign((size_t)M, 0.0); c.var.assign((size_t)M, 0.0); c.inv.assign((size_t)M, 0.0);
  out.assign((size_t)(M * D), 0.f);
  for (int64_t r = 0; r < M; ++r) {
    const float* xr = x + r * D;
    double mean = 0; for (int64_t j = 0; j < D; ++j) mean += xr[j]; mean /= D;
    double var = 0; for (int64_t j = 0; j < D; ++j) { double d = xr[j] - mean; var += d * d; } var /= D;
    float inv = (float)(1.0 / std::sqrt(var + eps));
    c.mean[r] = mean; c.var[r] = var; c.inv[r] = inv;
    for (int64_t j = 0; j < D; ++j) {
      float v = (xr[j] - (float)mean) * inv * w[j];
      out[r * D + j] = b ? v + b[j] : v;
    }
  }
}

}  // namespace

struct SjPretrain::Impl {
  SjConfig cfg;
  PStore P;
  int64_t adam_t = 0;
  int D = 0, H = 0, Dh = 0, F = 0, V = 0;
  std::mt19937 rng{7};
  std::string base_dir;
};

SjPretrain::SjPretrain(const std::string& ckpt_dir) : impl_(std::make_unique<Impl>()) {
  Impl& im = *impl_;
  im.base_dir = ckpt_dir;
  json cfg;
  {
    std::ifstream f(fs::path(ckpt_dir) / "sj_config.json");
    if (!f) throw std::runtime_error("SjPretrain: missing sj_config.json in " + ckpt_dir);
    f >> cfg;
  }
  SafeTensors w = SafeTensors::load((fs::path(ckpt_dir) / "model.safetensors").string());
  auto m = SjModel::load(cfg, w, ckpt_dir);
  const SjConfig& c = m->cfg();
  im.cfg = c;
  im.D = c.hidden_size; im.H = c.num_heads; im.Dh = im.D / im.H;
  im.F = c.intermediate_size; im.V = c.vocab_size;

  auto pull = [&](const std::string& name) {
    if (!w.has(name)) throw std::runtime_error("SjPretrain: missing " + name);
    auto sh = w.shape_of(name);
    int64_t n = 1; for (auto d : sh) n *= d;
    auto dat = w.data_f32(name);
    im.P.add(name, std::vector<float>(dat.get(), dat.get() + n));
  };

  pull("sj.embeddings.word.weight");
  pull("sj.embeddings.position.weight");
  pull("sj.embeddings.norm.weight");
  pull("sj.embeddings.norm.bias");
  for (int i = 0; i < c.num_layers; ++i) {
    std::string p = "sj.encoder.layers." + std::to_string(i) + ".";
    for (const char* s : {"attn_norm.weight","attn_norm.bias","attn.q.weight","attn.q.bias",
                          "attn.k.weight","attn.k.bias","attn.v.weight","attn.v.bias",
                          "attn.o.weight","attn.o.bias","ffn_norm.weight","ffn_norm.bias",
                          "ffn.w1.weight","ffn.w1.bias","ffn.w2.weight","ffn.w2.bias"})
      pull(p + s);
  }
  pull("sj.encoder.final_norm.weight");
  pull("sj.encoder.final_norm.bias");

  // MLM head: initialize from the word embedding (tied-ish init) or small noise.
  if (w.has("sj.mlm.weight")) {
    pull("sj.mlm.weight");
    pull("sj.mlm.bias");
  } else {
    // Tie to word embeddings at init (standard BERT initialization).
    auto emb = w.data_f32("sj.embeddings.word.weight");
    std::vector<float> mlm_w(emb.get(), emb.get() + (size_t)c.vocab_size * c.hidden_size);
    im.P.add("sj.mlm.weight", std::move(mlm_w));
    im.P.add("sj.mlm.bias", std::vector<float>((size_t)c.vocab_size, 0.f));
  }
}

SjPretrain::~SjPretrain() = default;

void SjPretrain::zero_grad() { impl_->P.zero_grad(); }

std::vector<std::string> SjPretrain::param_names() const { return impl_->P.names; }
int64_t SjPretrain::param_numel(const std::string& n) const { return impl_->P.numel(n); }
float SjPretrain::param_get(const std::string& n, int64_t i) const { return impl_->P.at(n)[i]; }
void SjPretrain::param_set(const std::string& n, int64_t i, float v) { impl_->P.at(n)[i] = v; }
float SjPretrain::grad_at(const std::string& n, int64_t i) const { return impl_->P.g.at(n)[i]; }

namespace {

// Forward + loss for a single masked sequence, with caches for backward.
struct EncFwd {
  int64_t L = 0;
  std::vector<int64_t> ids, masked_ids;   // masked_ids: token ids after masking
  std::vector<int> mask_pos;              // positions to predict
  LnC emb_norm;
  struct Layer {
    LnC an, fn;
    GemmC q, k, v, o, f1, f2;
    std::vector<float> attn_probs;        // [H, L, L]
    std::vector<float> X_in;              // residual input [L,D]
  };
  std::vector<Layer> layers;
  LnC final_norm;
  std::vector<float> h;                   // [L, D] final hidden
  GemmC mlm;                              // mlm head (act 0)
  std::vector<float> logits;              // [L, V]
};

void forward_enc(SjPretrain::Impl& im, EncFwd& f) {
  const int D = im.D, H = im.H, Dh = im.Dh, F = im.F;
  const int64_t L = f.L;
  // word + position embeddings, then LayerNorm
  std::vector<float> emb((size_t)(L * D));
  for (int64_t i = 0; i < L; ++i) {
    float* row = emb.data() + i * D;
    std::memcpy(row, im.P.at("sj.embeddings.word.weight") + f.masked_ids[i] * D, sizeof(float) * D);
    const float* pos = im.P.at("sj.embeddings.position.weight") + i * D;
    for (int j = 0; j < D; ++j) row[j] += pos[j];
  }
  std::vector<float> X;
  ln_c(emb.data(), L, D, im.P.at("sj.embeddings.norm.weight"), im.P.at("sj.embeddings.norm.bias"), im.cfg.norm_eps, f.emb_norm, X);

  f.layers.resize((size_t)im.cfg.num_layers);
  for (int li = 0; li < im.cfg.num_layers; ++li) {
    std::string p = "sj.encoder.layers." + std::to_string(li) + ".";
    EncFwd::Layer& ly = f.layers[li];
    ly.X_in = X;
    std::vector<float> y; ln_c(X.data(), L, D, im.P.at(p + "attn_norm.weight"), im.P.at(p + "attn_norm.bias"), im.cfg.norm_eps, ly.an, y);
    std::vector<float> q, k, v;
    gemm_c(y.data(), L, D, im.P.at(p + "attn.q.weight"), D, im.P.at(p + "attn.q.bias"), 0, ly.q, q);
    gemm_c(y.data(), L, D, im.P.at(p + "attn.k.weight"), D, im.P.at(p + "attn.k.bias"), 0, ly.k, k);
    gemm_c(y.data(), L, D, im.P.at(p + "attn.v.weight"), D, im.P.at(p + "attn.v.bias"), 0, ly.v, v);
    std::vector<float> Q((size_t)(H * L * Dh)), Kt((size_t)(H * L * Dh)), V((size_t)(H * L * Dh));
    for (int64_t i = 0; i < L; ++i)
      for (int h = 0; h < H; ++h) {
        std::memcpy(Q.data() + (h * L + i) * Dh, q.data() + i * D + h * Dh, sizeof(float) * Dh);
        std::memcpy(Kt.data() + (h * L + i) * Dh, k.data() + i * D + h * Dh, sizeof(float) * Dh);
        std::memcpy(V.data() + (h * L + i) * Dh, v.data() + i * D + h * Dh, sizeof(float) * Dh);
      }
    std::vector<float> At((size_t)(H * L * Dh));
    ly.attn_probs.resize((size_t)(H * L * L));
    attention_forward_one(Q.data(), Kt.data(), V.data(), H, L, Dh, L, 0, At.data(), ly.attn_probs.data());
    std::vector<float> ctx((size_t)(L * D));
    for (int64_t i = 0; i < L; ++i)
      for (int h = 0; h < H; ++h)
        std::memcpy(ctx.data() + i * D + h * Dh, At.data() + (h * L + i) * Dh, sizeof(float) * Dh);
    std::vector<float> proj; gemm_c(ctx.data(), L, D, im.P.at(p + "attn.o.weight"), D, im.P.at(p + "attn.o.bias"), 0, ly.o, proj);
    for (int64_t i = 0; i < L * D; ++i) X[i] += proj[i];
    std::vector<float> y2; ln_c(X.data(), L, D, im.P.at(p + "ffn_norm.weight"), im.P.at(p + "ffn_norm.bias"), im.cfg.norm_eps, ly.fn, y2);
    std::vector<float> f1; gemm_c(y2.data(), L, D, im.P.at(p + "ffn.w1.weight"), F, im.P.at(p + "ffn.w1.bias"), 1, ly.f1, f1);
    std::vector<float> f2; gemm_c(f1.data(), L, F, im.P.at(p + "ffn.w2.weight"), D, im.P.at(p + "ffn.w2.bias"), 0, ly.f2, f2);
    for (int64_t i = 0; i < L * D; ++i) X[i] += f2[i];
  }
  std::vector<float> h;
  ln_c(X.data(), L, D, im.P.at("sj.encoder.final_norm.weight"), im.P.at("sj.encoder.final_norm.bias"), im.cfg.norm_eps, f.final_norm, h);
  f.h = std::move(h);
  std::vector<float> logits;
  gemm_c(f.h.data(), L, D, im.P.at("sj.mlm.weight"), im.V, im.P.at("sj.mlm.bias"), 0, f.mlm, logits);
  f.logits = std::move(logits);
}

double mlm_loss(SjPretrain::Impl& im, EncFwd& f) {
  const int64_t L = f.L, V = im.V;
  double loss = 0; int n = 0;
  for (int pos : f.mask_pos) {
    const float* lg = f.logits.data() + (int64_t)pos * V;
    float mx = lg[0];
    for (int j = 1; j < V; ++j) mx = std::max(mx, lg[j]);
    double s = 0;
    for (int j = 0; j < V; ++j) s += std::exp((double)lg[j] - mx);
    int gold = (int)f.ids[pos];
    loss += -std::log(std::exp((double)lg[gold] - mx) / s);
    ++n;
  }
  return n ? loss / n : -1.0;
}

void backward_enc(SjPretrain::Impl& im, EncFwd& f) {
  const int D = im.D, H = im.H, Dh = im.Dh, F = im.F, V = im.V;
  const int64_t L = f.L;

  // dlogits at masked positions (softmax CE gradient)
  std::vector<float> dlogits((size_t)(L * V), 0.f);
  for (int pos : f.mask_pos) {
    const float* lg = f.logits.data() + (int64_t)pos * V;
    float mx = lg[0];
    for (int j = 1; j < V; ++j) mx = std::max(mx, lg[j]);
    std::vector<double> p((size_t)V);
    double s = 0;
    for (int j = 0; j < V; ++j) { p[j] = std::exp((double)lg[j] - mx); s += p[j]; }
    for (auto& v : p) v /= s;
    int gold = (int)f.ids[pos];
    double scale = 1.0 / f.mask_pos.size();
    for (int j = 0; j < V; ++j) {
      double g = p[j] - (j == gold ? 1.0 : 0.0);
      dlogits[(int64_t)pos * V + j] = (float)(g * scale);
    }
  }

  // MLM head: logits = h @ mlm_w^T (+b)
  std::vector<float> dh((size_t)(L * D), 0.f);
  {
    std::vector<float> dA((size_t)(L * D), 0.f), dW((size_t)(V * D), 0.f), db((size_t)V, 0.f);
    gemm_nt_backward(dlogits.data(), f.mlm.pre.data(), f.mlm.A.data(), im.P.at("sj.mlm.weight"), L, V, D, 0, dA.data(), dW.data(), db.data());
    for (int64_t i = 0; i < V * D; ++i) im.P.g["sj.mlm.weight"][i] += dW[i];
    for (int64_t i = 0; i < V; ++i) im.P.g["sj.mlm.bias"][i] += db[i];
    for (int64_t i = 0; i < L * D; ++i) dh[i] += dA[i];
  }

  // final norm
  std::vector<float> dX((size_t)(L * D)), dw((size_t)D, 0.f), db((size_t)D, 0.f);
  layer_norm_backward(dh.data(), f.final_norm.x.data(), im.P.at("sj.encoder.final_norm.weight"), L, D, im.cfg.norm_eps, true, dX.data(), dw.data(), db.data());
  for (int64_t i = 0; i < D; ++i) { im.P.g["sj.encoder.final_norm.weight"][i] += dw[i]; im.P.g["sj.encoder.final_norm.bias"][i] += db[i]; }

  // layers in reverse
  for (int li = im.cfg.num_layers - 1; li >= 0; --li) {
    std::string p = "sj.encoder.layers." + std::to_string(li) + ".";
    EncFwd::Layer& ly = f.layers[li];
    // ffn.w2
    std::vector<float> dA((size_t)(L * F), 0.f), dW((size_t)(D * F), 0.f), dbb((size_t)D, 0.f);
    gemm_nt_backward(dX.data(), ly.f2.pre.data(), ly.f2.A.data(), im.P.at(p + "ffn.w2.weight"), L, D, F, 0, dA.data(), dW.data(), dbb.data());
    for (int64_t i = 0; i < D * F; ++i) im.P.g[p + "ffn.w2.weight"][i] += dW[i];
    for (int64_t i = 0; i < D; ++i) im.P.g[p + "ffn.w2.bias"][i] += dbb[i];
    // ffn.w1 GELU
    std::vector<float> dA2((size_t)(L * D), 0.f), dW2((size_t)(F * D), 0.f), db2((size_t)F, 0.f);
    gemm_nt_backward(dA.data(), ly.f1.pre.data(), ly.f1.A.data(), im.P.at(p + "ffn.w1.weight"), L, F, D, 1, dA2.data(), dW2.data(), db2.data());
    for (int64_t i = 0; i < F * D; ++i) im.P.g[p + "ffn.w1.weight"][i] += dW2[i];
    for (int64_t i = 0; i < F; ++i) im.P.g[p + "ffn.w1.bias"][i] += db2[i];
    // ffn_norm
    std::vector<float> dxl((size_t)(L * D)), dw3((size_t)D, 0.f), db3((size_t)D, 0.f);
    layer_norm_backward(dA2.data(), ly.fn.x.data(), im.P.at(p + "ffn_norm.weight"), L, D, im.cfg.norm_eps, true, dxl.data(), dw3.data(), db3.data());
    for (int64_t i = 0; i < D; ++i) { im.P.g[p + "ffn_norm.weight"][i] += dw3[i]; im.P.g[p + "ffn_norm.bias"][i] += db3[i]; }
    // residual: dX += dxl
    for (int64_t i = 0; i < L * D; ++i) dX[i] += dxl[i];

    // attn.o
    std::vector<float> dproj = dX;
    std::vector<float> dao((size_t)(L * D), 0.f), dWo((size_t)(D * D), 0.f), dbo((size_t)D, 0.f);
    gemm_nt_backward(dproj.data(), ly.o.pre.data(), ly.o.A.data(), im.P.at(p + "attn.o.weight"), L, D, D, 0, dao.data(), dWo.data(), dbo.data());
    for (int64_t i = 0; i < D * D; ++i) im.P.g[p + "attn.o.weight"][i] += dWo[i];
    for (int64_t i = 0; i < D; ++i) im.P.g[p + "attn.o.bias"][i] += dbo[i];
    // attention backward
    std::vector<float> dAt((size_t)(H * L * Dh));
    for (int64_t i = 0; i < L; ++i)
      for (int h = 0; h < H; ++h)
        std::memcpy(dAt.data() + (h * L + i) * Dh, dao.data() + i * D + h * Dh, sizeof(float) * Dh);
    std::vector<float> Q((size_t)(H * L * Dh)), Kt((size_t)(H * L * Dh)), Vv((size_t)(H * L * Dh));
    for (int64_t i = 0; i < L; ++i)
      for (int h = 0; h < H; ++h) {
        std::memcpy(Q.data() + (h * L + i) * Dh, ly.q.pre.data() + i * D + h * Dh, sizeof(float) * Dh);
        std::memcpy(Kt.data() + (h * L + i) * Dh, ly.k.pre.data() + i * D + h * Dh, sizeof(float) * Dh);
        std::memcpy(Vv.data() + (h * L + i) * Dh, ly.v.pre.data() + i * D + h * Dh, sizeof(float) * Dh);
      }
    std::vector<float> dQ((size_t)(H * L * Dh), 0.f), dK((size_t)(H * L * Dh), 0.f), dV((size_t)(H * L * Dh), 0.f);
    attention_backward_one(dAt.data(), Q.data(), Kt.data(), Vv.data(), ly.attn_probs.data(), H, L, Dh, L, 0, dQ.data(), dK.data(), dV.data());
    std::vector<float> dq((size_t)(L * D), 0.f), dk((size_t)(L * D), 0.f), dv((size_t)(L * D), 0.f);
    for (int64_t i = 0; i < L; ++i)
      for (int h = 0; h < H; ++h) {
        std::memcpy(dq.data() + i * D + h * Dh, dQ.data() + (h * L + i) * Dh, sizeof(float) * Dh);
        std::memcpy(dk.data() + i * D + h * Dh, dK.data() + (h * L + i) * Dh, sizeof(float) * Dh);
        std::memcpy(dv.data() + i * D + h * Dh, dV.data() + (h * L + i) * Dh, sizeof(float) * Dh);
      }
    std::vector<float> dy((size_t)(L * D), 0.f);
    auto proj_back = [&](const std::string& proj, const std::vector<float>& dp, const GemmC& cache) {
      std::vector<float> dA3((size_t)(L * D), 0.f), dW3((size_t)(D * D), 0.f), db4((size_t)D, 0.f);
      gemm_nt_backward(dp.data(), cache.pre.data(), cache.A.data(), im.P.at(p + proj + ".weight"), L, D, D, 0, dA3.data(), dW3.data(), db4.data());
      for (int64_t i = 0; i < D * D; ++i) im.P.g[p + proj + ".weight"][i] += dW3[i];
      for (int64_t i = 0; i < D; ++i) im.P.g[p + proj + ".bias"][i] += db4[i];
      for (int64_t i = 0; i < L * D; ++i) dy[i] += dA3[i];
    };
    proj_back("attn.q", dq, ly.q);
    proj_back("attn.k", dk, ly.k);
    proj_back("attn.v", dv, ly.v);
    // attn_norm
    std::vector<float> dxl1((size_t)(L * D)), dw4((size_t)D, 0.f), db5((size_t)D, 0.f);
    layer_norm_backward(dy.data(), ly.an.x.data(), im.P.at(p + "attn_norm.weight"), L, D, im.cfg.norm_eps, true, dxl1.data(), dw4.data(), db5.data());
    for (int64_t i = 0; i < D; ++i) { im.P.g[p + "attn_norm.weight"][i] += dw4[i]; im.P.g[p + "attn_norm.bias"][i] += db5[i]; }
    // residual: dX = dxl1 + dproj (attn residual)
    for (int64_t i = 0; i < L * D; ++i) dX[i] = dxl1[i] + dproj[i];
  }

  // embeddings norm
  std::vector<float> demb((size_t)(L * D)), dw5((size_t)D, 0.f), db6((size_t)D, 0.f);
  layer_norm_backward(dX.data(), f.emb_norm.x.data(), im.P.at("sj.embeddings.norm.weight"), L, D, im.cfg.norm_eps, true, demb.data(), dw5.data(), db6.data());
  for (int64_t i = 0; i < D; ++i) { im.P.g["sj.embeddings.norm.weight"][i] += dw5[i]; im.P.g["sj.embeddings.norm.bias"][i] += db6[i]; }
  // word embedding backward (scatter-add)
  embedding_backward(demb.data(), f.masked_ids.data(), L, D, im.P.g["sj.embeddings.word.weight"].data());
  // position embedding backward (each position unique)
  for (int64_t i = 0; i < L; ++i) {
    float* dst = im.P.g["sj.embeddings.position.weight"].data() + i * D;
    const float* src = demb.data() + i * D;
    for (int j = 0; j < D; ++j) dst[j] += src[j];
  }
}

}  // namespace

double SjPretrain::step(const std::vector<int64_t>& ids, bool backward) {
  Impl& im = *impl_;
  EncFwd f;
  f.L = (int64_t)ids.size();
  f.ids = ids;
  f.masked_ids = ids;
  // choose mask positions deterministically from the ids (a hash of the
  // sequence), so step() and loss_double() always mask the same tokens for the
  // same input. Real training still sees varied masks because real batches are
  // different text; the deterministic choice is what makes gradient checking
  // (and reproducible overfit) correct.
  uint64_t h = 1469598103934665603ULL;
  for (int64_t id : ids) { h ^= (uint64_t)id + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2); }
  std::mt19937 rng(static_cast<uint32_t>(h));
  std::vector<int> candidates;
  for (int i = 0; i < (int)ids.size(); ++i) {
    int64_t id = ids[i];
    if (id == im.cfg.cls_id || id == im.cfg.sep_id || id == im.cfg.pad_id) continue;
    candidates.push_back(i);
  }
  if (candidates.empty()) return -1.0;
  int nmask = std::max(1, (int)(candidates.size() * 0.15));
  std::shuffle(candidates.begin(), candidates.end(), rng);
  f.mask_pos.assign(candidates.begin(), candidates.begin() + nmask);
  std::sort(f.mask_pos.begin(), f.mask_pos.end());
  std::uniform_real_distribution<double> u(0, 1);
  for (int pos : f.mask_pos) {
    double r = u(rng);
    if (r < 0.8) f.masked_ids[pos] = im.cfg.mask_id;
    else if (r < 0.9) f.masked_ids[pos] = (int64_t)(u(rng) * im.V);
    // else keep
  }

  forward_enc(im, f);
  double loss = mlm_loss(im, f);
  if (backward && loss >= 0) backward_enc(im, f);
  return loss;
}

double SjPretrain::loss_double(const std::vector<int64_t>& ids) {
  Impl& im = *impl_;
  EncFwd f;
  f.L = (int64_t)ids.size();
  f.ids = ids; f.masked_ids = ids;
  uint64_t h = 1469598103934665603ULL;
  for (int64_t id : ids) { h ^= (uint64_t)id + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2); }
  std::mt19937 rng(static_cast<uint32_t>(h));
  std::vector<int> candidates;
  for (int i = 0; i < (int)ids.size(); ++i) {
    int64_t id = ids[i];
    if (id == im.cfg.cls_id || id == im.cfg.sep_id || id == im.cfg.pad_id) continue;
    candidates.push_back(i);
  }
  if (candidates.empty()) return -1.0;
  int nmask = std::max(1, (int)(candidates.size() * 0.15));
  std::shuffle(candidates.begin(), candidates.end(), rng);
  f.mask_pos.assign(candidates.begin(), candidates.begin() + nmask);
  std::sort(f.mask_pos.begin(), f.mask_pos.end());
  std::uniform_real_distribution<double> u(0, 1);
  for (int pos : f.mask_pos) {
    double r = u(rng);
    if (r < 0.8) f.masked_ids[pos] = im.cfg.mask_id;
    else if (r < 0.9) f.masked_ids[pos] = (int64_t)(u(rng) * im.V);
  }
  forward_enc(im, f);
  return mlm_loss(im, f);
}

void SjPretrain::optimizer_step(double lr) {
  Impl& im = *impl_;
  ++im.adam_t;
  double b1 = 0.9, b2 = 0.999, wd = 0.01, eps = 1e-8;
  for (const auto& n : im.P.names) {
    float* w = im.P.at(n);
    float* g = im.P.g[n].data();
    float* m1 = im.P.m1[n].data();
    float* m2 = im.P.m2[n].data();
    int64_t sz = im.P.numel(n);
    for (int64_t i = 0; i < sz; ++i) {
      double gi = g[i] + wd * w[i];
      m1[i] = (float)(b1 * m1[i] + (1 - b1) * gi);
      m2[i] = (float)(b2 * m2[i] + (1 - b2) * gi * gi);
      double m1h = m1[i] / (1 - std::pow(b1, im.adam_t));
      double m2h = m2[i] / (1 - std::pow(b2, im.adam_t));
      w[i] -= (float)(lr * m1h / (std::sqrt(m2h) + eps));
    }
  }
}

void SjPretrain::save_checkpoint(const std::string& out_dir) {
  Impl& im = *impl_;
  fs::create_directories(out_dir);
  SafeTensors base = SafeTensors::load((fs::path(im.base_dir) / "model.safetensors").string());
  std::vector<TensorOut> out;
  for (const auto& name : base.names()) {
    // keep head/scorer/type_emb from base; encoder + embeddings are ours
    if (name.rfind("sj.embeddings", 0) == 0 || name.rfind("sj.encoder", 0) == 0) continue;
    TensorOut t; t.name = name; t.shape = base.shape_of(name);
    t.data = base.data_f32(name); t.as_f16 = true;
    out.push_back(std::move(t));
  }
  for (const auto& n : im.P.names) {
    if (n == "sj.mlm.weight" || n == "sj.mlm.bias") continue;  // not part of the checkpoint
    TensorOut t; t.name = n; t.shape = base.shape_of(n); t.as_f16 = true;
    auto sp = std::shared_ptr<float[]>(new float[(size_t)im.P.numel(n)]);
    std::memcpy(sp.get(), im.P.at(n), sizeof(float) * (size_t)im.P.numel(n));
    t.data = std::move(sp);
    out.push_back(std::move(t));
  }
  save_safetensors((fs::path(out_dir) / "model.safetensors").string(), out);
  std::ifstream cfgf(fs::path(im.base_dir) / "sj_config.json");
  std::ofstream cfgo(fs::path(out_dir) / "sj_config.json");
  cfgo << cfgf.rdbuf();
  if (fs::exists(fs::path(im.base_dir) / "tokenizer"))
    fs::copy(fs::path(im.base_dir) / "tokenizer", fs::path(out_dir) / "tokenizer",
             fs::copy_options::recursive | fs::copy_options::overwrite_existing);
}

}  // namespace snapjudge
