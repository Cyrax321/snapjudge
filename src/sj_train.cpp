#include "snapjudge/sj_train.hpp"

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
#include "snapjudge/common.hpp"
#include "snapjudge/safetensors.hpp"
#include "snapjudge/safewrite.hpp"
#include "snapjudge/sj_model.hpp"
#include "snapjudge/tokenizer.hpp"

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

// ------------------------ parameter store ------------------------------------
struct PStore {
  std::map<std::string, std::vector<float>> w, shape_t, m1, m2, g;
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

// ------------------------ loss (proper scoring rule) --------------------------
struct LossParams {
  double w_nll = 1.0, w_sph = 0.5, w_rps = 1.0, label_smoothing = 0.0, temperature = 1.0;
};

void smoothed_target(const std::vector<double>& t, double eps, std::vector<double>& ts) {
  ts.assign(t.begin(), t.end());
  if (eps > 0) for (size_t i = 0; i < t.size(); ++i) ts[i] = (1.0 - eps) * t[i] + eps / t.size();
}

// d(loss)/d(logits)
void loss_grad(const float* logits, const std::vector<double>& target, int qtype,
               const LossParams& lp, float* dz) {
  const int64_t k = (int64_t)target.size();
  std::vector<double> ts; smoothed_target(target, lp.label_smoothing, ts);
  const double invT = 1.0 / lp.temperature;
  std::vector<double> q((size_t)k);
  double mx = logits[0] * invT;
  for (int64_t i = 1; i < k; ++i) mx = std::max(mx, (double)logits[i] * invT);
  double s = 0;
  for (int64_t i = 0; i < k; ++i) { q[i] = std::exp((double)logits[i] * invT - mx); s += q[i]; }
  for (auto& v : q) v /= s;

  std::vector<double> gq((size_t)k, 0.0);
  const double log_floor = -9.21;
  for (int64_t i = 0; i < k; ++i) {
    double qc = std::max(q[i], 1e-12);
    if (std::log(qc) > log_floor) gq[i] -= lp.w_nll * ts[i] / qc;
  }
  double tq = 0, qn2 = 0;
  for (int64_t i = 0; i < k; ++i) { tq += ts[i] * q[i]; qn2 += q[i] * q[i]; }
  double qn = std::sqrt(std::max(qn2, 1e-18));
  for (int64_t i = 0; i < k; ++i) gq[i] += -lp.w_sph * (ts[i] * qn - tq * q[i]) / (qn * qn);
  if (qtype == 1 && k >= 2) {
    double cq = 0, ct = 0;
    std::vector<double> cd((size_t)k);
    for (int64_t j = 0; j < k; ++j) { cq += q[j]; ct += ts[j]; cd[j] = cq - ct; }
    for (int64_t i = 0; i < k; ++i) {
      double acc = 0;
      for (int64_t j = i; j < k; ++j) acc += cd[j];
      gq[i] += lp.w_rps * (2.0 / (k - 1)) * acc;
    }
  }
  double dot = 0;
  for (int64_t i = 0; i < k; ++i) dot += gq[i] * q[i];
  for (int64_t i = 0; i < k; ++i) dz[i] = (float)(invT * q[i] * (gq[i] - dot));
}

double loss_fwd(const float* logits, const std::vector<double>& target, int qtype,
                const LossParams& lp) {
  const int64_t k = (int64_t)target.size();
  std::vector<double> ts; smoothed_target(target, lp.label_smoothing, ts);
  const double invT = 1.0 / lp.temperature;
  std::vector<double> q((size_t)k);
  double mx = logits[0] * invT;
  for (int64_t i = 1; i < k; ++i) mx = std::max(mx, (double)logits[i] * invT);
  double s = 0;
  for (int64_t i = 0; i < k; ++i) { q[i] = std::exp((double)logits[i] * invT - mx); s += q[i]; }
  for (auto& v : q) v /= s;
  const double log_floor = -9.21;
  double log_score = 0, tq = 0, qn2 = 0;
  for (int64_t i = 0; i < k; ++i) {
    double lq = std::log(std::max(q[i], 1e-12));
    if (lq < log_floor) lq = log_floor;
    log_score += ts[i] * lq;
    tq += ts[i] * q[i];
    qn2 += q[i] * q[i];
  }
  double r = lp.w_nll * log_score + lp.w_sph * tq / std::sqrt(std::max(qn2, 1e-18));
  if (qtype == 1 && k >= 2) {
    double cq = 0, ct = 0, rps = 0;
    for (int64_t j = 0; j < k; ++j) { cq += q[j]; ct += ts[j]; double d = cq - ct; rps += d * d; }
    r -= lp.w_rps * rps / (k - 1);
  }
  return -r;
}

// ------------------------ GEMM / LN forward caches ----------------------------
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

struct LnC { std::vector<float> x; };
void ln_c(const float* x, int64_t M, int64_t D, const float* w, const float* b, float eps,
          LnC& c, std::vector<float>& out) {
  c.x.assign(x, x + M * D);
  out.assign((size_t)(M * D), 0.f);
  for (int64_t r = 0; r < M; ++r) {
    const float* xr = x + r * D;
    double mean = 0; for (int64_t j = 0; j < D; ++j) mean += xr[j]; mean /= D;
    double var = 0; for (int64_t j = 0; j < D; ++j) { double d = xr[j] - mean; var += d * d; } var /= D;
    float inv = (float)(1.0 / std::sqrt(var + eps));
    for (int64_t j = 0; j < D; ++j) {
      float v = (xr[j] - (float)mean) * inv * w[j];
      out[r * D + j] = b ? v + b[j] : v;
    }
  }
}

}  // namespace

struct SjTrainModel::Impl {
  std::unique_ptr<SjModel> frozen;
  std::shared_ptr<Tokenizer> tok;
  PStore P;
  int64_t adam_t = 0;
  LossParams lp;
  int D = 0, H = 0, Dh = 0, head_layers = 0, head_inter = 0, n_types = 0;
  float norm_eps = 1e-5f;
  int max_len = 512, head_max_len = 192;
  std::string base_dir;
};

SjTrainModel::SjTrainModel(const std::string& ckpt_dir) : impl_(std::make_unique<Impl>()) {
  std::string dir = ckpt_dir;
  if (!fs::exists(dir)) throw std::runtime_error("SjTrainModel: no such dir " + ckpt_dir);
  impl_->base_dir = dir;
  json cfg;
  { std::ifstream f(fs::path(dir) / "sj_config.json");
    if (!f) throw std::runtime_error("SjTrainModel: missing sj_config.json");
    f >> cfg; }
  SafeTensors w = SafeTensors::load((fs::path(dir) / "model.safetensors").string());
  impl_->frozen = SjModel::load(cfg, w, dir);

  const SjConfig& c = impl_->frozen->cfg();
  Impl& im = *impl_;
  im.D = c.hidden_size; im.H = c.num_heads; im.Dh = im.D / im.H;
  im.head_layers = c.head_layers; im.head_inter = c.head_intermediate;
  im.n_types = c.n_types; im.norm_eps = c.norm_eps;
  im.max_len = c.max_len; im.head_max_len = c.head_max_len;

  if (fs::exists(fs::path(dir) / "tokenizer")) {
    im.tok = Tokenizer::cached_from_dir((fs::path(dir) / "tokenizer").string());
  }

  auto pull = [&](const std::string& name) {
    if (!w.has(name)) throw std::runtime_error("SjTrainModel: missing " + name);
    auto sh = w.shape_of(name);
    int64_t n = 1; for (auto d : sh) n *= d;
    auto dat = w.data_f32(name);
    im.P.add(name, std::vector<float>(dat.get(), dat.get() + n));
  };
  pull("sj.type_emb.weight");
  for (int i = 0; i < im.head_layers; ++i) {
    std::string p = "sj.head.layers." + std::to_string(i) + ".";
    for (const char* s : {"norm1.weight","norm1.bias","norm2.weight","norm2.bias",
                          "attn.q.weight","attn.q.bias","attn.k.weight","attn.k.bias",
                          "attn.v.weight","attn.v.bias","attn.o.weight","attn.o.bias",
                          "ffn.w1.weight","ffn.w1.bias","ffn.w2.weight","ffn.w2.bias"})
      pull(p + s);
  }
  for (const char* s : {"sj.scorer.norm.weight","sj.scorer.norm.bias",
                        "sj.scorer.w1.weight","sj.scorer.w1.bias",
                        "sj.scorer.w2.weight","sj.scorer.w2.bias"})
    pull(s);

  n_params_ = 0; for (const auto& n : im.P.names) n_params_ += im.P.numel(n);
}

SjTrainModel::~SjTrainModel() = default;

void SjTrainModel::set_loss(double a, double b, double c, double ls) {
  impl_->lp = LossParams{a, b, c, ls, impl_->lp.temperature};
}
void SjTrainModel::set_loss_temperature(double T) { impl_->lp.temperature = T; }
void SjTrainModel::zero_grad() { impl_->P.zero_grad(); }

std::vector<std::string> SjTrainModel::param_names() const { return impl_->P.names; }
int64_t SjTrainModel::param_numel(const std::string& n) const { return impl_->P.numel(n); }
float SjTrainModel::param_get(const std::string& n, int64_t i) const { return impl_->P.at(n)[i]; }
void SjTrainModel::param_set(const std::string& n, int64_t i, float v) { impl_->P.at(n)[i] = v; }
float SjTrainModel::grad_at(const std::string& n, int64_t i) const { return impl_->P.g.at(n)[i]; }

// ---------------------------------------------------------------------------
// Per-row forward context: encoder output is frozen; only the head is cached.
// ---------------------------------------------------------------------------
namespace {

struct RowCtx {
  std::vector<int64_t> ids, markers;
  int64_t L = 0, K = 0;
  int qtype = 0;
  std::vector<float> h_enc;   // [L, D] frozen encoder output
  struct HC {
    LnC ln1, ln2;
    GemmC q, k, v, o, f1, f2;
    std::vector<float> attn_probs;   // [H, L, L]
  };
  std::vector<HC> heads;
  LnC sc_norm; GemmC sc_w1, sc_w2;
  std::vector<float> logits;
};

void forward_row(SjTrainModel::Impl& im, RowCtx& ctx) {
  const int64_t L = ctx.L, K = ctx.K, D = im.D;
  std::vector<float> X = ctx.h_enc;   // [L, D]
  // type_emb add
  for (int64_t i = 0; i < L; ++i) {
    float* xr = X.data() + i * D;
    const float* te = im.P.at("sj.type_emb.weight") + ctx.qtype * D;
    for (int j = 0; j < D; ++j) xr[j] += te[j];
  }
  for (int li = 0; li < im.head_layers; ++li) {
    std::string p = "sj.head.layers." + std::to_string(li) + ".";
    RowCtx::HC hc;
    std::vector<float> y; ln_c(X.data(), L, D, im.P.at(p + "norm1.weight"), im.P.at(p + "norm1.bias"), im.norm_eps, hc.ln1, y);
    std::vector<float> q, k, v;
    gemm_c(y.data(), L, D, im.P.at(p + "attn.q.weight"), D, im.P.at(p + "attn.q.bias"), 0, hc.q, q);
    gemm_c(y.data(), L, D, im.P.at(p + "attn.k.weight"), D, im.P.at(p + "attn.k.bias"), 0, hc.k, k);
    gemm_c(y.data(), L, D, im.P.at(p + "attn.v.weight"), D, im.P.at(p + "attn.v.bias"), 0, hc.v, v);
    // split [L,D] -> [H,L,Dh]
    std::vector<float> Q((size_t)(im.H * L * im.Dh)), Kt((size_t)(im.H * L * im.Dh)), V((size_t)(im.H * L * im.Dh));
    for (int64_t i = 0; i < L; ++i)
      for (int h = 0; h < im.H; ++h) {
        std::memcpy(Q.data() + (h * L + i) * im.Dh, q.data() + i * D + h * im.Dh, sizeof(float) * im.Dh);
        std::memcpy(Kt.data() + (h * L + i) * im.Dh, k.data() + i * D + h * im.Dh, sizeof(float) * im.Dh);
        std::memcpy(V.data() + (h * L + i) * im.Dh, v.data() + i * D + h * im.Dh, sizeof(float) * im.Dh);
      }
    std::vector<float> At((size_t)(im.H * L * im.Dh));
    hc.attn_probs.resize((size_t)(im.H * L * L));
    attention_forward_one(Q.data(), Kt.data(), V.data(), im.H, L, im.Dh, L, 0, At.data(), hc.attn_probs.data());
    std::vector<float> ao((size_t)(L * D));
    for (int64_t i = 0; i < L; ++i)
      for (int h = 0; h < im.H; ++h)
        std::memcpy(ao.data() + i * D + h * im.Dh, At.data() + (h * L + i) * im.Dh, sizeof(float) * im.Dh);
    std::vector<float> proj; gemm_c(ao.data(), L, D, im.P.at(p + "attn.o.weight"), D, im.P.at(p + "attn.o.bias"), 0, hc.o, proj);
    for (int64_t i = 0; i < L * D; ++i) X[i] += proj[i];
    std::vector<float> y2; ln_c(X.data(), L, D, im.P.at(p + "norm2.weight"), im.P.at(p + "norm2.bias"), im.norm_eps, hc.ln2, y2);
    std::vector<float> f1; gemm_c(y2.data(), L, D, im.P.at(p + "ffn.w1.weight"), im.head_inter, im.P.at(p + "ffn.w1.bias"), 1, hc.f1, f1);
    std::vector<float> f2; gemm_c(f1.data(), L, im.head_inter, im.P.at(p + "ffn.w2.weight"), D, im.P.at(p + "ffn.w2.bias"), 0, hc.f2, f2);
    for (int64_t i = 0; i < L * D; ++i) X[i] += f2[i];
    ctx.heads.push_back(std::move(hc));
  }
  // gather markers
  std::vector<float> m((size_t)(K * D));
  for (int64_t kk = 0; kk < K; ++kk) {
    int64_t pos = ctx.markers[(size_t)kk];
    if (pos < 0) pos = 0; if (pos >= L) pos = L - 1;
    std::memcpy(m.data() + kk * D, X.data() + pos * D, sizeof(float) * D);
  }
  std::vector<float> s1; ln_c(m.data(), K, D, im.P.at("sj.scorer.norm.weight"), im.P.at("sj.scorer.norm.bias"), im.norm_eps, ctx.sc_norm, s1);
  std::vector<float> s2; gemm_c(s1.data(), K, D, im.P.at("sj.scorer.w1.weight"), D, im.P.at("sj.scorer.w1.bias"), 1, ctx.sc_w1, s2);
  std::vector<float> lg; gemm_c(s2.data(), K, D, im.P.at("sj.scorer.w2.weight"), 1, im.P.at("sj.scorer.w2.bias"), 0, ctx.sc_w2, lg);
  ctx.logits = std::move(lg);
}

void backward_row(SjTrainModel::Impl& im, RowCtx& ctx, const std::vector<double>& target,
                  int qtype_int, double scale) {
  const int64_t L = ctx.L, K = ctx.K, D = im.D;
  std::vector<float> dz((size_t)K);
  loss_grad(ctx.logits.data(), target, qtype_int, im.lp, dz.data());
  for (auto& v : dz) v = (float)(v * scale);

  // scorer.w2 [1,D]
  {
    std::vector<float> dA((size_t)(K * D), 0.f), dW((size_t)D, 0.f), db(1, 0.f);
    gemm_nt_backward(dz.data(), ctx.sc_w2.pre.data(), ctx.sc_w2.A.data(), im.P.at("sj.scorer.w2.weight"), K, 1, D, 0, dA.data(), dW.data(), db.data());
    for (int64_t i = 0; i < D; ++i) im.P.g["sj.scorer.w2.weight"][i] += dW[i];
    im.P.g["sj.scorer.w2.bias"][0] += db[0];
    // scorer.w1 [D,D] GELU
    std::vector<float> dA2((size_t)(K * D), 0.f), dW2((size_t)(D * D), 0.f), db2((size_t)D, 0.f);
    gemm_nt_backward(dA.data(), ctx.sc_w1.pre.data(), ctx.sc_w1.A.data(), im.P.at("sj.scorer.w1.weight"), K, D, D, 1, dA2.data(), dW2.data(), db2.data());
    for (int64_t i = 0; i < D * D; ++i) im.P.g["sj.scorer.w1.weight"][i] += dW2[i];
    for (int64_t i = 0; i < D; ++i) im.P.g["sj.scorer.w1.bias"][i] += db2[i];
    // scorer norm
    std::vector<float> dxm((size_t)(K * D)), dw((size_t)D, 0.f), dbm((size_t)D, 0.f);
    layer_norm_backward(dA2.data(), ctx.sc_norm.x.data(), im.P.at("sj.scorer.norm.weight"), K, D, im.norm_eps, true, dxm.data(), dw.data(), dbm.data());
    for (int64_t i = 0; i < D; ++i) { im.P.g["sj.scorer.norm.weight"][i] += dw[i]; im.P.g["sj.scorer.norm.bias"][i] += dbm[i]; }
    // scatter back into dh
    std::vector<float> dh((size_t)(L * D), 0.f);
    for (int64_t kk = 0; kk < K; ++kk) {
      int64_t pos = ctx.markers[(size_t)kk];
      if (pos < 0) pos = 0; if (pos >= L) pos = L - 1;
      float* dst = dh.data() + pos * D; const float* src = dxm.data() + kk * D;
      for (int64_t j = 0; j < D; ++j) dst[j] += src[j];
    }
    std::vector<float> dX = std::move(dh);
    for (int li = im.head_layers - 1; li >= 0; --li) {
      std::string p = "sj.head.layers." + std::to_string(li) + ".";
      RowCtx::HC& hc = ctx.heads[li];
      // ffn.w2
      std::vector<float> dA((size_t)(L * im.head_inter), 0.f), dW((size_t)(D * im.head_inter), 0.f), db((size_t)D, 0.f);
      gemm_nt_backward(dX.data(), hc.f2.pre.data(), hc.f2.A.data(), im.P.at(p + "ffn.w2.weight"), L, D, im.head_inter, 0, dA.data(), dW.data(), db.data());
      for (int64_t i = 0; i < D * im.head_inter; ++i) im.P.g[p + "ffn.w2.weight"][i] += dW[i];
      for (int64_t i = 0; i < D; ++i) im.P.g[p + "ffn.w2.bias"][i] += db[i];
      // ffn.w1 GELU
      std::vector<float> dA2((size_t)(L * D), 0.f), dW2((size_t)(im.head_inter * D), 0.f), db2((size_t)(im.head_inter), 0.f);
      gemm_nt_backward(dA.data(), hc.f1.pre.data(), hc.f1.A.data(), im.P.at(p + "ffn.w1.weight"), L, im.head_inter, D, 1, dA2.data(), dW2.data(), db2.data());
      for (int64_t i = 0; i < im.head_inter * D; ++i) im.P.g[p + "ffn.w1.weight"][i] += dW2[i];
      for (int64_t i = 0; i < im.head_inter; ++i) im.P.g[p + "ffn.w1.bias"][i] += db2[i];
      // ln2
      std::vector<float> dxl((size_t)(L * D)), dw((size_t)D, 0.f), db3((size_t)D, 0.f);
      layer_norm_backward(dA2.data(), hc.ln2.x.data(), im.P.at(p + "norm2.weight"), L, D, im.norm_eps, true, dxl.data(), dw.data(), db3.data());
      for (int64_t i = 0; i < D; ++i) { im.P.g[p + "norm2.weight"][i] += dw[i]; im.P.g[p + "norm2.bias"][i] += db3[i]; }
      // residual: dX_mid = dX + dxl
      for (int64_t i = 0; i < L * D; ++i) dxl[i] += dX[i];
      // attn.o
      std::vector<float> dproj = dxl;
      std::vector<float> dao((size_t)(L * D), 0.f), dWo((size_t)(D * D), 0.f), dbo((size_t)D, 0.f);
      gemm_nt_backward(dproj.data(), hc.o.pre.data(), hc.o.A.data(), im.P.at(p + "attn.o.weight"), L, D, D, 0, dao.data(), dWo.data(), dbo.data());
      for (int64_t i = 0; i < D * D; ++i) im.P.g[p + "attn.o.weight"][i] += dWo[i];
      for (int64_t i = 0; i < D; ++i) im.P.g[p + "attn.o.bias"][i] += dbo[i];
      // attention backward: dao [L,D] -> [H,L,Dh]
      std::vector<float> dAt((size_t)(im.H * L * im.Dh));
      for (int64_t i = 0; i < L; ++i)
        for (int h = 0; h < im.H; ++h)
          std::memcpy(dAt.data() + (h * L + i) * im.Dh, dao.data() + i * D + h * im.Dh, sizeof(float) * im.Dh);
      std::vector<float> Q((size_t)(im.H * L * im.Dh)), Kt((size_t)(im.H * L * im.Dh)), V((size_t)(im.H * L * im.Dh));
      for (int64_t i = 0; i < L; ++i)
        for (int h = 0; h < im.H; ++h) {
          std::memcpy(Q.data() + (h * L + i) * im.Dh, hc.q.pre.data() + i * D + h * im.Dh, sizeof(float) * im.Dh);
          std::memcpy(Kt.data() + (h * L + i) * im.Dh, hc.k.pre.data() + i * D + h * im.Dh, sizeof(float) * im.Dh);
          std::memcpy(V.data() + (h * L + i) * im.Dh, hc.v.pre.data() + i * D + h * im.Dh, sizeof(float) * im.Dh);
        }
      std::vector<float> dQ((size_t)(im.H * L * im.Dh), 0.f), dK((size_t)(im.H * L * im.Dh), 0.f), dV((size_t)(im.H * L * im.Dh), 0.f);
      attention_backward_one(dAt.data(), Q.data(), Kt.data(), V.data(), hc.attn_probs.data(), im.H, L, im.Dh, L, 0, dQ.data(), dK.data(), dV.data());
      // back to [L,D] each
      std::vector<float> dq((size_t)(L * D), 0.f), dk((size_t)(L * D), 0.f), dv((size_t)(L * D), 0.f);
      for (int64_t i = 0; i < L; ++i)
        for (int h = 0; h < im.H; ++h) {
          std::memcpy(dq.data() + i * D + h * im.Dh, dQ.data() + (h * L + i) * im.Dh, sizeof(float) * im.Dh);
          std::memcpy(dk.data() + i * D + h * im.Dh, dK.data() + (h * L + i) * im.Dh, sizeof(float) * im.Dh);
          std::memcpy(dv.data() + i * D + h * im.Dh, dV.data() + (h * L + i) * im.Dh, sizeof(float) * im.Dh);
        }
      // q/k/v projections -> sum into dy (grad w.r.t. ln1 output)
      std::vector<float> dy((size_t)(L * D), 0.f);
      auto proj_back = [&](const std::string& proj, const std::vector<float>& dproj_,
                           const GemmC& cache) {
        std::vector<float> dA((size_t)(L * D), 0.f), dW((size_t)(D * D), 0.f), db((size_t)D, 0.f);
        gemm_nt_backward(dproj_.data(), cache.pre.data(), cache.A.data(),
                         im.P.at(p + proj + ".weight"), L, D, D, 0, dA.data(), dW.data(), db.data());
        for (int64_t i = 0; i < D * D; ++i) im.P.g[p + proj + ".weight"][i] += dW[i];
        for (int64_t i = 0; i < D; ++i) im.P.g[p + proj + ".bias"][i] += db[i];
        for (int64_t i = 0; i < L * D; ++i) dy[i] += dA[i];
      };
      proj_back("attn.q", dq, hc.q);
      proj_back("attn.k", dk, hc.k);
      proj_back("attn.v", dv, hc.v);
      // ln1
      std::vector<float> dxl1((size_t)(L * D)), dw0((size_t)D, 0.f), db0((size_t)D, 0.f);
      layer_norm_backward(dy.data(), hc.ln1.x.data(), im.P.at(p + "norm1.weight"), L, D, im.norm_eps, true, dxl1.data(), dw0.data(), db0.data());
      for (int64_t i = 0; i < D; ++i) { im.P.g[p + "norm1.weight"][i] += dw0[i]; im.P.g[p + "norm1.bias"][i] += db0[i]; }
      // residual through this head layer: dX = dxl1 + dmain (the residual path)
      for (int64_t i = 0; i < L * D; ++i) dX[i] = dxl1[i] + dX[i];
    }
    // type_emb add: dte[qtype] = sum over positions of dX
    float* gte = im.P.g["sj.type_emb.weight"].data() + ctx.qtype * D;
    for (int64_t i = 0; i < L; ++i) {
      const float* src = dX.data() + i * D;
      for (int64_t j = 0; j < D; ++j) gte[j] += src[j];
    }
    // encoder frozen: drop the gradient that would flow into h_enc
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// public training API
// ---------------------------------------------------------------------------
SjTrainModel::StepStats SjTrainModel::step(const TrainRow& row, bool accumulate_only) {
  Impl& im = *impl_;
  if (!im.tok) throw std::runtime_error("SjTrainModel: no tokenizer in checkpoint");
  std::vector<InternalQ> internals;
  for (const auto& qd : row.qdefs) internals.push_back(to_internal(qd));

  // build one RowCtx per question (encode is per-question here; tiny models)
  std::vector<RowCtx> rows;
  for (size_t i = 0; i < row.qids.size(); ++i) {
    auto [seq, markers] = build_sequence(*im.tok, row.state, internals[i], im.max_len, im.head_max_len, row.state.is_array());
    RowCtx ctx;
    ctx.qtype = QTYPES.at(internals[i].t);
    ctx.ids.assign(seq.begin(), seq.end());
    ctx.markers = markers;
    ctx.L = (int64_t)ctx.ids.size();
    ctx.K = (int64_t)markers.size();
    // encode the single sequence through the frozen encoder
    std::vector<std::vector<int64_t>> ids = {ctx.ids};
    std::vector<std::vector<int64_t>> mask = {std::vector<int64_t>((size_t)ctx.L, 1)};
    auto h = im.frozen->encode(ids, mask);   // [1, L, D]
    ctx.h_enc.assign(h[0][0].begin(), h[0][0].end());
    for (size_t t = 1; t < h[0].size(); ++t)
      ctx.h_enc.insert(ctx.h_enc.end(), h[0][t].begin(), h[0][t].end());
    rows.push_back(std::move(ctx));
  }

  double total_loss = 0; int correct = 0; int nrows = 0;
  const double scale = 1.0 / std::max<size_t>(1, rows.size());
  for (size_t i = 0; i < rows.size(); ++i) {
    RowCtx& ctx = rows[i];
    forward_row(im, ctx);
    // argmax accuracy
    int best = 0; for (int j = 1; j < ctx.K; ++j) if (ctx.logits[j] > ctx.logits[best]) best = j;
    int tbest = 0; for (size_t j = 1; j < row.targets[i].size(); ++j) if (row.targets[i][j] > row.targets[i][tbest]) tbest = (int)j;
    if (best == tbest) ++correct;
    total_loss += loss_fwd(ctx.logits.data(), row.targets[i], ctx.qtype, im.lp);
    ++nrows;
    backward_row(im, ctx, row.targets[i], ctx.qtype, scale);
  }
  StepStats st;
  st.loss = total_loss / std::max(1, nrows);
  st.acc = nrows ? (double)correct / nrows : 0.0;
  st.rows = nrows;
  return st;
}

void SjTrainModel::optimizer_step(double lr) {
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

double SjTrainModel::forward_loss_double(const TrainRow& row) {
  Impl& im = *impl_;
  std::vector<InternalQ> internals;
  for (const auto& qd : row.qdefs) internals.push_back(to_internal(qd));
  double total = 0; int n = 0;
  for (size_t i = 0; i < row.qids.size(); ++i) {
    auto [seq, markers] = build_sequence(*im.tok, row.state, internals[i], im.max_len, im.head_max_len, row.state.is_array());
    RowCtx ctx;
    ctx.qtype = QTYPES.at(internals[i].t);
    ctx.ids.assign(seq.begin(), seq.end());
    ctx.markers = markers; ctx.L = (int64_t)ctx.ids.size(); ctx.K = (int64_t)markers.size();
    std::vector<std::vector<int64_t>> ids = {ctx.ids};
    std::vector<std::vector<int64_t>> mask = {std::vector<int64_t>((size_t)ctx.L, 1)};
    auto h = im.frozen->encode(ids, mask);
    ctx.h_enc.assign(h[0][0].begin(), h[0][0].end());
    for (size_t t = 1; t < h[0].size(); ++t) ctx.h_enc.insert(ctx.h_enc.end(), h[0][t].begin(), h[0][t].end());
    forward_row(im, ctx);
    total += loss_fwd(ctx.logits.data(), row.targets[i], ctx.qtype, im.lp);
    ++n;
  }
  return total / std::max(1, n);
}

void SjTrainModel::save_checkpoint(const std::string& out_dir) {
  Impl& im = *impl_;
  fs::create_directories(out_dir);
  // copy tokenizer
  if (fs::exists(fs::path(im.base_dir) / "tokenizer"))
    fs::copy(fs::path(im.base_dir) / "tokenizer", fs::path(out_dir) / "tokenizer",
             fs::copy_options::recursive | fs::copy_options::overwrite_existing);
  // collect trainable tensors, keep the frozen encoder's tensors as-is by
  // re-reading the base safetensors.
  SafeTensors base = SafeTensors::load((fs::path(im.base_dir) / "model.safetensors").string());
  std::vector<TensorOut> out;
  // frozen encoder + embeddings: copy from base
  for (const auto& name : base.names()) {
    if (name.rfind("sj.head", 0) == 0 || name.rfind("sj.type_emb", 0) == 0 ||
        name.rfind("sj.scorer", 0) == 0) continue;
    TensorOut t; t.name = name; t.shape = base.shape_of(name);
    t.data = base.data_f32(name); t.as_f16 = true;
    out.push_back(std::move(t));
  }
  // trained head params
  for (const auto& n : im.P.names) {
    TensorOut t; t.name = n;
    // reconstruct shape from the base
    std::vector<int64_t> sh = base.shape_of(n);
    t.shape = sh; t.as_f16 = true;
    auto sp = std::shared_ptr<float[]>(new float[(size_t)im.P.numel(n)]);
    std::memcpy(sp.get(), im.P.at(n), sizeof(float) * (size_t)im.P.numel(n));
    t.data = std::move(sp);
    out.push_back(std::move(t));
  }
  save_safetensors((fs::path(out_dir) / "model.safetensors").string(), out);
  // copy sj_config.json
  std::ifstream cfgf(fs::path(im.base_dir) / "sj_config.json");
  std::ofstream cfgo(fs::path(out_dir) / "sj_config.json");
  cfgo << cfgf.rdbuf();
}

}  // namespace snapjudge
