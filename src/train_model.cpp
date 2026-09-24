// snapjudge/src/train_model.cpp — head-finetuning trainer (adapter pattern).
//
// The encoder weights are FROZEN: they run forward-only through the verified
// inference path (model.cpp encode()). Trainable: type_emb, the head
// transformer layers, and the scorer — the parts that carry checkpoint
// specialization on typed-decisions. The act head is frozen (the gold
// distributions carry no act targets).
//
// Loss = negative proper score (offline form of the RLCD rule):
//     reward = Σ t_i·log q_i + 0.5·(t·q)/|q|        (clamp q ≥ 1e-12, floor -9.21)
//     score rows also subtract RPS = Σ (cdf_q - cdf_t)² / (k-1)
//
// Gradients are analytic; tests/test_train.cpp finite-difference checks every
// trainable tensor on a tiny synthetic checkpoint before any real run.

#include "snapjudge/train.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <set>

#include "snapjudge/backward.hpp"
#include "snapjudge/common.hpp"
#include "snapjudge/model.hpp"
#include "snapjudge/safetensors.hpp"
#include "snapjudge/safewrite.hpp"
#include "snapjudge/tokenizer.hpp"

namespace snapjudge {

namespace fs = std::filesystem;

extern "C" {
void cblas_sgemm(int Order, int TransA, int TransB, int M, int N, int K,
                 float alpha, const float* A, int lda, const float* B, int ldb,
                 float beta, float* C, int ldc);
}

namespace {
constexpr int RM = 101, NO_T = 111, TR = 112;

// ------------------------ parameter store ---------------------------------------
struct PStore {
  std::map<std::string, std::vector<float>> w;       // fp32 trainable weights
  std::map<std::string, std::vector<int64_t>> shape;
  std::vector<std::string> names;                    // deterministic order
  std::map<std::string, std::vector<float>> m1, m2, g;  // Adam state + grads

  void add(const std::string& n, std::vector<float> v,
           const std::vector<int64_t>& sh) {
    w[n] = std::move(v);
    shape[n] = sh;
    names.push_back(n);
    m1[n] = std::vector<float>(w[n].size(), 0.f);
    m2[n] = std::vector<float>(w[n].size(), 0.f);
    g[n] = std::vector<float>(w[n].size(), 0.f);
  }
  float* at(const std::string& n) { return w.at(n).data(); }
  const float* at(const std::string& n) const { return w.at(n).data(); }
  int64_t numel(const std::string& n) const { return (int64_t)w.at(n).size(); }
  void zero_grad() {
    for (auto& n : names) std::fill(g[n].begin(), g[n].end(), 0.f);
  }
  double grad_norm() const {
    double s = 0;
    for (const auto& n : names)
      for (float v : g.at(n)) s += (double)v * v;
    return std::sqrt(s);
  }
};

// ------------------------ cached forward primitives ------------------------------
struct GemmC { std::vector<float> A, W, pre; };

std::vector<float> gemm_c(const float* A, int64_t M, int64_t K, const float* W,
                          int64_t N, const float* bias, int act, GemmC& c) {
  c.A.assign(A, A + M * K);
  c.W.assign(W, W + N * K);
  std::vector<float> out(static_cast<size_t>(M * N));
  cblas_sgemm(RM, NO_T, TR, (int)M, (int)N, (int)K, 1.0f, A, (int)K, W, (int)K, 0.0f,
              out.data(), (int)N);
  c.pre = out;
  if (bias)
    for (int64_t i = 0; i < M; ++i)
      for (int64_t j = 0; j < N; ++j) c.pre[static_cast<size_t>(i * N + j)] += bias[j];
  out = c.pre;
  if (act == 1)
    for (auto& v : out) v = 0.5f * v * (1.0f + std::erf(v * 0.70710678118654757f));
  else if (act == 2)
    for (auto& v : out) v = std::max(v, 0.f);
  return out;
}

struct LnC { std::vector<float> x, w, b; };

std::vector<float> ln_c(const float* x, int64_t M, int64_t D, const float* w,
                        const float* b, float eps, LnC& c) {
  c.x.assign(x, x + M * D);
  c.w.assign(w, w + D);
  if (b) c.b.assign(b, b + D); else c.b.clear();
  std::vector<float> out(static_cast<size_t>(M * D));
  for (int64_t r = 0; r < M; ++r) {
    const float* xr = x + r * D;
    double mean = 0;
    for (int64_t j = 0; j < D; ++j) mean += xr[j];
    mean /= D;
    double var = 0;
    for (int64_t j = 0; j < D; ++j) { double d = xr[j] - mean; var += d * d; }
    var /= D;
    float inv = static_cast<float>(1.0 / std::sqrt(var + eps));
    for (int64_t j = 0; j < D; ++j) {
      float v = (xr[j] - (float)mean) * inv * w[j];
      out[static_cast<size_t>(r * D + j)] = b ? v + b[j] : v;
    }
  }
  return out;
}

// ----------------------- proper-score loss ----------------------------------------
// reward = Σ t·logq + 0.5·(t·q)/|q| (+ RPS on score rows); loss = -reward.
static void loss_grad(const float* logits, const std::vector<double>& target,
                      int qtype_int, float* dz) {
  const int64_t k = (int64_t)target.size();
  std::vector<double> q((size_t)k);
  double mx = logits[0];
  for (int64_t i = 1; i < k; ++i) mx = std::max(mx, (double)logits[i]);
  double s = 0;
  for (int64_t i = 0; i < k; ++i) { q[(size_t)i] = std::exp((double)logits[i] - mx); s += q[(size_t)i]; }
  for (auto& v : q) v /= s;

  std::vector<double> gq((size_t)k, 0.0);
  const double log_floor = -9.21;
  for (int64_t i = 0; i < k; ++i) {
    double qc = std::max(q[(size_t)i], 1e-12);
    if (std::log(qc) > log_floor)
      gq[(size_t)i] -= target[(size_t)i] / qc;
  }
  double tq = 0, qn2 = 0;
  for (int64_t i = 0; i < k; ++i) { tq += target[(size_t)i] * q[(size_t)i]; qn2 += q[(size_t)i] * q[(size_t)i]; }
  double qn = std::sqrt(std::max(qn2, 1e-18));
  for (int64_t i = 0; i < k; ++i)
    gq[(size_t)i] += -0.5 * (target[(size_t)i] * qn - tq * q[(size_t)i]) / (qn * qn);
  if (qtype_int == 1 && k >= 2) {
    double cq = 0, ct = 0;
    std::vector<double> cdiff((size_t)k);
    for (int64_t j = 0; j < k; ++j) { cq += q[(size_t)j]; ct += target[(size_t)j]; cdiff[(size_t)j] = cq - ct; }
    for (int64_t i = 0; i < k; ++i) {
      double acc = 0;
      for (int64_t j = i; j < k; ++j) acc += cdiff[(size_t)j];
      gq[(size_t)i] += (2.0 / (k - 1)) * acc;
    }
  }
  double dot = 0;
  for (int64_t i = 0; i < k; ++i) dot += gq[(size_t)i] * q[(size_t)i];
  for (int64_t i = 0; i < k; ++i)
    dz[i] = (float)(q[(size_t)i] * (gq[(size_t)i] - dot));
}

static double loss_fwd(const float* logits, const std::vector<double>& target,
                       int qtype_int) {
  const int64_t k = (int64_t)target.size();
  std::vector<double> q((size_t)k);
  double mx = logits[0];
  for (int64_t i = 1; i < k; ++i) mx = std::max(mx, (double)logits[i]);
  double s = 0;
  for (int64_t i = 0; i < k; ++i) { q[(size_t)i] = std::exp((double)logits[i] - mx); s += q[(size_t)i]; }
  for (auto& v : q) v /= s;
  const double log_floor = -9.21;
  double log_score = 0, tq = 0, qn2 = 0;
  for (int64_t i = 0; i < k; ++i) {
    double qc = std::max(q[(size_t)i], 1e-12);
    double lq = std::log(qc);
    if (lq < log_floor) lq = log_floor;
    log_score += target[(size_t)i] * lq;
    tq += target[(size_t)i] * q[(size_t)i];
    qn2 += q[(size_t)i] * q[(size_t)i];
  }
  double sph = tq / std::sqrt(std::max(qn2, 1e-18));
  double r = log_score + 0.5 * sph;
  if (qtype_int == 1 && k >= 2) {
    double cq = 0, ct = 0, rps = 0;
    for (int64_t j = 0; j < k; ++j) {
      cq += q[(size_t)j]; ct += target[(size_t)j];
      double d = cq - ct;
      rps += d * d;
    }
    r -= rps / (k - 1);
  }
  return -r;
}

// ------------------------- row forward context -------------------------------------
struct RowCtx {
  std::vector<int64_t> ids, markers;
  int64_t L = 0, K = 0;
  int qtype = 0;
  struct HC {
    LnC ln1, ln2;
    GemmC in_proj, out_proj, ffn1, ffn2;
    std::vector<float> attn_probs;    // [H, L, L]
    std::vector<float> X_in, X_mid;   // [L, D]
    std::vector<float> attn_out;      // [L, D] (pre out_proj, reshaped back)
  };
  std::vector<HC> heads;
  LnC sc_ln;
  GemmC sc_l1, sc_l2;
  std::vector<float> logits;          // [K]
  std::vector<float> h_encoder;       // [L, D] frozen encoder output (input to head)
  std::vector<double> probs;          // cached softmax
};

}  // namespace

struct TrainModel::Impl {
  std::unique_ptr<DecisionModel> frozen;  // encoder forward-only
  PStore P;
  int64_t adam_t = 0;
  double b1 = 0.9, b2 = 0.999, wd = 0.01;
  const double eps_adam = 1e-8;
  int D = 0, H = 0, Dh = 0, head_layers = 0;
  float norm_eps = 1e-5f;
  int max_len = 512, head_max_len = 192;
};

// ------------------------------ ctor ----------------------------------------------
TrainModel::TrainModel(const std::string& ckpt_dir) : impl_(std::make_unique<Impl>()) {
  base_dir_ = ckpt_dir;
  {
    std::ifstream f(fs::path(ckpt_dir) / "rl_agent_config.json");
    if (!f) throw std::runtime_error("TrainModel: missing rl_agent_config.json in " + ckpt_dir);
    f >> cfg_;
  }
  nlohmann::json enc_cfg;
  {
    std::ifstream f(fs::path(ckpt_dir) / "encoder" / "config.json");
    if (!f) throw std::runtime_error("TrainModel: missing encoder/config.json");
    f >> enc_cfg;
  }
  tok_ = Tokenizer::cached_from_dir((fs::path(ckpt_dir) / "tokenizer").string());

  auto w = SafeTensors::load((fs::path(ckpt_dir) / "model.safetensors").string());
  impl_->frozen = DecisionModel::load(cfg_, enc_cfg, w, ckpt_dir);

  Impl& im = *impl_;
  auto& mc = impl_->frozen->cfg();
  im.D = mc.hidden_size;
  im.H = mc.num_attention_heads;
  im.Dh = im.D / im.H;
  im.head_layers = cfg_.value("head_layers", 2);
  im.norm_eps = mc.norm_eps;
  im.max_len = cfg_.value("max_len", 512);
  im.head_max_len = cfg_.value("head_max_len", 192);
  const int D = im.D;

  // seed trainable copies from the checkpoint
  auto pull = [&](const std::string& name) {
    auto sh = w.shape_of(name);
    int64_t n = 1;
    for (auto d : sh) n *= d;
    auto dat = w.data_f32(name);
    std::vector<float> v(dat.get(), dat.get() + n);
    im.P.add(name, std::move(v), sh);
  };
  pull("type_emb.weight");
  for (int i = 0; i < im.head_layers; ++i) {
    std::string p = "head.layers." + std::to_string(i) + ".";
    for (const char* s : {"norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
                          "self_attn.in_proj_weight", "self_attn.in_proj_bias",
                          "self_attn.out_proj.weight", "self_attn.out_proj.bias",
                          "linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias"})
      pull(p + s);
  }
  pull("scorer.0.weight");
  if (w.has("scorer.0.bias")) pull("scorer.0.bias");
  pull("scorer.1.weight");
  if (w.has("scorer.1.bias")) pull("scorer.1.bias");
  pull("scorer.3.weight");
  if (w.has("scorer.3.bias")) pull("scorer.3.bias");

  n_params_ = 0;
  for (const auto& n : im.P.names) n_params_ += im.P.numel(n);
}

TrainModel::~TrainModel() = default;

// ----------------------------- row forward ----------------------------------------
namespace {

// h2 = h_enc + type_emb[qtype]; head layers; gather markers; scorer.
// Returns ctx.logits.
void forward_row(TrainModel::Impl& im, RowCtx& ctx) {
  const int64_t L = ctx.L, K = ctx.K, D = im.D;
  std::vector<float>& X = ctx.h_encoder;   // [L, D] — owned by ctx, mutated
  for (int64_t i = 0; i < L; ++i) {
    float* xr = X.data() + i * D;
    const float* te = im.P.at("type_emb.weight") + ctx.qtype * D;
    for (int j = 0; j < D; ++j) xr[j] += te[j];
  }

  const float ln_eps = 1e-5f;
  for (int li = 0; li < im.head_layers; ++li) {
    std::string p = "head.layers." + std::to_string(li) + ".";
    RowCtx::HC hc;
    hc.ln1.x = X;
    hc.ln1.x.resize((size_t)(L * D));
    hc.X_in = X;
    hc.X_in.resize((size_t)(L * D));

    // ln1
    LnC ln1c;
    std::vector<float> y = ln_c(X.data(), L, D, im.P.at(p + "norm1.weight"),
                                im.P.at(p + "norm1.bias"), ln_eps, ln1c);
    hc.ln1 = ln1c;
    // in_proj
    GemmC gc;
    std::vector<float> qkv = gemm_c(y.data(), L, D, im.P.at(p + "self_attn.in_proj_weight"),
                                    3 * D, im.P.at(p + "self_attn.in_proj_bias"), 0, gc);
    hc.in_proj = gc;
    // attention: split [L, 3D] into per-head
    std::vector<float> Q((size_t)(im.H * L * im.Dh)), Kt((size_t)(im.H * L * im.Dh)),
        V((size_t)(im.H * L * im.Dh));
    int64_t HxDh = (int64_t)im.H * im.Dh;
    for (int64_t i = 0; i < L; ++i) {
      const float* row = qkv.data() + i * 3 * HxDh;
      for (int h = 0; h < im.H; ++h) {
        std::memcpy(Q.data() + (h * L + i) * im.Dh, row + h * im.Dh, sizeof(float) * im.Dh);
        std::memcpy(Kt.data() + (h * L + i) * im.Dh, row + HxDh + h * im.Dh, sizeof(float) * im.Dh);
        std::memcpy(V.data() + (h * L + i) * im.Dh, row + 2 * HxDh + h * im.Dh, sizeof(float) * im.Dh);
      }
    }
    std::vector<float> At((size_t)(im.H * L * im.Dh));
    hc.attn_probs.resize((size_t)(im.H * L * L));
    attention_forward_one(Q.data(), Kt.data(), V.data(), im.H, L, im.Dh, L, 0,
                          At.data(), hc.attn_probs.data());
    // back to [L, D]
    std::vector<float> ao((size_t)(L * D));
    for (int64_t i = 0; i < L; ++i)
      for (int h = 0; h < im.H; ++h)
        std::memcpy(ao.data() + i * D + h * im.Dh, At.data() + (h * L + i) * im.Dh,
                    sizeof(float) * im.Dh);
    hc.attn_out = ao;
    GemmC gc2;
    std::vector<float> proj = gemm_c(ao.data(), L, D, im.P.at(p + "self_attn.out_proj.weight"),
                                     D, im.P.at(p + "self_attn.out_proj.bias"), 0, gc2);
    hc.out_proj = gc2;
    for (int64_t i = 0; i < L * D; ++i) X[i] += proj[i];   // residual
    hc.X_mid = X;
    // ln2 + ffn
    LnC ln2c;
    std::vector<float> y2 = ln_c(X.data(), L, D, im.P.at(p + "norm2.weight"),
                                 im.P.at(p + "norm2.bias"), ln_eps, ln2c);
    hc.ln2 = ln2c;
    GemmC gf1, gf2;
    std::vector<float> f1 = gemm_c(y2.data(), L, D, im.P.at(p + "linear1.weight"), 4 * D,
                                   im.P.at(p + "linear1.bias"), 2, gf1);
    hc.ffn1 = gf1;
    std::vector<float> f2 = gemm_c(f1.data(), L, 4 * D, im.P.at(p + "linear2.weight"),
                                   D, im.P.at(p + "linear2.bias"), 0, gf2);
    hc.ffn2 = gf2;
    for (int64_t i = 0; i < L * D; ++i) X[i] += f2[i];
    ctx.heads.push_back(std::move(hc));
  }

  // gather marker rows
  std::vector<float> m((size_t)(K * D));
  for (int64_t kk = 0; kk < K; ++kk) {
    int64_t pos = ctx.markers[(size_t)kk];
    if (pos < 0) pos = 0;
    if (pos >= L) pos = L - 1;
    std::memcpy(m.data() + kk * D, X.data() + pos * D, sizeof(float) * D);
  }
  // scorer: LN + Linear(d,d) GELU + Linear(d,1)
  std::vector<float> s1 = ln_c(m.data(), K, D, im.P.at("scorer.0.weight"),
                               im.P.names.end() != std::find(im.P.names.begin(), im.P.names.end(), "scorer.0.bias")
                                   ? im.P.at("scorer.0.bias") : nullptr,
                               im.norm_eps, ctx.sc_ln);
  std::vector<float> s2 = gemm_c(s1.data(), K, D, im.P.at("scorer.1.weight"), D,
                                 im.P.names.end() != std::find(im.P.names.begin(), im.P.names.end(), "scorer.1.bias")
                                     ? im.P.at("scorer.1.bias") : nullptr,
                                 1, ctx.sc_l1);
  std::vector<float> lg = gemm_c(s2.data(), K, D, im.P.at("scorer.3.weight"), 1,
                                 im.P.names.end() != std::find(im.P.names.begin(), im.P.names.end(), "scorer.3.bias")
                                     ? im.P.at("scorer.3.bias") : nullptr,
                                 0, ctx.sc_l2);
  // training softmax has no -1e4 masking: only valid markers are present (K = true count)
  ctx.logits.assign(lg.begin(), lg.end());
  ctx.probs.resize((size_t)K);
  double mx = ctx.logits[0];
  for (int64_t i = 1; i < K; ++i) mx = std::max(mx, (double)ctx.logits[i]);
  double s = 0;
  for (int64_t i = 0; i < K; ++i) { ctx.probs[(size_t)i] = std::exp((double)ctx.logits[i] - mx); s += ctx.probs[(size_t)i]; }
  for (auto& v : ctx.probs) v /= s;
}

// ----------------------------- row backward -----------------------------------------
void backward_row(TrainModel::Impl& im, RowCtx& ctx, const std::vector<double>& target,
                  int qtype_int, double scale) {
  const int64_t L = ctx.L, K = ctx.K, D = im.D;
  const int64_t HxDh = (int64_t)im.H * im.Dh;

  std::vector<float> dz((size_t)K);
  loss_grad(ctx.logits.data(), target, qtype_int, dz.data());
  for (auto& v : dz) v = (float)(v * scale);   // mean over rows

  // scorer L2: logits = s2 @ w2^T (+b2), w2 [1, D]
  {
    std::string n = "scorer.3";
    std::vector<float> dA((size_t)(K * D), 0.f), dW((size_t)D, 0.f), db(1, 0.f);
    gemm_nt_backward(dz.data(), ctx.sc_l2.pre.data(), ctx.sc_l2.A.data(),
                     im.P.at(n + ".weight"), K, 1, D, 0, dA.data(), dW.data(), db.data());
    auto& g = im.P.g[n + ".weight"];
    for (int64_t i = 0; i < D; ++i) g[i] += dW[i];
