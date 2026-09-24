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
    // bias exists only if we registered it
    if (im.P.w.count(n + ".bias"))
      im.P.g[n + ".bias"][0] += db[0];
    // propagate into s2
    // L1: s2 = gelu(s1 @ w1^T (+b1)), w1 [D, D]
    std::vector<float> dPre = dA;   // K x D
    std::vector<float> dA2((size_t)(K * D), 0.f), dW2((size_t)(D * D), 0.f),
        db2((size_t)D, 0.f);
    gemm_nt_backward(dPre.data(), ctx.sc_l1.pre.data(), ctx.sc_l1.A.data(),
                     im.P.at("scorer.1.weight"), K, D, D, 1, dA2.data(), dW2.data(),
                     db2.data());
    auto& g2 = im.P.g["scorer.1.weight"];
    for (int64_t i = 0; i < D * D; ++i) g2[i] += dW2[i];
    if (im.P.w.count("scorer.1.bias")) {
      auto& gb = im.P.g["scorer.1.bias"];
      for (int64_t i = 0; i < D; ++i) gb[i] += db2[i];
    }
    // LN backward -> gradient onto gathered rows
    std::vector<float> dxm((size_t)(K * D)), dw((size_t)D, 0.f), dbm((size_t)D, 0.f);
    layer_norm_backward(dA2.data(), ctx.sc_ln.x.data(), im.P.at("scorer.0.weight"), K, D,
                        im.norm_eps, im.P.w.count("scorer.0.bias"),
                        dxm.data(), dw.data(), dbm.data());
    auto& glnw = im.P.g["scorer.0.weight"];
    for (int64_t i = 0; i < D; ++i) glnw[i] += dw[i];
    if (im.P.w.count("scorer.0.bias")) {
      auto& glnb = im.P.g["scorer.0.bias"];
      for (int64_t i = 0; i < D; ++i) glnb[i] += dbm[i];
    }
    // scatter dxm rows into dh at marker positions
    std::vector<float> dh((size_t)(L * D), 0.f);
    for (int64_t kk = 0; kk < K; ++kk) {
      int64_t pos = ctx.markers[(size_t)kk];
      if (pos < 0) pos = 0;
      if (pos >= L) pos = L - 1;
      float* dst = dh.data() + pos * D;
      const float* src = dxm.data() + kk * D;
      for (int64_t j = 0; j < D; ++j) dst[j] += src[j];
    }
    // flow dh backward through the head layers, accumulating into X-grad chain
    std::vector<float> dX = dh;   // gradient w.r.t. head output X (= h_enc + type_emb input path)
    for (int li = im.head_layers - 1; li >= 0; --li) {
      std::string p = "head.layers." + std::to_string(li) + ".";
      RowCtx::HC& hc = ctx.heads[li];
      // residual: X_out = X_mid + f2; dX_mid = dX ; d_ffn2 pre = dX
      // ffn2: f2 = f1 @ w2^T(+b2), w2 [D, 4D]
      {
        std::vector<float> dA((size_t)(L * 4 * D), 0.f), dW((size_t)(D * 4 * D), 0.f),
            db((size_t)D, 0.f);
        gemm_nt_backward(dX.data(), hc.ffn2.pre.data(), hc.ffn2.A.data(),
                         im.P.at(p + "linear2.weight"), L, D, 4 * D, 0, dA.data(),
                         dW.data(), db.data());
        auto& g = im.P.g[p + "linear2.weight"];
        for (int64_t i = 0; i < D * 4 * D; ++i) g[i] += dW[i];
        auto& gb = im.P.g[p + "linear2.bias"];
        for (int64_t i = 0; i < D; ++i) gb[i] += db[i];
        // ffn1: f1 = relu(y2 @ w1^T(+b1)), w1 [4D, D]
        std::vector<float> dA2((size_t)(L * D), 0.f), dW2((size_t)(4 * D * D), 0.f),
            db2((size_t)(4 * D), 0.f);
        gemm_nt_backward(dA.data(), hc.ffn1.pre.data(), hc.ffn1.A.data(),
                         im.P.at(p + "linear1.weight"), L, 4 * D, D, 2, dA2.data(),
                         dW2.data(), db2.data());
        auto& g2 = im.P.g[p + "linear1.weight"];
        for (int64_t i = 0; i < 4 * D * D; ++i) g2[i] += dW2[i];
        auto& gb2 = im.P.g[p + "linear1.bias"];
        for (int64_t i = 0; i < 4 * D; ++i) gb2[i] += db2[i];
        // ln2 backward
        std::vector<float> dxl((size_t)(L * D)), dw((size_t)D, 0.f), db3((size_t)D, 0.f);
        layer_norm_backward(dA2.data(), hc.ln2.x.data(), im.P.at(p + "norm2.weight"), L, D,
                            1e-5f, true, dxl.data(), dw.data(), db3.data());
        auto& gln = im.P.g[p + "norm2.weight"];
        auto& glnb = im.P.g[p + "norm2.bias"];
        for (int64_t i = 0; i < D; ++i) { gln[i] += dw[i]; glnb[i] += db3[i]; }
        // X was X_mid = X_in + proj; dX_in = dX + dxl ; dproj = dxl... careful: X_mid
        // feeds ln2. But residual: after ffn: X_out = X_mid + f2. So
        //   dX_mid = dX_out (main path) + dxl (through ln2/ffn branch)
        std::vector<float>& dXmid = dxl;   // gradients of the branch
        for (int64_t i = 0; i < L * D; ++i) {
          float total = dX[i] + dxl[i];
          dxl[i] = total;
        }
        // now gradients w.r.t. attention out proj and X_in:
        // X_mid = X_in + out_proj(out). out = reshape(At). in_proj path:
        std::vector<float> dproj = dxl;   // == d(X_mid)
        std::vector<float>& dmain = dxl;  // dX path through residual
        // out_proj backward: proj = ao @ wo^T(+bo), wo [D, D]
        std::vector<float> dao((size_t)(L * D), 0.f), dWo((size_t)(D * D), 0.f),
            dbo((size_t)D, 0.f);
        gemm_nt_backward(dproj.data(), hc.out_proj.pre.data(), hc.out_proj.A.data(),
                         im.P.at(p + "self_attn.out_proj.weight"), L, D, D, 0,
                         dao.data(), dWo.data(), dbo.data());
        auto& go = im.P.g[p + "self_attn.out_proj.weight"];
        for (int64_t i = 0; i < D * D; ++i) go[i] += dWo[i];
        auto& gob = im.P.g[p + "self_attn.out_proj.bias"];
        for (int64_t i = 0; i < D; ++i) gob[i] += dbo[i];
        // attention backward: dao [L,D] -> per-head [H,L,Dh]
        std::vector<float> dAt((size_t)(im.H * L * im.Dh), 0.f);
        for (int64_t i = 0; i < L; ++i)
          for (int h2 = 0; h2 < im.H; ++h2)
            std::memcpy(dAt.data() + (h2 * L + i) * im.Dh, dao.data() + i * D + h2 * im.Dh,
                        sizeof(float) * im.Dh);
        // need q/k/v cached: they were built from hc.in_proj pre-act? in_proj has no
        // act, so q,k,v are columns of in_proj output = gc.out = pre (no act).
        std::vector<float> Q((size_t)(im.H * L * im.Dh)), Kt((size_t)(im.H * L * im.Dh)),
            Vv((size_t)(im.H * L * im.Dh));
        for (int64_t i = 0; i < L; ++i) {
          const float* row = hc.in_proj.pre.data() + i * 3 * HxDh;
          for (int h2 = 0; h2 < im.H; ++h2) {
            std::memcpy(Q.data() + (h2 * L + i) * im.Dh, row + h2 * im.Dh,
                        sizeof(float) * im.Dh);
            std::memcpy(Kt.data() + (h2 * L + i) * im.Dh, row + HxDh + h2 * im.Dh,
                        sizeof(float) * im.Dh);
            std::memcpy(Vv.data() + (h2 * L + i) * im.Dh, row + 2 * HxDh + h2 * im.Dh,
                        sizeof(float) * im.Dh);
          }
        }
        std::vector<float> dQ((size_t)(im.H * L * im.Dh), 0.f),
            dK((size_t)(im.H * L * im.Dh), 0.f), dV((size_t)(im.H * L * im.Dh), 0.f);
        attention_backward_one(dAt.data(), Q.data(), Kt.data(), Vv.data(),
                               hc.attn_probs.data(), im.H, L, im.Dh, L, 0,
                               dQ.data(), dK.data(), dV.data());
        // back to [L, 3D]
        std::vector<float> dqkv((size_t)(L * 3 * D), 0.f);
        for (int64_t i = 0; i < L; ++i)
          for (int h2 = 0; h2 < im.H; ++h2) {
            float* row = dqkv.data() + i * 3 * HxDh;
            std::memcpy(row + h2 * im.Dh, dQ.data() + (h2 * L + i) * im.Dh,
                        sizeof(float) * im.Dh);
            std::memcpy(row + HxDh + h2 * im.Dh, dK.data() + (h2 * L + i) * im.Dh,
                        sizeof(float) * im.Dh);
            std::memcpy(row + 2 * HxDh + h2 * im.Dh, dV.data() + (h2 * L + i) * im.Dh,
                        sizeof(float) * im.Dh);
          }
        // in_proj backward (no act, bias)
        std::vector<float> dA3((size_t)(L * D), 0.f), dW3((size_t)(3 * D * D), 0.f),
            dbi((size_t)(3 * D), 0.f);
        gemm_nt_backward(dqkv.data(), hc.in_proj.pre.data(), hc.in_proj.A.data(),
                         im.P.at(p + "self_attn.in_proj_weight"), L, 3 * D, D, 0,
                         dA3.data(), dW3.data(), dbi.data());
        auto& gi = im.P.g[p + "self_attn.in_proj_weight"];
        for (int64_t i = 0; i < 3 * D * D; ++i) gi[i] += dW3[i];
        auto& gib = im.P.g[p + "self_attn.in_proj_bias"];
        for (int64_t i = 0; i < 3 * D; ++i) gib[i] += dbi[i];
        // ln1 backward
        std::vector<float> dxl1((size_t)(L * D)), dw0((size_t)D, 0.f),
            db0((size_t)D, 0.f);
        layer_norm_backward(dA3.data(), hc.ln1.x.data(), im.P.at(p + "norm1.weight"),
                            L, D, 1e-5f, true, dxl1.data(), dw0.data(), db0.data());
        auto& gln1 = im.P.g[p + "norm1.weight"];
        auto& gln1b = im.P.g[p + "norm1.bias"];
        for (int64_t i = 0; i < D; ++i) { gln1[i] += dw0[i]; gln1b[i] += db0[i]; }
        // residual: X_in feeds both main (through to X_mid) and branch. The forward was:
        //   X = X_in            (after ln1 cache)
        //   X_mid = X_in + proj_out
        //   X_out = X_mid + ffn_out
        // grads: dX_in gets dxl1 (branch) + dmain (residual).
        std::vector<float> dXin((size_t)(L * D));
        for (int64_t i = 0; i < L * D; ++i) dXin[i] = dmain[i] + dxl1[i];
        dX = std::move(dXin);
      }
    }
    // type_emb add: h = h_enc + te[qtype]. dh_encoder = dX (encoder frozen, drop
    // the grad), dte[qtype] = Σ_i dX_i
    auto& gte = im.P.g["type_emb.weight"];
    for (int64_t i = 0; i < L; ++i) {
      const float* src = dX.data() + i * D;
      float* dst = gte.data() + (int64_t)ctx.qtype * D;
      for (int64_t j = 0; j < D; ++j) dst[j] += src[j];
    }
  }
}

}  // namespace

// ----------------------------- public API -------------------------------------------
namespace {

// split one TrainRow into per-question RowCtx with ids+markers
std::vector<RowCtx> encode_row(const TrainRow& row, const std::vector<InternalQ>& internals,
                               Tokenizer& tok, int64_t max_len, int64_t head_max_len,
                               TrainModel::Impl& im) {
  std::vector<RowCtx> out;
  // tokenize the whole state's questions
  std::vector<std::vector<int64_t>> ids_b;
  for (size_t i = 0; i < row.qids.size(); ++i) {
    auto [seq, markers] = build_sequence(tok, row.state, internals[i], max_len,
                                         head_max_len,
                                         row.state.is_array());
    RowCtx ctx;
    ctx.qtype = QTYPES.at(internals[i].t);
    ctx.ids.assign(seq.begin(), seq.end());
    ctx.markers = markers;
    ctx.L = (int64_t)ctx.ids.size();
    ctx.K = (int64_t)markers.size();
    ids_b.push_back(ctx.ids);
    // frozen encoder forward for this row
    std::vector<std::vector<int64_t>> one{ctx.ids};
    std::vector<std::vector<int64_t>> mask{std::vector<int64_t>((size_t)ctx.L, 1)};
    auto h = im.frozen->encode(one, mask);   // [1, L, D]
    ctx.h_encoder.assign((size_t)(ctx.L * im.D), 0.f);
    for (int64_t p = 0; p < ctx.L; ++p)
      for (int64_t d = 0; d < im.D; ++d)
        ctx.h_encoder[(size_t)(p * im.D + d)] = h[0][(size_t)p][(size_t)d];
    out.push_back(std::move(ctx));
  }
  return out;
}

}  // namespace

TrainModel::StepStats TrainModel::step(const TrainRow& row, bool accumulate_only) {
  (void)accumulate_only;
  Impl& im = *impl_;
  std::vector<InternalQ> internals;
  for (const auto& qd : row.qdefs) internals.push_back(to_internal(qd));
  auto rows = encode_row(row, internals, *tok_, im.max_len, im.head_max_len, im);

  double loss_sum = 0;
  int acc_right = 0;
  for (size_t i = 0; i < rows.size(); ++i) {
    RowCtx& ctx = rows[i];
    forward_row(im, ctx);
    const std::vector<double>& target = row.targets[i];
    loss_sum += loss_fwd(ctx.logits.data(), target, ctx.qtype);
    int argmax = 0, gold = 0;
    for (int j = 0; j < (int)ctx.K; ++j) {
      if (ctx.logits[j] > ctx.logits[argmax]) argmax = j;
      if (target[j] > target[gold]) gold = j;
    }
    acc_right += argmax == gold;
    backward_row(im, ctx, target, ctx.qtype, 1.0 / std::max<size_t>(1, rows.size()));
  }
  int nrows = (int)rows.size();
  return {loss_sum / std::max(1, nrows), (double)acc_right / std::max(1, nrows), nrows};
}

void TrainModel::zero_grad() { impl_->P.zero_grad(); }

void TrainModel::optimizer_step(double lr) {
  Impl& im = *impl_;
  im.adam_t++;
  double bc1 = 1 - std::pow(im.b1, (double)im.adam_t);
  double bc2 = 1 - std::pow(im.b2, (double)im.adam_t);
  for (const auto& n : im.P.names) {
    auto& w = im.P.w[n];
    auto& g = im.P.g[n];
    auto& m1 = im.P.m1[n];
    auto& m2 = im.P.m2[n];
    for (size_t i = 0; i < w.size(); ++i) {
      m1[i] = (float)(im.b1 * m1[i] + (1 - im.b1) * g[i]);
      m2[i] = (float)(im.b2 * m2[i] + (1 - im.b2) * g[i] * g[i]);
      double mh = m1[i] / bc1, vh = m2[i] / bc2;
      w[i] = (float)(w[i] - lr * (mh / (std::sqrt(vh) + im.eps_adam) + im.wd * w[i]));
    }
  }
}

// ---- checkpoint save -------------------------------------------------------------
void TrainModel::save_checkpoint(const std::string& out_dir,
                                 const std::vector<double>& temperature,
                                 const std::map<std::string, double>& temp_by_options,
                                 const ordered_json& train_meta) {
  Impl& im = *impl_;
  fs::create_directories(out_dir);
  // full tensor list = base weights with trained weights spliced in
  auto w = SafeTensors::load((fs::path(base_dir_) / "model.safetensors").string());
  std::vector<TensorOut> out;
  for (const auto& name : w.names()) {
    auto sh = w.shape_of(name);
    int64_t n = 1;
    for (auto d : sh) n *= d;
    bool is_fp32 = name == "temperature";
    if (im.P.w.count(name)) {
      // trained tensor: copy from the store
      std::shared_ptr<float[]> buf(new float[(size_t)n]);
      std::memcpy(buf.get(), im.P.at(name), sizeof(float) * n);
      out.push_back({name, sh, buf, !is_fp32});
    } else {
      std::shared_ptr<float[]> buf = w.data_f32(name);
      out.push_back({name, sh, buf, !is_fp32});
    }
  }
  save_safetensors((fs::path(out_dir) / "model.safetensors").string(), out);

  ordered_json cfg = cfg_;
  cfg["temperature"] = temperature;
  ordered_json tbo = ordered_json::object();
  for (const auto& [k, v] : temp_by_options) tbo[k] = v;
  cfg["temperature_by_options"] = tbo;
  cfg["training"] = train_meta;
  std::ofstream(fs::path(out_dir) / "rl_agent_config.json") << cfg.dump(2);

  // copy tokenizer + encoder dirs
  for (const char* sub : {"tokenizer", "encoder"}) {
    fs::path src = fs::path(base_dir_) / sub;
    if (fs::is_directory(src))
      fs::copy(src, fs::path(out_dir) / sub,
               fs::copy_options::recursive | fs::copy_options::overwrite_existing);
  }
}

// ---- gradient-check helpers --------------------------------------------------------
std::vector<std::string> TrainModel::param_names() const {
  return std::vector<std::string>(impl_->P.names.begin(), impl_->P.names.end());
}
int64_t TrainModel::param_numel(const std::string& name) const {
  return impl_->P.numel(name);
}
float TrainModel::param_get(const std::string& name, int64_t i) const {
  return impl_->P.w.at(name)[(size_t)i];
}
void TrainModel::param_set(const std::string& name, int64_t i, float v) {
  impl_->P.w.at(name)[(size_t)i] = v;
}
std::vector<float> TrainModel::grad_of(const std::string& name) {
  return impl_->P.g.at(name);
}

double TrainModel::forward_loss_double(const TrainRow& row) {
  Impl& im = *impl_;
  std::vector<InternalQ> internals;
  for (const auto& qd : row.qdefs) internals.push_back(to_internal(qd));
  auto rows = encode_row(row, internals, *tok_, im.max_len, im.head_max_len, im);
  double total = 0;
  for (size_t i = 0; i < rows.size(); ++i) {
    RowCtx& ctx = rows[i];
    forward_row(im, ctx);
    total += loss_fwd(ctx.logits.data(), row.targets[i], ctx.qtype);
  }
  return total / std::max<size_t>(1, rows.size());
}

// ----------------------------- JSONL dataset ---------------------------------------

std::vector<TrainRow> load_train_rows(const std::string& jsonl_path) {
  std::ifstream f(jsonl_path);
  if (!f) throw std::runtime_error("load_train_rows: cannot open " + jsonl_path);
  std::vector<TrainRow> rows;
  std::string line;
  while (std::getline(f, line)) {
    while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
    if (line.empty()) continue;
    ordered_json j;
    try {
      j = ordered_json::parse(line);
    } catch (const std::exception& e) {
      throw std::runtime_error("load_train_rows: bad JSON in " + jsonl_path + ": " +
                               e.what());
    }
    TrainRow r;
    r.state = j["state"];
    r.workflow = j.value("workflow", "");
    const ordered_json& qs = j["questions"];
    const ordered_json& gd = j["gold"];
    for (auto it = qs.begin(); it != qs.end(); ++it) {
      const std::string& qid = it.key();
      if (!gd.contains(qid) || !gd[qid].contains("probabilities")) continue;
      const ordered_json& probs = gd[qid]["probabilities"];
      const ordered_json& qdef = qs[qid];
      std::string t = qdef.value("type", "");
      std::vector<double> tgt;
      if (t == "noul") {
        double pf = probs.value("false", 0.0), pt = probs.value("true", 0.0);
        double s = pf + pt;
        tgt = s > 0 ? std::vector<double>{pf / s, pt / s}
                    : std::vector<double>{0.5, 0.5};
      } else if (t == "choice") {
        const ordered_json& crit = qdef["criteria"];
        std::vector<double> raw;
        for (auto it2 = crit.begin(); it2 != crit.end(); ++it2)
          raw.push_back(probs.value(it2.key(), 0.0));
        double s = 0;
        for (double v : raw) s += v;
        if (s <= 0) continue;
        for (double v : raw) tgt.push_back(v / s);
      } else if (t == "score") {
        int k = (int)qdef["criteria"].size();
        for (int i = 0; i < k; ++i)
          tgt.push_back(probs.value(std::to_string(i), 0.0));
        double s = 0;
        for (double v : tgt) s += v;
        if (s <= 0) continue;
        for (double& v : tgt) v /= s;
      } else continue;
      r.qids.push_back(qid);
      r.qdefs.push_back(qdef);
      r.targets.push_back(std::move(tgt));
    }
    if (!r.qids.empty()) rows.push_back(std::move(r));
  }
  return rows;
}

void TrainModel::eval_logits(const std::vector<TrainRow>& rows,
                             std::vector<std::vector<float>>* logits_out,
                             std::vector<std::vector<double>>* targets_out,
                             std::vector<int>* qtypes_out) {
  Impl& im = *impl_;
  logits_out->clear(); targets_out->clear(); qtypes_out->clear();
  for (const auto& row : rows) {
    std::vector<InternalQ> internals;
    for (const auto& qd : row.qdefs) internals.push_back(to_internal(qd));
    auto ctxs = encode_row(row, internals, *tok_, im.max_len, im.head_max_len, im);
    for (size_t i = 0; i < ctxs.size(); ++i) {
      forward_row(im, ctxs[i]);
      logits_out->push_back(ctxs[i].logits);
      targets_out->push_back(row.targets[i]);
      qtypes_out->push_back(ctxs[i].qtype);
    }
  }
}

const std::vector<std::string>& TrainModel::training_param_names() const {
  return impl_->P.names;
}

}  // namespace snapjudge
