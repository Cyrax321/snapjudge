#include "snapjudge/sj_model.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include "snapjudge/safetensors.hpp"
#include "snapjudge/tensor.hpp"

namespace snapjudge {

using nlohmann::json;

namespace {

[[noreturn]] void fail(const std::string& id, const std::string& msg) {
  throw std::runtime_error("SjModel '" + id + "': " + msg);
}

void require_shape(const SafeTensors& w, const std::string& name,
                   std::initializer_list<int64_t> want, const std::string& id) {
  if (!w.has(name)) fail(id, "missing tensor " + name);
  if (w.shape_of(name) != std::vector<int64_t>(want)) {
    std::ostringstream os;
    os << "tensor " << name << " expected [";
    bool first = true;
    for (int64_t v : want) { os << (first ? "" : ",") << v; first = false; }
    os << "] but has [";
    first = true;
    for (int64_t v : w.shape_of(name)) { os << (first ? "" : ",") << v; first = false; }
    os << "]";
    fail(id, os.str());
  }
}

// Non-owning fp32 matrix view for GEMM (shares the pointer, never frees).
Tensor view_of(const float* p, int64_t M, int64_t K) {
  return Tensor::wrap({M, K}, std::shared_ptr<float[]>(const_cast<float*>(p), [](float*) {}));
}

// Multi-head attention with separate q/k/v/o projections (each [D, D] + bias).
// Y is [M, D] (M = B*L). Returns the projected output [M, D].
Tensor self_attention(const Tensor& Y, int B, int H, int L, int Dh,
                      const std::vector<int64_t>& lens,
                      const float* qw, const float* qb,
                      const float* kw, const float* kb,
                      const float* vw, const float* vb,
                      const float* ow, const float* ob) {
  int D = H * Dh;
  Tensor Q = gemm_nt(Y, view_of(qw, D, D), qb, 0);   // [M, D]
  Tensor Kt = gemm_nt(Y, view_of(kw, D, D), kb, 0);
  Tensor V = gemm_nt(Y, view_of(vw, D, D), vb, 0);

  // permute [M, D] -> [B, H, L, Dh]
  Tensor q({B, H, L, Dh}), k({B, H, L, Dh}), v({B, H, L, Dh});
  for (int b = 0; b < B; ++b)
    for (int i = 0; i < L; ++i) {
      const float* qr = Q.data_ptr() + (b * L + i) * D;
      const float* kr = Kt.data_ptr() + (b * L + i) * D;
      const float* vr = V.data_ptr() + (b * L + i) * D;
      for (int h = 0; h < H; ++h) {
        std::memcpy(q.mutable_data() + ((b * H + h) * L + i) * Dh, qr + h * Dh,
                    sizeof(float) * Dh);
        std::memcpy(k.mutable_data() + ((b * H + h) * L + i) * Dh, kr + h * Dh,
                    sizeof(float) * Dh);
        std::memcpy(v.mutable_data() + ((b * H + h) * L + i) * Dh, vr + h * Dh,
                    sizeof(float) * Dh);
      }
    }
  Tensor At = attention(q, k, v, lens, 0);   // [B, H, L, Dh]

  // permute back to [M, D]
  Tensor AO({B * L, D});
  for (int b = 0; b < B; ++b)
    for (int i = 0; i < L; ++i)
      for (int h = 0; h < H; ++h)
        std::memcpy(AO.mutable_data() + (b * L + i) * D + h * Dh,
                    At.data_ptr() + ((b * H + h) * L + i) * Dh,
                    sizeof(float) * Dh);
  return gemm_nt(AO, view_of(ow, D, D), ob, 0);
}

}  // namespace

std::unique_ptr<SjModel> SjModel::load(const json& cfg_json, const SafeTensors& w,
                                      const std::string& model_id) {
  if (!cfg_json.contains("vocab_size") || !cfg_json.contains("hidden_size") ||
      !cfg_json.contains("num_layers") || !cfg_json.contains("num_heads"))
    fail(model_id, "config missing required keys (vocab_size/hidden_size/num_layers/num_heads)");

  auto m = std::make_unique<SjModel>();
  SjConfig& c = m->cfg_;
  c.vocab_size = cfg_json.value("vocab_size", 0);
  c.hidden_size = cfg_json.value("hidden_size", 0);
  c.num_layers = cfg_json.value("num_layers", 0);
  c.num_heads = cfg_json.value("num_heads", 0);
  c.intermediate_size = cfg_json.value("intermediate_size", 4 * c.hidden_size);
  c.max_positions = cfg_json.value("max_positions", 512);
  c.norm_eps = cfg_json.value("norm_eps", 1e-5f);
  c.head_layers = cfg_json.value("head_layers", 1);
  c.head_intermediate = cfg_json.value("head_intermediate", 4 * c.hidden_size);
  c.max_len = cfg_json.value("max_len", 512);
  c.head_max_len = cfg_json.value("head_max_len", 192);
  c.n_types = cfg_json.value("n_types", 3);
  c.cls_id = cfg_json.value("cls_id", 0);
  c.sep_id = cfg_json.value("sep_id", 0);
  c.mask_id = cfg_json.value("mask_id", 0);
  c.pad_id = cfg_json.value("pad_id", 0);

  const int D = c.hidden_size;
  const int H = c.num_heads;
  const int F = c.intermediate_size;
  if (D % H != 0) fail(model_id, "hidden_size must be divisible by num_heads");

  // embeddings
  require_shape(w, "sj.embeddings.word.weight", {c.vocab_size, D}, model_id);
  m->word_emb_ = w.data_f32("sj.embeddings.word.weight");
  require_shape(w, "sj.embeddings.position.weight", {c.max_positions, D}, model_id);
  m->pos_emb_ = w.data_f32("sj.embeddings.position.weight");
  require_shape(w, "sj.embeddings.norm.weight", {D}, model_id);
  m->emb_norm_w_ = w.data_f32("sj.embeddings.norm.weight");
  require_shape(w, "sj.embeddings.norm.bias", {D}, model_id);
  m->emb_norm_b_ = w.data_f32("sj.embeddings.norm.bias");

  // encoder layers
  m->enc_.resize(static_cast<size_t>(c.num_layers));
  for (int i = 0; i < c.num_layers; ++i) {
    EncLayer& ly = m->enc_[static_cast<size_t>(i)];
    std::string p = "sj.encoder.layers." + std::to_string(i) + ".";
    require_shape(w, p + "attn_norm.weight", {D}, model_id);
    ly.an_w = w.data_f32(p + "attn_norm.weight");
    require_shape(w, p + "attn_norm.bias", {D}, model_id);
    ly.an_b = w.data_f32(p + "attn_norm.bias");
    require_shape(w, p + "attn.q.weight", {D, D}, model_id); ly.q_w = w.data_f32(p + "attn.q.weight");
    require_shape(w, p + "attn.q.bias", {D}, model_id);       ly.q_b = w.data_f32(p + "attn.q.bias");
    require_shape(w, p + "attn.k.weight", {D, D}, model_id); ly.k_w = w.data_f32(p + "attn.k.weight");
    require_shape(w, p + "attn.k.bias", {D}, model_id);       ly.k_b = w.data_f32(p + "attn.k.bias");
    require_shape(w, p + "attn.v.weight", {D, D}, model_id); ly.v_w = w.data_f32(p + "attn.v.weight");
    require_shape(w, p + "attn.v.bias", {D}, model_id);       ly.v_b = w.data_f32(p + "attn.v.bias");
    require_shape(w, p + "attn.o.weight", {D, D}, model_id); ly.o_w = w.data_f32(p + "attn.o.weight");
    require_shape(w, p + "attn.o.bias", {D}, model_id);       ly.o_b = w.data_f32(p + "attn.o.bias");
    require_shape(w, p + "ffn_norm.weight", {D}, model_id);
    ly.fn_w = w.data_f32(p + "ffn_norm.weight");
    require_shape(w, p + "ffn_norm.bias", {D}, model_id);
    ly.fn_b = w.data_f32(p + "ffn_norm.bias");
    require_shape(w, p + "ffn.w1.weight", {F, D}, model_id); ly.w1_w = w.data_f32(p + "ffn.w1.weight");
    require_shape(w, p + "ffn.w1.bias", {F}, model_id);       ly.w1_b = w.data_f32(p + "ffn.w1.bias");
    require_shape(w, p + "ffn.w2.weight", {D, F}, model_id); ly.w2_w = w.data_f32(p + "ffn.w2.weight");
    require_shape(w, p + "ffn.w2.bias", {D}, model_id);       ly.w2_b = w.data_f32(p + "ffn.w2.bias");
  }

  require_shape(w, "sj.encoder.final_norm.weight", {D}, model_id);
  m->final_norm_w_ = w.data_f32("sj.encoder.final_norm.weight");
  require_shape(w, "sj.encoder.final_norm.bias", {D}, model_id);
  m->final_norm_b_ = w.data_f32("sj.encoder.final_norm.bias");

  require_shape(w, "sj.type_emb.weight", {c.n_types, D}, model_id);
  m->type_emb_ = w.data_f32("sj.type_emb.weight");

  // head layers
  m->head_.resize(static_cast<size_t>(c.head_layers));
  for (int i = 0; i < c.head_layers; ++i) {
    HeadLayer& hl = m->head_[static_cast<size_t>(i)];
    std::string p = "sj.head.layers." + std::to_string(i) + ".";
    require_shape(w, p + "norm1.weight", {D}, model_id); hl.n1_w = w.data_f32(p + "norm1.weight");
    require_shape(w, p + "norm1.bias", {D}, model_id);   hl.n1_b = w.data_f32(p + "norm1.bias");
    require_shape(w, p + "norm2.weight", {D}, model_id); hl.n2_w = w.data_f32(p + "norm2.weight");
    require_shape(w, p + "norm2.bias", {D}, model_id);   hl.n2_b = w.data_f32(p + "norm2.bias");
    require_shape(w, p + "attn.q.weight", {D, D}, model_id); hl.q_w = w.data_f32(p + "attn.q.weight");
    require_shape(w, p + "attn.q.bias", {D}, model_id);       hl.q_b = w.data_f32(p + "attn.q.bias");
    require_shape(w, p + "attn.k.weight", {D, D}, model_id); hl.k_w = w.data_f32(p + "attn.k.weight");
    require_shape(w, p + "attn.k.bias", {D}, model_id);       hl.k_b = w.data_f32(p + "attn.k.bias");
    require_shape(w, p + "attn.v.weight", {D, D}, model_id); hl.v_w = w.data_f32(p + "attn.v.weight");
    require_shape(w, p + "attn.v.bias", {D}, model_id);       hl.v_b = w.data_f32(p + "attn.v.bias");
    require_shape(w, p + "attn.o.weight", {D, D}, model_id); hl.o_w = w.data_f32(p + "attn.o.weight");
    require_shape(w, p + "attn.o.bias", {D}, model_id);       hl.o_b = w.data_f32(p + "attn.o.bias");
    require_shape(w, p + "ffn.w1.weight", {c.head_intermediate, D}, model_id);
    hl.w1_w = w.data_f32(p + "ffn.w1.weight");
    require_shape(w, p + "ffn.w1.bias", {c.head_intermediate}, model_id);
    hl.w1_b = w.data_f32(p + "ffn.w1.bias");
    require_shape(w, p + "ffn.w2.weight", {D, c.head_intermediate}, model_id);
    hl.w2_w = w.data_f32(p + "ffn.w2.weight");
    require_shape(w, p + "ffn.w2.bias", {D}, model_id);
    hl.w2_b = w.data_f32(p + "ffn.w2.bias");
  }

  // scorer + act head
  require_shape(w, "sj.scorer.norm.weight", {D}, model_id);
  m->scorer_n_w_ = w.data_f32("sj.scorer.norm.weight");
  require_shape(w, "sj.scorer.norm.bias", {D}, model_id);
  m->scorer_n_b_ = w.data_f32("sj.scorer.norm.bias");
  require_shape(w, "sj.scorer.w1.weight", {D, D}, model_id);
  m->scorer_w1_ = w.data_f32("sj.scorer.w1.weight");
  require_shape(w, "sj.scorer.w1.bias", {D}, model_id);
  m->scorer_b1_ = w.data_f32("sj.scorer.w1.bias");
  require_shape(w, "sj.scorer.w2.weight", {1, D}, model_id);
  m->scorer_w2_ = w.data_f32("sj.scorer.w2.weight");
  require_shape(w, "sj.scorer.w2.bias", {1}, model_id);
  m->scorer_b2_ = w.data_f32("sj.scorer.w2.bias");

  return m;
}

Tensor SjModel::run_encoder(const std::vector<std::vector<int64_t>>& input_ids,
                            const std::vector<std::vector<int64_t>>& attention_mask) const {
  const SjConfig& c = cfg_;
  const int D = c.hidden_size, H = c.num_heads, F = c.intermediate_size;
  const int Dh = D / H;
  const int64_t B = static_cast<int64_t>(input_ids.size());
  const int64_t L = B ? static_cast<int64_t>(input_ids[0].size()) : 0;
  const int64_t M = B * L;
  const float eps = c.norm_eps;

  std::vector<int64_t> lens(static_cast<size_t>(B), L);
  for (int64_t b = 0; b < B; ++b) {
    int64_t s = 0;
    for (int64_t v : attention_mask[static_cast<size_t>(b)]) s += v;
    lens[static_cast<size_t>(b)] = s;
  }

  // word + position embeddings, then LayerNorm
  Tensor X({M, D});
  {
    std::vector<int64_t> flat_ids;
    flat_ids.reserve(static_cast<size_t>(M));
    for (const auto& row : input_ids) flat_ids.insert(flat_ids.end(), row.begin(), row.end());
    Tensor wemb = embedding(view_of(word_emb_.get(), c.vocab_size, D), flat_ids.data(), M);
    // add position embedding: for each row r, position = r % L
    Tensor pemb({M, D});
    for (int64_t r = 0; r < M; ++r) {
      int64_t pos = r % L;
      if (pos >= c.max_positions) pos = c.max_positions - 1;
      std::memcpy(pemb.mutable_data() + r * D, pos_emb_.get() + pos * D,
                  sizeof(float) * D);
    }
    add_(wemb, pemb);
    X = layer_norm(wemb, emb_norm_w_.get(), emb_norm_b_.get(), eps);
  }

  Tensor Y;
  for (const auto& ly : enc_) {
    Y = layer_norm(X, ly.an_w.get(), ly.an_b.get(), eps);
    Y = self_attention(Y, static_cast<int>(B), H, static_cast<int>(L), Dh, lens,
                      ly.q_w.get(), ly.q_b.get(), ly.k_w.get(), ly.k_b.get(),
                      ly.v_w.get(), ly.v_b.get(), ly.o_w.get(), ly.o_b.get());
    add_(X, Y);
    Y = layer_norm(X, ly.fn_w.get(), ly.fn_b.get(), eps);
    Y = gemm_nt(Y, view_of(ly.w1_w.get(), F, D), ly.w1_b.get(), 1);   // GELU
    Y = gemm_nt(Y, view_of(ly.w2_w.get(), D, F), ly.w2_b.get(), 0);
    add_(X, Y);
  }
  return layer_norm(X, final_norm_w_.get(), final_norm_b_.get(), eps);
}

std::vector<std::vector<std::vector<float>>> SjModel::encode(
    const std::vector<std::vector<int64_t>>& input_ids,
    const std::vector<std::vector<int64_t>>& attention_mask) const {
  const int64_t B = static_cast<int64_t>(input_ids.size());
  const int64_t L = B ? static_cast<int64_t>(input_ids[0].size()) : 0;
  const int D = cfg_.hidden_size;
  std::vector<std::vector<std::vector<float>>> out(static_cast<size_t>(B));
  if (!B) return out;

  Tensor h = run_encoder(input_ids, attention_mask);
  for (int64_t b = 0; b < B; ++b) {
    out[static_cast<size_t>(b)].resize(static_cast<size_t>(L));
    for (int64_t i = 0; i < L; ++i) {
      const float* src = h.data_ptr() + (b * L + i) * D;
      out[static_cast<size_t>(b)][static_cast<size_t>(i)].assign(src, src + D);
    }
  }
  return out;
}

void SjModel::forward(const std::vector<std::vector<int64_t>>& input_ids,
                      const std::vector<std::vector<int64_t>>& attention_mask,
                      const std::vector<std::vector<int64_t>>& marker_pos,
                      const std::vector<std::vector<uint8_t>>& marker_mask,
                      const std::vector<int64_t>& qtype,
                      std::vector<std::vector<float>>& logits,
                      std::vector<std::vector<float>>& act) const {
  const int64_t B = static_cast<int64_t>(input_ids.size());
  if (B == 0) return;
  const int64_t L = static_cast<int64_t>(input_ids[0].size());
  const int64_t K = static_cast<int64_t>(marker_pos[0].size());
  const int D = cfg_.hidden_size;
  const int H = cfg_.num_heads;
  const int Dh = D / H;
  const float eps = cfg_.norm_eps;

  Tensor h = run_encoder(input_ids, attention_mask);   // [M, D]

  // add type embedding per row
  for (int64_t b = 0; b < B; ++b) {
    const float* te = type_emb_.get() + qtype[static_cast<size_t>(b)] * D;
    for (int64_t i = 0; i < L; ++i) {
      float* hr = h.mutable_data() + (b * L + i) * D;
      for (int j = 0; j < D; ++j) hr[j] += te[j];
    }
  }

  // head transformer layers
  {
    Tensor Xf = h;
    std::vector<int64_t> lens(static_cast<size_t>(B), L);
    for (int64_t b = 0; b < B; ++b) {
      int64_t s = 0;
      for (int64_t v : attention_mask[static_cast<size_t>(b)]) s += v;
      lens[static_cast<size_t>(b)] = s;
    }
    for (const auto& hl : head_) {
      Tensor Y = layer_norm(Xf, hl.n1_w.get(), hl.n1_b.get(), eps);
      Y = self_attention(Y, static_cast<int>(B), H, static_cast<int>(L), Dh, lens,
                        hl.q_w.get(), hl.q_b.get(), hl.k_w.get(), hl.k_b.get(),
                        hl.v_w.get(), hl.v_b.get(), hl.o_w.get(), hl.o_b.get());
      add_(Xf, Y);
      Y = layer_norm(Xf, hl.n2_w.get(), hl.n2_b.get(), eps);
      Y = gemm_nt(Y, view_of(hl.w1_w.get(), cfg_.head_intermediate, D), hl.w1_b.get(), 1);
      Y = gemm_nt(Y, view_of(hl.w2_w.get(), D, cfg_.head_intermediate), hl.w2_b.get(), 0);
      add_(Xf, Y);
    }
    h = std::move(Xf);
  }

  // scorer: gather marker rows -> LN -> Linear(D,D) GELU -> Linear(1)
  {
    std::vector<int64_t> flat_pos;
    flat_pos.reserve(static_cast<size_t>(B * K));
    for (const auto& row : marker_pos) flat_pos.insert(flat_pos.end(), row.begin(), row.end());
    Tensor m({B * K, D});
    for (int64_t b = 0; b < B; ++b)
      for (int64_t kk = 0; kk < K; ++kk) {
        int64_t p = flat_pos[static_cast<size_t>(b * K + kk)];
        if (p < 0) p = 0;
        if (p >= L) p = L - 1;
        std::memcpy(m.mutable_data() + (b * K + kk) * D,
                    h.data_ptr() + (b * L + p) * D, sizeof(float) * D);
      }
    Tensor s = layer_norm(m, scorer_n_w_.get(), scorer_n_b_.get(), eps);
    s = gemm_nt(s, view_of(scorer_w1_.get(), D, D), scorer_b1_.get(), 1);
    s = gemm_nt(s, view_of(scorer_w2_.get(), 1, D), scorer_b2_.get(), 0);   // [BK, 1]

    logits.assign(static_cast<size_t>(B), std::vector<float>(static_cast<size_t>(K)));
    for (int64_t b = 0; b < B; ++b)
      for (int64_t kk = 0; kk < K; ++kk) {
        float lg = s.data_ptr()[b * K + kk];
        if (!marker_mask[static_cast<size_t>(b)][static_cast<size_t>(kk)]) lg = -1e4f;
        logits[static_cast<size_t>(b)][static_cast<size_t>(kk)] = lg;
      }
  }

  // act head: pooled [CLS] row + decision statistics -> softmax over 2 acts
  act.assign(static_cast<size_t>(B), std::vector<float>(2, 0.0f));
  for (int64_t b = 0; b < B; ++b) {
    const std::vector<float>& lg = logits[static_cast<size_t>(b)];
    float mx = *std::max_element(lg.begin(), lg.end());
    std::vector<double> p(lg.size());
    double s = 0;
    for (size_t j = 0; j < lg.size(); ++j) { p[j] = std::exp(lg[j] - mx); s += p[j]; }
    for (double& v : p) v /= s;
    // simple, deterministic act: 1 - max-probability concentration
    double top1 = *std::max_element(p.begin(), p.end());
    act[static_cast<size_t>(b)][0] = static_cast<float>(top1);
    act[static_cast<size_t>(b)][1] = static_cast<float>(1.0 - top1);
  }
}

}  // namespace snapjudge
