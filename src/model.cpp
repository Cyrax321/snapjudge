#include "snapjudge/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include "snapjudge/safetensors.hpp"
#include "snapjudge/tensor.hpp"

namespace snapjudge {

using nlohmann::json;

// ---- helpers ------------------------------------------------------------------
static void require_shape(const SafeTensors& w, const std::string& name,
                          std::initializer_list<int64_t> want, const std::string& id) {
  if (!w.has(name))
    throw std::runtime_error("Model weights incomplete for '" + id +
                             "': missing tensor " + name);
  if (w.shape_of(name) != std::vector<int64_t>(want)) {
    std::ostringstream os;
    os << "Model architecture mismatch for '" << id << "': " << name << " expected [";
    bool first = true;
    for (int64_t v : want) { os << (first ? "" : ",") << v; first = false; }
    os << "] but checkpoint has [";
    first = true;
    for (int64_t v : w.shape_of(name)) { os << (first ? "" : ",") << v; first = false; }
    os << "]";
    throw std::runtime_error(os.str());
  }
}

std::unique_ptr<DecisionModel> DecisionModel::load(const json& rl_cfg, const json& enc_cfg,
                                                   const SafeTensors& w,
                                                   const std::string& model_id) {
  // --- mirror _verify_compatibility step 1: required config keys
  for (const char* k : {"encoder", "head_layers"}) {
    if (!rl_cfg.contains(k))
      throw std::runtime_error("Incompatible model config for '" + model_id +
                               "': missing configuration keys ['" + k +
                               "']. Ensure this is a valid RL Agent decision model.");
  }
  // --- step 2: required component prefixes
  for (const char* p : {"encoder.", "type_emb.", "scorer.", "act_head."}) {
    bool any = false;
    for (const auto& n : w.names())
      if (n.rfind(p, 0) == 0) { any = true; break; }
    if (!any)
      throw std::runtime_error("Incompatible model weights for '" + model_id +
                               "': checkpoint is missing '" + p +
                               "' parameters. Expected an RL Agent decision model with "
                               "encoder and decision heads.");
  }

  auto m = std::make_unique<DecisionModel>();
  ModelConfig& c = m->cfg_;
  c.encoder = rl_cfg.value("encoder", std::string());
  c.head_layers = rl_cfg.value("head_layers", 2);
  c.max_len = rl_cfg.value("max_len", 512);
  c.head_max_len = rl_cfg.value("head_max_len", 192);
  {
    int n_costs = 0;
    if (rl_cfg.contains("act_costs") && rl_cfg["act_costs"].is_object())
      n_costs = static_cast<int>(rl_cfg["act_costs"].size());
    c.n_act = n_costs + 1;
  }

  c.hidden_size = enc_cfg.value("hidden_size", 0);
  c.num_hidden_layers = enc_cfg.value("num_hidden_layers", 0);
  c.num_attention_heads = enc_cfg.value("num_attention_heads", 0);
  c.intermediate_size = enc_cfg.value("intermediate_size", 0);
  c.max_position_embeddings = enc_cfg.value("max_position_embeddings", 8192);
  c.vocab_size = enc_cfg.value("vocab_size", 0);
  c.norm_eps = enc_cfg.value("norm_eps", 1e-5f);
  c.norm_bias = enc_cfg.value("norm_bias", false);
  c.local_attention = enc_cfg.value("local_attention", 0);
  if (enc_cfg.contains("layer_types"))
    c.layer_types = enc_cfg["layer_types"].get<std::vector<std::string>>();
  if (enc_cfg.contains("rope_parameters")) {
    const auto& rp = enc_cfg["rope_parameters"];
    if (rp.contains("full_attention"))
      c.rope_theta_full = rp["full_attention"].value("rope_theta", c.rope_theta_full);
    if (rp.contains("sliding_attention"))
      c.rope_theta_sliding = rp["sliding_attention"].value("rope_theta", c.rope_theta_sliding);
  }

  m->D_ = c.hidden_size;
  m->H_ = c.num_attention_heads;
  m->Dh_ = m->D_ / m->H_;
  m->F_ = c.intermediate_size;
  const int D = m->D_;

  // --- load encoder tensors (strict shapes)
  require_shape(w, "encoder.embeddings.tok_embeddings.weight", {c.vocab_size, D}, model_id);
  m->emb_w_ = w.data_f32("encoder.embeddings.tok_embeddings.weight");
  require_shape(w, "encoder.embeddings.norm.weight", {D}, model_id);
  m->emb_norm_ = w.data_f32("encoder.embeddings.norm.weight");
  require_shape(w, "encoder.final_norm.weight", {D}, model_id);
  m->final_norm_ = w.data_f32("encoder.final_norm.weight");

  int win = c.local_attention > 0 ? c.local_attention / 2 : 0;
  m->layers_.resize(static_cast<size_t>(c.num_hidden_layers));
  for (int i = 0; i < c.num_hidden_layers; ++i) {
    LayerWeights& ly = m->layers_[static_cast<size_t>(i)];
    std::string pfx = "encoder.layers." + std::to_string(i) + ".";
    ly.attention_type = i < static_cast<int>(c.layer_types.size())
                            ? c.layer_types[static_cast<size_t>(i)]
                            : "full_attention";
    ly.window = ly.attention_type == "sliding_attention" ? win : 0;
    std::string an = pfx + "attn_norm.weight";
    ly.has_attn_norm = w.has(an);
    if (ly.has_attn_norm) {
      require_shape(w, an, {D}, model_id);
      ly.attn_norm.d = w.data_f32(an);
    }
    require_shape(w, pfx + "attn.Wqkv.weight", {3 * D, D}, model_id);
    ly.wqkv.d = w.data_f32(pfx + "attn.Wqkv.weight");
    require_shape(w, pfx + "attn.Wo.weight", {D, D}, model_id);
    ly.wo.d = w.data_f32(pfx + "attn.Wo.weight");
    require_shape(w, pfx + "mlp_norm.weight", {D}, model_id);
    ly.mlp_norm.d = w.data_f32(pfx + "mlp_norm.weight");
    require_shape(w, pfx + "mlp.Wi.weight", {2 * m->F_, D}, model_id);
    ly.wi.d = w.data_f32(pfx + "mlp.Wi.weight");
    require_shape(w, pfx + "mlp.Wo.weight", {D, m->F_}, model_id);
    ly.wo2.d = w.data_f32(pfx + "mlp.Wo.weight");
  }

  require_shape(w, "type_emb.weight", {3, D}, model_id);
  m->type_emb_ = w.data_f32("type_emb.weight");

  // --- decision head layers
  m->head_.resize(static_cast<size_t>(c.head_layers));
  for (int i = 0; i < c.head_layers; ++i) {
    HeadLayer& hl = m->head_[static_cast<size_t>(i)];
    std::string pfx = "head.layers." + std::to_string(i) + ".";
    auto get = [&](const char* suffix, std::initializer_list<int64_t> sh) {
      require_shape(w, pfx + suffix, sh, model_id);
      return w.data_f32(pfx + suffix);
    };
    hl.n1w = get("norm1.weight", {D});
    hl.n1b = get("norm1.bias", {D});
    hl.n2w = get("norm2.weight", {D});
    hl.n2b = get("norm2.bias", {D});
    hl.in_w = get("self_attn.in_proj_weight", {3 * D, D});
    hl.in_b = get("self_attn.in_proj_bias", {3 * D});
    hl.out_w = get("self_attn.out_proj.weight", {D, D});
    hl.out_b = get("self_attn.out_proj.bias", {D});
    hl.l1w = get("linear1.weight", {4 * D, D});
    hl.l1b = get("linear1.bias", {4 * D});
    hl.l2w = get("linear2.weight", {D, 4 * D});
    hl.l2b = get("linear2.bias", {D});
  }

  // --- scorer + act head
  require_shape(w, "scorer.0.weight", {D}, model_id);
  m->scorer_ln_w_ = w.data_f32("scorer.0.weight");
  m->scorer_ln_b_ = w.has("scorer.0.bias") ? w.data_f32("scorer.0.bias") : nullptr;
  require_shape(w, "scorer.1.weight", {D, D}, model_id);
  m->scorer_w1_ = w.data_f32("scorer.1.weight");
  m->scorer_b1_ = w.has("scorer.1.bias") ? w.data_f32("scorer.1.bias") : nullptr;
  require_shape(w, "scorer.3.weight", {1, D}, model_id);
  m->scorer_w2_ = w.data_f32("scorer.3.weight");
  m->scorer_b2_ = w.has("scorer.3.bias") ? w.data_f32("scorer.3.bias") : nullptr;
  require_shape(w, "act_head.0.weight", {256, D + 4}, model_id);
  m->act_w1_ = w.data_f32("act_head.0.weight");
  m->act_b1_ = w.data_f32("act_head.0.bias");
  require_shape(w, "act_head.2.weight", {c.n_act, 256}, model_id);
  m->act_w2_ = w.data_f32("act_head.2.weight");
  m->act_b2_ = w.data_f32("act_head.2.bias");

  // --- RoPE tables (fp64 compute like torch outer(pos, inv_freq) then float)
  const int64_t half = m->Dh_ / 2;
  const int64_t L = c.max_position_embeddings;
  auto build_table = [&](double theta, std::shared_ptr<float[]>& cos_t,
                         std::shared_ptr<float[]>& sin_t) {
    cos_t.reset(new float[static_cast<size_t>(L * half)]);
    sin_t.reset(new float[static_cast<size_t>(L * half)]);
    for (int64_t pos = 0; pos < L; ++pos) {
      for (int64_t d = 0; d < half; ++d) {
        double inv = 1.0 / std::pow(theta, 2.0 * static_cast<double>(d) / m->Dh_);
        double fr = static_cast<double>(pos) * inv;
        cos_t[pos * half + d] = static_cast<float>(std::cos(fr));
        sin_t[pos * half + d] = static_cast<float>(std::sin(fr));
      }
    }
  };
  build_table(c.rope_theta_full, m->rope_cos_full_, m->rope_sin_full_);
  build_table(c.rope_theta_sliding, m->rope_cos_slide_, m->rope_sin_slide_);

  // optional encoder-output LoRA adapter (rank r): A [r, D], B [D, r]
  if (w.has("lora_A.weight") && w.has("lora_B.weight")) {
    // shapes: A [r, D], B [D, r]; r = A.numel / D
    auto ash = w.shape_of("lora_A.weight");
    auto bsh = w.shape_of("lora_B.weight");
    if (ash.size() != 2 || bsh.size() != 2 || ash[0] != bsh[1] ||
        ash[1] != static_cast<int64_t>(D) || bsh[0] != static_cast<int64_t>(D))
      throw std::runtime_error("Model 'lora_A.weight'/'lora_B.weight' must be [r,D] and [D,r]");
    m->lora_r_ = static_cast<int>(ash[0]);
    m->lora_a_ = w.data_f32("lora_A.weight");
    m->lora_b_ = w.data_f32("lora_B.weight");
  }
  return m;
}

// ---- forward ---------------------------------------------------------------

namespace {
struct MatView {  // non-owning fp32 [M, K] view for gemm calls
  Tensor t;
  static MatView of(const float* p, int64_t M, int64_t K) {
    MatView v;
    // Wrap a copy-free view: Tensor::wrap shares the pointer.
    auto sp = std::shared_ptr<float[]>(const_cast<float*>(p), [](float*) {});
    v.t = Tensor::wrap({M, K}, sp);
    return v;
  }
};
}  // namespace

std::vector<std::vector<std::vector<float>>> DecisionModel::encode(
    const std::vector<std::vector<int64_t>>& input_ids,
    const std::vector<std::vector<int64_t>>& attention_mask) const {
  const int64_t B = static_cast<int64_t>(input_ids.size());
  const int64_t L = B ? static_cast<int64_t>(input_ids[0].size()) : 0;
  std::vector<std::vector<std::vector<float>>> out(static_cast<size_t>(B));
  if (!B) return out;

  std::vector<int64_t> lens(static_cast<size_t>(B), L);
  for (int64_t b = 0; b < B; ++b) {
    int64_t s = 0;
    for (int64_t v : attention_mask[static_cast<size_t>(b)]) s += v;
    lens[static_cast<size_t>(b)] = s;
  }

  // --- duplicate of the encoder half of forward(); kept separate so the
  // headless path never allocates the decision-head scratch.
  const int64_t M = B * L;
  Tensor X({M, D_});
  {
    std::vector<int64_t> flat_ids;
    flat_ids.reserve(static_cast<size_t>(M));
    for (const auto& row : input_ids) flat_ids.insert(flat_ids.end(), row.begin(), row.end());
    Tensor emb = embedding(Tensor::wrap({cfg_.vocab_size, D_}, emb_w_), flat_ids.data(), M);
    X = layer_norm(emb, emb_norm_.get(), nullptr, cfg_.norm_eps);
  }
  const int64_t HxDh = static_cast<int64_t>(H_) * Dh_;
  Tensor Y;
  Tensor qkv({M, D_ * 3});
  Tensor O({M, D_});
  for (size_t li = 0; li < layers_.size(); ++li) {
    const LayerWeights& ly = layers_[li];
    Y = ly.has_attn_norm ? layer_norm(X, ly.attn_norm.d.get(), nullptr, cfg_.norm_eps) : X;
    qkv = gemm_nt(Y, MatView::of(ly.wqkv.d.get(), 3 * D_, D_).t, nullptr, 0);
    const float *ct, *st;
    if (ly.attention_type == "sliding_attention") {
      ct = rope_cos_slide_.get();
      st = rope_sin_slide_.get();
    } else {
      ct = rope_cos_full_.get();
      st = rope_sin_full_.get();
    }
    rope_inplace(qkv, ct, st, L, H_, Dh_);
    Tensor Q({B, H_, L, Dh_}), Kt({B, H_, L, Dh_}), V({B, H_, L, Dh_});
    for (int64_t b = 0; b < B; ++b)
      for (int64_t i = 0; i < L; ++i) {
        const float* row = qkv.data_ptr() + (b * L + i) * 3 * HxDh;
        for (int64_t h = 0; h < H_; ++h) {
          std::memcpy(Q.mutable_data() + ((b * H_ + h) * L + i) * Dh_, row + h * Dh_,
                      sizeof(float) * static_cast<size_t>(Dh_));
          std::memcpy(Kt.mutable_data() + ((b * H_ + h) * L + i) * Dh_,
                      row + HxDh + h * Dh_, sizeof(float) * static_cast<size_t>(Dh_));
          std::memcpy(V.mutable_data() + ((b * H_ + h) * L + i) * Dh_,
                      row + 2 * HxDh + h * Dh_, sizeof(float) * static_cast<size_t>(Dh_));
        }
      }
    Tensor At = attention(Q, Kt, V, lens, ly.window);
    for (int64_t b = 0; b < B; ++b)
      for (int64_t i = 0; i < L; ++i)
        for (int64_t h = 0; h < H_; ++h)
          std::memcpy(O.mutable_data() + (b * L + i) * HxDh + h * Dh_,
                      At.data_ptr() + ((b * H_ + h) * L + i) * Dh_,
                      sizeof(float) * static_cast<size_t>(Dh_));
    Y = gemm_nt(O, MatView::of(ly.wo.d.get(), D_, D_).t, nullptr, 0);
    add_(X, Y);
    Y = layer_norm(X, ly.mlp_norm.d.get(), nullptr, cfg_.norm_eps);
    Tensor G = gemm_geglu(Y, MatView::of(ly.wi.d.get(), 2 * F_, D_).t);
    Y = gemm_nt(G, MatView::of(ly.wo2.d.get(), D_, F_).t, nullptr, 0);
    add_(X, Y);
  }
  Tensor h = layer_norm(X, final_norm_.get(), nullptr, cfg_.norm_eps);
  if (lora_r_ > 0) {
    // h [M, D] += (h @ A^T [M,r]) @ B^T [r,D]
    Tensor z = gemm_nt(h, MatView::of(lora_a_.get(), lora_r_, D_).t, nullptr, 0);
    Tensor delta = gemm_nt(z, MatView::of(lora_b_.get(), D_, lora_r_).t, nullptr, 0);
    add_(h, delta);
  }
  for (int64_t b = 0; b < B; ++b) {
    out[static_cast<size_t>(b)].resize(static_cast<size_t>(L));
    for (int64_t i = 0; i < L; ++i) {
      const float* src = h.data_ptr() + (b * L + i) * D_;
      out[static_cast<size_t>(b)][static_cast<size_t>(i)].assign(src, src + D_);
    }
  }
  return out;
}

void DecisionModel::forward(const std::vector<std::vector<int64_t>>& input_ids,
                            const std::vector<std::vector<int64_t>>& attention_mask,
                            const std::vector<std::vector<int64_t>>& marker_pos,
                            const std::vector<std::vector<uint8_t>>& marker_mask,
                            const std::vector<int64_t>& qtype,
                            std::vector<std::vector<float>>& logits_out,
                            std::vector<std::vector<float>>& act_out) const {
  const int64_t B = static_cast<int64_t>(input_ids.size());
  if (B == 0) return;
  const int64_t L = static_cast<int64_t>(input_ids[0].size());
  const int64_t K = static_cast<int64_t>(marker_pos[0].size());
  const int64_t M = B * L;
  const int D = D_;
  const float eps = cfg_.norm_eps;

  // valid lengths per row (for attention)
  std::vector<int64_t> lens(static_cast<size_t>(B), L);
  for (int64_t b = 0; b < B; ++b) {
    int64_t s = 0;
    for (int64_t v : attention_mask[static_cast<size_t>(b)]) s += v;
    lens[static_cast<size_t>(b)] = s;
  }

  // ---- encoder: embeddings + norm (fp32 residual stream X)
  Tensor X({M, D});
  {
    std::vector<int64_t> flat_ids;
    flat_ids.reserve(static_cast<size_t>(M));
    for (const auto& row : input_ids) flat_ids.insert(flat_ids.end(), row.begin(), row.end());
    Tensor emb = embedding(Tensor::wrap({cfg_.vocab_size, D}, emb_w_), flat_ids.data(), M);
    X = layer_norm(emb, emb_norm_.get(), nullptr, eps);
  }

  const int64_t HxDh = static_cast<int64_t>(H_) * Dh_;
  Tensor Y;             // branch output buffer
  Tensor qkv({M, 3 * D});
  Tensor O({M, D});

  for (size_t li = 0; li < layers_.size(); ++li) {
    const LayerWeights& ly = layers_[li];
    // Y = attn_norm(X) (layer 0: X itself — attn_norm is Identity)
    if (ly.has_attn_norm)
      Y = layer_norm(X, ly.attn_norm.d.get(), nullptr, eps);
    else
      Y = X;
    // qkv = Y @ Wqkv^T (no bias)
    qkv = gemm_nt(Y, MatView::of(ly.wqkv.d.get(), 3 * D, D).t, nullptr, 0);
    // rope on q,k packed (q|k|v)(head)(dim)
    const float *ct, *st;
    if (ly.attention_type == "sliding_attention") {
      ct = rope_cos_slide_.get();
      st = rope_sin_slide_.get();
    } else {
      ct = rope_cos_full_.get();
      st = rope_sin_full_.get();
    }
    rope_inplace(qkv, ct, st, L, H_, Dh_);
    // attention: reshape views [B, H, L, Dh] slices of qkv — qkv rows already
    // have (q|k|v) per row, so we permute to separate q,k,v [B,H,L,Dh] blocks.
    Tensor Q({B, H_, L, Dh_}), Kt({B, H_, L, Dh_}), V({B, H_, L, Dh_});
    for (int64_t b = 0; b < B; ++b)
      for (int64_t i = 0; i < L; ++i) {
        const float* row = qkv.data_ptr() + (b * L + i) * 3 * HxDh;
        for (int64_t h = 0; h < H_; ++h) {
          std::memcpy(Q.mutable_data() + ((b * H_ + h) * L + i) * Dh_, row + h * Dh_,
                      sizeof(float) * static_cast<size_t>(Dh_));
          std::memcpy(Kt.mutable_data() + ((b * H_ + h) * L + i) * Dh_,
                      row + HxDh + h * Dh_, sizeof(float) * static_cast<size_t>(Dh_));
          std::memcpy(V.mutable_data() + ((b * H_ + h) * L + i) * Dh_,
                      row + 2 * HxDh + h * Dh_, sizeof(float) * static_cast<size_t>(Dh_));
        }
      }
    Tensor At = attention(Q, Kt, V, lens, ly.window);  // [B, H, L, Dh]
    for (int64_t b = 0; b < B; ++b)
      for (int64_t i = 0; i < L; ++i)
        for (int64_t h = 0; h < H_; ++h)
          std::memcpy(O.mutable_data() + (b * L + i) * HxDh + h * Dh_,
                      At.data_ptr() + ((b * H_ + h) * L + i) * Dh_,
                      sizeof(float) * static_cast<size_t>(Dh_));
    // Y = attn out projection
    Y = gemm_nt(O, MatView::of(ly.wo.d.get(), D, D).t, nullptr, 0);
    // X += Y ; Y = mlp_norm(X)
    add_(X, Y);
    Y = layer_norm(X, ly.mlp_norm.d.get(), nullptr, eps);
    // GEGLU + down proj
    Tensor G = gemm_geglu(Y, MatView::of(ly.wi.d.get(), 2 * F_, D).t);
    Y = gemm_nt(G, MatView::of(ly.wo2.d.get(), D, F_).t, nullptr, 0);
    // X += Y ; Y = next norm(X)  — we just add; the next layer's norm handles it.
    add_(X, Y);
  }
  // final norm
  Tensor h = layer_norm(X, final_norm_.get(), nullptr, eps);  // [M, D]

  if (lora_r_ > 0) {
    // h [M, D] += (h @ A^T [M,r]) @ B^T [r,D]
    Tensor z = gemm_nt(h, MatView::of(lora_a_.get(), lora_r_, D).t, nullptr, 0);
    Tensor delta = gemm_nt(z, MatView::of(lora_b_.get(), D, lora_r_).t, nullptr, 0);
    add_(h, delta);
  }

  // h = h + type_emb[qtype]
  for (int64_t b = 0; b < B; ++b) {
    const float* te = type_emb_.get() + qtype[static_cast<size_t>(b)] * D;
    for (int64_t i = 0; i < L; ++i) {
      float* hr = h.mutable_data() + (b * L + i) * D;
      for (int j = 0; j < D; ++j) hr[j] += te[j];
    }
  }

  // ---- head: N x pre-norm TransformerEncoderLayer (relu ffn, biases, eps 1e-5)
  {
    Tensor Xf = h;  // fp32 in-place residual stream [M, D]
    Tensor ln_out;
    for (const HeadLayer& hl : head_) {
      // attention branch
      ln_out = layer_norm(Xf, hl.n1w.get(), hl.n1b.get(), 1e-5f);
      Tensor qkv2 = gemm_nt(ln_out, MatView::of(hl.in_w.get(), 3 * D, D).t,
                            hl.in_b.get(), 0);
      // reshape/per-mute into [B, H, L, Dh]
      Tensor Q({B, H_, L, Dh_}), Kt({B, H_, L, Dh_}), V({B, H_, L, Dh_});
      for (int64_t b = 0; b < B; ++b)
        for (int64_t i = 0; i < L; ++i) {
          const float* row = qkv2.data_ptr() + (b * L + i) * 3 * HxDh;
          for (int64_t hh = 0; hh < H_; ++hh) {
            std::memcpy(Q.mutable_data() + ((b * H_ + hh) * L + i) * Dh_, row + hh * Dh_,
                        sizeof(float) * static_cast<size_t>(Dh_));
            std::memcpy(Kt.mutable_data() + ((b * H_ + hh) * L + i) * Dh_,
                        row + HxDh + hh * Dh_, sizeof(float) * static_cast<size_t>(Dh_));
            std::memcpy(V.mutable_data() + ((b * H_ + hh) * L + i) * Dh_,
                        row + 2 * HxDh + hh * Dh_, sizeof(float) * static_cast<size_t>(Dh_));
          }
        }
      // nn.MultiheadAttention with src_key_padding_mask=pad -> mask keys where pad
      Tensor At = attention(Q, Kt, V, lens, 0);
      Tensor AO({M, D});
      for (int64_t b = 0; b < B; ++b)
        for (int64_t i = 0; i < L; ++i)
          for (int64_t hh = 0; hh < H_; ++hh)
            std::memcpy(AO.mutable_data() + (b * L + i) * HxDh + hh * Dh_,
                        At.data_ptr() + ((b * H_ + hh) * L + i) * Dh_,
                        sizeof(float) * static_cast<size_t>(Dh_));
      Tensor proj = gemm_nt(AO, MatView::of(hl.out_w.get(), D, D).t, hl.out_b.get(), 0);
      add_(Xf, proj);
      // FFN branch
      ln_out = layer_norm(Xf, hl.n2w.get(), hl.n2b.get(), 1e-5f);
      Tensor f1 = gemm_nt(ln_out, MatView::of(hl.l1w.get(), 4 * D, D).t, hl.l1b.get(), 2);
      Tensor f2 = gemm_nt(f1, MatView::of(hl.l2w.get(), D, 4 * D).t, hl.l2b.get(), 0);
      add_(Xf, f2);
    }
    h = std::move(Xf);
  }

  // ---- scorer: gather marker rows, then LN + Linear + GELU + Linear(1)
  {
    std::vector<int64_t> flat_pos;
    flat_pos.reserve(static_cast<size_t>(B * K));
    for (const auto& row : marker_pos)
      flat_pos.insert(flat_pos.end(), row.begin(), row.end());
    // gather: h here is [M, D] flattened from [B, L, D] — index by b*L + p
    Tensor m({B * K, D});
    for (int64_t b = 0; b < B; ++b)
      for (int64_t kk = 0; kk < K; ++kk) {
        int64_t p = flat_pos[static_cast<size_t>(b * K + kk)];
        if (p < 0) p = 0;
        if (p >= L) p = L - 1;
        std::memcpy(m.mutable_data() + (b * K + kk) * D,
                    h.data_ptr() + (b * L + p) * D, sizeof(float) * static_cast<size_t>(D));
      }
    Tensor s = layer_norm(m, scorer_ln_w_.get(), scorer_ln_b_.get(), eps);
    s = gemm_nt(s, MatView::of(scorer_w1_.get(), D, D).t, scorer_b1_.get(), 1);   // GELU
    s = gemm_nt(s, MatView::of(scorer_w2_.get(), 1, D).t, scorer_b2_.get(), 0);   // [BK, 1]
    logits_out.assign(static_cast<size_t>(B), std::vector<float>(static_cast<size_t>(K)));
    for (int64_t b = 0; b < B; ++b)
      for (int64_t kk = 0; kk < K; ++kk) {
        float lg = s.data_ptr()[b * K + kk];
        if (!marker_mask[static_cast<size_t>(b)][static_cast<size_t>(kk)]) lg = -1e4f;
        logits_out[static_cast<size_t>(b)][static_cast<size_t>(kk)] = lg;
      }
  }

  // ---- act head: [pooled CLS row, top1, top1-top2, ent, k/255] -> logits -> softmax
  act_out.assign(static_cast<size_t>(B), std::vector<float>(static_cast<size_t>(cfg_.n_act)));
  for (int64_t b = 0; b < B; ++b) {
    // softmax over logits row (after -1e4 masking)
    const std::vector<float>& lg = logits_out[static_cast<size_t>(b)];
    float mx = *std::max_element(lg.begin(), lg.end());
    std::vector<double> p(lg.size());
    double s = 0;
    for (size_t j = 0; j < lg.size(); ++j) { p[j] = std::exp(lg[j] - mx); s += p[j]; }
    for (double& v : p) v /= s;
    double top1 = 0, top2v = 0;
    for (double v : p) {
      if (v > top1) { top2v = top1; top1 = v; }
      else if (v > top2v) top2v = v;
    }
    double ksum = 0;
    for (int64_t kk = 0; kk < K; ++kk) ksum += marker_mask[static_cast<size_t>(b)][static_cast<size_t>(kk)];
    double kf = std::max<double>(2.0, ksum);
    double ent = 0;
    for (double v : p) ent -= v * std::log(std::max(v, 1e-9));
    ent /= std::log(kf);

    const float* pooled = h.data_ptr() + (b * L) * D;   // position 0 = [CLS]
    std::vector<float> feats(static_cast<size_t>(D + 4));
    std::memcpy(feats.data(), pooled, sizeof(float) * static_cast<size_t>(D));
    feats[static_cast<size_t>(D) + 0] = static_cast<float>(top1);
    feats[static_cast<size_t>(D) + 1] = static_cast<float>(top1 - top2v);
    feats[static_cast<size_t>(D) + 2] = static_cast<float>(ent);
    feats[static_cast<size_t>(D) + 3] = static_cast<float>(kf / 255.0);

    Tensor ft = Tensor::wrap({1, D + 4},
        std::shared_ptr<float[]>(feats.data(), [](float*) {}));
    Tensor a1 = gemm_nt(ft, MatView::of(act_w1_.get(), 256, D + 4).t, act_b1_.get(), 1);
    Tensor a2 = gemm_nt(a1, MatView::of(act_w2_.get(), cfg_.n_act, 256).t, act_b2_.get(), 0);
    // softmax over n_act
    float amx = a2.data_ptr()[0];
    for (int j = 1; j < cfg_.n_act; ++j) amx = std::max(amx, a2.data_ptr()[j]);
    double as = 0;
    float* a2d = a2.mutable_data();
    for (int j = 0; j < cfg_.n_act; ++j) {
      a2d[j] = std::exp(a2d[j] - amx);
      as += a2d[j];
    }
    for (int j = 0; j < cfg_.n_act; ++j)
      act_out[static_cast<size_t>(b)][static_cast<size_t>(j)] =
          static_cast<float>(a2.data_ptr()[j] / as);
  }
}

}  // namespace snapjudge
