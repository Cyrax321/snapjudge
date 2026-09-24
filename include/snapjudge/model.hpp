#pragma once
// snapjudge model.hpp: DecisionModel — ModernBERT/mmBERT encoder + typed
// decision head, port of common.py::DecisionModel + build_model.
// All compute fp32 (CPU parity target: python device="cpu").

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace snapjudge {

struct ModelConfig {
  // from rl_agent_config.json
  std::string encoder;         // e.g. "answerdotai/ModernBERT-large"
  int head_layers = 2;
  int max_len = 512;
  int head_max_len = 192;
  int n_act = 2;               // len(act_costs) + 1

  // from encoder/config.json
  int hidden_size = 0;
  int num_hidden_layers = 0;
  int num_attention_heads = 0;
  int intermediate_size = 0;
  int max_position_embeddings = 0;
  int vocab_size = 0;
  float norm_eps = 1e-5f;
  bool norm_bias = false;
  int local_attention = 0;                    // window = local_attention / 2
  std::vector<std::string> layer_types;       // "full_attention" | "sliding_attention"
  double rope_theta_full = 10000.0;
  double rope_theta_sliding = 10000.0;
};

struct LayerWeights {
  bool has_attn_norm = false;                 // layer 0: no attn_norm (Identity)
  // fp32 weights kept as shared arrays
  struct W { std::vector<int64_t> shape; std::shared_ptr<float[]> d; };
  W attn_norm, wqkv, wo, mlp_norm, wi, wo2;
  std::string attention_type;                 // layer_types[i]
  int window = 0;                             // sliding: local_attention//2, else 0
};

class DecisionModel {
 public:
  // Load from parsed configs + a safetensors handle. Strict checking mirrors
  // agent.py::_verify_compatibility: required prefixes, required cfg keys, and
  // shape mismatches are errors naming the offending tensors.
  static std::unique_ptr<DecisionModel> load(const nlohmann::json& rl_cfg,
                                             const nlohmann::json& enc_cfg,
                                             const class SafeTensors& weights,
                                             const std::string& model_id);

  // Encoder-only forward: hidden states [B, L, D] after final_norm, WITHOUT
  // the decision head. Feeds shortlist's embed_fn_from_agent (mean pooling in
  // shortlist.py takes embeddings from the encoder output).
  std::vector<std::vector<std::vector<float>>> encode(
      const std::vector<std::vector<int64_t>>& input_ids,
      const std::vector<std::vector<int64_t>>& attention_mask) const;

  // forward: port of DecisionModel.forward at inference (detach_encoder=False).
  void forward(const std::vector<std::vector<int64_t>>& input_ids,
               const std::vector<std::vector<int64_t>>& attention_mask,
               const std::vector<std::vector<int64_t>>& marker_pos,
               const std::vector<std::vector<uint8_t>>& marker_mask,
               const std::vector<int64_t>& qtype,
               std::vector<std::vector<float>>& logits_out,
               std::vector<std::vector<float>>& act_out) const;

  const ModelConfig& cfg() const { return cfg_; }

 private:
  ModelConfig cfg_;
  // encoder
  std::shared_ptr<float[]> emb_w_;       // [V, D]
  std::shared_ptr<float[]> emb_norm_;    // [D]
  std::vector<LayerWeights> layers_;
  std::shared_ptr<float[]> final_norm_;
  // rope tables per layer type, [max_len, Dh/2]
  std::shared_ptr<float[]> rope_cos_full_, rope_sin_full_;
  std::shared_ptr<float[]> rope_cos_slide_, rope_sin_slide_;
  // type embedding [3, D]
  std::shared_ptr<float[]> type_emb_;
  // head layers (nn.TransformerEncoderLayer, norm_first, relu ffn, biases)
  struct HeadLayer {
    std::shared_ptr<float[]> n1w, n1b, n2w, n2b;
    std::shared_ptr<float[]> in_w, in_b, out_w, out_b;  // in_w [3D, D]
    std::shared_ptr<float[]> l1w, l1b, l2w, l2b;        // l1 [4D, D]
  };
  std::vector<HeadLayer> head_;
  // scorer: LayerNorm + Linear(d,d) + GELU + Linear(d,1)
  std::shared_ptr<float[]> scorer_ln_w_;  // [D]
  std::shared_ptr<float[]> scorer_ln_b_;  // [D]
  std::shared_ptr<float[]> scorer_w1_;    // [D, D]
  std::shared_ptr<float[]> scorer_b1_;    // [D]
  std::shared_ptr<float[]> scorer_w2_;    // [1, D]
  std::shared_ptr<float[]> scorer_b2_;    // [1]
  // act_head: Linear(d+4, 256) + GELU + Linear(256, n_act)
  std::shared_ptr<float[]> act_w1_;      // [256, D+4]
  std::shared_ptr<float[]> act_b1_;      // [256]
  std::shared_ptr<float[]> act_w2_;      // [n_act, 256]
  std::shared_ptr<float[]> act_b2_;      // [n_act]

  int D_ = 0, H_ = 0, Dh_ = 0, F_ = 0;

  // The CUDA fast path (cuda/fast.cpp) reads these member fields directly.
  friend class FastSnapjudge;
};

}  // namespace snapjudge
