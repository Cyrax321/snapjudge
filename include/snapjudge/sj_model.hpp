#pragma once
// snapjudge sj_model.hpp: the native snapjudge encoder + typed decision head.
//
// This is snapjudge's own model, deliberately a different architecture from the
// ModernBERT/mmBERT backbones the decision-model field standardised on:
//
//   * learned absolute position embeddings (not RoPE)
//   * GELU activations with bias terms (not GeGLU, not bias-free)
//   * full bidirectional attention (no sliding window)
//   * separate Q/K/V/O projections (no fused Wqkv)
//
// Weights load from a single sj_config.json + model.safetensors in our own
// schema (tensor names prefixed sj.*). The forward pass reuses the shared fp32
// tensor primitives in tensor.hpp.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace snapjudge {

class SafeTensors;
struct Tensor;

struct SjConfig {
  // model shape
  int vocab_size = 0;
  int hidden_size = 0;
  int num_layers = 0;
  int num_heads = 0;
  int intermediate_size = 0;
  int max_positions = 0;
  float norm_eps = 1e-5f;
  int head_layers = 1;
  int head_intermediate = 0;   // ffn width of the head transformer (0 = 4*D)
  int max_len = 0;             // sequence length cap
  int head_max_len = 0;        // option budget (markers + instructions)
  int n_types = 3;             // choice / score / noul

  // special ids (must match the tokenizer's added_tokens)
  int32_t cls_id = 0;
  int32_t sep_id = 0;
  int32_t mask_id = 0;
  int32_t pad_id = 0;
};

class SjModel {
 public:
  static std::unique_ptr<SjModel> load(const nlohmann::json& cfg_json,
                                       const SafeTensors& weights,
                                       const std::string& model_id);

  // Encoder-only: hidden states [B, L, D] after final_norm (no head).
  std::vector<std::vector<std::vector<float>>> encode(
      const std::vector<std::vector<int64_t>>& input_ids,
      const std::vector<std::vector<int64_t>>& attention_mask) const;

  // Full forward: encoder + typed decision head. `marker_pos` [B, K] gives the
  // positions of each option's [MASK] token; `marker_mask` [B, K] marks valid
  // options; `qtype` [B] is the question type (0 choice / 1 score / 2 noul).
  // Writes `logits` [B, K] and `act` [B] (act head, n_act = 2).
  void forward(const std::vector<std::vector<int64_t>>& input_ids,
               const std::vector<std::vector<int64_t>>& attention_mask,
               const std::vector<std::vector<int64_t>>& marker_pos,
               const std::vector<std::vector<uint8_t>>& marker_mask,
               const std::vector<int64_t>& qtype,
               std::vector<std::vector<float>>& logits,
               std::vector<std::vector<float>>& act) const;

  const SjConfig& cfg() const { return cfg_; }
  int D() const { return cfg_.hidden_size; }
  int H() const { return cfg_.num_heads; }
  int Dh() const { return cfg_.hidden_size / cfg_.num_heads; }

 private:
  SjConfig cfg_;

  // embeddings
  std::shared_ptr<float[]> word_emb_;    // [V, D]
  std::shared_ptr<float[]> pos_emb_;     // [max_positions, D]
  std::shared_ptr<float[]> emb_norm_w_;  // [D]
  std::shared_ptr<float[]> emb_norm_b_;  // [D]

  struct EncLayer {
    // attention: pre-norm -> q/k/v/o with biases -> residual
    std::shared_ptr<float[]> an_w, an_b;   // [D]
    std::shared_ptr<float[]> q_w, q_b;     // [D, D], [D]
    std::shared_ptr<float[]> k_w, k_b;
    std::shared_ptr<float[]> v_w, v_b;
    std::shared_ptr<float[]> o_w, o_b;
    // ffn: pre-norm -> w1 (D->4D) GELU -> w2 (4D->D) with biases -> residual
    std::shared_ptr<float[]> fn_w, fn_b;   // [D]
    std::shared_ptr<float[]> w1_w, w1_b;   // [4D, D], [4D]
    std::shared_ptr<float[]> w2_w, w2_b;   // [D, 4D], [D]
  };
  std::vector<EncLayer> enc_;
  std::shared_ptr<float[]> final_norm_w_;  // [D]
  std::shared_ptr<float[]> final_norm_b_;  // [D]

  // typed decision head
  std::shared_ptr<float[]> type_emb_;      // [n_types, D]
  struct HeadLayer {
    std::shared_ptr<float[]> n1_w, n1_b, n2_w, n2_b;      // [D]
    std::shared_ptr<float[]> q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b;  // [D,D]
    std::shared_ptr<float[]> w1_w, w1_b, w2_w, w2_b;      // ffn
  };
  std::vector<HeadLayer> head_;
  // scorer: LN -> Linear(D,D) GELU -> Linear(D,1)
  std::shared_ptr<float[]> scorer_n_w_, scorer_n_b_;   // [D]
  std::shared_ptr<float[]> scorer_w1_, scorer_b1_;     // [D, D], [D]
  std::shared_ptr<float[]> scorer_w2_, scorer_b2_;     // [1, D], [1]

  // Encoder forward shared by encode() and forward(); returns final-norm hidden
  // states [B*L, D].
  struct Tensor run_encoder(const std::vector<std::vector<int64_t>>& input_ids,
                            const std::vector<std::vector<int64_t>>& attention_mask) const;

  friend class SjTrainModel;
};

}  // namespace snapjudge
