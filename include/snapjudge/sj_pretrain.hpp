#pragma once
// snapjudge sj_pretrain.hpp: masked-language-model pretraining for the native
// snapjudge encoder (SjModel).
//
// Trains the FULL encoder — word/position embeddings, every encoder layer, and
// the final norm — on a masked-language-modeling objective (mask ~15% of tokens,
// predict them). This is what gives the own model real weights before the head
// is fine-tuned on decision data. An optional output head (tied or separate) maps
// the encoder hidden states to vocabulary logits.
//
// Single-sequence (B=1) forward/backward for a correct, verifiable first path;
// batching is a later optimization. All fp32, AdamW, gradient-checked by
// tests/test_sj_pretrain.cpp.

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace snapjudge {

class SjPretrain {
 public:
  // Load a native sj checkpoint (sj_config.json + model.safetensors) as the
  // initialization. If `init_random` is true and no checkpoint exists, the
  // caller may pass an empty dir; otherwise the checkpoint is required.
  explicit SjPretrain(const std::string& ckpt_dir);
  ~SjPretrain();

  // One training step over a single token sequence. `ids` are token ids (the
  // special ids from the config). Masks ~15% of non-special positions, runs the
  // encoder + MLM head, computes cross-entropy at masked positions, and (when
  // `backward`) accumulates gradients. Returns the mean loss (and -1.0 if no
  // position was maskable).
  double step(const std::vector<int64_t>& ids, bool backward = true);

  void zero_grad();
  void optimizer_step(double lr);

  // numeric-gradient access for tests
  std::vector<std::string> param_names() const;
  int64_t param_numel(const std::string& name) const;
  float param_get(const std::string& name, int64_t i) const;
  void param_set(const std::string& name, int64_t i, float v);
  float grad_at(const std::string& name, int64_t i) const;
  double loss_double(const std::vector<int64_t>& ids);

  // Save the current encoder weights as a native sj checkpoint (embeddings +
  // encoder + final norm), reusing the head/scorer from the base if present.
  void save_checkpoint(const std::string& out_dir);

 public:
  // Impl is exposed for the file-local forward/backward helpers in
  // src/sj_pretrain.cpp; treat it as internal.
  struct Impl;

 private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace snapjudge
