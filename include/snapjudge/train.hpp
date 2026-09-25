#pragma once
// snapjudge train.hpp: the training engine. Owns a training copy of the
// checkpoint (fp32 weights), the AdamW optimizer state, and the training
// forward/backward pass. Verified against numeric gradients in
// tests/test_train.cpp.

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace snapjudge {

using nlohmann::ordered_json;
class Tokenizer;
class SafeTensors;

// One training example: state + the typed questions + gold target
// distributions (from the typed-decisions corpus or any compatible JSONL).
struct TrainRow {
  ordered_json state;
  std::vector<std::string> qids;
  std::vector<ordered_json> qdefs;                 // internal-normalized defs
  std::vector<std::vector<double>> targets;        // per-question gold probs
  std::string workflow;
};

// Load JSONL rows as produced by snapjudge/python/export_typed_decisions.py:
// {"state": {...}, "questions": {...}, "gold": {...}, "workflow": "..."}
std::vector<TrainRow> load_train_rows(const std::string& jsonl_path);

// Post-hoc temperature fitting (snapjudge/src/fit_temperature.cpp): fits one
// scalar per (qtype, bucket) and per qtype on eval logits/targets.
std::pair<std::vector<double>, std::map<std::string, double>> fit_temperatures(
    const std::vector<std::vector<float>>& logits,
    const std::vector<std::vector<double>>& targets,
    const std::vector<int>& qtypes);

// ---- training model -----------------------------------------------------------
class TrainModel {
 public:
  // Load from a checkpoint dir (rl_agent_config.json + encoder/config.json +
  // model.safetensors + tokenizer/). Weights are read into fp32 (trainable).
  explicit TrainModel(const std::string& ckpt_dir);
  ~TrainModel();

  // One micro-step for a state's worth of items (each question = one row).
  // Runs forward + backward, accumulates grads. Returns the mean proper-score
  // reward (higher = better; loss = -reward) and accuracy of argmax predictions.
  struct StepStats { double loss; double acc; int rows; };
  StepStats step(const TrainRow& row, bool accumulate_only);

  void optimizer_step(double lr);          // AdamW over accumulated grads
  void zero_grad();
  int64_t param_count() const { return n_params_; }

  // Configure the proper-score loss weights + label smoothing before training.
  // w_nll/w_sph/w_rps scale the log / spherical / ranked-probability terms;
  // label_smoothing ∈ [0,1) mixes the gold distribution toward uniform.
  void set_loss(double w_nll, double w_sph, double w_rps, double label_smoothing);

  // Enable/disable the rank-r encoder-output LoRA adapter. r=0 disables it
  // (trainable set = head + type_emb + scorer only, the frozen-encoder default).
  void set_lora_r(int r);

  // Eval forward over rows through the trained head: per-train-question logits,
  // target and qtype. Used by temperature fitting.
  void eval_logits(const std::vector<TrainRow>& rows,
                   std::vector<std::vector<float>>* logits_out,
                   std::vector<std::vector<double>>* targets_out,
                   std::vector<int>* qtypes_out);

  // Trainable parameter names in deterministic order.
  const std::vector<std::string>& training_param_names() const;

  // Save the current weights as a snapjudge checkpoint at out_dir (fp16 weights,
  // fp32 temperature, config + tokenizer + encoder copied from base).
  void save_checkpoint(const std::string& out_dir, const std::vector<double>& temperature,
                       const std::map<std::string, double>& temp_by_options,
                       const ordered_json& train_meta);

  const nlohmann::json& cfg() const { return cfg_; }
  Tokenizer* tokenizer() { return tok_.get(); }
  const std::string& base_dir() const { return base_dir_; }

  // Numeric-gradient access for tests: parameter read/write by name+index.
  std::vector<std::string> param_names() const;
  int64_t param_numel(const std::string& name) const;
  float param_get(const std::string& name, int64_t i) const;
  void param_set(const std::string& name, int64_t i, float v);
  std::vector<float> grad_of(const std::string& name);  // zero's + recomputes

  // Full-graph double loss for gradient checking.
  double forward_loss_double(const TrainRow& row);

 public:
  // Impl is exposed for the file-local forward/backward helpers in
  // src/train_model.cpp; treat it as internal.
  struct Impl;
 private:
  std::unique_ptr<Impl> impl_;
  nlohmann::json cfg_;
  std::string base_dir_;
  std::shared_ptr<Tokenizer> tok_;
  int64_t n_params_ = 0;
};

}  // namespace snapjudge
