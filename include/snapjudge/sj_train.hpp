#pragma once
// snapjudge sj_train.hpp: trainer for the native snapjudge model (SjModel).
//
// Same pattern as the head-finetuning trainer: the encoder is FROZEN (runs
// forward-only through SjModel::encode); the trainable set is the typed
// decision head — type_emb, the head transformer layers, and the scorer. The
// loss is the offline proper-scoring rule (reward = w_nll·Σt·logq + w_sph·(t·q)/|q|
// minus an RPS term on score rows), i.e. the model is rewarded for honest,
// calibrated probabilities.
//
// This is the OWN training path: it operates on the `sj.*` checkpoint schema
// (sj_config.json + model.safetensors + tokenizer/), so an end-to-end own
// checkpoint can be trained with zero external tooling.

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "snapjudge/train.hpp"   // TrainRow + load_train_rows (shared JSONL schema)

namespace snapjudge {

using nlohmann::ordered_json;
class SjModel;
class Tokenizer;

class SjTrainModel {
 public:
  explicit SjTrainModel(const std::string& ckpt_dir);
  ~SjTrainModel();

  struct StepStats { double loss; double acc; int rows; };
  StepStats step(const TrainRow& row, bool accumulate_only);
  void optimizer_step(double lr);
  void zero_grad();
  int64_t param_count() const { return n_params_; }

  void set_loss(double w_nll, double w_sph, double w_rps, double label_smoothing);
  void set_loss_temperature(double T);

  // numeric-gradient access for tests
  std::vector<std::string> param_names() const;
  int64_t param_numel(const std::string& name) const;
  float param_get(const std::string& name, int64_t i) const;
  void param_set(const std::string& name, int64_t i, float v);
  float grad_at(const std::string& name, int64_t i) const;   // last accumulated grad
  double forward_loss_double(const TrainRow& row);

  // Save the current weights as a native sj checkpoint (sj_config.json +
  // model.safetensors), copying the tokenizer from the base dir.
  void save_checkpoint(const std::string& out_dir);

 public:
  // Impl is exposed for the file-local forward/backward helpers in
  // src/sj_train.cpp; treat it as internal.
  struct Impl;

 private:
  std::unique_ptr<Impl> impl_;
  int64_t n_params_ = 0;
};

}  // namespace snapjudge
