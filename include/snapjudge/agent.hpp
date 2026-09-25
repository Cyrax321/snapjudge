#pragma once
// snapjudge agent.hpp: high-level System-1 decision runtime, port of
// agent.py::Agent.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace snapjudge {

class Tokenizer;
class DecisionModel;

class Agent {
 public:
  // model_id_or_path: local directory or HF repo id (resolved via hub.cpp).
  // subfolder picks one checkpoint from a bundled repo. device: "cpu"|"cuda".
  explicit Agent(const std::string& model_id_or_path = "snapjudge/snapjudge",
                 const std::string& device = "",
                 const std::string& token = "",
                 const std::string& subfolder = "");

  ~Agent();
  Agent(Agent&&) noexcept;
  Agent& operator=(Agent&&) noexcept;
  Agent(const Agent&) = delete;
  Agent& operator=(const Agent&) = delete;

  // system_one / predict: same semantics + output shape as Python.
  nlohmann::ordered_json system_one(const nlohmann::ordered_json& state,
                                    const nlohmann::ordered_json& questions) const;
  nlohmann::ordered_json predict(const nlohmann::ordered_json& state,
                                 const nlohmann::ordered_json& questions) const {
    return system_one(state, questions);
  }
  // Coarse-to-fine choice: shortlist high-cardinality choice questions (option
  // count > `threshold`) to `k` candidates by encoder-cosine before the single
  // forward pass, so 50+ option questions don't starve their token budget.
  // k=0/negative disables shortlisting (single-pass fallback). `threshold`
  // controls the auto-trigger; 0 shortlists every choice question.
  nlohmann::ordered_json system_one_shortlist(
      const nlohmann::ordered_json& state, const nlohmann::ordered_json& questions,
      int k = 20, int threshold = 20) const;
  // Throughput path: same questions over many states.
  std::vector<nlohmann::ordered_json> predict_batch(
      const std::vector<nlohmann::ordered_json>& states,
      const nlohmann::ordered_json& questions, int batch_size = 0) const;

  // Mean-pooled encoder embeddings (shortlist.py embed_fn_from_agent).
  std::vector<std::vector<float>> embed(const std::vector<std::string>& texts,
                                        int max_length = 512, int batch_size = 32) const;

  const std::string& device() const { return device_; }

  const nlohmann::json& cfg() const { return cfg_; }

  // Swap in the CUDA fused-kernel fast path (snapjudge agent.accelerate).
  // No-op on non-CUDA builds: returns false with a warning on stderr.
  bool accelerate();
  bool accelerated() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string device_ = "cpu";
  std::string tokenizer_dir_;
  nlohmann::json cfg_;              // parsed rl_agent_config.json (raw values kept)
  std::vector<double> temperature_;
  std::unordered_map<std::string, double> temperature_by_options_;

  nlohmann::ordered_json run_batch(const std::vector<std::vector<nlohmann::ordered_json>>& items_groups,
                                   const std::vector<std::string>& ids,
                                   const nlohmann::ordered_json& internal) const;
};

}  // namespace snapjudge
