#pragma once
// snapjudge common.hpp: sequence construction + answer math, port of
// common.py. Question/state shapes use nlohmann::json to preserve
// Python dict/list/str/number semantics and insertion order is handled by
// ordered_json at API boundaries.

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace snapjudge {

// Python dicts preserve insertion order; nlohmann::json sorts keys. Use
// ordered_json at all API boundaries so probabilities/legends/results keep
// Python order exactly.
using Json = nlohmann::ordered_json;
using OrderedJson = nlohmann::ordered_json;

extern const std::unordered_map<std::string, int> QTYPES;

inline const double TEMP_MIN = 0.5, TEMP_MAX = 5.0;

// json.dumps(v, ensure_ascii=False) replica.
std::string serialize_state(const Json& state);
// render_criterion from common.py (strings pass through, else compact JSON).
std::string render_criterion(const Json& v);

struct InternalQ {
  std::string t;                      // "choice" | "score" | "noul"
  std::string ins;                    // instructions as text
  Json crit;                          // object or array (or empty object for noul)
  bool has_labels = false;
  Json labels;                        // {"false": "...", "true": "..."}
};

// render_options(q) — option texts in label-index order.
std::vector<std::string> render_options(const InternalQ& q);

class Tokenizer;
// build_sequence from common.py: returns (ids, markers).
std::pair<std::vector<int32_t>, std::vector<int64_t>> build_sequence(
    const Tokenizer& tok, const Json& state, const InternalQ& q,
    int64_t max_len, int64_t head_max_len, bool truncate_left = false);

double clamp_temperature(const Json& t);   // NaN/inf/type errors -> 1.0
double clamp_temperature(double t);

std::string temp_bucket(int qtype, int k);

// confidence_from_probs: 1 - H(p)/log(k), clipped [0,1].
double confidence_from_probs(const double* p, int k);

// round to 4 decimals like Python round()
double round4(double v);

// ece_score from common.py
double ece_score(const std::vector<double>& conf, const std::vector<double>& correct,
                 int bins = 15);

// proper_reward (inference-portable version; fp64 math over row-probability inputs)
// q: [N, K] reported distributions, target: [N, K], qtype: [N], mask: [N, K]
std::vector<double> proper_reward(const std::vector<std::vector<double>>& q,
                                  const std::vector<std::vector<double>>& target,
                                  const std::vector<int64_t>& qtype,
                                  const std::vector<std::vector<double>>& mask,
                                  double w_sph = 0.5, double w_rps = 1.0,
                                  double log_floor = -9.21);

// td_lambda_targets (common.py): TD(lambda) targets for multi-turn trajectories.
// batch: target_flat [N,2], ep_group [N] (-1 = no group), ep_step [N].
std::vector<double> td_lambda_targets(const std::vector<double>& p_true,
                                      const std::vector<double>& target_flat,
                                      const std::vector<int64_t>& ep_group,
                                      const std::vector<int64_t>& ep_step,
                                      double lam = 1.0);

// Internal normalization from question dict to InternalQ (agent.py::_to_internal)
InternalQ to_internal(const Json& qdef);

// Question shape validation (agent.py::_check_question); throws std::invalid_argument
void check_question(const std::string& qid, const Json& qdef);

}  // namespace snapjudge
