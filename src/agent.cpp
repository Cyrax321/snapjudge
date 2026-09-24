#include "snapjudge/agent.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "snapjudge/common.hpp"
#include "snapjudge/hub.hpp"
#include "snapjudge/model.hpp"
#include "snapjudge/safetensors.hpp"
#include "snapjudge/tokenizer.hpp"
#if defined(SNAPJUDGE_WITH_CUDA)
#include "snapjudge/fast.hpp"
#endif

using json = nlohmann::ordered_json;

namespace snapjudge {

namespace fs = std::filesystem;

struct Agent::Impl {
  std::shared_ptr<Tokenizer> tok;
  std::unique_ptr<DecisionModel> model;
#if defined(SNAPJUDGE_WITH_CUDA)
  std::unique_ptr<FastSnapjudge> fast;   // snapjudge::FastSnapjudge (cuda/fast.cpp)
#endif
};

Agent::~Agent() = default;
Agent::Agent(Agent&&) noexcept = default;
Agent& Agent::operator=(Agent&&) noexcept = default;

Agent::Agent(const std::string& model_id_or_path, const std::string& device,
             const std::string& token, const std::string& subfolder) {
  std::string model_dir = model_id_or_path;
  if (!fs::exists(model_dir)) {
    std::string prefix = subfolder.empty() ? "" : subfolder + "/";
    std::vector<std::string> patterns;
    for (const char* n :
         {"rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"}) {
      patterns.push_back(prefix + n);
    }
    model_dir = resolve_checkpoint(model_id_or_path, patterns,
                                   token.empty() ? (std::getenv("HF_TOKEN") ? std::getenv("HF_TOKEN") : "") : token);
  }
  if (!subfolder.empty()) {
    model_dir = (fs::path(model_dir) / subfolder).string();
    if (!fs::is_directory(model_dir))
      throw std::invalid_argument("Subfolder '" + subfolder + "' not found in '" +
                                  model_id_or_path + "'.");
  }

  fs::path dir = model_dir;
  fs::path cfg_path = dir / "rl_agent_config.json";
  if (!fs::exists(cfg_path))
    throw std::runtime_error("Incompatible model: '" + model_id_or_path +
                             "' does not contain 'rl_agent_config.json'. That file ships "
                             "with the weights of a snapjudge/snapjudge checkpoint.");
  std::ifstream(cfg_path) >> cfg_;

  fs::path weights_path = dir / "model.safetensors";
  if (!fs::exists(weights_path))
    throw std::runtime_error("Incompatible model: 'model.safetensors' not found in '" +
                             model_id_or_path + "'.");

  impl_ = std::make_unique<Impl>();
  impl_->tok = Tokenizer::cached_from_dir((dir / "tokenizer").string());
  tokenizer_dir_ = (dir / "tokenizer").string();

  json enc_cfg;
  {
    std::ifstream f(dir / "encoder" / "config.json");
    if (!f.good()) throw std::runtime_error("snapjudge: missing " +
                                            (dir / "encoder" / "config.json").string());
    f >> enc_cfg;
  }
  auto w = SafeTensors::load(weights_path.string());
  impl_->model = DecisionModel::load(cfg_, enc_cfg, w, model_id_or_path);

  device_ = device.empty() ? std::string(std::getenv("SNAPJUDGE_DEVICE") ?
                                         std::getenv("SNAPJUDGE_DEVICE") : "cpu")
                           : device;
#if !defined(SNAPJUDGE_WITH_CUDA)
  if (device_ == "cuda") {
    fprintf(stderr, "snapjudge: CUDA requested but this build has no CUDA support; "
                    "running on CPU.\n");
    device_ = "cpu";
  }
#endif

  // temperature handling (agent.py): keep raw in cfg_, clamp applied values.
  auto temps = cfg_.value("temperature", json::array({1.0, 1.0, 1.0}));
  temperature_.clear();
  for (const auto& t : temps) temperature_.push_back(clamp_temperature(static_cast<const snapjudge::Json&>(t)));
  temperature_by_options_.clear();
  std::vector<std::string> rejected;
  if (cfg_.contains("temperature_by_options") &&
      cfg_["temperature_by_options"].is_object()) {
    for (auto it = cfg_["temperature_by_options"].begin();
         it != cfg_["temperature_by_options"].end(); ++it) {
      double raw = it.value().is_number() ? it.value().get<double>() : NAN;
      double applied = clamp_temperature(static_cast<const snapjudge::Json&>(it.value()));
      temperature_by_options_[it.key()] = applied;
      if (!std::isnan(raw) && raw != applied) rejected.push_back(it.key());
    }
  }
  if (temps.is_array()) {
    for (size_t i = 0; i < temps.size(); ++i) {
      if (temps[i].is_number()) {
        double raw = temps[i].get<double>();
        if (raw != temperature_[i])
          rejected.push_back("temperature[" + std::to_string(i) + "]");
      }
    }
  }
  if (!rejected.empty()) {
    fprintf(stderr,
            "snapjudge: this checkpoint ships invalid temperatures or values outside "
            "[%g, %g]; clamping. Treat confidence from the affected entries as "
            "uncalibrated.\n", 0.5, 5.0);
  }
}

namespace {

struct Item {
  std::vector<int32_t> ids;
  std::vector<int64_t> markers;
  int qtype;
};

}  // namespace

std::vector<json> Agent::predict_batch(const std::vector<json>& states,
                                       const json& questions, int batch_size) const {
  if (states.empty()) return {};
  std::vector<std::string> ids;
  ids.reserve(questions.size());
  for (auto it = questions.begin(); it != questions.end(); ++it) ids.push_back(it.key());
  if (ids.empty()) {
    return std::vector<json>(states.size(),
                             json{{"model", "snapjudge-rl-agent"},
                                  {"answers", json::object()},
                                  {"usage", {{"input_tokens", 0}, {"output_tokens", 0}}}});
  }
  for (const auto& qid : ids) check_question(qid, questions[qid]);
  std::unordered_map<std::string, InternalQ> internal;
  for (const auto& qid : ids) internal.emplace(qid, to_internal(questions[qid]));

  int chunk = (batch_size > 0) ? batch_size : static_cast<int>(states.size());
  std::vector<json> results;
  results.reserve(states.size());

  const int64_t max_len = cfg_.value("max_len", 512);
  const int64_t head_max_len = cfg_.value("head_max_len", 192);

  for (size_t start = 0; start < states.size(); start += static_cast<size_t>(chunk)) {
    size_t end = std::min(states.size(), start + static_cast<size_t>(chunk));
    std::vector<std::vector<Item>> per_state_items;
    for (size_t si = start; si < end; ++si) {
      bool truncate_left = states[si].is_array();
      std::vector<Item> items;
      items.reserve(ids.size());
      for (const auto& qid : ids) {
        const InternalQ& q = internal[qid];
        auto [seq, markers] =
            build_sequence(*impl_->tok, states[si], q, max_len, head_max_len, truncate_left);
        if (static_cast<int64_t>(markers.size()) !=
            static_cast<int64_t>(render_options(q).size()))
          throw std::invalid_argument("question \"" + qid +
                                      "\" options exceed head_max_len=" +
                                      std::to_string(head_max_len));
        items.push_back({std::move(seq), std::move(markers), QTYPES.at(q.t)});
      }
      per_state_items.push_back(std::move(items));
    }

    // collate (common.py::collate_items, target omitted — inference only)
    std::vector<std::vector<int64_t>> input_ids, attention_mask, marker_pos;
    std::vector<std::vector<uint8_t>> marker_mask;
    std::vector<int64_t> qtype_v;
    {
      std::vector<const Item*> all;
      for (const auto& group : per_state_items)
        for (const auto& it : group) all.push_back(&it);
      size_t Lmax = 0, Kmax = 0;
      for (const Item* it : all) {
        Lmax = std::max(Lmax, it->ids.size());
        Kmax = std::max(Kmax, it->markers.size());
      }
      int32_t pad_id = impl_->tok->pad_id;
      input_ids.assign(all.size(), std::vector<int64_t>(Lmax, pad_id));
      attention_mask.assign(all.size(), std::vector<int64_t>(Lmax, 0));
      marker_pos.assign(all.size(), std::vector<int64_t>(Kmax, 0));
      marker_mask.assign(all.size(), std::vector<uint8_t>(Kmax, 0));
      for (size_t r = 0; r < all.size(); ++r) {
        const Item* it = all[r];
        for (size_t j = 0; j < it->ids.size(); ++j) {
          input_ids[r][j] = it->ids[j];
          attention_mask[r][j] = 1;
        }
        for (size_t kk = 0; kk < it->markers.size(); ++kk) {
          marker_pos[r][kk] = it->markers[kk];
          marker_mask[r][kk] = 1;
        }
        qtype_v.push_back(it->qtype);
      }
    }

    std::vector<std::vector<float>> logits, act;
    impl_->model->forward(input_ids, attention_mask, marker_pos, marker_mask, qtype_v,
                          logits, act);

    // decode answers per state (agent.py::_decode_answers)
    size_t row = 0;
    for (const auto& items : per_state_items) {
      int64_t n_tokens = 0;
      for (size_t r = row; r < row + items.size(); ++r)
        for (int64_t v : attention_mask[r]) n_tokens += v;

      json answers = json::object();
      for (size_t j = 0; j < items.size(); ++j) {
        const std::string& qid = ids[j];
        const InternalQ& q = internal[qid];
        int k = static_cast<int>(items[j].markers.size());
        int qt = items[j].qtype;
        double t_scale;
        auto it2 = temperature_by_options_.find(temp_bucket(qt, k));
        t_scale = it2 != temperature_by_options_.end() ? it2->second
                                                       : temperature_[static_cast<size_t>(qt)];
        std::vector<double> z(static_cast<size_t>(k));
        for (int kk = 0; kk < k; ++kk) z[static_cast<size_t>(kk)] = logits[row + j][kk] / t_scale;
        double mx = *std::max_element(z.begin(), z.end());
        double s = 0;
        std::vector<double> p(static_cast<size_t>(k));
        for (int kk = 0; kk < k; ++kk) {
          p[static_cast<size_t>(kk)] = std::exp(z[static_cast<size_t>(kk)] - mx);
          s += p[static_cast<size_t>(kk)];
        }
        for (double& v : p) v /= s;

        double conf = round4(confidence_from_probs(p.data(), k));
        json ext = {{"act_probability", round4(act[row + j][0])}};

        if (q.t == "choice") {
          std::vector<std::string> keys;
          for (auto it = q.crit.begin(); it != q.crit.end(); ++it) keys.push_back(it.key());
          int argmax = 0;
          for (int kk = 1; kk < k; ++kk) if (p[kk] > p[argmax]) argmax = kk;
          json probs = json::object();
          for (int kk = 0; kk < k; ++kk) probs[keys[kk]] = round4(p[kk]);
          answers[qid] = {{"type", "choice"},
                          {"choice", keys[argmax]},
                          {"probabilities", probs},
                          {"confidence", conf},
                          {"action", ext}};
        } else if (q.t == "score") {
          double exp_score = 0;
          for (int kk = 0; kk < k; ++kk) exp_score += kk * p[kk];
          json legend = json::object(), probs = json::object();
          int i = 0;
          for (const auto& c : q.crit) {
            legend[std::to_string(i)] = c;
            probs[std::to_string(i)] = round4(p[i]);
            ++i;
          }
          answers[qid] = {{"type", "score"},
                          {"score", round4(exp_score)},
                          {"legend", legend},
                          {"probabilities", probs},
                          {"confidence", conf},
                          {"action", ext}};
        } else {
          double p1 = p.size() > 1 ? p[1] : 0.0;
          answers[qid] = {{"type", "noul"},
                          {"noul", round4(p1)},
                          {"confidence", round4(std::max(p1, 1.0 - p1))},
                          {"action", ext}};
        }
      }
      row += items.size();
      results.push_back({{"model", "snapjudge-rl-agent"},
                         {"answers", answers},
                         {"usage", {{"input_tokens", n_tokens},
                                    {"output_tokens", 0}}}});
    }
  }
  return results;
}

json Agent::system_one(const json& state, const json& questions) const {
  return predict_batch({state}, questions, 0).front();
}

// ---- embed: mean-pooled encoder hidden states over tokenized texts ----------
// Port of shortlist.py::embed_fn_from_agent: encode each text (truncated to
// max_length), encoder-only forward, then mean over non-padding positions.
std::vector<std::vector<float>> Agent::embed(const std::vector<std::string>& texts,
                                             int max_length, int batch_size) const {
  if (max_length < 1)
    throw std::invalid_argument("max_length must be a positive integer, got " +
                                std::to_string(max_length));
  if (batch_size < 1)
    throw std::invalid_argument("batch_size must be a positive integer, got " +
                                std::to_string(batch_size));
  std::vector<std::string> rows(texts.begin(), texts.end());
  if (rows.empty()) return {};

  std::vector<std::vector<float>> out(rows.size());
  for (size_t start = 0; start < rows.size(); start += static_cast<size_t>(batch_size)) {
    size_t end = std::min(rows.size(), start + static_cast<size_t>(batch_size));
    std::vector<std::vector<int64_t>> enc;
    size_t Lmax = 0;
    for (size_t r = start; r < end; ++r) {
      std::vector<int32_t> ids = impl_->tok->encode(rows[r]);
      if (static_cast<int>(ids.size()) > max_length)
        ids.resize(static_cast<size_t>(max_length));
      std::vector<int64_t> v(ids.begin(), ids.end());
      Lmax = std::max(Lmax, v.size());
      enc.push_back(std::move(v));
    }
    std::vector<std::vector<int64_t>> ids_b, mask_b;
    for (const auto& v : enc) {
      std::vector<int64_t> row = v;
      std::vector<int64_t> m(Lmax, 0);
      for (size_t k = 0; k < row.size(); ++k) m[k] = 1;
      row.resize(Lmax, impl_->tok->pad_id);
      ids_b.push_back(std::move(row));
      mask_b.push_back(std::move(m));
    }
    auto hidden = impl_->model->encode(ids_b, mask_b);  // [b, L, D]
    for (size_t r = 0; r < enc.size(); ++r) {
      size_t n = std::max<size_t>(1, enc[r].size());
      size_t D2 = hidden[r].empty() ? 0 : hidden[r][0].size();
      std::vector<float> pooled(D2, 0.0f);
      for (size_t i = 0; i < enc[r].size(); ++i)
        for (size_t d = 0; d < D2; ++d) pooled[d] += hidden[r][i][d];
      for (float& v : pooled) v /= static_cast<float>(n);
      out[start + r] = std::move(pooled);
    }
  }
  return out;
}

// ---- CUDA fast path (agent.py::accelerate contract) -------------------------
bool Agent::accelerate() {
#if defined(SNAPJUDGE_WITH_CUDA)
  if (impl_->fast) return true;
  if (device_ != "cuda") {
    fprintf(stderr, "snapjudge: fast path needs a CUDA device (device=%s)\n",
            device_.c_str());
    return false;
  }
  try {
    impl_->fast = std::make_unique<FastSnapjudge>(
        impl_->model.get(), cfg_.value("max_len", 512), true);
    return true;
  } catch (const std::exception& e) {
    fprintf(stderr, "snapjudge: fast path unavailable (%s); using the CPU forward.\n",
            e.what());
    return false;
  }
#else
  fprintf(stderr,
          "snapjudge: this build has no CUDA fast path (SNAPJUDGE_CUDA=OFF); "
          "using the CPU forward.\n");
  return false;
#endif
}

bool Agent::accelerated() const {
#if defined(SNAPJUDGE_WITH_CUDA)
  return impl_->fast != nullptr;
#else
  return false;
#endif
}

}  // namespace snapjudge
