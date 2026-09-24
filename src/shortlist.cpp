#include "snapjudge/shortlist.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "snapjudge/agent.hpp"
#include "snapjudge/common.hpp"

namespace snapjudge {

namespace {

int check_k(int k, bool from_caller = true) {
  if (k < 1)
    throw std::invalid_argument("k must be a positive integer, got " + std::to_string(k));
  return k;
}

struct Items {
  std::vector<std::pair<std::string, ordered_json>> items;
};

// _criteria_items from shortlist.py: order-preserving, duplicate-rejecting.
std::vector<std::pair<std::string, ordered_json>> criteria_items(const ordered_json& criteria) {
  std::vector<std::pair<std::string, ordered_json>> items;
  if (criteria.is_object()) {
    for (auto it = criteria.begin(); it != criteria.end(); ++it) items.emplace_back(it.key(), it.value());
  } else if (criteria.is_array()) {
    for (const auto& c : criteria) items.emplace_back(c.get<std::string>(), nullptr);
  } else {
    throw std::invalid_argument("choice criteria must be a dict or list, got " +
                                std::string(criteria.type_name()));
  }
  if (items.empty())
    throw std::invalid_argument("choice criteria must contain at least one option");
  std::unordered_map<std::string, bool> seen;
  for (const auto& [k, v] : items)
    if (!seen.insert({k, true}).second)
      throw std::invalid_argument("choice criteria label \"" + k + "\" is duplicated");
  return items;
}

std::vector<std::string> option_texts(const std::vector<std::pair<std::string, ordered_json>>& items) {
  ordered_json crit = ordered_json::object();
  for (const auto& [k, v] : items) crit[k] = v;
  InternalQ q;
  q.t = "choice";
  q.ins = "";
  q.crit = crit;
  return render_options(q);
}

std::string query_text(const ordered_json& state, const std::string& instructions) {
  std::string body = serialize_state(state);
  if (instructions.empty()) return body;
  return instructions + "\n" + body;
}

std::vector<double> cosine(const std::vector<double>& q,
                           const std::vector<std::vector<double>>& docs) {
  double qn = 0;
  for (double v : q) qn += v * v;
  qn = std::sqrt(qn);
  std::vector<double> sims(docs.size(), 0.0);
  if (qn == 0.0) return sims;
  for (size_t i = 0; i < docs.size(); ++i) {
    double dn = 0, dot = 0;
    for (size_t d = 0; d < docs[i].size(); ++d) {
      dn += docs[i][d] * docs[i][d];
      if (d < q.size()) dot += docs[i][d] * q[d];
    }
    dn = std::sqrt(dn);
    double denom = dn * qn;
    if (denom > 0.0) sims[i] = std::max(-1.0, std::min(1.0, dot / denom));
  }
  return sims;
}

struct Rank {
  std::vector<std::string> labels;
  std::vector<double> scores;  // empty when passthrough
  bool passthrough = false;
  size_t n = 0;
};

Rank rank_impl(const ordered_json& state, const ordered_json& criteria,
               const EmbedFn& embed_fn, int k, const std::string& instructions) {
  check_k(k);
  auto items = criteria_items(criteria);
  size_t n = items.size();
  std::vector<std::string> keys;
  for (const auto& [kk, vv] : items) keys.push_back(kk);
  if (static_cast<size_t>(k) >= n) return {keys, {}, true, n};
  if (!embed_fn) throw std::invalid_argument("embed_fn must be callable");
  std::vector<std::string> texts = {query_text(state, instructions)};
  auto opts = option_texts(items);
  texts.insert(texts.end(), opts.begin(), opts.end());
  std::vector<std::vector<double>> M = embed_fn(texts);
  if (M.size() != texts.size())
    throw std::invalid_argument("embed_fn must return an array of shape (" +
                                std::to_string(texts.size()) + ", dim)");
  std::vector<double> sims = cosine(M[0], std::vector<std::vector<double>>(M.begin() + 1, M.end()));
  // stable argsort by descending score (mergesort keeps earlier labels on ties)
  std::vector<size_t> order(sims.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](size_t a, size_t b) { return sims[a] > sims[b]; });
  order.resize(static_cast<size_t>(k));
  Rank r;
  r.passthrough = false;
  r.n = n;
  for (size_t i : order) {
    r.labels.push_back(keys[i]);
    r.scores.push_back(sims[i]);
  }
  return r;
}

ordered_json subset_criteria(const ordered_json& criteria, const std::vector<std::string>& labels) {
  if (criteria.is_object()) {
    ordered_json out = ordered_json::object();
    for (const auto& l : labels) out[l] = criteria.at(l);
    return out;
  }
  ordered_json out = ordered_json::array();
  for (const auto& l : labels) out.push_back(l);
  return out;
}

}  // namespace

std::vector<std::string> shortlist_choice(const ordered_json& state,
                                          const ordered_json& criteria,
                                          const EmbedFn& embed_fn, int k,
                                          const std::string& instructions) {
  return rank_impl(state, criteria, embed_fn, check_k(k), instructions).labels;
}

ordered_json predict_shortlist(
    const std::function<ordered_json(const ordered_json&, const ordered_json&)>& runner,
    const ordered_json& state, const ordered_json& questions,
    const EmbedFn& embed_fn, int k) {
  if (!questions.is_object())
    throw std::invalid_argument("questions must be a dict of question id -> definition");
  int checked = check_k(k);
  ordered_json reduced = ordered_json::object();
  ordered_json meta = ordered_json::object();
  for (auto it = questions.begin(); it != questions.end(); ++it) {
    const std::string& qid = it.key();
    const ordered_json& qdef = it.value();
    if (!qdef.is_object() || qdef.value("type", "") != "choice") {
      reduced[qid] = qdef;
      continue;
    }
    if (!qdef.contains("criteria"))
      throw std::invalid_argument("question \"" + qid + "\" is a choice but has no criteria");
    Rank r = rank_impl(state, qdef["criteria"], embed_fn, checked,
                       qdef.value("instructions", ""));
    meta[qid] = {{"labels", r.labels},
                 {"scores", r.passthrough ? ordered_json(nullptr) : ordered_json(r.scores)},
                 {"k", checked},
                 {"n", r.n},
                 {"passthrough", r.passthrough}};
    if (r.passthrough) {
      reduced[qid] = qdef;
      continue;
    }
    ordered_json updated = qdef;
    updated["criteria"] = subset_criteria(qdef["criteria"], r.labels);
    reduced[qid] = updated;
  }
  ordered_json result = runner(state, reduced);
  result["shortlist"] = meta;
  return result;
}

EmbedFn embed_fn_from_agent(const Agent& agent, int max_length, int batch_size) {
  const Agent* a = &agent;
  return [a, max_length, batch_size](const std::vector<std::string>& texts) {
    auto pooled = a->embed(texts, max_length, batch_size);
    std::vector<std::vector<double>> out(pooled.size());
    for (size_t i = 0; i < pooled.size(); ++i)
      out[i].assign(pooled[i].begin(), pooled[i].end());
    return out;
  };
}

}  // namespace snapjudge
