#include "snapjudge/common.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <unordered_set>

#include "snapjudge/tokenizer.hpp"

namespace snapjudge {

const std::unordered_map<std::string, int> QTYPES = {
    {"choice", 0}, {"score", 1}, {"noul", 2}};

namespace {

// Python json.dumps(v, ensure_ascii=False) with default separators (", ", ": ").
std::string py_json(const Json& v) {
  if (v.is_null()) return "null";
  if (v.is_string()) {
    // JSON string escaping without ASCII escaping
    Json esc = v;
    return esc.dump(-1, ' ', false);  // dump quotes the string; ensure no non-ASCII escaping
  }
  if (v.is_boolean()) return v.get<bool>() ? "true" : "false";
  if (v.is_number()) return v.dump();
  if (v.is_array()) {
    std::string out = "[";
    bool first = true;
    for (const auto& e : v) {
      if (!first) out += ", ";
      out += py_json(e);
      first = false;
    }
    return out + "]";
  }
  if (v.is_object()) {
    std::string out = "{";
    bool first = true;
    for (auto it = v.begin(); it != v.end(); ++it) {
      if (!first) out += ", ";
      out += Json(it.key()).dump(-1, ' ', false);
      out += ": ";
      out += py_json(it.value());
      first = false;
    }
    return out + "}";
  }
  return "null";
}

}  // namespace

std::string serialize_state(const Json& state) {
  if (state.is_string()) return state.get<std::string>();
  return py_json(state);
}

std::string render_criterion(const Json& v) {
  if (v.is_string()) return v.get<std::string>();
  // json.dumps(v, ensure_ascii=False, separators=(", ", ": "))
  return py_json(v);
}

// _resolve_noul_labels from common.py — throws std::invalid_argument.
static std::pair<std::string, std::string> resolve_noul_labels(const Json* labels) {
  static const std::pair<std::string, std::string> DEF{"false", "true"};
  if (!labels) return DEF;
  const Json& l = *labels;
  if (!l.is_object() || !l.contains("false") || !l.contains("true") || l.size() != 2)
    throw std::invalid_argument(
        "noul labels must map exactly 'false' and 'true' to distinct non-empty strings");
  if (!l["false"].is_string() || !l["true"].is_string())
    throw std::invalid_argument(
        "noul labels must map exactly 'false' and 'true' to distinct non-empty strings");
  auto trim = [](std::string s) {
    size_t a = s.find_first_not_of(" \t\n\r");
    size_t b = s.find_last_not_of(" \t\n\r");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
  };
  std::string f = trim(l["false"].get<std::string>());
  std::string t = trim(l["true"].get<std::string>());
  if (f.empty() || t.empty() || f == t)
    throw std::invalid_argument(
        "noul labels must map exactly 'false' and 'true' to distinct non-empty strings");
  return {f, t};
}

std::vector<std::string> render_options(const InternalQ& q) {
  if (q.t != "noul" && q.has_labels)
    throw std::invalid_argument("labels is only supported for noul questions");
  std::vector<std::string> out;
  if (q.t == "choice") {
    for (auto it = q.crit.begin(); it != q.crit.end(); ++it) {
      const Json& v = it.value();
      if (v.is_null() || (v.is_string() && v.get<std::string>().empty()))
        out.push_back(it.key());
      else
        out.push_back(it.key() + ": " + render_criterion(v));
    }
    return out;
  }
  if (q.t == "score") {
    int i = 0;
    for (const auto& c : q.crit)
      out.push_back("level " + std::to_string(i++) + ": " + render_criterion(c));
    return out;
  }
  // noul
  Json crit = q.crit.is_object() ? q.crit : Json::object();
  std::pair<std::string, std::string> labels =
      resolve_noul_labels(q.has_labels ? &q.labels : nullptr);
  std::string fcrit, tcrit;
  bool has_f = crit.contains("false") && !crit["false"].is_null() &&
               !(crit["false"].is_string() && crit["false"].get<std::string>().empty());
  bool has_t = crit.contains("true") && !crit["true"].is_null() &&
               !(crit["true"].is_string() && crit["true"].get<std::string>().empty());
  std::string f =
      labels.first + ": " +
      (has_f ? render_criterion(crit["false"]) : "no, the statement does not hold");
  std::string t =
      labels.second + ": " +
      (has_t ? render_criterion(crit["true"]) : "yes, the statement holds");
  return {f, t};
}

static std::string replace_all_str(std::string s, const std::string& from,
                                   const std::string& to) {
  if (from.empty()) return s;
  size_t pos = 0;
  while ((pos = s.find(from, pos)) != std::string::npos) {
    s.replace(pos, from.size(), to);
    pos += to.size();
  }
  return s;
}

std::pair<std::vector<int32_t>, std::vector<int64_t>> build_sequence(
    const Tokenizer& tok, const Json& state, const InternalQ& q,
    int64_t max_len, int64_t head_max_len, bool truncate_left) {
  const std::string& mask_tok = tok.mask_token;
  std::vector<std::string> opts = render_options(q);
  std::string ins = replace_all_str(q.ins, mask_tok, " ");
  std::vector<int32_t> head_ids = tok.encode(q.t + " question: " + ins);
  std::vector<std::vector<int32_t>> opt_ids;
  for (const std::string& o : opts) {
    std::vector<int32_t> ids = tok.encode(" " + replace_all_str(o, mask_tok, " "));
    ids.insert(ids.begin(), tok.mask_id);
    if (ids.size() > 49) ids.resize(49);  // [mask] + 48 tokens
    opt_ids.push_back(std::move(ids));
  }
  auto total = [&]() {
    int64_t s = 0;
    for (const auto& o : opt_ids) s += static_cast<int64_t>(o.size());
    return s;
  };
  int64_t opt_budget = head_max_len - total();
  if (opt_budget < 16) {
    int64_t per = std::max<int64_t>(
        4, (head_max_len - 16) / std::max<int64_t>(1, static_cast<int64_t>(opt_ids.size())));
    for (auto& o : opt_ids)
      if (static_cast<int64_t>(o.size()) > per) o.resize(static_cast<size_t>(per));
    opt_budget = head_max_len - total();
  }
  if (static_cast<int64_t>(head_ids.size()) > std::max<int64_t>(8, opt_budget))
    head_ids.resize(static_cast<size_t>(std::max<int64_t>(8, opt_budget)));

  std::vector<int32_t> ids;
  ids.reserve(static_cast<size_t>(max_len));
  ids.push_back(tok.cls_id);
  ids.insert(ids.end(), head_ids.begin(), head_ids.end());
  ids.push_back(tok.sep_id);
  std::vector<int64_t> markers;
  for (const auto& o : opt_ids) {
    markers.push_back(static_cast<int64_t>(ids.size()));
    ids.insert(ids.end(), o.begin(), o.end());
  }
  ids.push_back(tok.sep_id);
  int64_t room = std::max<int64_t>(0, max_len - static_cast<int64_t>(ids.size()) - 1);
  std::vector<int32_t> st = tok.encode(replace_all_str(serialize_state(state), mask_tok, " "));
  std::vector<int32_t> st_trimmed;
  if (truncate_left) {
    int64_t start = std::max<int64_t>(0, static_cast<int64_t>(st.size()) - room);
    st_trimmed.assign(st.begin() + start, st.end());
  } else if (static_cast<int64_t>(st.size()) > room) {
    st_trimmed.assign(st.begin(), st.begin() + room);
  } else {
    st_trimmed = std::move(st);
  }
  ids.insert(ids.end(), st_trimmed.begin(), st_trimmed.end());
  ids.push_back(tok.sep_id);
  if (static_cast<int64_t>(ids.size()) > max_len) ids.resize(static_cast<size_t>(max_len));
  markers.erase(std::remove_if(markers.begin(), markers.end(),
                               [&](int64_t m) { return m >= max_len; }),
                markers.end());
  return {std::move(ids), std::move(markers)};
}

double clamp_temperature(double t) {
  if (std::isnan(t) || std::isinf(t)) return 1.0;
  return std::min(TEMP_MAX, std::max(TEMP_MIN, t));
}

double clamp_temperature(const Json& t) {
  // Python `float(t)` coercion: reject bools here the way callers do (bool is
  // not a meaningful temperature) and convert str like float('2.5').
  if (t.is_number()) return clamp_temperature(t.get<double>());
  if (t.is_string()) {
    try {
      size_t pos;
      double v = std::stod(t.get<std::string>(), &pos);
      if (pos == t.get<std::string>().size()) return clamp_temperature(v);
    } catch (...) {
    }
  }
  return 1.0;
}

std::string temp_bucket(int qtype, int k) {
  static const char* names[3] = {"choice", "score", "noul"};
  const char* size = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+";
  return std::string(names[qtype]) + ":" + size;
}

double confidence_from_probs(const double* p, int k) {
  if (k < 2) return 1.0;
  double ent = 0;
  for (int i = 0; i < k; ++i) ent -= p[i] * std::log(std::max(p[i], 1e-12));
  return std::min(1.0, std::max(0.0, 1.0 - ent / std::log(static_cast<double>(k))));
}

double round4(double v) {
  // Python round() is banker's rounding at .5; replicate for exactness.
  double scaled = v * 1e4;
  double r = std::nearbyint(scaled);
  return r / 1e4;
}

double ece_score(const std::vector<double>& conf, const std::vector<double>& correct, int bins) {
  if (conf.empty()) return std::nan("");
  double e = 0.0;
  for (int i = 0; i < bins; ++i) {
    double lo = static_cast<double>(i) / bins, hi = static_cast<double>(i + 1) / bins;
    double w = 0, cs = 0, ks = 0;
    for (size_t j = 0; j < conf.size(); ++j) {
      bool sel = (i == 0 ? conf[j] >= lo : conf[j] > lo) && conf[j] <= hi;
      if (sel) { w += 1; cs += conf[j]; ks += correct[j]; }
    }
    if (w > 0) {
      e += (w / conf.size()) * std::fabs(cs / w - ks / w);
    }
  }
  return e;
}

std::vector<double> proper_reward(const std::vector<std::vector<double>>& q_in,
                                  const std::vector<std::vector<double>>& target,
                                  const std::vector<int64_t>& qtype,
                                  const std::vector<std::vector<double>>& mask,
                                  double w_sph, double w_rps, double log_floor) {
  size_t N = q_in.size();
  std::vector<double> out(N, 0.0);
  for (size_t n = 0; n < N; ++n) {
    size_t K = q_in[n].size();
    double log_score = 0, qnorm = 0, sph_num = 0;
    for (size_t k = 0; k < K; ++k) {
      double qk = q_in[n][k] * mask[n][k];
      log_score += target[n][k] * std::max(std::log(std::max(qk, 1e-12)), log_floor);
      sph_num += target[n][k] * qk;
      qnorm += qk * qk;
    }
    double r = log_score + w_sph * sph_num / std::max(std::sqrt(qnorm), 1e-9);
    if (qtype[n] == 1) {  // score
      double kk = 0;
      for (size_t k = 0; k < K; ++k) kk += mask[n][k];
      kk = std::max(2.0, kk);
      double cdf_q = 0, cdf_t = 0, rps = 0;
      for (size_t k = 0; k < K; ++k) {
        cdf_q += q_in[n][k] * mask[n][k];
        cdf_t += target[n][k];
        double d = cdf_q - cdf_t;
        rps += d * d * (mask[n][k] ? 1.0 : 0.0);
      }
      r -= w_rps * (rps / (kk - 1));
    }
    out[n] = r;
  }
  return out;
}

// Port of common.py::td_lambda_targets. Returns a copy of `target_flat` with
// grouped episodes' columns replaced by TD(lambda) backups.
std::vector<double> td_lambda_targets(const std::vector<double>& p_true,
                                      const std::vector<double>& target_flat,
                                      const std::vector<int64_t>& ep_group,
                                      const std::vector<int64_t>& ep_step,
                                      double lam) {
  std::vector<double> target = target_flat;
  const int64_t N = static_cast<int64_t>(p_true.size());
  // groups >= 0, unique, in first-appearance order after sorting by value
  std::vector<int64_t> groups;
  {
    std::vector<int64_t> all;
    for (int64_t g : ep_group)
      if (g >= 0) all.push_back(g);
    std::sort(all.begin(), all.end());
    all.erase(std::unique(all.begin(), all.end()), all.end());
    groups = std::move(all);
  }
  for (int64_t g : groups) {
    std::vector<int64_t> idx;
    for (int64_t i = 0; i < N; ++i)
      if (ep_group[static_cast<size_t>(i)] == g) idx.push_back(i);
    std::sort(idx.begin(), idx.end(),
              [&](int64_t a, int64_t b) {
                return ep_step[static_cast<size_t>(a)] < ep_step[static_cast<size_t>(b)];
              });
    double G = target_flat[static_cast<size_t>(idx.back() * 2 + 1)];
    for (int64_t j = static_cast<int64_t>(idx.size()) - 1; j >= 0; --j) {
      if (j < static_cast<int64_t>(idx.size()) - 1) {
        G = (1.0 - lam) * p_true[static_cast<size_t>(idx[static_cast<size_t>(j + 1)])] +
            lam * G;
      }
      target[static_cast<size_t>(idx[static_cast<size_t>(j)] * 2 + 0)] = 1.0 - G;
      target[static_cast<size_t>(idx[static_cast<size_t>(j)] * 2 + 1)] = G;
    }
  }
  return target;
}

InternalQ to_internal(const Json& qdef) {
  InternalQ q;
  q.t = qdef.at("type").get<std::string>();
  q.crit = qdef.contains("criteria") ? qdef["criteria"] : Json();
  if (q.t == "choice" && q.crit.is_array()) {
    Json obj = Json::object();
    for (const auto& c : q.crit) obj[c.get<std::string>()] = nullptr;
    q.crit = std::move(obj);
  } else if (q.t == "noul" && q.crit.is_object()) {
    // Normalize boolean literal keys to string keys ("true"/"false")
    Json obj = Json::object();
    for (auto it = q.crit.begin(); it != q.crit.end(); ++it) {
      std::string k = it.key();
      std::transform(k.begin(), k.end(), k.begin(), ::tolower);
      obj[k] = it.value();
    }
    q.crit = std::move(obj);
  }
  if (qdef.contains("instructions")) {
    if (qdef["instructions"].is_string())
      q.ins = qdef["instructions"].get<std::string>();
    else
      q.ins = py_json(qdef["instructions"]);
  }
  if (qdef.contains("labels")) {
    q.has_labels = true;
    q.labels = qdef["labels"];
  }
  return q;
}

void check_question(const std::string& qid, const Json& qdef) {
  auto err = [&](const std::string& m) {
    throw std::invalid_argument("question \"" + qid + "\": " + m);
  };
  if (!qdef.is_object())
    err("definition must be a dict, got " + std::string(qdef.type_name()));
  std::string t = qdef.value("type", "");
  if (!QTYPES.count(t))
    err("unknown type '" + t + "'; use one of [choice, noul, score]");
  if (!qdef.contains("instructions"))
    err("no 'instructions'; add the text the model should answer");
  bool has_crit = qdef.contains("criteria") && !qdef["criteria"].is_null();
  if (t == "choice") {
    if (has_crit && !(qdef["criteria"].is_object() || qdef["criteria"].is_array()))
      err("a choice question takes 'criteria' as a dict of label -> description, "
          "or a list of labels");
    if (!has_crit || qdef["criteria"].empty())
      err("a choice question needs at least one criterion");
  } else if (t == "score") {
    if (!has_crit || !qdef["criteria"].is_array())
      err("a score question takes 'criteria' as a list of level descriptions, "
          "index 0 first");
    if (qdef["criteria"].empty())
      err("a score question needs at least one level");
  } else if (has_crit && !qdef["criteria"].is_object()) {
    err("a noul question takes 'criteria' as a dict with optional 'true'/'false' "
        "descriptions, or omits it");
  }
  if (qdef.contains("labels")) {
    if (t != "noul")
      err("'labels' is only supported for noul questions");
    try {
      resolve_noul_labels(&qdef["labels"]);
    } catch (const std::invalid_argument& e) {
      err(e.what());
    }
  }
}

}  // namespace snapjudge
