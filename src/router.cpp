#include "snapjudge/router.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <set>
#include <stdexcept>

#include "snapjudge/agent.hpp"
#include "snapjudge/lang.hpp"

namespace snapjudge {

namespace {

const std::pair<std::string, std::string> kBundleRepo{"snapjudge/snapjudge", ""};

const std::unordered_map<std::string, std::pair<std::string, std::string>>& default_models() {
  static const std::unordered_map<std::string, std::pair<std::string, std::string>> m = {
      {"english", {"snapjudge/snapjudge", ""}},
      {"multilingual", {"snapjudge/snapjudge", "multilingual"}},
      {"typed-decisions", {"snapjudge/snapjudge", "typed-decisions"}},
  };
  return m;
}

const std::unordered_map<std::string, std::string>& standalone_models() {
  static const std::unordered_map<std::string, std::string> m = {
      {"english", "snapjudge/snapjudge"},
      {"multilingual", "snapjudge/multilingual"},
      {"typed-decisions", "snapjudge/typed-decisions"},
  };
  return m;
}

const std::unordered_map<std::string, std::string>& aliases() {
  static const std::unordered_map<std::string, std::string> m = {
      {"en", "english"}, {"snapjudge", "english"}, {"default", "english"},
      {"multi", "multilingual"}, {"ml", "multilingual"},
      {"snapjudge-multilingual", "multilingual"},
      {"typed", "typed-decisions"}, {"typed_decisions", "typed-decisions"},
      {"snapjudge-typed-decisions", "typed-decisions"}, {"decisions", "typed-decisions"},
  };
  return m;
}

const std::unordered_map<std::string, std::set<std::string>>& typed_workflows() {
  static const std::unordered_map<std::string, std::set<std::string>> m = {
      {"agent_trace_observability", {"action", "needs_review", "outcome", "risk", "urgency"}},
      {"customer_service", {"action", "category", "churn_risk", "needs_human", "urgency"}},
      {"invoice_processing", {"discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"}},
      {"security_incidents", {"credential_compromise", "disposition", "severity", "true_positive", "urgency"}},
  };
  return m;
}

std::string lower_trim(std::string s) {
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);
  return s;
}

// _english_from_code from router.py: True/False/""
std::string english_from_code(const std::string& value) {
  if (value.empty()) return "";
  std::string code = lower_trim(value);
  if (code.empty()) return "";
  size_t dot = code.find('.');
  if (dot != std::string::npos) code = code.substr(0, dot);
  std::replace(code.begin(), code.end(), '_', '-');
  std::string primary = code.substr(0, code.find('-'));
  if (primary.empty()) return "";
  return (primary == "en" || primary == "eng" || primary == "english") ? "true" : "false";
}

}  // namespace

std::string normalise_name(const std::string& name) {
  std::string key = lower_trim(name);
  auto it = aliases().find(key);
  if (it != aliases().end()) key = it->second;
  if (!default_models().count(key)) {
    throw std::invalid_argument(
        "unknown model '" + name +
        "'; choose one of english, multilingual, typed-decisions (or an alias)");
  }
  return key;
}

std::string match_typed_decisions_workflow(const ordered_json& questions) {
  if (!questions.is_object()) return "";
  std::set<std::string> ids;
  for (auto it = questions.begin(); it != questions.end(); ++it) ids.insert(it.key());
  for (const auto& [wf, sig] : typed_workflows()) {
    if (ids == sig) return wf;
  }
  return "";
}

struct Router::Impl {
  std::unordered_map<std::string, std::pair<std::string, std::string>> models;
  std::string device;
  std::string token;
  int max_loaded = 2;
  std::string default_checkpoint = "english";
  bool auto_task_detection = false;
  std::function<std::string(const ordered_json&)> lang_guess;
  std::unordered_map<std::string, std::shared_ptr<Agent>> agents;
  std::deque<std::string> order;  // LRU first
  mutable std::recursive_mutex lock;
};

Router::Router() : Router(Options{}) {}

Router::Router(Options opts) : impl_(std::make_unique<Impl>()) {
  if (opts.standalone_repos) {
    for (const auto& [k, v] : standalone_models()) impl_->models[k] = {v, ""};
  } else {
    impl_->models = default_models();
  }
  for (const auto& [k, v] : opts.models) impl_->models[normalise_name(k)] = v;
  impl_->device = opts.device;
  impl_->token = opts.token.empty()
                     ? (std::getenv("HF_TOKEN") ? std::getenv("HF_TOKEN") : "")
                     : opts.token;
  impl_->max_loaded = std::max(1, opts.max_loaded);
  impl_->default_checkpoint = normalise_name(opts.default_checkpoint);
  impl_->auto_task_detection = opts.auto_task_detection;
  impl_->lang_guess = opts.lang_guess;
  (void)opts.preload_names;
  if (opts.preload) preload(opts.preload_names);
}

Router::~Router() = default;

namespace {
std::string repo_str(const std::pair<std::string, std::string>& spec) {
  return spec.second.empty() ? spec.first : (spec.first + "/" + spec.second);
}

void touch(std::deque<std::string>& order, const std::string& key) {
  order.erase(std::remove(order.begin(), order.end(), key), order.end());
  order.push_back(key);
}
}  // namespace

Agent& Router::load(const std::string& name) {
  std::string key = normalise_name(name);
  std::lock_guard<std::recursive_mutex> lk(impl_->lock);
  auto it = impl_->agents.find(key);
  if (it != impl_->agents.end()) {
    touch(impl_->order, key);
    return *it->second;
  }
  auto spec = impl_->models[key];
  auto agent = std::make_shared<Agent>(spec.first, impl_->device, impl_->token, spec.second);
  impl_->agents[key] = agent;
  impl_->order.push_back(key);
  // evict
  while (static_cast<int>(impl_->order.size()) > impl_->max_loaded) {
    std::string victim = impl_->order.front();
    impl_->order.pop_front();
    impl_->agents.erase(victim);
  }
  // keep views consistent (router.py)
  for (auto it = impl_->agents.begin(); it != impl_->agents.end();) {
    if (std::find(impl_->order.begin(), impl_->order.end(), it->first) == impl_->order.end())
      it = impl_->agents.erase(it);
    else
      ++it;
  }
  return *agent;
}

Router& Router::attach(const std::string& name, std::shared_ptr<Agent> agent) {
  std::string key = normalise_name(name);
  std::lock_guard<std::recursive_mutex> lk(impl_->lock);
  impl_->agents[key] = agent;
  touch(impl_->order, key);
  impl_->max_loaded = std::max<int>(impl_->max_loaded, impl_->agents.size());
  return *this;
}

Router& Router::preload(const std::vector<std::string>& names) {
  std::vector<std::string> want;
  if (names.empty())
    for (const auto& [k, v] : impl_->models) want.push_back(k);
  else
    for (const auto& n : names) want.push_back(normalise_name(n));
  std::lock_guard<std::recursive_mutex> lk(impl_->lock);
  std::set<std::string> all(want.begin(), want.end());
  for (const auto& [k, v] : impl_->agents) all.insert(k);
  impl_->max_loaded = std::max<int>(impl_->max_loaded, all.size());
  for (const auto& n : want)
    if (!impl_->agents.count(n)) load(n);
  return *this;
}

void Router::unload(const std::string& name) {
  std::lock_guard<std::recursive_mutex> lk(impl_->lock);
  if (name.empty()) {
    impl_->agents.clear();
    impl_->order.clear();
    return;
  }
  std::string key = normalise_name(name);
  impl_->agents.erase(key);
  impl_->order.erase(std::remove(impl_->order.begin(), impl_->order.end(), key),
                     impl_->order.end());
}

std::vector<std::string> Router::loaded() const {
  std::lock_guard<std::recursive_mutex> lk(impl_->lock);
  return {impl_->order.begin(), impl_->order.end()};
}

Agent* Router::agent_for(const std::string& name) const {
  std::lock_guard<std::recursive_mutex> lk(impl_->lock);
  auto it = impl_->agents.find(name);
  if (it != impl_->agents.end()) return it->second.get();
  try {
    it = impl_->agents.find(normalise_name(name));
    if (it != impl_->agents.end()) return it->second.get();
  } catch (...) {
  }
  return nullptr;
}

RouteDecision Router::route(const ordered_json& state, const ordered_json& questions,
                            const std::string& model, const std::string& task,
                            const std::string& lang, const std::string& lang_guess) const {
  auto decision = [&](const std::string& m, const std::string& reason,
                      const ordered_json& detection, const ordered_json& workflow) {
    ordered_json d = ordered_json::object();
    d["model"] = m;
    d["repo"] = repo_str(impl_->models.at(m));
    d["reason"] = reason;
    d["detection"] = detection;
    d["workflow"] = workflow;
    return d;
  };

  if (!model.empty()) {
    std::string key = normalise_name(model);
    return decision(key, "explicit model='" + model + "'", nullptr, nullptr);
  }
  if (!task.empty()) {
    std::string t = lower_trim(task);
    std::replace(t.begin(), t.end(), '-', '_');
    std::string key = (t == "typed_decisions") ? "typed-decisions" : normalise_name(task);
    return decision(key, "explicit task='" + task + "'", nullptr, nullptr);
  }

  std::string workflow = match_typed_decisions_workflow(questions);
  ordered_json wf = workflow.empty() ? ordered_json(nullptr) : ordered_json(workflow);
  if (!workflow.empty() && impl_->auto_task_detection) {
    return decision("typed-decisions",
                    "question ids match the '" + workflow + "' typed-decisions workflow",
                    nullptr, wf);
  }

  if (!lang.empty()) {
    std::string e = english_from_code(lang);
    std::string key = (e == "true") ? "english" : "multilingual";
    return decision(key, "explicit lang='" + lang + "'", nullptr, wf);
  }

  // caller-supplied lang_guess
  {
    std::string resolved;
    if (!lang_guess.empty()) resolved = english_from_code(lang_guess);
    else if (impl_->lang_guess) resolved = english_from_code(impl_->lang_guess(state));
    if (!resolved.empty()) {
      std::string key = (resolved == "true") ? "english" : "multilingual";
      return decision(key, "lang_guess: the caller identified this as " +
                               std::string(resolved == "true" ? "English" : "non-English") +
                               " text",
                      nullptr, wf);
    }
  }

  ordered_json det = analyse(state);
  std::string key, reason;
  std::string script = det["script"].get<std::string>();
  if (script == "unknown") {
    key = impl_->default_checkpoint;
    reason = "no letters detected in state; using default (" + key + ")";
  } else if (script != "latin") {
    key = "multilingual";
    char buf[128];
    snprintf(buf, sizeof(buf), "%.0f", 100.0 * det["non_latin_fraction"].get<double>());
    reason = "non-Latin script (" + script + ", " + buf +
             "% of letters); the English checkpoint cannot read it";
  } else if (!det["is_english"].get<bool>()) {
    key = "multilingual";
    if (!det["language"].is_null()) {
      reason = "Latin script but language looks like '" +
               det["language"].get<std::string>() + "', not English";
    } else {
      char buf[128];
      snprintf(buf, sizeof(buf), "%.0f", 100.0 * det["diacritic_rate"].get<double>());
      reason = std::string("Latin script, language not identified but ") + buf +
               "% non-English letters; not safe for the English checkpoint";
    }
  } else if (det["language_undecided"].get<bool>()) {
    key = impl_->default_checkpoint;
    reason = "Latin script, language not identified and no non-English letters; "
             "using default (" + key + ")";
  } else {
    key = "english";
    reason = "English Latin text";
  }
  return decision(key, reason, det, wf);
}

ordered_json Router::predict(const ordered_json& state, const ordered_json& questions,
                             const std::string& model, const std::string& task,
                             const std::string& lang, const std::string& lang_guess) {
  RouteDecision d = route(state, questions, model, task, lang, lang_guess);
  Agent& agent = load(d["model"].get<std::string>());
  ordered_json result = agent.system_one(state, questions);
  result["routing"] = d;
  return result;
}

}  // namespace snapjudge
