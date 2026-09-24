#include "snapjudge/client.hpp"

// Port of integrations/langchain.py. The pydantic/RunnableSerializable
// base machinery does not exist in C++; the decision logic, HTTP remote mode,
// and (error) payload shapes are all preserved.

#include <cstdio>
#include <stdexcept>

#if defined(SNAPJUDGE_HAVE_CURL)
#include <curl/curl.h>
#endif

#include "snapjudge/agent.hpp"
#include "snapjudge/router.hpp"

namespace snapjudge {

namespace detail {

static const char* CANDIDATE_KEYS[] = {"input", "text", "query", "prompt",
                                       "message", "body", "content"};

ordered_json from_message_or_value(const ordered_json& val) {
  if (val.is_object() && val.contains("content")) return val["content"];
  return val;
}

ordered_json from_messages_list(const ordered_json& msgs) {
  if (!msgs.is_array() || msgs.empty()) return "";
  // most recent human/user message, scanning backwards
  for (auto it = msgs.rbegin(); it != msgs.rend(); ++it) {
    const ordered_json& m = *it;
    if (m.is_object()) {
      std::string role = m.value("type", m.value("role", std::string()));
      if (role == "human" || role == "user")
        return m.contains("content") ? m["content"] : ordered_json(m);
    }
  }
  const ordered_json& last = msgs.back();
  return last.is_object() && last.contains("content") ? last["content"]
                                                      : ordered_json(last);
}

ordered_json extract_text(const ordered_json& input, const std::string& state_key) {
  if (!state_key.empty() && input.is_object() && input.contains(state_key))
    return from_message_or_value(input[state_key]);
  if (input.is_string()) return input;
  if (input.is_object()) {
    for (const char* k : CANDIDATE_KEYS)
      if (input.contains(k)) return from_message_or_value(input[k]);
    if (input.contains("messages")) return from_messages_list(input["messages"]);
    return input;
  }
  if (input.is_array()) return from_messages_list(input);
  return input;
}

}  // namespace detail

// ---- Backend ------------------------------------------------------------------

namespace {
Router& default_router() {
  static std::unique_ptr<Router> r = [] { return std::make_unique<Router>(); }();
  return *r;
}

#if defined(SNAPJUDGE_HAVE_CURL)
size_t curl_write_str(char* ptr, size_t size, size_t nmemb, void* ud) {
  static_cast<std::string*>(ud)->append(ptr, size * nmemb);
  return size * nmemb;
}

ordered_json call_remote(const std::string& base_url, const ordered_json& state,
                         const ordered_json& questions, const std::string& api_key,
                         const std::string& model, double timeout_s) {
  std::string url = base_url;
  while (!url.empty() && url.back() == '/') url.pop_back();
  if (url.size() < 13 || url.rfind("/v1/systemone") != url.size() - 13)
    url += "/v1/systemone";

  ordered_json payload = {{"state", state}, {"questions", questions}};
  if (!model.empty()) payload["model"] = model;
  std::string data = payload.dump();

  CURL* h = curl_easy_init();
  if (!h) throw std::runtime_error("snapjudge: curl init failed");
  std::string body;
  struct curl_slist* hdrs = nullptr;
  hdrs = curl_slist_append(hdrs, "Content-Type: application/json");
  if (!api_key.empty())
    hdrs = curl_slist_append(hdrs, ("Authorization: Bearer " + api_key).c_str());
  curl_easy_setopt(h, CURLOPT_URL, url.c_str());
  curl_easy_setopt(h, CURLOPT_POSTFIELDS, data.c_str());
  curl_easy_setopt(h, CURLOPT_HTTPHEADER, hdrs);
  curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, curl_write_str);
  curl_easy_setopt(h, CURLOPT_WRITEDATA, &body);
  curl_easy_setopt(h, CURLOPT_TIMEOUT_MS, static_cast<long>(timeout_s * 1000));
  CURLcode rc = curl_easy_perform(h);
  long code = 0;
  curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &code);
  curl_slist_free_all(hdrs);
  curl_easy_cleanup(h);
  if (rc != CURLE_OK)
    throw std::runtime_error("Failed to connect to snapjudge server at " + url + ": " +
                             curl_easy_strerror(rc));
  if (code != 200)
    throw std::runtime_error("snapjudge server error " + std::to_string(code) + ": " + body);
  return ordered_json::parse(body);
}
#endif
}  // namespace

ordered_json Backend::execute(const ordered_json& state,
                              const ordered_json& questions) const {
  if (!base_url.empty()) {
#if defined(SNAPJUDGE_HAVE_CURL)
    return call_remote(base_url, state, questions, api_key, model, timeout_s);
#else
    throw std::runtime_error("snapjudge: this build has no libcurl; use an "
                             "in-process Router/Agent backend");
#endif
  }
  Router& r = router ? *router : default_router();
  Agent* a = agent;
  if (a) return a->predict(state, questions);
  return r.predict(state, questions, model);
}

// ---- RouterRunnable --------------------------------------------------------------
std::string RouterRunnable::invoke(const ordered_json& input) {
  ordered_json text = detail::extract_text(input, state_key);
  ordered_json crit = ordered_json::object();
  for (const auto& [k, v] : criteria) crit[k] = v;
  ordered_json questions = ordered_json{
      {"route", ordered_json{{"type", "choice"},
                             {"instructions", instructions},
                             {"criteria", crit}}}};
  ordered_json res = backend.execute(text, questions);
  last_decision = res;
  const ordered_json& ans = res["answers"]["route"];
  std::string choice = ans["choice"].get<std::string>();
  double conf = ans.value("confidence", 1.0);
  if (confidence_threshold > 0.0 && conf < confidence_threshold && !fallback.empty())
    return fallback;
  return choice;
}

// ---- Guardrail -------------------------------------------------------------------
ordered_json Guardrail::invoke(const ordered_json& input) {
  ordered_json text = detail::extract_text(input, state_key);
  ordered_json qdefs =
      (questions.is_null() || questions.empty()) ? guard_questions() : questions;
  ordered_json res = backend.execute(text, qdefs);
  const ordered_json& answers = res.value("answers", ordered_json::object());

  ordered_json violations = ordered_json::object();
  for (auto it = answers.begin(); it != answers.end(); ++it) {
    const ordered_json& ans = it.value();
    std::string t = ans.value("type", "");
    if (t == "noul" && ans.value("noul", 0.0) >= threshold)
      violations[it.key()] = {{"probability", ans["noul"]},
                              {"confidence", ans.value("confidence", 0.0)}};
    else if (t == "score" && ans.value("score", 0.0) >= threshold)
      violations[it.key()] = {{"score", ans["score"]},
                              {"confidence", ans.value("confidence", 0.0)}};
  }
  bool is_safe = violations.empty();

  if (!is_safe && action == "raise") {
    std::string names;
    bool first = true;
    for (auto it = violations.begin(); it != violations.end(); ++it) {
      if (!first) names += ", ";
      names += it.key();
      first = false;
    }
    throw GuardrailError("snapjudge guardrail policy violation detected: [" + names + "]",
                         violations, res);
  }
  if (!is_safe && action == "filter") {
    if (input.is_object()) {
      ordered_json filtered = input;
      filtered["output"] = rejection_message;
      return filtered;
    }
    return rejection_message;
  }
  if (action == "annotate") {
    ordered_json guardrails = {{"passed", is_safe},
                               {"violations", violations},
                               {"answers", answers}};
    if (input.is_object()) {
      ordered_json annotated = input;
      annotated["guardrails"] = guardrails;
      return annotated;
    }
    return ordered_json{{"input", input}, {"guardrails", guardrails}};
  }
  return input;
}

// ---- Triage ----------------------------------------------------------------------
ordered_json Triage::invoke(const ordered_json& state) {
  ordered_json text = detail::extract_text(state, state_key);
  ordered_json res = backend.execute(text, triage_questions());
  const ordered_json& ans = res.value("answers", ordered_json::object());
  auto noul = [&](const char* k) {
    return ans.contains(k) ? ans[k].value("noul", 0.0) >= 0.5 : false;
  };
  ordered_json triage_info = {
      {"intent", ans.contains("intent") && ans["intent"].contains("choice")
                     ? ans["intent"]["choice"] : ordered_json()},
      {"intent_confidence", ans.contains("intent") && ans["intent"].contains("confidence")
                                ? ordered_json(ans["intent"]["confidence"]) : ordered_json()},
      {"is_urgent", noul("is_urgent")},
      {"frustration_score",
       ans.contains("frustration") && ans["frustration"].contains("score")
           ? ordered_json(ans["frustration"]["score"]) : ordered_json()},
      {"churn_risk", noul("churn_risk")},
      {"refund_requested", noul("refund_requested")}};
  if (state.is_object()) {
    ordered_json updated = state;
    updated["triage"] = triage_info;
    return updated;
  }
  return ordered_json{{"input", state}, {"triage", triage_info}};
}

// ---- Evaluator ---------------------------------------------------------------------
ordered_json Evaluator::invoke(const ordered_json& input) {
  ordered_json text = detail::extract_text(input, state_key);
  ordered_json res = backend.execute(text, questions);
  return res.value("answers", ordered_json::object());
}

ordered_json Evaluator::evaluate_strings(const std::string& prediction,
                                         const std::string& input) {
  ordered_json state =
      input.empty() ? ordered_json(prediction)
                    : ordered_json{{"input", input}, {"prediction", prediction}};
  ordered_json res = backend.execute(state, questions);
  return res.value("answers", ordered_json::object());
}

}  // namespace snapjudge
