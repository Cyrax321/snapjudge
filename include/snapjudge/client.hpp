#pragma once
// snapjudge client.hpp: LangChain-integration equivalents for C++ callers,
// port of integrations/langchain.py. LangChain itself has no C++
// equivalent; these are plain classes with the same decision logic and the
// same optional remote (HTTP) mode against a snapjudge-serve instance.

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include <nlohmann/json.hpp>

#include "snapjudge/presets.hpp"

namespace snapjudge {

using nlohmann::ordered_json;
class Agent;
class Router;

// Raised when a guardrail policy is violated (langchain.py GuardrailError).
class GuardrailError : public std::invalid_argument {
 public:
  ordered_json violations, raw_decision;
  GuardrailError(const std::string& msg, ordered_json v, ordered_json r)
      : std::invalid_argument(msg), violations(std::move(v)), raw_decision(std::move(r)) {}
};

namespace detail {
// _extract_text: try state_key first, then common message keys, then the
// most-recent human message.
ordered_json extract_text(const ordered_json& input, const std::string& state_key);
}  // namespace detail

// Backend for the integration classes: either in-process Router/Agent or an
// HTTP URL pointing at a snapjudge-serve (or snapjudge-serve) instance. Set exactly
// one of router / agent / base_url; empty falls back to a process-wide default
// Router (the Python `_get_default_router` behavior).
struct Backend {
  Router* router = nullptr;
  Agent* agent = nullptr;
  std::string base_url;                 // e.g. http://localhost:8000
  std::string api_key;
  std::string model;                    // optional model hint
  double timeout_s = 10.0;

  ordered_json execute(const ordered_json& state, const ordered_json& questions) const;
};

// RouterRunnable equivalent: choose a branch label via a choice question.
class RouterRunnable {
 public:
  std::unordered_map<std::string, std::string> criteria;
  std::string instructions = "Which route should handle this request?";
  double confidence_threshold = 0.0;
  std::string fallback;                 // empty = no fallback
  std::string state_key;
  Backend backend;
  ordered_json last_decision;           // log of the raw result

  std::string invoke(const ordered_json& input);
  std::string operator()(const ordered_json& input) { return invoke(input); }
};

// Guardrail equivalent: action "raise" | "filter" | "annotate".
class Guardrail {
 public:
  ordered_json questions;               // not set = guard_questions() preset
  std::string action = "raise";
  std::string rejection_message =
      "I cannot fulfill this request because it violates safety guidelines.";
  double threshold = 0.5;
  std::string state_key;
  Backend backend;

  ordered_json invoke(const ordered_json& input);
  ordered_json operator()(const ordered_json& input) { return invoke(input); }
};

// Triage equivalent: enrich state with the triage preset's answers.
class Triage {
 public:
  std::string state_key;
  Backend backend;
  ordered_json invoke(const ordered_json& state);
  ordered_json operator()(const ordered_json& state) { return invoke(state); }
};

// Evaluator equivalent: rubric grading of a prediction.
class Evaluator {
 public:
  ordered_json questions;
  std::string state_key;
  Backend backend;
  ordered_json invoke(const ordered_json& input);
  ordered_json operator()(const ordered_json& input) { return invoke(input); }
  ordered_json evaluate_strings(const std::string& prediction,
                                const std::string& input = "");
};

}  // namespace snapjudge
