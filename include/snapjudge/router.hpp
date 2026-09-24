#pragma once
// snapjudge router.hpp: lazily-loaded checkpoint routing, port of
// router.py.

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

namespace snapjudge {

using nlohmann::ordered_json;
class Agent;

// RouteDecision fields: model, repo, reason, detection, workflow (nullable).
using RouteDecision = ordered_json;

std::string normalise_name(const std::string& name);
// workflow name when the question ids exactly match, else "".
std::string match_typed_decisions_workflow(const ordered_json& questions);

class Router {
 public:
  struct Options {
    std::unordered_map<std::string, std::pair<std::string, std::string>> models;
    std::string device;
    std::string token;
    int max_loaded = 2;
    std::string default_checkpoint = "english";
    bool auto_task_detection = false;
    bool standalone_repos = false;
    bool preload = false;
    std::vector<std::string> preload_names;
    // opt-in language hint: language code, nullopt to abstain.
    std::function<std::string(const ordered_json&)> lang_guess;
  };

  Router();
  explicit Router(Options opts);
  ~Router();

  Agent& load(const std::string& name);
  Router& attach(const std::string& name, std::shared_ptr<Agent> agent);
  Router& preload(const std::vector<std::string>& names = {});
  void unload(const std::string& name = "");
  std::vector<std::string> loaded() const;
  Agent* agent_for(const std::string& name) const;   // nullptr if not resident

  // route() — no model load, no forward pass. `lang`/`task`/`model` explicit
  // overrides, `lang_guess` per-call hint.
  RouteDecision route(const ordered_json& state, const ordered_json& questions,
                      const std::string& model = "", const std::string& task = "",
                      const std::string& lang = "",
                      const std::string& lang_guess = "") const;

  // route, then answer on the chosen checkpoint. Result gets a "routing" key.
  ordered_json predict(const ordered_json& state, const ordered_json& questions,
                       const std::string& model = "", const std::string& task = "",
                       const std::string& lang = "",
                       const std::string& lang_guess = "");
  ordered_json system_one(const ordered_json& state, const ordered_json& questions) {
    return predict(state, questions);
  }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace snapjudge
