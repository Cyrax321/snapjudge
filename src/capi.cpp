#include "snapjudge/capi.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "nlohmann/json.hpp"
#include "snapjudge/agent.hpp"

using nlohmann::ordered_json;

// One engine instance = a set of named checkpoints + an error buffer. The C
// API owns Agents directly (arbitrary caller-chosen names), rather than going
// through the Router's built-in english/multilingual/typed-decisions taxonomy,
// because an embedding host decides its own checkpoint names.
struct sj_engine {
  std::unordered_map<std::string, std::shared_ptr<snapjudge::Agent>> agents;
  std::string default_name;   // first loaded checkpoint
  std::string device;
  std::string token;
  std::string error;
  std::mutex mu;
};

namespace {

char* strdup_owned(const std::string& s) {
  char* p = static_cast<char*>(std::malloc(s.size() + 1));
  if (!p) return nullptr;
  std::memcpy(p, s.data(), s.size());
  p[s.size()] = '\0';
  return p;
}

snapjudge::Agent* pick(sj_engine* e, const char* model) {
  // model empty/NULL -> default; otherwise look up by name.
  if (model && *model) {
    auto it = e->agents.find(std::string(model));
    if (it != e->agents.end()) return it->second.get();
    return nullptr;
  }
  if (e->default_name.empty()) return nullptr;
  auto it = e->agents.find(e->default_name);
  return it != e->agents.end() ? it->second.get() : nullptr;
}

}  // namespace

extern "C" {

int sj_version(void) { return 100; }  // 0.1.0

sj_engine* sj_engine_new(const char* options_json) {
  auto* e = new sj_engine();
  if (options_json && *options_json) {
    try {
      ordered_json o = ordered_json::parse(options_json);
      if (o.is_object()) {
        if (o.contains("device") && o["device"].is_string())
          e->device = o["device"].get<std::string>();
        if (o.contains("token") && o["token"].is_string())
          e->token = o["token"].get<std::string>();
      }
    } catch (...) {
      // Malformed options: fall back to defaults.
    }
  }
  return e;
}

void sj_engine_free(sj_engine* e) { delete e; }

int sj_load(sj_engine* e, const char* name, const char* model_id_or_path) {
  if (!e || !name || !model_id_or_path) {
    if (e) e->error = "sj_load: null argument";
    return -1;
  }
  try {
    auto agent = std::make_shared<snapjudge::Agent>(std::string(model_id_or_path),
                                                    e->device, e->token);
    std::lock_guard<std::mutex> lk(e->mu);
    e->agents[std::string(name)] = std::move(agent);
    if (e->default_name.empty()) e->default_name = name;
    e->error.clear();
    return 0;
  } catch (const std::exception& ex) {
    std::lock_guard<std::mutex> lk(e->mu);
    e->error = ex.what();
    return -1;
  }
}

char* sj_predict(sj_engine* e, const char* state_json, const char* questions_json,
                 const char* model) {
  if (!e || !state_json || !questions_json) {
    if (e) e->error = "sj_predict: null argument";
    return nullptr;
  }
  try {
    ordered_json state = ordered_json::parse(state_json);
    ordered_json questions = ordered_json::parse(questions_json);
    snapjudge::Agent* a = pick(e, model);
    if (!a) {
      std::lock_guard<std::mutex> lk(e->mu);
      e->error = "sj_predict: no such checkpoint '" + std::string(model ? model : "") + "'";
      return nullptr;
    }
    ordered_json result = a->system_one(state, questions);
    std::lock_guard<std::mutex> lk(e->mu);
    e->error.clear();
    return strdup_owned(result.dump(-1, ' ', false));
  } catch (const std::exception& ex) {
    std::lock_guard<std::mutex> lk(e->mu);
    e->error = ex.what();
    return nullptr;
  }
}

char* sj_route(sj_engine* e, const char* state_json, const char* questions_json) {
  // Routing across arbitrary named checkpoints is a higher-level policy the
  // host owns. The C API exposes prediction; routing stays in the C++ Router.
  (void)state_json;
  (void)questions_json;
  if (e) {
    std::lock_guard<std::mutex> lk(e->mu);
    e->error = "sj_route: not supported at the C API level";
  }
  return nullptr;
}

const char* sj_error_message(sj_engine* e) {
  if (!e) return nullptr;
  return e->error.empty() ? nullptr : e->error.c_str();
}

void sj_string_free(char* s) { std::free(s); }

}  // extern "C"
