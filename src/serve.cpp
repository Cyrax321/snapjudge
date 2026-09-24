#include "snapjudge/serve.hpp"

// serve.py semantics in C++:
//   GET  /health                 -> {status, loaded, device}
//   POST /v1/systemone           -> Jev-compatible decision payload
//   Bearer auth when SNAPJUDGE_API_KEY is set; unknown model ids fall back to
//   auto-routing; errors -> 422 {detail}; single-worker inference queue.

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_set>

#include "httplib.h"
#include "snapjudge/router.hpp"

namespace snapjudge {

using nlohmann::ordered_json;

namespace {

bool env_bool(const char* name, bool def) {
  const char* v = std::getenv(name);
  if (!v) return def;
  std::string s = v;
  for (auto& c : s) c = std::tolower(static_cast<unsigned char>(c));
  return s == "1" || s == "true" || s == "yes" || s == "on";
}

std::vector<std::string> env_list(const char* name) {
  const char* v = std::getenv(name);
  if (!v || !*v) return {};
  std::vector<std::string> out;
  std::string s = v, cur;
  while (true) {
    size_t pos = s.find(',');
    if (pos == std::string::npos) {
      cur = s;
      s.clear();
    } else {
      cur = s.substr(0, pos);
      s = s.substr(pos + 1);
    }
    size_t a = cur.find_first_not_of(" \t");
    size_t b = cur.find_last_not_of(" \t");
    if (a != std::string::npos) out.push_back(cur.substr(a, b - a + 1));
    if (s.empty()) break;
  }
  return out;
}

const std::unordered_set<std::string>& known_models() {
  static const std::unordered_set<std::string> m = {"english", "multilingual",
                                                    "typed-decisions"};
  return m;
}

const std::unordered_map<std::string, std::string>& published_ids() {
  static const std::unordered_map<std::string, std::string> m = {
      {"snapjudge/multilingual", "multilingual"},
      {"snapjudge/typed-decisions", "typed-decisions"},
  };
  return m;
}

// _resolve_model from serve.py
std::string resolve_model(const ordered_json& body) {
  if (!body.contains("model") || !body["model"].is_string()) return "";
  std::string model = body["model"].get<std::string>();
  {
    // strip + lower
    size_t a = model.find_first_not_of(" \t");
    size_t b = model.find_last_not_of(" \t");
    if (a == std::string::npos) return "";
    model = model.substr(a, b - a + 1);
    for (auto& c : model) c = std::tolower(static_cast<unsigned char>(c));
  }
  auto pit = published_ids().find(model);
  if (pit != published_ids().end()) return pit->second;
  try {
    std::string key = normalise_name(model);
    return known_models().count(key) ? key : "";
  } catch (...) {
    return "";
  }
}

}  // namespace

std::unique_ptr<Router> build_router_from_env() {
  Router::Options o;
  if (const char* d = std::getenv("SNAPJUDGE_DEVICE"); d && *d) o.device = d;
  o.auto_task_detection = env_bool("SNAPJUDGE_AUTO_TASK", false);
  auto out = std::make_unique<Router>(o);
  if (env_bool("SNAPJUDGE_PRELOAD", true)) out->preload(env_list("SNAPJUDGE_MODELS"));
  return out;
}

void serve(std::unique_ptr<Router> router) {
  httplib::Server svr;

  std::string api_key = std::getenv("SNAPJUDGE_API_KEY")
                            ? std::getenv("SNAPJUDGE_API_KEY")
                            : "";

  // One inference at a time (serve.py's ThreadPoolExecutor(1) + asyncio.Lock).
  std::mutex infer_mu;

  svr.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
    ordered_json body = {{"status", "ok"},
                         {"loaded", router->loaded()},
                         {"device", std::getenv("SNAPJUDGE_DEVICE")
                                        ? std::string(std::getenv("SNAPJUDGE_DEVICE"))
                                        : "auto"}};
    res.set_content(body.dump(2, ' ', false), "application/json");
  });

  svr.Post("/v1/systemone", [&](const httplib::Request& req, httplib::Response& res) {
    if (!api_key.empty()) {
      std::string auth = req.get_header_value("Authorization");
      if (auth != "Bearer " + api_key) {
        ordered_json e = {{"detail", "invalid or missing bearer token"}};
        res.status = 401;
        res.set_content(e.dump(), "application/json");
        return;
      }
    }
    ordered_json body;
    try {
      body = ordered_json::parse(req.body);
    } catch (...) {
      res.status = 400;
      res.set_content(ordered_json{{"detail", "request body must be an object with a 'questions' field"}}.dump(),
                      "application/json");
      return;
    }
    if (!body.is_object() || !body.contains("questions")) {
      res.status = 400;
      res.set_content(ordered_json{{"detail", "request body must be an object with a 'questions' field"}}.dump(),
                      "application/json");
      return;
    }
    ordered_json state = body.contains("state") ? body["state"] : ordered_json();
    std::string model = resolve_model(body);
    try {
      std::lock_guard<std::mutex> lk(infer_mu);
      ordered_json result = router->predict(state, body["questions"], model);
      res.set_content(result.dump(-1, ' ', false), "application/json");
    } catch (const std::exception& e) {
      res.status = 422;
      res.set_content(ordered_json{{"detail", e.what()}}.dump(), "application/json");
    }
  });

  std::string host = std::getenv("SNAPJUDGE_HOST") ? std::getenv("SNAPJUDGE_HOST")
                                                   : "0.0.0.0";
  int port = std::getenv("SNAPJUDGE_PORT") ? std::atoi(std::getenv("SNAPJUDGE_PORT"))
                                           : 8000;
  fprintf(stderr, "snapjudge-serve: listening on %s:%d\n", host.c_str(), port);
  if (!svr.listen(host, port)) {
    fprintf(stderr, "snapjudge-serve: failed to bind %s:%d\n", host.c_str(), port);
  }
}

}  // namespace snapjudge
