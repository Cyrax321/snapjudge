#include "snapjudge/mcp.hpp"

// Model Context Protocol over stdio: newline-delimited JSON-RPC 2.0 messages.
//
// Lifecycle follows the MCP spec: an `initialize` request is answered with the
// negotiated protocol version and a `tools` capability; the client then sends
// `notifications/initialized`; `tools/list` and `tools/call` drive the actual
// work. Errors on tool calls are returned as structured results (the tool's own
// `{error, message}` payload) rather than JSON-RPC errors, so the caller can
// surface them per-tool.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <nlohmann/json.hpp>

#include "snapjudge/agent.hpp"
#include "snapjudge/presets.hpp"
#include "snapjudge/router.hpp"

namespace snapjudge {

using nlohmann::ordered_json;

namespace {

// A user-facing tool error: carried as a structured result, not a protocol
// error, so a single bad argument doesn't look like a transport failure.
struct ToolError : std::exception {
  std::string code, message;
  ToolError(std::string c, std::string m) : code(std::move(c)), message(std::move(m)) {}
  const char* what() const noexcept override { return message.c_str(); }
};

const std::unordered_set<std::string> VALID_TYPES = {"choice", "score", "noul"};
const std::unordered_set<std::string> VALID_MODELS = {"auto", "english", "multilingual",
                                                      "typed-decisions"};

// ---------------------------------------------------------------------------
// Preset workflow registry (name -> builder). Adding a preset only needs an
// entry here plus the builder in presets.cpp.
// ---------------------------------------------------------------------------
const std::unordered_map<std::string, ordered_json (*)()>& preset_builders() {
  static const std::unordered_map<std::string, ordered_json (*)()> m = {
      {"guard", guard_questions},
      {"moderation", moderation_questions},
      {"triage", triage_questions},
      {"model_router", router_questions},
  };
  return m;
}

// ---------------------------------------------------------------------------
// Argument validation. Each tool re-validates its inputs and returns a clean,
// specific error instead of a generic "bad request".
// ---------------------------------------------------------------------------
ordered_json validate_state(const ordered_json& state) {
  if (!state.is_object() || state.empty())
    throw ToolError("invalid_state", "state must be a non-empty JSON object");
  return state;
}

ordered_json validate_questions(const ordered_json& questions) {
  if (!questions.is_object() || questions.empty())
    throw ToolError("invalid_questions",
                    "questions must be a non-empty JSON object keyed by question name");
  ordered_json cleaned = ordered_json::object();
  for (auto it = questions.begin(); it != questions.end(); ++it) {
    const std::string& name = it.key();
    if (name.empty())
      throw ToolError("invalid_questions", "question name must be a non-empty string: " + name);
    const ordered_json& spec = it.value();
    if (!spec.is_object())
      throw ToolError("invalid_questions", "questions[" + name + "] must be an object");
    std::string qtype = spec.value("type", "");
    if (!VALID_TYPES.count(qtype))
      throw ToolError("invalid_questions",
                      "questions[" + name +
                          "].type must be one of [choice, noul, score], got '" + qtype + "'");
    if (!spec.contains("instructions") || !spec["instructions"].is_string() ||
        spec["instructions"].get<std::string>().empty())
      throw ToolError("invalid_questions",
                      "questions[" + name + "].instructions must be a non-empty string");
    ordered_json entry = {{"type", qtype}, {"instructions", spec["instructions"]}};
    if (spec.contains("criteria") && !spec["criteria"].is_null()) {
      const ordered_json& crit = spec["criteria"];
      if (qtype == "choice") {
        if (!crit.is_object() || crit.empty())
          throw ToolError("invalid_questions",
                          "questions[" + name +
                              "].criteria must be a non-empty object of label -> description");
        ordered_json c2 = ordered_json::object();
        for (auto it2 = crit.begin(); it2 != crit.end(); ++it2) {
          const ordered_json& cv = it2.value();
          c2[it2.key()] = cv.is_string() ? ordered_json(cv) : ordered_json(cv.dump(-1, ' ', false));
        }
        entry["criteria"] = c2;
      } else if (qtype == "score") {
        if (!crit.is_array() || crit.empty())
          throw ToolError("invalid_questions",
                          "questions[" + name +
                              "].criteria must be a non-empty list of rubric levels");
        ordered_json c2 = ordered_json::array();
        for (const auto& x : crit)
          c2.push_back(x.is_string() ? x.get<std::string>() : x.dump(-1, ' ', false));
        entry["criteria"] = c2;
      } else {
        if (!crit.is_object())
          throw ToolError("invalid_questions",
                          "questions[" + name +
                              "].criteria must be an object when present (noul)");
        ordered_json c2 = ordered_json::object();
        for (auto it2 = crit.begin(); it2 != crit.end(); ++it2) {
          const ordered_json& cv = it2.value();
          c2[it2.key()] = cv.is_string() ? ordered_json(cv) : ordered_json(cv.dump(-1, ' ', false));
        }
        entry["criteria"] = c2;
      }
    }
    cleaned[name] = entry;
  }
  return cleaned;
}

std::string validate_model(const ordered_json& model) {
  if (model.is_null()) return "auto";
  std::string m = model.is_string() ? model.get<std::string>() : "";
  if (!VALID_MODELS.count(m))
    throw ToolError("invalid_model",
                    "model must be one of [auto, english, multilingual, typed-decisions], got '" +
                        m + "'");
  return m;
}

// ---------------------------------------------------------------------------
// Lazy router construction, shared across all tools.
// ---------------------------------------------------------------------------
std::unique_ptr<Router> g_router;
std::mutex g_router_mu;

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
    cur = pos == std::string::npos ? s : s.substr(0, pos);
    size_t a = cur.find_first_not_of(" \t");
    size_t b = cur.find_last_not_of(" \t");
    if (a != std::string::npos) out.push_back(cur.substr(a, b - a + 1));
    if (pos == std::string::npos) break;
    s = s.substr(pos + 1);
  }
  return out;
}

Router& ensure_router() {
  std::lock_guard<std::mutex> lk(g_router_mu);
  if (!g_router) {
    Router::Options o;
    if (const char* d = std::getenv("SNAPJUDGE_DEVICE"); d && *d) o.device = d;
    g_router = std::make_unique<Router>(o);
    if (env_bool("SNAPJUDGE_PRELOAD", true)) {
      std::vector<std::string> names = {"english", "multilingual"};
      const char* m = std::getenv("SNAPJUDGE_MODELS");
      if (m && *m) names = env_list("SNAPJUDGE_MODELS");
      try {
        g_router->preload(names);
      } catch (const std::exception& e) {
        g_router.reset();
        throw ToolError("internal_error", std::string("router construction failed: ") + e.what());
      }
    }
  }
  return *g_router;
}

double round2(double v) { return std::round(v * 100.0) / 100.0; }

// ---------------------------------------------------------------------------
// Tool implementations.
// ---------------------------------------------------------------------------
ordered_json tool_status() {
  ordered_json ckpt = ordered_json::object();
  if (g_router) {
    for (const auto& n : g_router->loaded()) {
      Agent* a = g_router->agent_for(n);
      if (a) ckpt[n] = a->device();
    }
  }
  std::string actual = ckpt.empty() ? "" : ckpt.begin().value().get<std::string>();
  std::string configured = std::getenv("SNAPJUDGE_DEVICE") ? std::getenv("SNAPJUDGE_DEVICE")
                                                           : "auto";
  return ordered_json{
      {"device", actual.empty() ? configured : actual},
      {"device_is_preference", actual.empty()},
      {"checkpoint_devices", ckpt},
      {"loaded", g_router ? ordered_json(g_router->loaded()) : ordered_json::array()},
      {"router_preload", env_bool("SNAPJUDGE_PRELOAD", true)},
      {"router_ready", g_router != nullptr},
      {"package_versions", {{"snapjudge", "0.1.0"}}},
  };
}

ordered_json tool_predict(const ordered_json& state, const ordered_json& questions,
                          const ordered_json& model) {
  ordered_json state_d = validate_state(state);
  ordered_json questions_d = validate_questions(questions);
  std::string model_name = validate_model(model);
  Router& r = ensure_router();

  auto t0 = std::chrono::steady_clock::now();
  ordered_json result = model_name == "auto" ? r.predict(state_d, questions_d)
                                             : r.predict(state_d, questions_d, model_name);
  double latency_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

  ordered_json routing = result.contains("routing")
                             ? result["routing"]
                             : ordered_json{{"model", model_name},
                                            {"repo", nullptr},
                                            {"reason", "explicit model"}};
  ordered_json out = {{"answers", result["answers"]},
                      {"routing", routing},
                      {"latency_ms", round2(latency_ms)}};
  if (!routing.is_null() && routing.contains("model") && routing["model"].is_string()) {
    Agent* a = g_router ? g_router->agent_for(routing["model"].get<std::string>()) : nullptr;
    if (a) out["device"] = a->device();
  }
  return out;
}

ordered_json tool_route(const ordered_json& state, const ordered_json& questions) {
  ordered_json state_d = validate_state(state);
  ordered_json questions_d = validate_questions(questions);
  Router& r = ensure_router();
  ordered_json d = r.route(state_d, questions_d);
  return ordered_json{{"model", d["model"]}, {"repo", d["repo"]}, {"reason", d["reason"]}};
}

ordered_json tool_preset(const ordered_json& preset, const ordered_json& state) {
  if (!preset.is_string()) throw ToolError("invalid_preset", "preset must be a string");
  std::string p = preset.get<std::string>();
  auto it = preset_builders().find(p);
  if (it == preset_builders().end())
    throw ToolError("invalid_preset",
                    "preset must be one of [guard, model_router, moderation, triage], got '" + p +
                        "'");
  ordered_json state_d = validate_state(state);
  ordered_json questions = it->second();
  Router& r = ensure_router();
  auto t0 = std::chrono::steady_clock::now();
  ordered_json result = r.predict(state_d, questions);
  double latency_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  ordered_json routing = result.value("routing", ordered_json());
  ordered_json out = {{"answers", result["answers"]},
                      {"routing", routing},
                      {"latency_ms", round2(latency_ms)}};
  if (routing.contains("model") && routing["model"].is_string()) {
    Agent* a = g_router ? g_router->agent_for(routing["model"].get<std::string>()) : nullptr;
    if (a) out["device"] = a->device();
  }
  return out;
}

// ---------------------------------------------------------------------------
// Tool schemas for tools/list.
// ---------------------------------------------------------------------------
const char* kUsage = " Structured decisions only: choice (finite labels), score "
                     "(ordinal rubric), noul (calibrated P(true)). No text generation, "
                     "so nothing to parse and nothing to hallucinate.";

ordered_json tool_list() {
  ordered_json arr = ordered_json::array();
  arr.push_back({{"name", "snapjudge_status"},
                 {"description", "Report the device actually in use per loaded checkpoint (or the "
                                 "configured preference when nothing is loaded), loaded "
                                 "checkpoints, and package versions."},
                 {"inputSchema", {{"type", "object"}, {"properties", ordered_json::object()}}}});
  arr.push_back({{"name", "snapjudge_route"},
                 {"description",
                  std::string("Decide which checkpoint would answer, without running a forward "
                              "pass. Use to explain routing (english vs multilingual vs "
                              "typed-decisions) to the user.") +
                      kUsage},
                 {"inputSchema", {{"type", "object"},
                                  {"properties", {{"state", {{"type", "object"}}},
                                                  {"questions", {{"type", "object"}}}}},
                                  {"required", {"state", "questions"}}}}});
  arr.push_back({{"name", "snapjudge_predict"},
                 {"description",
                  std::string("Answer typed questions (choice/score/noul) over any state in one "
                              "forward pass. Returns answers with confidence, routing metadata and "
                              "the device of the checkpoint that answered.") +
                      kUsage},
                 {"inputSchema", {{"type", "object"},
                                  {"properties", {{"state", {{"type", "object"}}},
                                                  {"questions", {{"type", "object"}}},
                                                  {"model", {{"type", "string"},
                                                             {"default", "auto"}}}}},
                                  {"required", {"state", "questions"}}}}});
  arr.push_back({{"name", "snapjudge_preset"},
                 {"description",
                  std::string("Run a built-in workflow: 'guard' | 'moderation' | 'triage' | "
                              "'model_router'. Use when the task matches one of those presets "
                              "instead of hand-writing questions.") +
                      kUsage},
                 {"inputSchema", {{"type", "object"},
                                  {"properties", {{"preset", {{"type", "string"}}},
                                                  {"state", {{"type", "object"}}}}},
                                  {"required", {"preset", "state"}}}}});
  return arr;
}

// A successful tool result: text content plus the structured payload.
ordered_json result_text(const ordered_json& payload) {
  return ordered_json{
      {"content", ordered_json::array({ordered_json{{"type", "text"},
                                                    {"text", payload.dump(2, ' ', false)}}})},
      {"isError", false}};
}

ordered_json error_text(const std::string& code, const std::string& message) {
  ordered_json res = result_text(ordered_json{{"error", code}, {"message", message}});
  res["isError"] = true;
  return res;
}

}  // namespace

int mcp_main_loop() {
  if (env_bool("SNAPJUDGE_PRELOAD", true)) {
    try {
      ensure_router();
    } catch (const std::exception& e) {
      fprintf(stderr, "[snapjudge-mcp] preload failed (will retry on demand): %s\n", e.what());
    }
  }

  std::string line;
  while (std::getline(std::cin, line)) {
    while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
    if (line.empty()) continue;

    ordered_json req;
    try {
      req = ordered_json::parse(line);
    } catch (...) {
      continue;  // not a JSON message — ignore (protocol has no recovery for this)
    }
    if (!req.is_object()) continue;

    std::string method = req.value("method", "");
    bool is_notification = !req.contains("id");
    ordered_json id = req.value("id", ordered_json());
    ordered_json resp;
    bool respond = true;

    try {
      if (method == "initialize") {
        ordered_json out = {{"protocolVersion",
                             req.value("params", ordered_json::object())
                                 .value("protocolVersion", "2024-11-05")},
                            {"capabilities", {{"tools", ordered_json::object()}}},
                            {"serverInfo", {{"name", "snapjudge"}, {"version", "0.1.0"}}}};
        resp = {{"jsonrpc", "2.0"}, {"id", id}, {"result", out}};
      } else if (method == "notifications/initialized" || method == "initialized") {
        respond = false;
      } else if (method == "ping") {
        resp = {{"jsonrpc", "2.0"}, {"id", id}, {"result", ordered_json::object()}};
      } else if (method == "tools/list") {
        resp = {{"jsonrpc", "2.0"}, {"id", id}, {"result", {{"tools", tool_list()}}}};
      } else if (method == "tools/call") {
        const ordered_json& params = req["params"];
        std::string tool = params.value("name", "");
        const ordered_json args = params.value("arguments", ordered_json::object());
        ordered_json result;
        if (tool == "snapjudge_status") {
          result = result_text(tool_status());
        } else if (tool == "snapjudge_route") {
          result = result_text(
              tool_route(args.value("state", ordered_json()), args.value("questions", ordered_json())));
        } else if (tool == "snapjudge_predict") {
          result = result_text(tool_predict(args.value("state", ordered_json()),
                                            args.value("questions", ordered_json()),
                                            args.value("model", ordered_json())));
        } else if (tool == "snapjudge_preset") {
          result = result_text(
              tool_preset(args.value("preset", ordered_json()), args.value("state", ordered_json())));
        } else {
          resp = {{"jsonrpc", "2.0"},
                  {"id", id},
                  {"error", {{"code", -32602}, {"message", "unknown tool: " + tool}}}};
          std::cout << resp.dump() << "\n" << std::flush;
          continue;
        }
        resp = {{"jsonrpc", "2.0"}, {"id", id}, {"result", result}};
      } else {
        if (is_notification) continue;
        resp = {{"jsonrpc", "2.0"},
                {"id", id},
                {"error", {{"code", -32601}, {"message", "method not found: " + method}}}};
      }
    } catch (const ToolError& e) {
      resp = {{"jsonrpc", "2.0"}, {"id", id}, {"result", error_text(e.code, e.message)}};
    } catch (const std::exception& e) {
      resp = {{"jsonrpc", "2.0"},
              {"id", id},
              {"result", error_text("internal_error", std::string("RuntimeError: ") + e.what())}};
    }
    if (respond) std::cout << resp.dump() << "\n" << std::flush;
  }
  return 0;
}

}  // namespace snapjudge
