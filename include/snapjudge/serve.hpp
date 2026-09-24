#pragma once
// snapjudge serve.hpp: HTTP server over the TypeSafe Jev /v1/systemone wire
// protocol, port of serve.py. Built on vendored cpp-httplib.

#include <memory>
#include <string>

#include <nlohmann/json.hpp>

namespace snapjudge {

class Router;

// Build a Router from SNAPJUDGE_DEVICE / SNAPJUDGE_AUTO_TASK /
// SNAPJUDGE_PRELOAD / SNAPJUDGE_MODELS env (same contract as serve.py's
// SNAPJUDGE_* vars). Returned router is caller-owned.
std::unique_ptr<Router> build_router_from_env();

// Blocking server; reads SNAPJUDGE_HOST / SNAPJUDGE_PORT / SNAPJUDGE_API_KEY /
// SNAPJUDGE_THREADS / SNAPJUDGE_LOG_LEVEL. Returns after close().
void serve(std::unique_ptr<Router> router);

}  // namespace snapjudge
