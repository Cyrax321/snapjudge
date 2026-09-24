#pragma once
// snapjudge mcp.hpp: MCP stdio server (JSON-RPC 2.0 lines), port of
// mcp/server.py + tools.py + device.py.

#include <string>

namespace snapjudge {

// Run the MCP server loop on stdin/stdout until EOF or exit notification.
// Environment: SNAPJUDGE_DEVICE / SNAPJUDGE_PRELOAD / SNAPJUDGE_MODELS /
// SNAPJUDGE_THREADS (same meaning as snapjudge-serve, default preload list
// "english,multilingual").
int mcp_main_loop();

}  // namespace snapjudge
