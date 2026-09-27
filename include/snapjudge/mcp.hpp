#pragma once
// snapjudge mcp.hpp: Model Context Protocol server over stdio.
//
// Implements the MCP (JSON-RPC 2.0, newline-delimited over stdin/stdout)
// transport with a tools capability. The server exposes four tools:
//   snapjudge_status   — device/loaded-checkpoint report
//   snapjudge_route    — routing decision without a forward pass
//   snapjudge_predict  — typed questions answered in one forward pass
//   snapjudge_preset   — run a built-in question workflow
//
// Environment: SNAPJUDGE_DEVICE / SNAPJUDGE_PRELOAD / SNAPJUDGE_MODELS /
// SNAPJUDGE_THREADS (same meaning as snapjudge-serve, default preload list
// "english,multilingual").

#include <string>

namespace snapjudge {

// Run the MCP server loop on stdin/stdout until EOF. Returns a process exit
// code (0 on clean shutdown).
int mcp_main_loop();

}  // namespace snapjudge
