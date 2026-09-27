#pragma once
// snapjudge hub.hpp: HF hub resolution + download into a HF-compatible cache.
// Local directories pass through unchanged.

#include <string>
#include <vector>

namespace snapjudge {

// The HF cache directory snapjudge reads/writes. Respects HF_HOME, then
// HF_HUB_CACHE, then SNAPJUDGE_CACHE_DIR, else ~/.cache/huggingface/hub.
std::string hub_cache_dir();

// Resolve repo_id to a local snapshot directory. If `repo_id` is an existing
// local path it is returned unchanged. Otherwise the repo is downloaded into
// the hub cache (filtered by allow_patterns, subfolder prefix applied by
// callers) and the snapshot path returned. `token` overrides HF_TOKEN.
// Throws std::runtime_error on network/auth errors; std::invalid_argument
// when a local-looking path does not exist.
std::string resolve_checkpoint(const std::string& repo_id,
                               const std::vector<std::string>& allow_patterns,
                               const std::string& token = "");

}  // namespace snapjudge
