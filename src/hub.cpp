#include "snapjudge/hub.hpp"

// snapjudge hub.cpp: HF hub resolution + download.
//
// Cache layout mirrors huggingface_hub so checkpoints downloaded by either
// side are shared:
//   <cache>/models--org--repo/
//     refs/main                      -> <commit sha>
//     blobs/<etag or sha>
//     snapshots/<sha>/<path>         -> symlink to blob (POSIX)
//
// Download path: GET /api/models/{repo} for the file list + revision, filter
// by allow_patterns (same glob-lite as python), fetch each file from
// /resolve/{rev}/{path} (following HF's CDN redirects via libcurl), write
// atomically (tmp + rename), then link into snapshots/<sha>/.

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#ifndef _WIN32
#include <unistd.h>
#endif

#if defined(SNAPJUDGE_HAVE_CURL)
#include <curl/curl.h>
#endif

#include <nlohmann/json.hpp>

namespace snapjudge {

namespace fs = std::filesystem;
using nlohmann::json;

std::string hub_cache_dir() {
  if (const char* p = std::getenv("HF_HUB_CACHE"); p && *p) return p;
  if (const char* p = std::getenv("HF_HOME"); p && *p) return std::string(p) + "/hub";
  if (const char* p = std::getenv("SNAPJUDGE_CACHE_DIR"); p && *p) return p;
  const char* home = std::getenv("HOME");
  return (home ? std::string(home) : std::string(".")) + "/.cache/snapjudge/hub";
}

namespace {

// org/name -> models--org--name
std::string repo_cache_name(const std::string& repo_id) {
  std::string out = "models--";
  for (char c : repo_id) out += (c == '/') ? "--" : std::string(1, c);
  return out;
}

// glob-lite same semantics as python fnmatch for the patterns we use:
// '*' crosses '/', matches any substring (incl. prefix "tokenizer/*").
bool glob_match(const std::string& pat, const std::string& s) {
  size_t p = 0, m = 0, star_p = std::string::npos, star_m = 0;
  while (m < s.size()) {
    if (p < pat.size() && pat[p] == s[m]) { ++p; ++m; }
    else if (p < pat.size() && pat[p] == '*') { star_p = p++; star_m = m; }
    else if (star_p != std::string::npos) { p = star_p + 1; m = ++star_m; }
    else return false;
  }
  while (p < pat.size() && pat[p] == '*') ++p;
  return p == pat.size();
}

bool matches_patterns(const std::string& rel,
                      const std::vector<std::string>& allow_patterns) {
  if (allow_patterns.empty()) return true;
  for (const auto& p : allow_patterns)
    if (glob_match(p, rel)) return true;
  return false;
}

// ---- local cache resolution -------------------------------------------------
std::string cached_snapshot(const std::string& base, const std::string& repo_id) {
  fs::path snap_dir = fs::path(base) / repo_cache_name(repo_id) / "snapshots";
  std::error_code ec;
  if (!fs::exists(snap_dir, ec)) return "";
  // Follow refs/main when present, else newest snapshot dir name wins.
  fs::path ref = fs::path(base) / repo_cache_name(repo_id) / "refs" / "main";
  std::string main_rev;
  {
    std::ifstream f(ref);
    if (f) f >> main_rev;
  }
  fs::path best;
  if (!main_rev.empty() && fs::exists(snap_dir / main_rev, ec)) return (snap_dir / main_rev).string();
  for (const auto& e : fs::directory_iterator(snap_dir, ec))
    if (e.is_directory(ec)) best = e.path();
  return best.empty() ? "" : best.string();
}

#if defined(SNAPJUDGE_HAVE_CURL)
// ---- curl helpers -----------------------------------------------------------
class Curl {
 public:
  CURL* h;
  Curl() {
    h = curl_easy_init();
    if (!h) throw std::runtime_error("snapjudge: curl_easy_init failed");
    curl_easy_setopt(h, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(h, CURLOPT_FAILONERROR, 0L);
    curl_easy_setopt(h, CURLOPT_USERAGENT, "snapjudge-cpp/0.1");
  }
  ~Curl() { curl_easy_cleanup(h); }
};

size_t write_to_string(char* ptr, size_t size, size_t nmemb, void* userdata) {
  static_cast<std::string*>(userdata)->append(ptr, size * nmemb);
  return size * nmemb;
}

size_t write_to_file(char* ptr, size_t size, size_t nmemb, void* userdata) {
  return std::fwrite(ptr, size, nmemb, static_cast<FILE*>(userdata));
}

std::string hf_endpoint() {
  if (const char* p = std::getenv("HF_ENDPOINT"); p && *p) {
    std::string e = p;
    while (!e.empty() && e.back() == '/') e.pop_back();
    return e;
  }
  return "https://huggingface.co";
}

struct RepoInfo {
  std::string sha;
  std::vector<std::string> files;
};

RepoInfo fetch_repo_info(const std::string& repo_id, const std::string& token) {
  Curl c;
  std::string body;
  std::string url = hf_endpoint() + "/api/models/" + repo_id;
  curl_easy_setopt(c.h, CURLOPT_URL, url.c_str());
  curl_easy_setopt(c.h, CURLOPT_WRITEFUNCTION, write_to_string);
  curl_easy_setopt(c.h, CURLOPT_WRITEDATA, &body);
  struct curl_slist* hdrs = nullptr;
  if (!token.empty()) {
    std::string auth = "Authorization: Bearer " + token;
    hdrs = curl_slist_append(hdrs, auth.c_str());
    curl_easy_setopt(c.h, CURLOPT_HTTPHEADER, hdrs);
  }
  CURLcode rc = curl_easy_perform(c.h);
  long code = 0;
  curl_easy_getinfo(c.h, CURLINFO_RESPONSE_CODE, &code);
  if (hdrs) curl_slist_free_all(hdrs);
  if (rc != CURLE_OK)
    throw std::runtime_error("snapjudge: network error fetching " + url + ": " +
                             curl_easy_strerror(rc));
  if (code == 401 || code == 403)
    throw std::runtime_error("snapjudge: hub auth failed for " + repo_id +
                             " (set HF_TOKEN or pass token=)");
  if (code == 404)
    throw std::runtime_error("snapjudge: repo not found on hub: " + repo_id);
  if (code != 200)
    throw std::runtime_error("snapjudge: hub returned " + std::to_string(code) +
                             " for " + url);
  json j;
  try {
    j = json::parse(body);
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("snapjudge: bad hub JSON: ") + e.what());
  }
  RepoInfo info;
  info.sha = j.value("sha", std::string());
  if (info.sha.empty() && j.contains("lastModified")) info.sha = "";  // fallback below
  for (const auto& s : j.value("siblings", json::array())) {
    if (s.contains("rfilename") && s["rfilename"].is_string())
      info.files.push_back(s["rfilename"].get<std::string>());
  }
  return info;
}

void download_file(const std::string& repo_id, const std::string& rev,
                   const std::string& rel, const std::string& dest_tmp,
                   const std::string& token) {
  Curl c;
  std::string url = hf_endpoint() + "/" + repo_id + "/resolve/" + rev + "/" + rel;
  curl_easy_setopt(c.h, CURLOPT_URL, url.c_str());
  FILE* f = std::fopen(dest_tmp.c_str(), "wb");
  if (!f) throw std::runtime_error("snapjudge: cannot write " + dest_tmp + ": " +
                                   std::strerror(errno));
  curl_easy_setopt(c.h, CURLOPT_WRITEFUNCTION, write_to_file);
  curl_easy_setopt(c.h, CURLOPT_WRITEDATA, f);
  struct curl_slist* hdrs = nullptr;
  if (!token.empty()) {
    std::string auth = "Authorization: Bearer " + token;
    hdrs = curl_slist_append(hdrs, auth.c_str());
    curl_easy_setopt(c.h, CURLOPT_HTTPHEADER, hdrs);
  }
  CURLcode rc = curl_easy_perform(c.h);
  long code = 0;
  curl_easy_getinfo(c.h, CURLINFO_RESPONSE_CODE, &code);
  std::fclose(f);
  if (hdrs) curl_slist_free_all(hdrs);
  if (rc != CURLE_OK) {
    std::error_code ec;
    fs::remove(dest_tmp, ec);
    throw std::runtime_error("snapjudge: download failed for " + url + ": " +
                             curl_easy_strerror(rc));
  }
  if (code != 200) {
    std::error_code ec;
    fs::remove(dest_tmp, ec);
    throw std::runtime_error("snapjudge: download of " + rel + " returned HTTP " +
                             std::to_string(code));
  }
}
#endif

}  // namespace

std::string resolve_checkpoint(const std::string& repo_id,
                               const std::vector<std::string>& allow_patterns,
                               const std::string& token) {
  if (fs::exists(repo_id)) return repo_id;

  if (repo_id.rfind("/", 0) == 0 || repo_id.rfind("./", 0) == 0 ||
      repo_id.rfind("../", 0) == 0) {
    throw std::invalid_argument(
        "Local model path not found: '" + repo_id +
        "'. Check that the directory exists and that training saved the model "
        "successfully.");
  }

  // 1. Local caches first: HF's own cache, then snapjudge's.
  {
    std::string hf_hub;
    if (const char* p = std::getenv("HF_HUB_CACHE"); p && *p) hf_hub = p;
    else if (const char* p = std::getenv("HF_HOME"); p && *p) hf_hub = std::string(p) + "/hub";
    else if (const char* p = std::getenv("HOME"); p) hf_hub = std::string(p) + "/.cache/huggingface/hub";
    for (const std::string& base : {hf_hub, hub_cache_dir()}) {
      std::string hit = cached_snapshot(base, repo_id);
      if (!hit.empty()) return hit;
    }
  }

#if !defined(SNAPJUDGE_HAVE_CURL)
  (void)allow_patterns;
  (void)token;
  throw std::runtime_error(
      "snapjudge: " + repo_id +
      " is not cached and this build has no libcurl. Rebuild with CURL or "
      "pre-download with python huggingface_hub.");
#else
  // 2. Download into the snapjudge cache.
  fprintf(stderr, "snapjudge: downloading %s ...\n", repo_id.c_str());
  RepoInfo info = fetch_repo_info(repo_id, token);
  std::string rev = info.sha.empty() ? "main" : info.sha;
  fs::path root = fs::path(hub_cache_dir()) / repo_cache_name(repo_id);
  fs::path blobs = root / "blobs";
  fs::path snap = root / "snapshots" / rev;
  fs::create_directories(blobs);
  fs::create_directories(snap);

  bool any = false;
  for (const auto& rel : info.files) {
    if (!matches_patterns(rel, allow_patterns)) continue;
    any = true;
    fs::path final_blob = blobs / rel;  // readable name; unique via rel
    if (!fs::exists(final_blob)) {
      fs::path tmp = blobs / (rel + ".tmp");
      if (fs::exists(tmp)) fs::remove(tmp);
      download_file(repo_id, rev, rel, tmp.string(), token);
      fs::path parent = final_blob.parent_path();
      fs::create_directories(parent);
      fs::rename(tmp, final_blob);
    }
    // link into the snapshot
    fs::path link = snap / rel;
    fs::create_directories(link.parent_path());
    std::error_code ec;
    fs::remove(link, ec);
#ifndef _WIN32
    ::symlink(final_blob.c_str(), link.c_str());
#else
    fs::copy_file(final_blob, link);
#endif
  }
  if (!any)
    throw std::runtime_error("snapjudge: no files of repo " + repo_id +
                             " matched the requested patterns");

  // refs/main
  if (!info.sha.empty()) {
    fs::create_directories(root / "refs");
    std::ofstream(root / "refs" / "main") << info.sha;
  }
  return snap.string();
#endif
}

}  // namespace snapjudge
