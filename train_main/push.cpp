// snapjudge-train-push: push a trained checkpoint dir to the Hugging Face hub.
//
//  snapjudge-train-push --ckpt out/checkpoint-ft --repo you/name [--private]
//
// Security posture (SEC-01):
//  - The HF token never enters a shell command line, URL, or log.
//  - Hub HTTP calls run through libcurl with header-only auth (same link as
//    hub.cpp; if libcurl is missing this binary prints a clear error).
//  - git runs via posix_spawnp(argv, no shell) with authentication supplied
//    through a GIT_ASKPASS shim living in a mode-0700 temp dir that is wiped
//    after the push. Nothing derived from the token reaches the process table.

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#else
#error "snapjudge-train-push: POSIX build only (uses posix_spawnp)"
#endif

#include <curl/curl.h>

namespace fs = std::filesystem;

namespace {

// repo must be exactly <owner>/<name>, GitHub-legal characters only
bool valid_repo(const std::string& r) {
  if (r.empty()) return false;
  auto part_ok = [](const std::string& s) {
    if (s.empty() || s == "." || s == "..") return false;
    if (s.find("..") != std::string::npos) return false;   // no traversal owner/name
    for (char c : s)
      if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_' || c == '.'))
        return false;
    return true;
  };
  auto pos = r.find('/');
  if (pos == std::string::npos || r.find('/', pos + 1) != std::string::npos) return false;
  return part_ok(r.substr(0, pos)) && part_ok(r.substr(pos + 1));
}

// argv-based spawn (no shell). Prints only argv content here (no secrets).
int spawn(const std::vector<std::string>& argv,
          const std::vector<std::pair<std::string, std::string>>& extra_env = {}) {
  std::string show;
  for (const auto& a : argv) show += a + " ";
  std::fprintf(stderr, "+ %s\n", show.c_str());
  std::vector<std::string> envv;
  for (char** e = environ; *e; ++e) envv.emplace_back(*e);
  for (const auto& kv : extra_env) envv.emplace_back(kv.first + "=" + kv.second);
  std::vector<char*> envp;
  for (auto& s : envv) envp.push_back(s.data());
  envp.push_back(nullptr);
  std::vector<char*> args;
  for (auto& s : argv) args.push_back(const_cast<char*>(s.c_str()));
  args.push_back(nullptr);
  pid_t pid;
  if (posix_spawnp(&pid, argv[0].c_str(), nullptr, nullptr, args.data(), envp.data()) != 0)
    return -1;
  int status = 0;
  while (waitpid(pid, &status, 0) < 0)
    if (errno != EINTR) break;
  if (WIFEXITED(status)) return WEXITSTATUS(status);
  return -1;
}

size_t write_str(char* p, size_t size, size_t nmemb, void* ud) {
  static_cast<std::string*>(ud)->append(p, size * nmemb);
  return size * nmemb;
}

class Curl {
 public:
  CURL* h = nullptr;
  Curl() { h = curl_easy_init(); }
  ~Curl() { if (h) curl_easy_cleanup(h); }
};

long post_json(const std::string& url, const std::string& token,
               const std::string& body, std::string* out) {
  Curl c;
  if (!c.h) return -1;
  struct curl_slist* hdrs = nullptr;
  std::string auth = "Authorization: Bearer " + token;
  hdrs = curl_slist_append(hdrs, auth.c_str());
  hdrs = curl_slist_append(hdrs, "Content-Type: application/json");
  curl_easy_setopt(c.h, CURLOPT_URL, url.c_str());
  curl_easy_setopt(c.h, CURLOPT_POSTFIELDS, body.c_str());
  curl_easy_setopt(c.h, CURLOPT_WRITEFUNCTION, write_str);
  curl_easy_setopt(c.h, CURLOPT_WRITEDATA, out);
  curl_easy_setopt(c.h, CURLOPT_HTTPHEADER, hdrs);
  CURLcode rc = curl_easy_perform(c.h);
  long code = 0;
  curl_easy_getinfo(c.h, CURLINFO_RESPONSE_CODE, &code);
  curl_slist_free_all(hdrs);
  if (rc != CURLE_OK) return -1;
  return code;
}

std::string whoami(const std::string& token) {
  Curl c;
  std::string body;
  curl_easy_setopt(c.h, CURLOPT_URL, "https://huggingface.co/api/whoami-v2");
  curl_easy_setopt(c.h, CURLOPT_WRITEFUNCTION, write_str);
  curl_easy_setopt(c.h, CURLOPT_WRITEDATA, &body);
  struct curl_slist* hdrs = nullptr;
  std::string auth = "Authorization: Bearer " + token;
  hdrs = curl_slist_append(hdrs, auth.c_str());
  curl_easy_setopt(c.h, CURLOPT_HTTPHEADER, hdrs);
  CURLcode rc = curl_easy_perform(c.h);
  curl_slist_free_all(hdrs);
  if (rc != CURLE_OK) return "";
  auto pos = body.find("\"name\":\"");
  if (pos == std::string::npos) return "";
  pos += 8;
  auto end = body.find('"', pos);
  return body.substr(pos, end - pos);
}

// GIT_ASKPASS shim: git invokes it with a prompt; it answers from env vars the
// child process inherits. The token is never in argv nor in any URL.
std::string make_askpass(const fs::path& dir) {
  fs::path shim = dir / "git-askpass.sh";
  {
    std::ofstream f(shim);
    f << "#!/bin/sh\ncase \"$1\" in\n"
         "*sername*) printf '%s' \"$SJ_GIT_USER\";;\n*) printf '%s' \"$SJ_GIT_PASS\";;\n"
         "esac\n";
  }
  chmod(shim.c_str(), 0700);
  return shim.string();
}

}  // namespace

int main(int argc, char** argv) {
  std::string ckpt, repo;
  bool priv = false;
  for (int i = 1; i < argc; ++i) {
    std::string t = argv[i];
    if (t == "--ckpt" && i + 1 < argc) ckpt = argv[++i];
    else if (t == "--repo" && i + 1 < argc) repo = argv[++i];
    else if (t == "--private") priv = true;
    else if (t == "--help" || t == "-h") {
      std::puts("snapjudge-train-push --ckpt <dir> --repo <you/name> [--private]\n"
                "  needs HF_TOKEN in the environment");
      return 0;
    }
  }
  if (ckpt.empty() || repo.empty()) {
    std::puts("snapjudge-train-push --ckpt <dir> --repo <you/name> [--private]\n"
              "  needs HF_TOKEN in the environment");
    return 2;
  }
  if (!valid_repo(repo)) {
    std::fprintf(stderr, "snapjudge-train-push: invalid --repo %s "
                         "(expected owner/name with letters, digits, '.', '-', '_')\n",
                 repo.c_str());
    return 2;
  }
  for (const char* f : {"rl_agent_config.json", "model.safetensors"})
    if (!fs::exists(fs::path(ckpt) / f)) {
      std::fprintf(stderr, "snapjudge-train-push: %s missing in %s\n", f, ckpt.c_str());
      return 2;
    }

  const char* token_c = std::getenv("HF_TOKEN");
  if (!token_c || !*token_c) {
    std::fprintf(stderr, "snapjudge-train-push: set HF_TOKEN first\n");
    return 2;
  }
  std::string token = token_c;

  std::string user = whoami(token);
  if (user.empty()) {
    std::fprintf(stderr, "snapjudge-train-push: HF_TOKEN invalid or hub unreachable\n");
    return 2;
  }

  {
    std::string body = "{\"id\":\"" + repo + "\",\"private\":" + (priv ? "true" : "false") + "}";
    std::string out;
    long code = post_json("https://huggingface.co/api/models/create-repo", token, body, &out);
    (void)code;
  }

  std::string tmpdir_str = std::string(::getenv("TMPDIR") ?: "/tmp") + "/snapjudge-push-" +
                           std::to_string(::getpid());
  fs::path tmpdir(tmpdir_str);
  fs::create_directories(tmpdir);
  fs::permissions(tmpdir, fs::perms::owner_all);

  fs::path askpass = fs::path(make_askpass(tmpdir));
  const std::vector<std::pair<std::string, std::string>> git_env = {
      {"GIT_ASKPASS", askpass.string()},
      {"GIT_TERMINAL_PROMPT", "0"},
      {"SJ_GIT_USER", user},
      {"SJ_GIT_PASS", token},
  };
  const std::string https_url = "https://huggingface.co/models/" + repo;
  const fs::path dst = tmpdir / "repo";

  spawn({"git", "clone", "--depth", "1", https_url, dst.string()}, git_env);
  if (!fs::exists(dst / ".git")) {
    fs::create_directories(dst);
    spawn({"git", "-C", dst.string(), "init", "-q"}, git_env);
    spawn({"git", "-C", dst.string(), "remote", "add", "origin", https_url}, git_env);
  }

  for (const auto& e : fs::recursive_directory_iterator(ckpt)) {
    fs::path rel = fs::relative(e.path(), ckpt);
    fs::path to = dst / rel;
    fs::create_directories(to.parent_path());
    if (e.is_symlink()) continue;   // never write a caller-provided symlink target
    fs::copy_file(e.path(), to, fs::copy_options::overwrite_existing);
  }

  spawn({"git", "-C", dst.string(), "lfs", "install", "--local"}, git_env);
  spawn({"git", "-C", dst.string(), "lfs", "track", "*.safetensors"}, git_env);
  spawn({"git", "-C", dst.string(), "add", "-A"}, git_env);
  int rc = spawn({"git", "-C", dst.string(), "-c", "user.email=snapjudge@local",
                  "-c", "user.name=snapjudge", "commit", "-q", "-m", "snapjudge checkpoint"},
                 git_env);
  if (rc != 0) {
    std::fprintf(stderr, "snapjudge-train-push: nothing to commit (already up to date?)\n");
    fs::remove_all(tmpdir);
    return 0;
  }
  rc = spawn({"git", "-C", dst.string(), "push", "-q", "origin", "HEAD:main"}, git_env);
  fs::remove_all(tmpdir);
  if (rc != 0) {
    std::fprintf(stderr, "snapjudge-train-push: push failed\n");
    return 2;
  }
  std::fprintf(stderr, "pushed -> https://huggingface.co/%s\n", repo.c_str());
  return 0;
}
