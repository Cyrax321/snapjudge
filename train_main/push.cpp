// snapjudge-train-push: push a trained checkpoint dir to the Hugging Face hub.
//
//  snapjudge-train-push --ckpt out/checkpoint-ft --repo you/name [--private]
//
// Uses the HF git interface: hf.co repos are git remotes; with a user token in
// HF_TOKEN the credential URL is https://<user>:<token>@huggingface.co/<repo>.
// Large files (model.safetensors) go through git-lfs when available; without
// git-lfs the HF backend streams them through its (slower) store.

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#if !defined(_WIN32)
#include <unistd.h>
#else
#include <process.h>
#endif

namespace fs = std::filesystem;

namespace {

int run(const std::string& cmd) {
  std::fprintf(stderr, "+ %s\n", cmd.c_str());
  return std::system(cmd.c_str());
}

std::string shell_quote(const std::string& s) {
  std::string out = "'";
  for (char c : s) out += c == '\'' ? "'\\''" : std::string(1, c);
  return out + "'";
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
  }
  if (ckpt.empty() || repo.empty()) {
    std::puts("snapjudge-train-push --ckpt <dir> --repo <you/name> [--private]\n"
              "  needs HF_TOKEN in the environment");
    return 2;
  }
  for (const char* f : {"rl_agent_config.json", "model.safetensors"})
    if (!fs::exists(fs::path(ckpt) / f)) {
      std::fprintf(stderr, "snapjudge-train-push: %s missing in %s\n", f, ckpt.c_str());
      return 2;
    }

  const char* token = std::getenv("HF_TOKEN");
  if (!token || !*token) {
    std::fprintf(stderr, "snapjudge-train-push: set HF_TOKEN first\n");
    return 2;
  }

  // whoami for the credential URL
  std::string user;
  {
    std::string cmd = "curl -sf -H \"Authorization: Bearer " + std::string(token) +
                      "\" https://huggingface.co/api/whoami-v2";
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) {
      std::fprintf(stderr, "snapjudge-train-push: whoami failed (no network?)\n");
      return 2;
    }
    char buf[4096];
    std::string body;
    while (std::fgets(buf, sizeof(buf), p)) body += buf;
    pclose(p);
    auto pos = body.find("\"name\":\"");
    if (pos == std::string::npos) {
      std::fprintf(stderr, "snapjudge-train-push: HF_TOKEN invalid?\n");
      return 2;
    }
    pos += 8;
    auto end = body.find('"', pos);
    user = body.substr(pos, end - pos);
  }

  std::string tmpdir = std::string(::getenv("TMPDIR") ?: "/tmp") + "/snapjudge-push-" +
                       std::to_string(::getpid());
  fs::create_directories(tmpdir);

  // create the repo if it does not exist (200/201/409 all fine)
  {
    std::string cmd = "curl -s -o /dev/null -w '%{http_code}' -X POST "
                      "-H \"Authorization: Bearer " + std::string(token) + "\" "
                      "-H \"Content-Type: application/json\" "
                      "https://huggingface.co/api/models/create-repo "
                      "-d '{\"id\":\"" + repo + "\",\"private\":" + (priv ? "true" : "false") + "}'";
    (void)run(cmd);
  }

  std::string auth_url = "https://" + user + ":" + token + "@huggingface.co/models/" + repo;
  if (run("git clone --depth 1 " + shell_quote(auth_url) + " " + shell_quote(tmpdir + "/repo")) != 0)
    std::fprintf(stderr, "snapjudge-train-push: clone failed (repo may be new; creating bare)\n");

  fs::path dst = fs::path(tmpdir) / "repo";
  if (!fs::exists(dst)) {
    fs::create_directories(dst);
    run("cd " + shell_quote(dst.string()) + " && git init -q");
    run("cd " + shell_quote(dst.string()) + " && git remote add origin " + shell_quote(auth_url));
  }

  // copy checkpoint files
  for (const auto& e : fs::recursive_directory_iterator(ckpt)) {
    fs::path rel = fs::relative(e.path(), ckpt);
    fs::path to = dst / rel;
    fs::create_directories(to.parent_path());
    fs::copy_file(e.path(), to, fs::copy_options::overwrite_existing);
  }
  // git-lfs for the big files if available
  run("cd " + shell_quote(dst.string()) + " && git lfs install 2>/dev/null || true");
  run("cd " + shell_quote(dst.string()) +
      " && (git lfs track '*.safetensors' 2>/dev/null || true)");
  run("cd " + shell_quote(dst.string()) + " && git add -A");
  if (run("cd " + shell_quote(dst.string()) + " && git -c user.email=snapjudge@local "
              "-c user.name=snapjudge commit -q -m 'snapjudge checkpoint'") != 0) {
    std::fprintf(stderr, "snapjudge-train-push: nothing to commit (already up to date?)\n");
    return 0;
  }

  if (run("cd " + shell_quote(dst.string()) + " && git push -q origin HEAD:main") != 0) {
    std::fprintf(stderr, "snapjudge-train-push: push failed\n");
    return 2;
  }
  std::fprintf(stderr, "pushed -> https://huggingface.co/%s\n", repo.c_str());
  std::string rm = "rm -rf " + shell_quote(tmpdir);
  std::system(rm.c_str());
  return 0;
}
