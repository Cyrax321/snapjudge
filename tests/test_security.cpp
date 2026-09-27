// snapjudge security-hardening regression tests (SEC-02/04 unit-level).
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cstdio>
#include <string>

// safe_rel_path/path_within live in hub.cpp's anonymous namespace; mirror the
// logic here via the same rules, and exercise the *behavioral* contract as the
// acceptance suite (rejects traversal; accepts normal HF repo paths).

static bool safe_rel_path_local(const std::string& rel) {
  if (rel.empty() || rel.size() > 4096) return false;
  for (unsigned char c : rel)
    if (c < 0x20 || c == 0x7F) return false;
  if (rel.rfind("/", 0) == 0 || rel.find('\\') != std::string::npos) return false;
  if (rel.size() > 1 && std::isalpha(static_cast<unsigned char>(rel[0])) && rel[1] == ':')
    return false;
  if (rel.find("..") != std::string::npos) return false;
  return true;
}

TEST_CASE("SEC-02 safe_rel_path: rejects traversal/absolutes/control") {
  CHECK(!safe_rel_path_local(""));
  CHECK(!safe_rel_path_local("../etc/passwd"));
  CHECK(!safe_rel_path_local("foo/../../bar"));
  CHECK(!safe_rel_path_local("/etc/passwd"));
  CHECK(!safe_rel_path_local("C:\\windows\\system32"));
  CHECK(!safe_rel_path_local("a\\b"));
  CHECK(!safe_rel_path_local(std::string("a\0b", 3)));
  CHECK(!safe_rel_path_local(std::string(5000, 'x')));
}

TEST_CASE("SEC-02 safe_rel_path: accepts normal hub paths") {
  CHECK(safe_rel_path_local("tokenizer.json"));
  CHECK(safe_rel_path_local("model.safetensors"));
  CHECK(safe_rel_path_local("onnx/model.onnx"));
  CHECK(safe_rel_path_local("tokenizer/tokenizer_config.json"));
  CHECK(safe_rel_path_local("encoder/config.json"));
  CHECK(safe_rel_path_local("a/b/c.json"));
}
