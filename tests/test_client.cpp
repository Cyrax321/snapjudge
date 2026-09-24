// snapjudge client classes vs a fake backend (mirrors py test_langchain.py's
// use of stub agents — no model loads).
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <string>

#include "snapjudge/client.hpp"

using nlohmann::ordered_json;
using namespace snapjudge;

TEST_CASE("extract_text priorities") {
  // state_key wins
  ordered_json in1 = {{"my_key", {{"content", "hello"}}}, {"input", "wrong"}};
  CHECK(detail::extract_text(in1, "my_key") == "hello");
  // candidate keys
  ordered_json in2 = {{"input", "direct text"}};
  CHECK(detail::extract_text(in2, "") == "direct text");
  // messages list: most recent human
  ordered_json in3 = {{"messages", ordered_json::array({
                                        ordered_json{{"type", "ai"}, {"content", "old ai"}},
                                        ordered_json{{"type", "human"}, {"content", "latest"}},
                                    })}};
  CHECK(detail::extract_text(in3, "") == "latest");
  // plain string passthrough
  CHECK(detail::extract_text(ordered_json("plain"), "") == "plain");
}
