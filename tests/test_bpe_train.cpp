// snapjudge BPE trainer test: train a tiny byte-level BPE vocab, serialize it,
// then load it through the inference Tokenizer and verify encoding is stable.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "snapjudge/bpe_train.hpp"
#include "snapjudge/tokenizer.hpp"

using namespace snapjudge;
using nlohmann::ordered_json;

TEST_CASE("byte-level BPE trainer produces a loadable tokenizer") {
  // A corpus with a repeating pattern so merges are unambiguous.
  std::string corpus =
      "the cat sat the cat sat the cat sat the cat sat "
      "the dog ran the dog ran the dog ran the dog ran "
      "refund refund refund refund billing billing billing";

  BpeVocab v = train_byte_bpe(corpus, 300, nullptr);

  // Vocab must be at least 256 base symbols plus at least one learned merge.
  CHECK(v.tokens.size() >= 257);
  CHECK(v.tokens.size() <= 300);
  CHECK(v.merges.size() == v.tokens.size() - 256);

  // Every merge references symbols that exist.
  for (const auto& [a, b] : v.merges) {
    bool a_ok = false, b_ok = false;
    for (const auto& t : v.tokens) { if (t == a) a_ok = true; if (t == b) b_ok = true; }
    CHECK(a_ok);
    CHECK(b_ok);
  }

  // Serialize -> parse -> encode must be stable.
  std::string json_text = bpe_vocab_to_tokenizer_json(v);
  auto tok = Tokenizer::from_tokenizer_json(json_text);
  REQUIRE(tok != nullptr);

  // Encoding is deterministic and repeatable.
  auto a_ids = tok->encode("the cat sat");
  auto b_ids = tok->encode("the cat sat");
  CHECK(a_ids == b_ids);
  CHECK(!a_ids.empty());
}

TEST_CASE("BPE trainer: empty corpus still yields the 256 base symbols") {
  BpeVocab v = train_byte_bpe("", 300, nullptr);
  CHECK(v.tokens.size() == 256);
  CHECK(v.merges.empty());
  std::string json_text = bpe_vocab_to_tokenizer_json(v);
  auto tok = Tokenizer::from_tokenizer_json(json_text);
  REQUIRE(tok != nullptr);
  CHECK(!tok->encode("hello").empty());
}
