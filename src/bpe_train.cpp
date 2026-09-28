#include "snapjudge/bpe_train.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "nlohmann/json.hpp"
#include "snapjudge/tokenizer.hpp"
#include "snapjudge/unicode.hpp"

namespace snapjudge {

using nlohmann::json;

// A single "word" in the BPE sense: the code points of one GPT-2 split piece.
// Internally we store the sequence of symbol indices (into a symbol table) so
// merges are integer comparisons, not string compares.
struct Word {
  std::vector<int32_t> sym;
};

namespace {

// Symbol table: byte-symbols (0..255, already mapped through the byte<->unicode
// table) come first; each learned merge appends a new symbol.
struct Trainer {
  std::vector<std::u32string> symbols;   // symbol id -> code-point string
  std::vector<std::pair<int32_t, int32_t>> merge_of;  // symbol -> (a, b)
  std::vector<Word> words;

  // pair (a, b) -> number of occurrences across all words this round.
  std::unordered_map<uint64_t, int64_t> counts;
};

uint64_t pair_key(int32_t a, int32_t b) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) |
         static_cast<uint64_t>(static_cast<uint32_t>(b));
}

// Map raw UTF-8 text into base byte-symbols via the shared byte<->unicode
// table, split on the GPT-2 rule, and turn each piece into a Word.
std::vector<Word> tokenize_corpus(const std::string& corpus) {
  const std::u32string& b2u = Tokenizer::byte_unicode();
  // inverse: unicode code point -> byte (for base symbols)
  std::unordered_map<char32_t, int32_t> u2b;
  for (int b = 0; b < 256; ++b) u2b[b2u[b]] = b;

  std::vector<Word> out;
  for (const auto& piece : Tokenizer::gpt2_split(utf8_decode(corpus))) {
    Word w;
    w.sym.reserve(piece.size());
    for (char32_t cp : piece) {
      // Each piece code point is one UTF-8 byte's unicode mapping.
      std::string bytes = utf8_encode(cp);
      for (unsigned char byte : bytes) {
        char32_t mapped = b2u[byte];
        auto it = u2b.find(mapped);
        w.sym.push_back(it != u2b.end() ? it->second : static_cast<int32_t>(mapped));
      }
    }
    if (!w.sym.empty()) out.push_back(std::move(w));
  }
  return out;
}

}  // namespace

BpeVocab train_byte_bpe(const std::string& corpus, int max_vocab,
                        void (*on_progress)(int round, int vocab_size)) {
  const std::u32string& b2u = Tokenizer::byte_unicode();
  if (max_vocab < 256) max_vocab = 256;

  Trainer t;
  // 256 base symbols.
  for (int b = 0; b < 256; ++b) t.symbols.emplace_back(1, b2u[b]);
  t.merge_of.resize(256, {-1, -1});

  t.words = tokenize_corpus(corpus);
  if (t.words.empty()) {
    BpeVocab v;
    for (int b = 0; b < 256; ++b) v.tokens.push_back(utf8_encode(b2u[b]));
    v.ids.resize(256);
    for (int i = 0; i < 256; ++i) v.ids[i] = i;
    return v;
  }

  int rounds = max_vocab - 256;
  for (int round = 0; round < rounds; ++round) {
    // Count adjacent pairs.
    t.counts.clear();
    for (const auto& w : t.words) {
      for (size_t i = 0; i + 1 < w.sym.size(); ++i) {
        ++t.counts[pair_key(w.sym[i], w.sym[i + 1])];
      }
    }
    // Find the most frequent pair (lowest ids break ties -> deterministic).
    uint64_t best_key = 0;
    int64_t best_count = -1;
    int32_t best_a = -1, best_b = -1;
    for (const auto& [k, c] : t.counts) {
      if (c > best_count || (c == best_count && k < best_key)) {
        best_count = c;
        best_key = k;
        best_a = static_cast<int32_t>(k >> 32);
        best_b = static_cast<int32_t>(k & 0xffffffffu);
      }
    }
    if (best_count < 2) break;  // no pair occurs twice; nothing left to merge

    int32_t new_sym = static_cast<int32_t>(t.symbols.size());
    t.symbols.push_back(t.symbols[best_a] + t.symbols[best_b]);
    t.merge_of.push_back({best_a, best_b});

    // Apply the merge everywhere.
    for (auto& w : t.words) {
      std::vector<int32_t> nxt;
      nxt.reserve(w.sym.size());
      size_t i = 0;
      while (i < w.sym.size()) {
        if (i + 1 < w.sym.size() && w.sym[i] == best_a && w.sym[i + 1] == best_b) {
          nxt.push_back(new_sym);
          i += 2;
        } else {
          nxt.push_back(w.sym[i]);
          i += 1;
        }
      }
      w.sym = std::move(nxt);
    }

    if (on_progress) on_progress(round + 1, static_cast<int>(t.symbols.size()));
  }

  // Build the result.
  BpeVocab v;
  for (size_t s = 256; s < t.symbols.size(); ++s) {
    auto [a, b] = t.merge_of[s];
    v.merges.emplace_back(utf8_encode(t.symbols[a]), utf8_encode(t.symbols[b]));
  }
  for (const auto& sym : t.symbols) v.tokens.push_back(utf8_encode(sym));
  v.ids.resize(v.tokens.size());
  for (size_t i = 0; i < v.tokens.size(); ++i) v.ids[i] = static_cast<int32_t>(i);
  return v;
}

std::string bpe_vocab_to_tokenizer_json(const BpeVocab& vocab) {
  // vocab object: token -> id (ids in insertion order).
  json j = json::object();
  for (size_t i = 0; i < vocab.tokens.size(); ++i)
    j[vocab.tokens[i]] = static_cast<int32_t>(i);

  // merges: list of "a b" strings in rank order.
  json merges = json::array();
  for (const auto& [a, b] : vocab.merges)
    merges.push_back(a + " " + b);

  json doc = {
      {"model",
       {{"type", "BPE"}, {"vocab", j}, {"merges", merges}}},
      {"pre_tokenizer",
       {{"type", "ByteLevel"},
        {"add_prefix_space", false},
        {"trim_offsets", true},
        {"use_regex", true}}},
      {"normalizer", {{"type", "NFC"}}},
      {"added_tokens", json::array()},
  };
  return doc.dump(2, ' ', false);
}

}  // namespace snapjudge
