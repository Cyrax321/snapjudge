#pragma once
// snapjudge tokenizer.hpp: dependency-free BPE tokenizer for HF tokenizer.json.
//
// Implements the two pre-tokenizers a decision checkpoint can ship with:
//   * ByteLevel   — GPT-2 style: bytes mapped through a byte<->unicode table,
//                   split on the GPT-2 regex, then BPE-merged by rank.
//   * Metaspace   — SentencePiece style: ' ' replaced by a word-boundary marker,
//                   then BPE-merged over unicode code points.
// Special-token handling follows the HF AddedVocabulary contract (normalized vs
// raw-text matching, lstrip semantics). No Python, no torch, no ONNX Runtime —
// just UTF-8, a couple of hash maps and a binary search.

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace snapjudge {

class Tokenizer {
 public:
  // Special-token ids, resolved from the tokenizer.json added_tokens/vocab when
  // available, otherwise the checkpoint defaults. Exposed because callers build
  // the [CLS] ... [SEP] sequences by hand.
  int32_t cls_id = 101, sep_id = 102, mask_id = 103, pad_id = 0, unk_id = 0;
  std::string mask_token = "[MASK]";

  // Encode text without wrapping special tokens (equivalent to
  // `tok(text, add_special_tokens=False)`).
  std::vector<int32_t> encode(const std::string& text) const;

  // Parse a tokenizer.json document. Returns nullptr on malformed input.
  static std::shared_ptr<Tokenizer> from_tokenizer_json(const std::string& json_text);

  // Load <dir>/tokenizer.json (falls back to <dir>/tokenizer/tokenizer.json).
  // Throws std::runtime_error when the file is absent or unparseable.
  static std::shared_ptr<Tokenizer> from_dir(const std::string& dir);

  // Process-wide cache keyed on (canonical dir, tokenizer_config mtime).
  // Falls back to a fresh parse when the mtime cannot be read.
  static std::shared_ptr<Tokenizer> cached_from_dir(const std::string& dir);

  // GPT-2 byte -> unicode code-point table for the ByteLevel path (single
  // shared table, built once). Public so the BPE trainer (bpe_train.hpp) uses
  // the exact same table the encoder uses.
  static const std::u32string& byte_unicode();

  // Scanner for the GPT-2 byte-level split rule:
  //   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
  // Public so the BPE trainer splits text exactly as the encoder does.
  static std::vector<std::u32string> gpt2_split(const std::u32string& text);

 private:
  enum class Kind { ByteLevel, Metaspace };
  Kind kind_ = Kind::ByteLevel;

  // Vocabulary, stored as sorted token strings + parallel ids so lookup is a
  // binary search and stays allocation-free on the hot path.
  std::vector<std::string> vocab_tokens_;
  std::vector<int32_t> vocab_ids_;
  // Merge ranks: pair key -> rank (lower rank merges first).
  std::unordered_map<uint64_t, int32_t> merge_rank_;
  // added_tokens content -> id (fast path when a token is both in vocab and
  // listed under added_tokens).
  std::unordered_map<std::string, int32_t> added_;

  // An added token. `lstrip` trims trailing whitespace of the gap that precedes
  // it (HF AddedToken behavior).
  struct AddedTok {
    std::string content;
    int32_t id;
    bool lstrip = false;
  };
  // normalized=false tokens are matched against the RAW text first;
  // normalized=true tokens against each normalized gap. Leftmost-longest wins
  // within each stage, matching HF AddedVocabulary.
  std::vector<AddedTok> added_pre_;   // normalized == false
  std::vector<AddedTok> added_post_;  // normalized == true

  // Normalizer "Replace" rules, applied in order (pattern, content).
  std::vector<std::pair<std::string, std::string>> replaces_;

  int32_t vocab_lookup(const std::string& tok) const;
  std::vector<std::u32string> bpe_word(std::vector<char32_t> word) const;

  struct Span {
    std::string text;   // raw gap text
    int32_t added_id;   // >= 0 when this span is an added token
  };
  static std::vector<Span> split_added(const std::string& text,
                                        const std::vector<AddedTok>& toks);

  std::vector<int32_t> encode_bytelevel_gap(const std::u32string& nfc_cps) const;
  std::vector<int32_t> encode_metaspace_gap(const std::string& text) const;
};

}  // namespace snapjudge
