#pragma once
// snapjudge tokenizer.hpp: dependency-free HF tokenizer (BPE byte-level +
// Metaspace), ported 1:1 from reference TS module src/tokenizer.ts which is parity-tested
// against HF tokenizers. Loads tokenizer.json directly.

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace snapjudge {

class Tokenizer {
 public:
  int32_t cls_id = 101, sep_id = 102, mask_id = 103, pad_id = 0, unk_id = 0;
  std::string mask_token = "[MASK]";

  // Encode text without special tokens (matches tok(text, add_special_tokens=False)).
  std::vector<int32_t> encode(const std::string& text) const;

  // Load from a parsed HF tokenizer.json string. Returns nullptr on parse failure.
  static std::shared_ptr<Tokenizer> from_tokenizer_json(const std::string& json_text);

  // Load tokenizer.json from a directory (looks for <dir>/tokenizer.json).
  // Throws std::runtime_error if not found/unparseable.
  static std::shared_ptr<Tokenizer> from_dir(const std::string& dir);

  // Process-wide cache keyed on (abs dir, file mtime) — port of
  // agent.py::_load_tokenizer. Falls back to from_dir on mtime failure.
  static std::shared_ptr<Tokenizer> cached_from_dir(const std::string& dir);

 private:
  enum class Kind { ByteLevel, Metaspace };
  Kind kind_ = Kind::ByteLevel;

  // vocab string -> id. Strings are UTF-8. Sorted arrays for lookup with binary
  // search; parallel id array. Merges: "a b" rank map.
  std::vector<std::string> vocab_tokens_;
  std::vector<int32_t> vocab_ids_;
  std::unordered_map<uint64_t, int32_t> merge_rank_;  // hash of "a\x00b" -> rank
  std::unordered_map<std::string, int32_t> added_;    // added_tokens content -> id
  // Added-token splitting, mirroring HF tokenizers' AddedVocabulary: tokens with
  // normalized=false are matched against the RAW text first; normalized=true
  // tokens are then matched against each normalized gap. Leftmost-longest wins.
  struct AddedTok { std::string content; int32_t id; bool lstrip = false; };
  std::vector<AddedTok> added_pre_;    // normalized == false
  std::vector<AddedTok> added_post_;   // normalized == true
  // GPT-2 byte -> unicode point map for the ByteLevel path.
  static const std::u32string& byte_unicode();
  // Normalizer Replace rules applied in order (pattern, content).
  std::vector<std::pair<std::string, std::string>> replaces_;

  int32_t vocab_lookup(const std::string& tok) const;
  std::vector<std::u32string> bpe_word(std::vector<char32_t> word) const;
  // Split `text` into spans: added tokens (id >= 0) vs. plain gaps (id < 0).
  struct Span { std::string text; int32_t added_id; };
  static std::vector<Span> split_added(const std::string& text,
                                       const std::vector<AddedTok>& toks);
  std::vector<int32_t> encode_bytelevel_gap(const std::u32string& nfc_cps) const;
  std::vector<int32_t> encode_metaspace_gap(const std::string& text) const;

  // Hand-written scanner reproducing HF's ByteLevel GPT2_SPLIT regex:
  //   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
  // Returns matched segments over the decoded code points.
  static std::vector<std::u32string> gpt2_split(const std::u32string& text);
};

}  // namespace snapjudge
