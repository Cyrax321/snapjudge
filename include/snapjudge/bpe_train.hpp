#pragma once
// snapjudge bpe_train.hpp: byte-level BPE tokenizer trainer.
//
// Trains a GPT-2 style byte-level BPE vocabulary from raw text, producing the
// same tokenizer.json schema the inference Tokenizer loads. This is what makes
// an own checkpoint possible: train a vocabulary on your own corpus, then the
// encoder + head use it end to end with zero dependency on any other tokenizer.
//
// Algorithm (the standard byte-pair encoding greedy merge):
//   1. map every byte through the GPT-2 byte<->unicode table;
//   2. split on the GPT-2 regex (same rule the encoder uses);
//   3. repeatedly merge the most frequent adjacent pair, up to `max_vocab`
//      entries, counting over the whole corpus each round.
//
// A vocabulary of N entries = 256 base byte-symbols + (N - 256) merges.

#include <cstdint>
#include <string>
#include <vector>

namespace snapjudge {

// Trained vocabulary: the ordered merge list (a, b) that the encoder applies
// by rank. `vocab` is the resulting token -> id map (id 0..size-1, base byte
// symbols first, then each merged token in merge order).
struct BpeVocab {
  // merge[i] = (a, b) — the i-th learned merge (rank i).
  std::vector<std::pair<std::string, std::string>> merges;
  // token -> id. Sorted by id so id == insertion order.
  std::vector<std::string> tokens;   // tokens[i] has id i
  std::vector<int32_t> ids;          // parallel id array (i)
};

// Train a byte-level BPE vocabulary. `corpus` is raw UTF-8 text (may be a
// single concatenated buffer). `max_vocab` is the target vocab size (>= 256).
// Progress is reported via `on_progress(round, vocab_size)` when non-null.
BpeVocab train_byte_bpe(const std::string& corpus, int max_vocab,
                        void (*on_progress)(int round, int vocab_size));

// Serialize the trained vocab as an HF tokenizer.json document (byte-level
// pre-tokenizer, no normalizer beyond NFC, GPT-2 merges).
std::string bpe_vocab_to_tokenizer_json(const BpeVocab& vocab);

}  // namespace snapjudge
