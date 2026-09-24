#include "snapjudge/tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "nlohmann/json.hpp"
#include "snapjudge/unicode.hpp"

namespace snapjudge {

namespace fs = std::filesystem;
using nlohmann::json;

// --------------------------------------------------------------------------
// GPT-2 byte <-> unicode tables (same mapping as HF ByteLevel pre-tokenizer).
// --------------------------------------------------------------------------
const std::u32string& Tokenizer::byte_unicode() {
  static const std::u32string table = [] {
    std::u32string t(256, 0);
    auto in_range = [](int b) {
      return (b >= 0x21 && b <= 0x7e) || (b >= 0xa1 && b <= 0xac) || (b >= 0xae && b <= 0xff);
    };
    int k = 0;
    for (int b = 0; b < 256; ++b) {
      t[b] = in_range(b) ? static_cast<char32_t>(b) : static_cast<char32_t>(0x100 + (k++));
    }
    return t;
  }();
  return table;
}

// --------------------------------------------------------------------------
// GPT2_SPLIT reimplementation.
//
// HF regex:  's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// Alternation order matters; a contraction is matched as its own piece only
// when it begins a token the regex would otherwise start at a letter — i.e.
// the apostrophe must not be preceded-in-piece by another letter (leftmost-
// longest alternation: "'s" only matches standing at a token start).
// --------------------------------------------------------------------------
// Matched length of 's|'t|'re|'ve|'m|'ll|'d at position i (0 if none).
// Order-sensitive like the regex: 2-char forms first? No — 're/'ve/'ll only fire
// when the full 3 chars are present; "'me" must NOT swallow the 'e'.
static int contraction_len(const std::u32string& s, size_t i) {
  size_t n = s.size();
  if (s[i] != U'\'' || i + 1 >= n) return 0;
  char32_t c1 = s[i + 1];
  if (c1 == U's' || c1 == U't' || c1 == U'm' || c1 == U'd') return 2;
  if (i + 2 < n) {
    if ((c1 == U'r' || c1 == U'v') && s[i + 2] == U'e') return 3;
    if (c1 == U'l' && s[i + 2] == U'l') return 3;
    if (c1 == U's' || c1 == U'd') return 2;  // covered above, kept for clarity
  }
  return 0;
}

// Character classes used by the regex: 1=\p{L}  2=\p{N}  3=\s  4=other
static inline int cp_class(char32_t cp) {
  if (uni_isletter(cp)) return 1;
  if (uni_isnumber(cp)) return 2;
  if (uni_isws(cp)) return 3;
  return 4;
}

std::vector<std::u32string> Tokenizer::gpt2_split(const std::u32string& s) {
  std::vector<std::u32string> out;
  size_t i = 0, n = s.size();
  while (i < n) {
    // 1. contraction (only when the apostrophe starts a piece)
    if (int cl = contraction_len(s, i)) {
      out.emplace_back(s.substr(i, cl));
      i += cl;
      continue;
    }
    int c = cp_class(s[i]);
    if (c == 1 || c == 2 || c == 4) {
      size_t j = i + 1;
      while (j < n && cp_class(s[j]) == c) ++j;
      out.emplace_back(s.substr(i, j - i));
      i = j;
      continue;
    }
    // c == 3: whitespace. ` ?` allows ONE literal space to prefix a letter /
    // number / punctuation token.
    if (s[i] == U' ' && i + 1 < n) {
      int cd = cp_class(s[i + 1]);
      if (cd == 1 || cd == 2 || cd == 4) {
        size_t j = i + 2;
        while (j < n && cp_class(s[j]) == cd) ++j;
        out.emplace_back(s.substr(i, j - i));
        i = j;
        continue;
      }
    }
    // Whitespace run: \s+(?!\S) keeps all but the last ws char when a non-ws
    // follows (the last one may lead the next token); \s+ is the fallback for
    // a lone ws char.
    size_t j = i + 1;
    while (j < n && cp_class(s[j]) == 3) ++j;
    if (j == n || j - i == 1) {
      out.emplace_back(s.substr(i, j - i));
      i = j;
    } else {
      out.emplace_back(s.substr(i, j - i - 1));
      i = j - 1;
    }
  }
  return out;
}

// --------------------------------------------------------------------------
// BPE merge: greedy rank-ordered, identical to tokens' merge loop.
// --------------------------------------------------------------------------
static inline uint64_t pair_key(const std::u32string& a, const std::u32string& b) {
  // FNV-1a over a\0b code points; collisions are practically impossible for
  // 580k pairs but correctness is still checked by tests.
  uint64_t h = 1469598103934665603ULL;
  auto mix = [&](char32_t cp) {
    h ^= cp;
    h *= 1099511628211ULL;
  };
  for (char32_t cp : a) mix(cp);
  mix(0);
  for (char32_t cp : b) mix(cp);
  return h;
}

std::vector<std::u32string> Tokenizer::bpe_word(std::vector<char32_t> word) const {
  if (word.size() <= 1) return {std::u32string(word.begin(), word.end())};
  std::vector<std::u32string> syms;
  syms.reserve(word.size());
  for (char32_t c : word) syms.emplace_back(1, c);
  for (;;) {
    int64_t best = INT64_MAX;
    size_t idx = SIZE_MAX;
    for (size_t i = 0; i + 1 < syms.size(); ++i) {
      auto it = merge_rank_.find(pair_key(syms[i], syms[i + 1]));
      if (it != merge_rank_.end() && it->second < best) {
        best = it->second;
        idx = i;
      }
    }
    if (idx == SIZE_MAX) break;
    syms[idx] += syms[idx + 1];
    syms.erase(syms.begin() + static_cast<long>(idx) + 1);
  }
  return syms;
}

int32_t Tokenizer::vocab_lookup(const std::string& tok) const {
  size_t lo = 0, hi = vocab_tokens_.size();
  while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    int c = vocab_tokens_[mid].compare(tok);
    if (c < 0) lo = mid + 1;
    else if (c > 0) hi = mid;
    else return vocab_ids_[mid];
  }
  // Check added_tokens too (they are usually also in vocab but be thorough).
  auto it = added_.find(tok);
  return it != added_.end() ? it->second : -1;
}

// --------------------------------------------------------------------------
// AddedVocabulary: leftmost-longest split on a set of added tokens.
// --------------------------------------------------------------------------
std::vector<Tokenizer::Span> Tokenizer::split_added(const std::string& text,
                                                    const std::vector<AddedTok>& toks) {
  std::vector<Span> out;
  std::string gap;
  size_t i = 0;
  while (i < text.size()) {
    int32_t best = -1;
    size_t best_len = 0;
    bool best_lstrip = false;
    for (const auto& t : toks) {
      size_t L = t.content.size();
      if (L > best_len && i + L <= text.size() &&
          text.compare(i, L, t.content) == 0) {
        best_len = L;
        best = t.id;
        best_lstrip = t.lstrip;
      }
    }
    if (best_len) {
      if (!gap.empty()) {
        // lstrip=true on the added token trims trailing whitespace of the
        // preceding gap (HF AddedToken behavior).
        if (best_lstrip) {
          while (!gap.empty()) {
            unsigned char c = static_cast<unsigned char>(gap.back());
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f') gap.pop_back();
            else break;
          }
        }
        if (!gap.empty()) out.push_back({std::move(gap), -1});
        gap.clear();
      }
      out.push_back({"", best});
      i += best_len;
    } else {
      gap.push_back(text[i++]);
    }
  }
  if (!gap.empty()) out.push_back({std::move(gap), -1});
  return out;
}

// --------------------------------------------------------------------------
// ByteLevel gap encode: NFC'd code points in, GPT2 split + byte map + BPE.
// --------------------------------------------------------------------------
std::vector<int32_t> Tokenizer::encode_bytelevel_gap(const std::u32string& cps) const {
  std::vector<int32_t> out;
  for (const auto& piece : gpt2_split(cps)) {
    std::vector<char32_t> mapped;
    for (char32_t cp : piece) {
      std::string b = utf8_encode(cp);
      for (unsigned char byte : b) mapped.push_back(byte_unicode()[byte]);
    }
    for (const auto& tok : bpe_word(std::move(mapped))) {
      int32_t id = vocab_lookup(utf8_encode(tok));
      out.push_back(id >= 0 ? id : unk_id);
    }
  }
  return out;
}

// --------------------------------------------------------------------------
// Metaspace gap encode (mmBERT): prepend ▁, split at ▁, newline runs as own
// pieces, BPE per piece. (Normalizer replaces applied by caller.)
// --------------------------------------------------------------------------
std::vector<int32_t> Tokenizer::encode_metaspace_gap(const std::string& t) const {
  std::vector<int32_t> out;
  const std::string MARK = "\xE2\x96\x81";  // ▁
  if (t.empty()) return out;
  std::vector<std::pair<bool, std::string>> segs;  // (is_newline_run, text)
  size_t i = 0;
  while (i < t.size()) {
    if (t[i] == '\n') {
      size_t j = i;
      while (j < t.size() && t[j] == '\n') ++j;
      segs.emplace_back(true, t.substr(i, j - i));
      i = j;
    } else {
      size_t j = i;
      while (j < t.size() && t[j] != '\n') ++j;
      segs.emplace_back(false, t.substr(i, j - i));
      i = j;
    }
  }
  auto push = [&](const std::string& piece) {
    if (piece.empty()) return;
    std::u32string cps2 = utf8_decode(piece);
    std::vector<char32_t> chars(cps2.begin(), cps2.end());
    for (const auto& tok : bpe_word(std::move(chars))) {
      int32_t id = vocab_lookup(utf8_encode(tok));
      out.push_back(id >= 0 ? id : unk_id);
    }
  };
  for (const auto& [is_nl, seg] : segs) {
    if (is_nl) { push(seg); continue; }
    std::string w = seg.rfind(MARK, 0) == 0 ? seg : MARK + seg;
    size_t pos = 0;
    while (pos < w.size()) {
      size_t next = w.find(MARK, pos);
      if (next == std::string::npos) break;
      size_t after = w.find(MARK, next + MARK.size());
      std::string chunk = after == std::string::npos ? w.substr(next) : w.substr(next, after - next);
      push(chunk);
      pos = after == std::string::npos ? w.size() : after;
    }
  }
  return out;
}

// --------------------------------------------------------------------------
// Full encode pipeline, mirroring HF tokenizers:
//   1. split raw text on added tokens with normalized=false
//   2. normalize each gap (NFC for bytelevel, Replace rules for metaspace)
//   3. split each normalized gap on added tokens with normalized=true
//   4. pre-tokenize + BPE each remaining gap
// --------------------------------------------------------------------------
std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
  std::vector<int32_t> out;
  if (text.empty()) return out;
  for (auto& span : split_added(text, added_pre_)) {
    if (span.added_id >= 0) { out.push_back(span.added_id); continue; }
    if (kind_ == Kind::ByteLevel) {
      std::string normed = nfc(span.text);
      for (auto& sub : split_added(normed, added_post_)) {
        if (sub.added_id >= 0) { out.push_back(sub.added_id); continue; }
        auto ids = encode_bytelevel_gap(utf8_decode(sub.text));
        out.insert(out.end(), ids.begin(), ids.end());
      }
    } else {
      std::string normed = span.text;
      for (const auto& [from, to] : replaces_) {
        if (from.empty()) continue;
        size_t pos = 0;
        while ((pos = normed.find(from, pos)) != std::string::npos) {
          normed.replace(pos, from.size(), to);
          pos += to.size();
        }
      }
      for (auto& sub : split_added(normed, added_post_)) {
        if (sub.added_id >= 0) { out.push_back(sub.added_id); continue; }
        auto ids = encode_metaspace_gap(sub.text);
        out.insert(out.end(), ids.begin(), ids.end());
      }
    }
  }
  return out;
}

// --------------------------------------------------------------------------
// tokenizer.json parsing
// --------------------------------------------------------------------------
namespace {
const char* CLS_ALIASES[] = {"[CLS]", "<bos>", "<s>"};
const char* SEP_ALIASES[] = {"[SEP]", "<eos>", "</s>"};
const char* PAD_ALIASES[] = {"[PAD]", "<pad>", "[PAD]"};
const char* MASK_ALIASES[] = {"[MASK]", "<mask>"};
const char* UNK_ALIASES[] = {"[UNK]", "<unk>", "[UNK]"};

bool node_has_type(const json* node, const char* want, int depth = 0) {
  if (!node || depth > 8 || !node->is_object()) return false;
  auto it = node->find("type");
  if (it != node->end() && it->is_string() && it->get<std::string>() == want) return true;
  for (const char* key : {"normalizers", "pre_tokenizers", "decoders"}) {
    auto it2 = node->find(key);
    if (it2 != node->end() && it2->is_array())
      for (const auto& c : *it2)
        if (node_has_type(&c, want, depth + 1)) return true;
  }
  return false;
}

void collect_replaces(const json* node, std::vector<std::pair<std::string, std::string>>& out,
                      int depth = 0) {
  if (!node || depth > 8 || !node->is_object()) return;
  auto ty = node->find("type");
  if (ty != node->end() && ty->is_string() && ty->get<std::string>() == "Replace") {
    auto pat = node->find("pattern");
    auto cont = node->find("content");
    if (pat != node->end() && pat->is_object()) {
      auto ps = pat->find("String");
      if (ps != pat->end() && ps->is_string() && cont != node->end() && cont->is_string()) {
        out.emplace_back(ps->get<std::string>(), cont->get<std::string>());
      }
    }
  }
  for (const char* key : {"normalizers", "pre_tokenizers", "decoders"}) {
    auto it = node->find(key);
    if (it != node->end() && it->is_array())
      for (const auto& c : *it) collect_replaces(&c, out, depth + 1);
  }
}
}  // namespace

std::shared_ptr<Tokenizer> Tokenizer::from_tokenizer_json(const std::string& json_text) {
  json r;
  try {
    r = json::parse(json_text);
  } catch (...) {
    return nullptr;
  }
  auto model_it = r.find("model");
  if (model_it == r.end() || !model_it->is_object()) return nullptr;
  auto vocab_it = model_it->find("vocab");
  if (vocab_it == model_it->end() || !vocab_it->is_object()) return nullptr;

  auto tok = std::make_shared<Tokenizer>();

  // vocab sorted by token string for binary search
  std::vector<std::pair<std::string, int32_t>> v;
  v.reserve(vocab_it->size());
  for (auto it = vocab_it->begin(); it != vocab_it->end(); ++it) {
    if (!it.value().is_number_integer()) continue;
    v.emplace_back(it.key(), it.value().get<int32_t>());
  }
  std::sort(v.begin(), v.end());
  tok->vocab_tokens_.reserve(v.size());
  tok->vocab_ids_.reserve(v.size());
  for (auto& p : v) {
    tok->vocab_tokens_.push_back(std::move(p.first));
    tok->vocab_ids_.push_back(p.second);
  }

  auto merges_it = model_it->find("merges");
  if (merges_it != model_it->end() && merges_it->is_array()) {
    int32_t rank = 0;
    for (const auto& m : *merges_it) {
      std::u32string a, b;
      if (m.is_string()) {
        std::string s = m.get<std::string>();
        size_t sp = s.find(' ');
        if (sp != std::string::npos) {
          a = utf8_decode(s.substr(0, sp));
          b = utf8_decode(s.substr(sp + 1));
        }
      } else if (m.is_array() && m.size() >= 2) {
        a = utf8_decode(m[0].get<std::string>());
        b = utf8_decode(m[1].get<std::string>());
      }
      if (!a.empty() || !b.empty()) tok->merge_rank_[pair_key(a, b)] = rank;
      ++rank;
    }
  }

  auto added_it = r.find("added_tokens");
  if (added_it != r.end() && added_it->is_array()) {
    for (const auto& t : *added_it) {
      auto c = t.find("content");
      auto id = t.find("id");
      if (c != t.end() && c->is_string() && id != t.end() && id->is_number_integer()) {
        const std::string content = c->get<std::string>();
        int32_t tid = id->get<int32_t>();
        tok->added_[content] = tid;
        // HF splits normalized=false tokens against the raw text and
        // normalized=true tokens against the normalized gaps.
        auto norm = t.find("normalized");
        bool normalized = norm != t.end() && norm->is_boolean() && norm->get<bool>();
        auto ls = t.find("lstrip");
        bool lstrip = ls != t.end() && ls->is_boolean() && ls->get<bool>();
        (normalized ? tok->added_post_ : tok->added_pre_).push_back({content, tid, lstrip});
      }
    }
    auto by_len = [](const Tokenizer::AddedTok& a, const Tokenizer::AddedTok& b) {
      return a.content.size() > b.content.size();
    };
    std::sort(tok->added_pre_.begin(), tok->added_pre_.end(), by_len);
    std::sort(tok->added_post_.begin(), tok->added_post_.end(), by_len);
  }

  auto pick = [&](const char** aliases, size_t n, int32_t fallback) -> std::pair<int32_t, std::string> {
    for (size_t k = 0; k < n; ++k) {
      auto it = tok->added_.find(aliases[k]);
      if (it != tok->added_.end()) return {it->second, aliases[k]};
      int32_t id = tok->vocab_lookup(aliases[k]);
      if (id >= 0) return {id, aliases[k]};
    }
    return {fallback, aliases[0]};
  };

  // Checkpoint-default fallbacks (ModernBERT added-token ids).
  auto cls = pick(CLS_ALIASES, 3, 50281);
  auto sep = pick(SEP_ALIASES, 3, 50282);
  auto pad = pick(PAD_ALIASES, 3, 50283);
  auto mask = pick(MASK_ALIASES, 2, 50284);
  auto unk = pick(UNK_ALIASES, 3, 50280);
  tok->cls_id = cls.first;
  tok->sep_id = sep.first;
  tok->pad_id = pad.first;
  tok->mask_id = mask.first;
  tok->unk_id = unk.first;
  tok->mask_token = mask.second;

  auto pt_it = r.find("pre_tokenizer");
  bool metaspace = pt_it != r.end() && node_has_type(&(*pt_it), "Metaspace");
  tok->kind_ = metaspace ? Kind::Metaspace : Kind::ByteLevel;

  auto norm_it = r.find("normalizer");
  if (norm_it != r.end()) collect_replaces(&(*norm_it), tok->replaces_);
  if (metaspace && tok->replaces_.empty()) {
    tok->replaces_.emplace_back(" ", "\xE2\x96\x81");
  }
  return tok;
}

std::shared_ptr<Tokenizer> Tokenizer::from_dir(const std::string& dir) {
  fs::path p = fs::path(dir) / "tokenizer.json";
  if (!fs::exists(p)) {
    // also allow passing the tokenizer/ dir directly
    p = fs::path(dir) / "tokenizer" / "tokenizer.json";
    if (!fs::exists(p))
      throw std::runtime_error("snapjudge: tokenizer.json not found under " + dir);
  }
  std::ifstream f(p);
  std::stringstream ss;
  ss << f.rdbuf();
  auto tok = from_tokenizer_json(ss.str());
  if (!tok) throw std::runtime_error("snapjudge: failed to parse " + p.string());
  return tok;
}

std::shared_ptr<Tokenizer> Tokenizer::cached_from_dir(const std::string& dir) {
  static std::mutex mu;
  static std::unordered_map<std::string, std::weak_ptr<Tokenizer>> cache;
  std::string key;
  try {
    fs::path cfg = fs::path(dir) / "tokenizer_config.json";
    std::error_code ec;
    auto mt = fs::last_write_time(cfg, ec);
    key = fs::weakly_canonical(dir).string() + "@" + (ec ? std::string("?") :
        std::to_string(static_cast<long long>(mt.time_since_epoch().count())));
  } catch (...) {
    return from_dir(dir);
  }
  std::lock_guard<std::mutex> lk(mu);
  auto it = cache.find(key);
  if (it != cache.end()) {
    if (auto sp = it->second.lock()) return sp;
  }
  auto tok = from_dir(dir);
  cache[key] = tok;
  return tok;
}

}  // namespace snapjudge
