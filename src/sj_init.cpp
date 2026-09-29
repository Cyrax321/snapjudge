#include "snapjudge/sj_init.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "snapjudge/safewrite.hpp"
#include "snapjudge/tokenizer.hpp"

namespace snapjudge {

namespace fs = std::filesystem;
using nlohmann::json;

namespace {

struct T {
  std::string name;
  std::vector<int64_t> shape;
  std::vector<float> w;
};

// He-initialization for linear weights, small normal for embeddings, LayerNorm
// weight = 1 / bias = 0.
void add_linear(std::vector<T>& out, const std::string& name, int64_t rows, int64_t cols,
                std::mt19937& rng) {
  float stdv = std::sqrt(2.0f / static_cast<float>(cols));
  std::normal_distribution<float> d(0.0f, stdv);
  T t; t.name = name; t.shape = {rows, cols};
  t.w.resize(static_cast<size_t>(rows * cols));
  for (auto& v : t.w) v = d(rng);
  out.push_back(std::move(t));
}
void add_bias(std::vector<T>& out, const std::string& name, int64_t n) {
  T t; t.name = name; t.shape = {n};
  t.w.assign(static_cast<size_t>(n), 0.0f);
  out.push_back(std::move(t));
}
void add_norm(std::vector<T>& out, const std::string& name, int64_t n) {
  T w; w.name = name; w.shape = {n};
  w.w.assign(static_cast<size_t>(n), 1.0f);
  out.push_back(std::move(w));
  add_bias(out, name, n);  // reuses name? no — bias has distinct name below
}

}  // namespace

void init_sj_checkpoint(const std::string& dir, int V, int D, int L, int H, int F,
                        int max_pos, int head_layers, uint64_t seed) {
  std::mt19937 rng(static_cast<uint32_t>(seed));
  std::normal_distribution<float> ed(0.0f, 0.02f);

  std::vector<T> tensors;

  // embeddings
  {
    T t; t.name = "sj.embeddings.word.weight"; t.shape = {V, D};
    t.w.resize(static_cast<size_t>(V * D));
    for (auto& v : t.w) v = ed(rng);
    tensors.push_back(std::move(t));
  }
  {
    T t; t.name = "sj.embeddings.position.weight"; t.shape = {max_pos, D};
    t.w.resize(static_cast<size_t>(max_pos * D));
    for (auto& v : t.w) v = ed(rng);
    tensors.push_back(std::move(t));
  }
  tensors.push_back({"sj.embeddings.norm.weight", {D}, std::vector<float>((size_t)D, 1.0f)});
  tensors.push_back({"sj.embeddings.norm.bias", {D}, std::vector<float>((size_t)D, 0.0f)});

  for (int i = 0; i < L; ++i) {
    std::string p = "sj.encoder.layers." + std::to_string(i) + ".";
    tensors.push_back({p + "attn_norm.weight", {D}, std::vector<float>((size_t)D, 1.0f)});
    tensors.push_back({p + "attn_norm.bias", {D}, std::vector<float>((size_t)D, 0.0f)});
    for (const char* s : {"q", "k", "v", "o"}) {
      add_linear(tensors, p + "attn." + s + ".weight", D, D, rng);
      add_bias(tensors, p + "attn." + s + ".bias", D);
    }
    tensors.push_back({p + "ffn_norm.weight", {D}, std::vector<float>((size_t)D, 1.0f)});
    tensors.push_back({p + "ffn_norm.bias", {D}, std::vector<float>((size_t)D, 0.0f)});
    add_linear(tensors, p + "ffn.w1.weight", F, D, rng);
    add_bias(tensors, p + "ffn.w1.bias", F);
    add_linear(tensors, p + "ffn.w2.weight", D, F, rng);
    add_bias(tensors, p + "ffn.w2.bias", D);
  }
  tensors.push_back({"sj.encoder.final_norm.weight", {D}, std::vector<float>((size_t)D, 1.0f)});
  tensors.push_back({"sj.encoder.final_norm.bias", {D}, std::vector<float>((size_t)D, 0.0f)});

  // head
  {
    T t; t.name = "sj.type_emb.weight"; t.shape = {3, D};
    t.w.resize(static_cast<size_t>(3 * D));
    for (auto& v : t.w) v = ed(rng);
    tensors.push_back(std::move(t));
  }
  for (int i = 0; i < head_layers; ++i) {
    std::string p = "sj.head.layers." + std::to_string(i) + ".";
    tensors.push_back({p + "norm1.weight", {D}, std::vector<float>((size_t)D, 1.0f)});
    tensors.push_back({p + "norm1.bias", {D}, std::vector<float>((size_t)D, 0.0f)});
    tensors.push_back({p + "norm2.weight", {D}, std::vector<float>((size_t)D, 1.0f)});
    tensors.push_back({p + "norm2.bias", {D}, std::vector<float>((size_t)D, 0.0f)});
    for (const char* s : {"q", "k", "v", "o"}) {
      add_linear(tensors, p + "attn." + s + ".weight", D, D, rng);
      add_bias(tensors, p + "attn." + s + ".bias", D);
    }
    add_linear(tensors, p + "ffn.w1.weight", 4 * D, D, rng);
    add_bias(tensors, p + "ffn.w1.bias", 4 * D);
    add_linear(tensors, p + "ffn.w2.weight", D, 4 * D, rng);
    add_bias(tensors, p + "ffn.w2.bias", D);
  }
  tensors.push_back({"sj.scorer.norm.weight", {D}, std::vector<float>((size_t)D, 1.0f)});
  tensors.push_back({"sj.scorer.norm.bias", {D}, std::vector<float>((size_t)D, 0.0f)});
  add_linear(tensors, "sj.scorer.w1.weight", D, D, rng);
  add_bias(tensors, "sj.scorer.w1.bias", D);
  add_linear(tensors, "sj.scorer.w2.weight", 1, D, rng);
  add_bias(tensors, "sj.scorer.w2.bias", 1);

  // write safetensors
  std::vector<TensorOut> out;
  for (auto& t : tensors) {
    TensorOut o;
    o.name = t.name;
    o.shape = t.shape;
    auto sp = std::shared_ptr<float[]>(new float[t.w.size()]);
    std::memcpy(sp.get(), t.w.data(), sizeof(float) * t.w.size());
    o.data = std::move(sp);
    o.as_f16 = true;
    out.push_back(std::move(o));
  }
  fs::create_directories(dir);
  save_safetensors((fs::path(dir) / "model.safetensors").string(), out);

  // config
  json cfg = {
      {"vocab_size", V}, {"hidden_size", D}, {"num_layers", L}, {"num_heads", H},
      {"intermediate_size", F}, {"max_positions", max_pos}, {"norm_eps", 1e-5},
      {"head_layers", head_layers}, {"head_intermediate", 4 * D},
      {"max_len", max_pos}, {"head_max_len", 128}, {"n_types", 3},
      {"cls_id", 1}, {"sep_id", 2}, {"mask_id", 4}, {"pad_id", 3},
  };
  std::ofstream cfgo(fs::path(dir) / "sj_config.json");
  cfgo << cfg.dump(2, ' ', false);

  // tokenizer: byte-level, V vocab, specials at ids 0..4
  const std::u32string& b2u = Tokenizer::byte_unicode();
  json vocab = json::object();
  vocab["[UNK]"] = 0; vocab["[CLS]"] = 1; vocab["[SEP]"] = 2; vocab["[PAD]"] = 3; vocab["[MASK]"] = 4;
  int next = 5;
  auto u32_to_utf8 = [](char32_t cp) {
    std::string utf8;
    if (cp < 0x80) utf8 += static_cast<char>(cp);
    else if (cp < 0x800) { utf8 += static_cast<char>(0xC0 | (cp >> 6)); utf8 += static_cast<char>(0x80 | (cp & 0x3F)); }
    else { utf8 += static_cast<char>(0xE0 | (cp >> 12)); utf8 += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); utf8 += static_cast<char>(0x80 | (cp & 0x3F)); }
    return utf8;
  };
  for (int b = 0; b < 256 && next < V; ++b) {
    std::string utf8 = u32_to_utf8(b2u[b]);
    if (!vocab.contains(utf8)) vocab[utf8] = next++;
  }
  while (next < V) vocab["\u0100" + std::to_string(next)] = next++;

  json tok = {
      {"model", {{"type", "BPE"}, {"vocab", vocab}, {"merges", json::array()}}},
      {"pre_tokenizer", {{"type", "ByteLevel"}, {"add_prefix_space", false},
                         {"trim_offsets", true}, {"use_regex", true}}},
      {"normalizer", {{"type", "NFC"}}},
      {"added_tokens", json::array({
          {{"id", 0}, {"content", "[UNK]"}, {"special", true}, {"normalized", false}},
          {{"id", 1}, {"content", "[CLS]"}, {"special", true}, {"normalized", false}},
          {{"id", 2}, {"content", "[SEP]"}, {"special", true}, {"normalized", false}},
          {{"id", 3}, {"content", "[PAD]"}, {"special", true}, {"normalized", false}},
          {{"id", 4}, {"content", "[MASK]"}, {"special", true}, {"normalized", false}},
      })},
  };
  fs::create_directories(fs::path(dir) / "tokenizer");
  std::ofstream tf(fs::path(dir) / "tokenizer" / "tokenizer.json");
  tf << tok.dump(2, ' ', false);
  std::ofstream tcf(fs::path(dir) / "tokenizer" / "tokenizer_config.json");
  tcf << json{{"tokenizer_class", "TokenizersBackend"}}.dump();
}

}  // namespace snapjudge
