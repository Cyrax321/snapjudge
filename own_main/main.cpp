// snapjudge-own: build a native snapjudge decision model from scratch.
//
// Three stages, in one command:
//   init      create a fresh random SjModel checkpoint (self-contained C++)
//   pretrain  masked-language-model the encoder on a raw text corpus
//   finetune  train the decision head on typed-decisions JSONL
//
// This is the OWN pipeline: no external tokenizer, no ModernBERT, no Laya.
// Everything runs in C++ over the native sj_* checkpoint schema.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "snapjudge/common.hpp"
#include "snapjudge/sj_init.hpp"
#include "snapjudge/sj_pretrain.hpp"
#include "snapjudge/sj_train.hpp"
#include "snapjudge/tokenizer.hpp"
#include "snapjudge/train.hpp"

namespace fs = std::filesystem;
using snapjudge::Tokenizer;

namespace {

struct Args {
  std::string stage;
  std::string out = "out/own";
  // init
  int vocab = 512, hidden = 128, layers = 4, heads = 8, intermediate = 512, max_pos = 128, head_layers = 1;
  uint64_t seed = 7;
  // pretrain
  std::string corpus;
  int p_steps = 1000;
  double p_lr = 1e-3;
  int p_every = 100;
  // finetune
  std::string data_jsonl, val_jsonl;
  int f_steps = 500;
  double f_lr = 2e-3;
  int f_every = 50;
  bool help = false;
};

void help() {
  std::puts(
      "snapjudge-own — build a native snapjudge decision model from scratch\n"
      "\n"
      "  snapjudge-own init      --out DIR [shape flags]\n"
      "  snapjudge-own pretrain  --out DIR --corpus FILE [--steps N --lr R]\n"
      "  snapjudge-own finetune  --out DIR --data train.jsonl [--val val.jsonl]\n"
      "\n"
      "init flags:\n"
      "  --vocab N --hidden N --layers N --heads N --intermediate N --max-pos N --seed N\n"
      "pretrain flags:\n"
      "  --corpus FILE (raw text, one document per line)  --steps N --lr R --every N\n"
      "finetune flags:\n"
      "  --data train.jsonl (typed-decisions schema)  --steps N --lr R --every N\n");
}

Args parse(int argc, char** argv) {
  Args a;
  a.stage = argc > 1 ? argv[1] : "";
  auto i = 2;
  auto take = [&](std::string& d) { if (i < argc) d = argv[i++]; };
  auto taked = [&](double& d) { if (i < argc) d = std::stod(argv[i++]); };
  auto takei = [&](int& d) { if (i < argc) d = std::stoi(argv[i++]); };
  auto takeu = [&](uint64_t& d) { if (i < argc) d = std::stoull(argv[i++]); };
  while (i < argc) {
    std::string t = argv[i++];
    if (t == "--out") take(a.out);
    else if (t == "--vocab") takei(a.vocab);
    else if (t == "--hidden") takei(a.hidden);
    else if (t == "--layers") takei(a.layers);
    else if (t == "--heads") takei(a.heads);
    else if (t == "--intermediate") takei(a.intermediate);
    else if (t == "--max-pos") takei(a.max_pos);
    else if (t == "--head-layers") takei(a.head_layers);
    else if (t == "--seed") takeu(a.seed);
    else if (t == "--corpus") take(a.corpus);
    else if (t == "--steps") { if (i < argc) { int n = std::stoi(argv[i++]); a.p_steps = n; a.f_steps = n; } }
    else if (t == "--lr") { if (i < argc) { double v = std::stod(argv[i++]); a.p_lr = v; a.f_lr = v; } }
    else if (t == "--every") { if (i < argc) { int n = std::stoi(argv[i++]); a.p_every = n; a.f_every = n; } }
    else if (t == "--data") take(a.data_jsonl);
    else if (t == "--val") take(a.val_jsonl);
    else if (t == "--help" || t == "-h") a.help = true;
  }
  return a;
}

// read a text corpus, one document per line, encode with the checkpoint's
// tokenizer, return the sequences (capped to max_len).
std::vector<std::vector<int64_t>> load_corpus(const std::string& path, const Tokenizer& tok,
                                              int64_t max_len) {
  std::vector<std::vector<int64_t>> out;
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open corpus " + path);
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    std::vector<int64_t> ids;
    ids.push_back(tok.cls_id);
    auto enc = tok.encode(line);
    for (auto id : enc) ids.push_back(id);
    ids.push_back(tok.sep_id);
    if ((int64_t)ids.size() > max_len) ids.resize((size_t)max_len);
    out.push_back(std::move(ids));
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  Args a = parse(argc, argv);
  if (a.help || a.stage.empty()) {
    help();
    return a.stage.empty() ? 2 : 0;
  }

  if (a.stage == "init") {
    snapjudge::init_sj_checkpoint(a.out, a.vocab, a.hidden, a.layers, a.heads,
                                  a.intermediate, a.max_pos, a.head_layers, a.seed);
    std::fprintf(stderr, "initialized native checkpoint -> %s (D=%d L=%d H=%d)\n",
                 a.out.c_str(), a.hidden, a.layers, a.heads);
    return 0;
  }

  if (a.stage == "pretrain") {
    if (a.corpus.empty()) { std::fprintf(stderr, "pretrain: --corpus required\n"); return 2; }
    snapjudge::SjPretrain p(a.out);
    auto tok = Tokenizer::cached_from_dir((fs::path(a.out) / "tokenizer").string());
    auto seqs = load_corpus(a.corpus, *tok, 128);
    std::fprintf(stderr, "pretrain: %zd sequences\n", seqs.size());
    int k = 0;
    for (int step = 0; step < a.p_steps; ++step) {
      p.zero_grad();
      const auto& s = seqs[k % seqs.size()];
      p.step(s, true);
      p.optimizer_step(a.p_lr);
      ++k;
      if (step % a.p_every == 0) {
        double loss = p.loss_double(s);
        std::fprintf(stderr, "pretrain step %d/%d loss %.4f\n", step, a.p_steps, loss);
      }
    }
    // save the pretrained encoder into a subdir for the next stage
    std::string pt = a.out + "-pretrained";
    p.save_checkpoint(pt);
    std::fprintf(stderr, "saved pretrained checkpoint -> %s\n", pt.c_str());
    return 0;
  }

  if (a.stage == "finetune") {
    if (a.data_jsonl.empty()) { std::fprintf(stderr, "finetune: --data required\n"); return 2; }
    // train on the pretrained checkpoint if present, else the init checkpoint
    std::string base = a.out;
    std::string pt = a.out + "-pretrained";
    if (fs::exists(pt)) base = pt;
    snapjudge::SjTrainModel tm(base);
    auto rows = snapjudge::load_train_rows(a.data_jsonl);
    std::fprintf(stderr, "finetune: %zd rows, %lld trainable params (base %s)\n",
                 rows.size(), (long long)tm.param_count(), base.c_str());
    for (int step = 0; step < a.f_steps; ++step) {
      tm.zero_grad();
      auto st = tm.step(rows[step % rows.size()], false);
      tm.optimizer_step(a.f_lr);
      if (step % a.f_every == 0)
        std::fprintf(stderr, "finetune step %d/%d loss %.4f acc %.3f\n",
                     step, a.f_steps, st.loss, st.acc);
    }
    std::string ft = a.out + "-ft";
    tm.save_checkpoint(ft);
    std::fprintf(stderr, "saved fine-tuned checkpoint -> %s\n", ft.c_str());
    return 0;
  }

  std::fprintf(stderr, "snapjudge-own: unknown stage '%s'\n", a.stage.c_str());
  help();
  return 2;
}
