// snapjudge-train: train a snapjudge decision head on typed-decisions-format
// JSONL data, fit temperatures on the eval split, write a full checkpoint dir.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "snapjudge/common.hpp"
#include "snapjudge/train.hpp"

using snapjudge::TrainModel;
using snapjudge::TrainRow;

namespace snapjudge {} // fit_temperatures declared in snapjudge/train.hpp

namespace {

struct Args {
  std::string base;            // checkpoint dir or hub id
  std::string train_jsonl;
  std::string val_jsonl;
  std::string out = "out/checkpoint-ft";
  double epochs = 1.0;
  int rows_per_step = 2;
  int accum = 8;
  double lr = 2e-5;
  double warmup = 0.03;
  int seed = 17;
  bool fit_temperature = true;
  bool help = false;
};

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string t = argv[i];
    auto take = [&](std::string& d) {
      if (i + 1 < argc) d = argv[++i];
    };
    auto taked = [&](double& d) {
      if (i + 1 < argc) d = std::stod(argv[++i]);
    };
    auto takei = [&](int& d) {
      if (i + 1 < argc) d = std::stoi(argv[++i]);
    };
    if (t == "--base") take(a.base);
    else if (t == "--data") take(a.train_jsonl);
    else if (t == "--val") take(a.val_jsonl);
    else if (t == "--out") take(a.out);
    else if (t == "--epochs") taked(a.epochs);
    else if (t == "--rows-per-step") takei(a.rows_per_step);
    else if (t == "--accum") takei(a.accum);
    else if (t == "--lr") taked(a.lr);
    else if (t == "--warmup") taked(a.warmup);
    else if (t == "--seed") takei(a.seed);
    else if (t == "--no-fit-temperature") a.fit_temperature = false;
    else if (t == "--help" || t == "-h") a.help = true;
  }
  return a;
}

void help() {
  std::puts(
      "snapjudge-train — fine-tune a snapjudge decision checkpoint\n"
      "  --base <dir|hub-id>      base checkpoint (dir with rl_agent_config.json, required)\n"
      "  --data <train.jsonl>     training data (typed-decisions JSONL)\n"
      "  --val <val.jsonl>        validation data (default: data with val.jsonl suffix swap)\n"
      "  --out <dir>              output checkpoint dir (default out/checkpoint-ft)\n"
      "  --epochs N  --lr R  --rows-per-step N  --accum N  --warmup F  --seed N\n"
      "  --no-fit-temperature     skip post-hoc temperature fitting\n");
}

}  // namespace

int main(int argc, char** argv) {
  Args a = parse(argc, argv);
  if (a.help || a.base.empty() || a.train_jsonl.empty()) {
    help();
    if (a.base.empty()) {
      std::fprintf(stderr, "snapjudge-train: --base is required\n");
      return 2;
    }
    if (a.train_jsonl.empty()) {
      std::fprintf(stderr, "snapjudge-train: --data is required\n");
      return 2;
    }
    return 0;
  }

  if (a.val_jsonl.empty()) {
    // replace .jsonl with val.jsonl if present, else train itself
    std::string v = a.train_jsonl;
    auto pos = v.rfind(".jsonl");
    if (pos != std::string::npos) v.replace(pos, 6, "val.jsonl");
    a.val_jsonl = std::ifstream(v) ? v : a.train_jsonl;
  }

  std::fprintf(stderr, "snapjudge-train: base=%s data=%s val=%s out=%s\n",
               a.base.c_str(), a.train_jsonl.c_str(), a.val_jsonl.c_str(), a.out.c_str());

  auto rows = snapjudge::load_train_rows(a.train_jsonl);
  auto val_rows = snapjudge::load_train_rows(a.val_jsonl);
  std::fprintf(stderr, "loaded %zd train rows, %zd val rows\n",
               rows.size(), val_rows.size());

  TrainModel tm(a.base);
  std::fprintf(stderr, "trainable: %lld params across %zd tensors\n",
               (long long)tm.param_count(), tm.training_param_names().size());

  std::mt19937 rng(a.seed);
  auto t0 = std::chrono::steady_clock::now();
  int64_t total_micro = (int64_t)std::ceil(a.epochs * rows.size() / a.rows_per_step);
  int64_t warmup_steps = (int64_t)(total_micro * a.warmup);
  int64_t micro = 0;
  for (int epoch = 0; epoch < (int)std::ceil(a.epochs); ++epoch) {
    std::shuffle(rows.begin(), rows.end(), rng);
    for (size_t start = 0; start < rows.size(); start += (size_t)a.rows_per_step) {
      tm.zero_grad();
      int64_t nrows = 0;
      double loss = 0, acc = 0;
      size_t end = std::min(rows.size(), start + (size_t)a.rows_per_step);
      for (size_t si = start; si < end; ++si) {
        auto st = tm.step(rows[si], true);
        loss += st.loss * st.rows;
        acc += st.acc * st.rows;
        nrows += st.rows;
      }
      loss /= std::max<int64_t>(1, nrows);
      acc /= std::max<int64_t>(1, nrows);

      double lr = a.lr;
      if (micro < warmup_steps) lr = a.lr * (double)micro / std::max<int64_t>(1, warmup_steps);
      double p = (double)(micro - warmup_steps) / std::max<int64_t>(1, total_micro - warmup_steps);
      if (p > 0) lr = a.lr * 0.5 * (1 + std::cos(M_PI * std::min(1.0, p)));

      if ((micro + 1) % a.accum == 0 || micro + 1 == total_micro) {
        tm.optimizer_step(lr);
        tm.zero_grad();
      }
      if (micro % 10 == 0) {
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "step %5lld/%lld  loss %.4f  acc %.3f  lr %.2e  (%.0fs)\n",
                     (long long)micro, (long long)total_micro, loss, acc, lr, el);
      }
      ++micro;
      if (micro >= total_micro) break;
    }
    if (micro >= total_micro) break;
  }

  std::vector<std::vector<float>> logits;
  std::vector<std::vector<double>> targets;
  std::vector<int> qtypes;
  // eval with the trained head
  tm.eval_logits(val_rows, &logits, &targets, &qtypes);
  {
    // pre-temp accuracy
    int right = 0, total = 0;
    for (size_t r = 0; r < logits.size(); ++r) {
      int argmax = 0, gold = 0;
      for (size_t j = 0; j < targets[r].size(); ++j) {
        if (logits[r][j] > logits[r][argmax]) argmax = (int)j;
        if (targets[r][j] > targets[r][gold]) gold = (int)j;
      }
      right += argmax == gold;
      ++total;
    }
    std::fprintf(stderr, "pre-temp accuracy: %.4f (%d/%d)\n",
                 (double)right / std::max(1, total), right, total);
  }

  std::vector<double> temperature = {1.0, 1.0, 1.0};
  std::map<std::string, double> tb;
  if (a.fit_temperature) {
    auto [t, buckets] = snapjudge::fit_temperatures(logits, targets, qtypes);
    temperature = t;
    tb = buckets;
    std::fprintf(stderr, "fitted temperature=[%.3f, %.3f, %.3f], %zd buckets\n",
                 t[0], t[1], t[2], buckets.size());
  }

  double hours = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 3600.;
  tm.save_checkpoint(a.out, temperature, tb,
                     nlohmann::ordered_json{
                         {"updates", (long long)total_micro},
                         {"epochs_completed", a.epochs},
                         {"hours", std::round(hours * 100) / 100},
                         {"dataset", "typed-decisions"},
                         {"runtime", "snapjudge"},
                         {"supervised_proper_score", true},
                     });
  std::fprintf(stderr, "saved checkpoint -> %s\n", a.out.c_str());
  return 0;
}
