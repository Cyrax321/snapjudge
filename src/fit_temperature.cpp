// snapjudge/src/fit_temperature.cpp — post-hoc temperature fitting for a
// trained checkpoint, run on the validation split. One scalar per
// (qtype, option-count bucket), fitted by 1-D golden-section NLL minimization.
//
// This is the step that moved mean ECE 0.466 -> 0.081 on the reference
// checkpoints; refit on data from your deployment domain before trusting
// confidence values.

#include "snapjudge/train.hpp"

#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "snapjudge/common.hpp"

namespace snapjudge {
namespace {

// forward all rows through the head after training (use TrainModel's public path).
// We reuse forward_loss pieces by re-implementing the eval forward:
// encoder(frozen) + head(trained) -> logits. TrainModel::step doubles as the
// eval forward when we read StepStats — but we need raw logits. Exposed in
// train.hpp via friend-free small bridge: `eval_logits`.

struct Bucket { int qtype; int klo, khi; std::string key; };

std::string bucket_key(int qtype, int k) {
  return temp_bucket(qtype, k);
}

// golden-section 1-D minimize
double fit_scalar(const std::vector<double>& logit, const std::vector<double>& tgt,
                  const std::vector<int>& ks) {
  // logit: concatenated rows; each row uses ks[r] leading entries
  auto nll = [&](double t) {
    double total = 0;
    size_t off = 0;
    for (size_t r = 0; r < ks.size(); ++r) {
      int k = ks[r];
      double mx = -1e30;
      for (int j = 0; j < k; ++j) mx = std::max(mx, logit[off + j] / t);
      double s = 0;
      std::vector<double> q(k);
      for (int j = 0; j < k; ++j) { q[j] = std::exp(logit[off + j] / t - mx); s += q[j]; }
      for (int j = 0; j < k; ++j) q[j] /= s;
      for (int j = 0; j < k; ++j) {
        double p = std::max(q[j], 1e-12);
        total += tgt[off + j] * std::log(p);
      }
      off += k;
    }
    return -total / std::max<size_t>(1, ks.size());
  };
  double a = 0.05, b = 5.0;
  const double gr = (std::sqrt(5.0) - 1) / 2;
  double c = b - gr * (b - a), d = a + gr * (b - a);
  double fc = nll(c), fd = nll(d);
  for (int i = 0; i < 80; ++i) {
    if (fc < fd) { b = d; d = c; fd = fc; c = b - gr * (b - a); fc = nll(c); }
    else { a = c; c = d; fc = fd; d = a + gr * (b - a); fd = nll(d); }
  }
  return (a + b) / 2;
}

}  // namespace

// Fit temperatures. Returns (temperature_by_qtype, temperature_by_options).
std::pair<std::vector<double>, std::map<std::string, double>> fit_temperatures(
    const std::vector<std::vector<float>>& logits,
    const std::vector<std::vector<double>>& targets,
    const std::vector<int>& qtypes) {
  // group by (qtype, bucket)
  std::map<std::string, std::vector<size_t>> groups;
  for (size_t r = 0; r < logits.size(); ++r)
    groups[temp_bucket(qtypes[r], (int)targets[r].size())].push_back(r);

  std::map<std::string, double> by_opt;
  for (auto& [key, idxs] : groups) {
    std::vector<double> L, T;
    std::vector<int> ks;
    for (size_t r : idxs) {
      for (size_t j = 0; j < targets[r].size(); ++j) {
        L.push_back(logits[r][j]);
        T.push_back(targets[r][j]);
      }
      ks.push_back((int)targets[r].size());
    }
    by_opt[key] = fit_scalar(L, T, ks);
  }
  // per qtype
  std::vector<double> per_type = {1.0, 1.0, 1.0};
  for (int qt = 0; qt < 3; ++qt) {
    std::vector<double> L, T;
    std::vector<int> ks;
    for (size_t r = 0; r < logits.size(); ++r) {
      if (qtypes[r] != qt) continue;
      for (size_t j = 0; j < targets[r].size(); ++j) {
        L.push_back(logits[r][j]);
        T.push_back(targets[r][j]);
      }
      ks.push_back((int)targets[r].size());
    }
    if (!ks.empty()) per_type[qt] = fit_scalar(L, T, ks);
  }
  return {per_type, by_opt};
}

}  // namespace snapjudge
