#pragma once
// snapjudge metrics.hpp: distribution/calibration metrics for eval reports and
// figures. Everything derives from per-question points — one implementation,
// used by snapjudge-eval (JSON fields) and snapjudge-plots (figures/tables).

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace snapjudge {

using nlohmann::ordered_json;

// One evaluated question: predicted distribution q, teacher distribution t,
// qtype int (0 choice / 1 score / 2 noul), reported confidence, gold-correct.
struct MetricPoint {
  int qtype;
  double conf;
  double correct;
  std::vector<double> q, t;
};

struct Metrics {
  int n = 0;
  double accuracy = NAN;          // argmax agreement
  double nll = NAN;               // mean cross-entropy vs teacher
  double soft = NAN;              // mean q·t
  double tv = NAN;                // mean total variation distance
  double brier = NAN;
  double ece = NAN;               // confidence-bin calibration error
  double mce = NAN;               // max calibration error across bins
  double score_mae = NAN;         // |E[q] - E[t]| on score rows (NAN if none)
  double spearman = NAN;          // E[q] vs E[t] rank corr on score rows
  double auroc_conf = NAN;        // AUROC of confidence predicting correctness
};

Metrics compute_metrics(const std::vector<MetricPoint>& points);

// Per-type accuracy breakdown (choice/noul/score).
std::map<std::string, std::pair<int, int>> accuracy_by_type(
    const std::vector<MetricPoint>& points);

// LaTeX booktabs table body for the paper (rows = metric).
std::string metrics_latex(const Metrics& m, const std::string& ckpt_name);

}  // namespace snapjudge
