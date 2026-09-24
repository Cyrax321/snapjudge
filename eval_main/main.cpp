// snapjudge-eval: evaluate a checkpoint on typed-decisions-format JSONL.
//
// Runs the REAL deployment decode path (snapjudge::Agent, temperatures from the
// checkpoint config applied) — eval numbers are what production will return.

#include <cmath>
#include <fstream>
#include <map>
#include <cstdio>
#include <string>
#include <vector>

#include "snapjudge/agent.hpp"
#include "snapjudge/common.hpp"
#include "snapjudge/train.hpp"
#include "snapjudge/metrics.hpp"

using snapjudge::Agent;
using snapjudge::TrainRow;
using nlohmann::ordered_json;

namespace {

// decode one train row through the agent; answers keyed per qid.
struct RowResult { int right, total; double brier, soft;
                   std::vector<double> conf, corr; };

struct QPoint { double conf; double correct; std::string type; std::string label_gold;
                std::string label_pred; std::string qid; std::string workflow;
                std::vector<double> q, t; };

RowResult eval_row(Agent& agent, const TrainRow& row, std::vector<QPoint>* points) {
  ordered_json qs = ordered_json::object();
  for (size_t i = 0; i < row.qids.size(); ++i) qs[row.qids[i]] = row.qdefs[i];
  ordered_json res = agent.system_one(row.state, qs);
  RowResult rr{0, 0, 0, 0, {}, {}};
  for (size_t i = 0; i < row.qids.size(); ++i) {
    const auto& qid = row.qids[i];
    const auto& t = row.targets[i];
    const ordered_json& ans = res["answers"][qid];
    std::string ty = ans.value("type", "");
    std::vector<double> q(t.size(), 0.0);
    if (ty == "choice") {
      auto it2 = row.qdefs[i];
      int j = 0;
      for (auto it = it2["criteria"].begin(); it != it2["criteria"].end(); ++it, ++j)
        q[j] = ans["probabilities"].value(it.key(), 0.0);
    } else if (ty == "score") {
      for (size_t j = 0; j < t.size(); ++j)
        q[j] = ans["probabilities"].value(std::to_string(j), 0.0);
    } else {  // noul: prob of true is index 1 -> q=[1-p, p]
      double p = ans.value("noul", 0.5);
      q = {1.0 - p, p};
    }
    int gold = 0;
    for (size_t j = 1; j < t.size(); ++j) if (t[j] > t[gold]) gold = (int)j;
    int pred = 0;
    for (size_t j = 1; j < q.size(); ++j) if (q[j] > q[pred]) pred = (int)j;
    rr.right += pred == gold;
    rr.total += 1;
    double b = 0, s = 0;
    for (size_t j = 0; j < q.size(); ++j) {
      double d = q[j] - t[j];
      b += d * d;
      s += q[j] * t[j];   // teacher-soft score: inner product
    }
    rr.brier += b;
    rr.soft += s;
    double conf = snapjudge::confidence_from_probs(q.data(), (int)q.size());
    rr.conf.push_back(conf);
    rr.corr.push_back(pred == gold ? 1.0 : 0.0);
    if (points) {
      points->push_back({conf, pred == gold ? 1.0 : 0.0, ty,
                         "", "", qid, row.workflow, q, t});
      auto& pt = points->back();
      // label names for choice/score rows
      if (ty == "choice") {
        int j = 0;
        for (auto it = row.qdefs[i]["criteria"].begin();
             it != row.qdefs[i]["criteria"].end(); ++it, ++j) {
          if (j == gold) pt.label_gold = it.key();
          if (j == pred) pt.label_pred = it.key();
        }
      } else if (ty == "score") {
        pt.label_gold = std::to_string(gold);
        pt.label_pred = std::to_string(pred);
      } else {
        pt.label_gold = gold ? "true" : "false";
        pt.label_pred = pred ? "true" : "false";
      }
    }
  }
  return rr;
}

}  // namespace

int main(int argc, char** argv) {
  std::string ckpt, data, json_out;
  int limit = 0;
  for (int i = 1; i < argc; ++i) {
    std::string t = argv[i];
    if (t == "--ckpt" && i + 1 < argc) ckpt = argv[++i];
    else if (t == "--data" && i + 1 < argc) data = argv[++i];
    else if (t == "--json" && i + 1 < argc) json_out = argv[++i];
    else if (t == "--n" && i + 1 < argc) limit = std::stoi(argv[++i]);
    else if (t == "--help" || t == "-h") {
      std::puts("snapjudge-eval --ckpt <dir|hub-id> --data <val.jsonl> [--n N] [--json out.json]");
      return 0;
    }
  }
  if (ckpt.empty() || data.empty()) {
    std::fprintf(stderr, "snapjudge-eval: --ckpt and --data are required\n");
    return 2;
  }

  auto rows = snapjudge::load_train_rows(data);
  if (limit > 0 && (size_t)limit < rows.size()) rows.resize((size_t)limit);
  Agent agent(ckpt, "cpu");

  int right = 0, total = 0;
  double brier = 0, soft = 0;
  std::vector<double> conf, corr;
  std::vector<QPoint> points;
  std::map<std::string, std::pair<int, int>> per_wf;

  size_t i = 0;
  for (const auto& row : rows) {
    auto rr = eval_row(agent, row, &points);
    right += rr.right; total += rr.total;
    brier += rr.brier; soft += rr.soft;
    for (double c : rr.conf) conf.push_back(c);
    for (double c : rr.corr) corr.push_back(c);
    auto& w = per_wf[row.workflow];
    w.first += rr.right; w.second += rr.total;
    if (++i % 25 == 0) std::fprintf(stderr, "  ... %zd/%zd rows\n", i, rows.size());
  }

  double acc = (double)right / std::max(1, total);
  std::printf("snapjudge eval: %s  (rows=%zd, questions=%d)\n", ckpt.c_str(),
              rows.size(), total);
  std::printf("  accuracy    %.4f\n", acc);
  std::printf("  soft score  %.4f   (mean q·teacher)\n", soft / std::max(1, total));
  std::printf("  brier       %.4f   (lower better)\n", brier / std::max(1, total));
  std::printf("  ece         %.4f   (lower better, post-temperature)\n",
              snapjudge::ece_score(conf, corr));
  std::printf("  per-workflow accuracy:\n");
  for (const auto& [wf, rc] : per_wf)
    std::printf("    %-28s %.4f (%d/%d)\n", wf.c_str(), (double)rc.first / rc.second,
                rc.first, rc.second);

  if (!json_out.empty()) {
    // machine-readable eval report for figure generation
    ordered_json rep;
    rep["checkpoint"] = ckpt;
    rep["data"] = data;
    rep["rows"] = rows.size();
    rep["questions"] = total;
    rep["accuracy"] = acc;
    rep["soft"] = soft / std::max(1, total);
    rep["brier"] = brier / std::max(1, total);
    rep["ece"] = snapjudge::ece_score(conf, corr);
    // extended publication metrics from the same point set
    {
      std::vector<snapjudge::MetricPoint> mpts;
      for (const auto& p : points) {
        snapjudge::MetricPoint mp;
        mp.qtype = p.type == "choice" ? 0 : p.type == "score" ? 1 : 2;
        mp.conf = p.conf;
        mp.correct = p.correct;
        mp.q = p.q;
        mp.t = p.t;
        mpts.push_back(std::move(mp));
      }
      auto mm = snapjudge::compute_metrics(mpts);
      rep["nll"] = mm.nll;
      rep["tv"] = mm.tv;
      rep["mce"] = mm.mce;
      rep["auroc_conf"] = mm.auroc_conf;
      if (!std::isnan(mm.score_mae)) rep["score_mae"] = mm.score_mae;
      if (!std::isnan(mm.spearman)) rep["spearman"] = mm.spearman;
    }
    ordered_json wfj = ordered_json::object();
    for (const auto& [wf, rc] : per_wf)
      wfj[wf] = {{"right", rc.first}, {"total", rc.second}};
    rep["per_workflow"] = wfj;
    ordered_json pts = ordered_json::array();
    for (const auto& p : points)
      pts.push_back({{"qid", p.qid}, {"workflow", p.workflow}, {"type", p.type},
                     {"conf", p.conf}, {"correct", p.correct},
                     {"label_gold", p.label_gold}, {"label_pred", p.label_pred},
                     {"q", p.q}, {"t", p.t}});
    rep["points"] = pts;
    std::ofstream of(json_out);
    if (!of) {
      std::fprintf(stderr, "snapjudge-eval: cannot write %s\n", json_out.c_str());
      return 2;
    }
    of << rep.dump(2);
    std::fprintf(stderr, "wrote eval report -> %s\n", json_out.c_str());
  }
  return 0;
}
