// snapjudge-plots: turn an eval --json report into SVG figures.
// No image library: hand-written SVG (the toolchain stays dependency-free).

#include <algorithm>
#include <cmath>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "canvas.hpp"
#include "snapjudge/metrics.hpp"

using nlohmann::ordered_json;

namespace {

// Figure records primitives once; they render to SVG and PNG identically.
struct Figure {
  int w = 800, h = 500;
  struct Prim {
    enum { RECT, LINE, TEXT } kind;
    double x, y, x2, y2;
    std::string a;      // color (fill/stroke or text) or text string (kind==TEXT)
    std::string b;      // secondary: dashed marker / css class ("label"/"tick"/"title")
    std::string c;      // text anchor
    double wpx = 1.0, opacity = 1.0;
  };
  std::ostringstream svg;
  std::vector<Prim> prims;

  Figure(int w_, int h_, const char* title) : w(w_), h(h_) {
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << w << "\" height=\""
        << h << "\">\n"
        << "<style>text{font-family:Helvetica,Arial,sans-serif;font-size:13px}"
        << ".label{fill:#333}.tick{fill:#777;font-size:11px}"
        << ".title{font-size:16px;font-weight:bold;fill:#111}</style>\n"
        << "<text x=\"20\" y=\"26\" class=\"title\">" << title << "</text>\n";
    prims.push_back({Prim::TEXT, 20, 26, 0, 0, title, "title", "start", 1.0, 1.0});
  }
  void rect(double x, double y, double rw, double rh, const char* fill, double opacity = 1.0) {
    prims.push_back({Prim::RECT, x, y, rw, rh, fill, "", "", 1.0, opacity});
    svg << "<rect x=\"" << x << "\" y=\"" << y << "\" width=\"" << rw << "\" height=\"" << rh
        << "\" fill=\"" << fill << "\" fill-opacity=\"" << opacity << "\"/>\n";
  }
  void line(double x1, double y1, double x2, double y2, const char* stroke,
            double wpx = 1.0, const char* dash = "") {
    prims.push_back({Prim::LINE, x1, y1, x2, y2, stroke, dash, "", wpx, 1.0});
    svg << "<line x1=\"" << x1 << "\" y1=\"" << y1 << "\" x2=\"" << x2 << "\" y2=\"" << y2
        << "\" stroke=\"" << stroke << "\" stroke-width=\"" << wpx << "\""
        << (std::string(dash).empty() ? "" : (std::string(" stroke-dasharray=\"") + dash + "\""))
        << "/>\n";
  }
  void text(double x, double y, const std::string& t, const char* cls = "label",
            const char* anchor = "start") {
    prims.push_back({Prim::TEXT, x, y, 0, 0, t, cls, anchor, 1.0, 1.0});
    svg << "<text x=\"" << x << "\" y=\"" << y << "\" class=\"" << cls
        << "\" text-anchor=\"" << anchor << "\">" << t << "</text>\n";
  }
  std::string svg_str() const { return svg.str() + "</svg>\n"; }

  // Rasterize to PNG.
  bool write_png(const std::string& path) const {
    snapjudge::plots::Canvas cx(w, h);
    cx.clear();
    for (const auto& p : prims) {
      if (p.kind == Prim::RECT) {
        uint32_t h6 = std::stoul(p.a.substr(1), nullptr, 16);
        cx.fill_rect((int)std::lround(p.x), (int)std::lround(p.y),
                     (int)std::lround(p.x2), (int)std::lround(p.y2),
                     snapjudge::plots::Color::hex(h6, (uint8_t)(p.opacity * 255)));
      } else if (p.kind == Prim::LINE) {
        uint32_t h6 = std::stoul(p.a.substr(1), nullptr, 16);
        if (!p.b.empty())
          cx.draw_dashed_line((int)std::lround(p.x), (int)std::lround(p.y),
                              (int)std::lround(p.x2), (int)std::lround(p.y2),
                              snapjudge::plots::Color::hex(h6), (int)std::lround(p.wpx));
        else
          cx.draw_line((int)std::lround(p.x), (int)std::lround(p.y),
                       (int)std::lround(p.x2), (int)std::lround(p.y2),
                       snapjudge::plots::Color::hex(h6), (int)std::lround(p.wpx));
      } else {
        uint32_t h6 = 0x333333;
        if (p.b == "tick") h6 = 0x777777;
        else if (p.b == "title") h6 = 0x111111;
        cx.draw_text((int)std::lround(p.x), (int)std::lround(p.y) - 7, p.a,
                     snapjudge::plots::Color::hex(h6), p.b == "title" ? 2 : 1,
                     p.c.c_str());
      }
    }
    return cx.write_png(path);
  }
};

// reliability diagram: mean confidence vs empirical accuracy, binned
void fig_calibration(const std::vector<double>& conf, const std::vector<double>& corr,
                     const std::string& title, const std::string& out) {
  const int B = 15;
  double bin_acc[15] = {0}, bin_conf[15] = {0};
  int bin_n[15] = {0};
  for (size_t i = 0; i < conf.size(); ++i) {
    int b = std::min(B - 1, (int)(conf[i] * B));
    bin_conf[b] += conf[i];
    bin_acc[b] += corr[i];
    bin_n[b]++;
  }
  Figure v(800, 500, title.c_str());
  double px0 = 80, py0 = 430, pw = 640, ph = 350;
  v.rect(px0, py0 - ph, pw, ph, "#f8f8fa");
  for (int t = 0; t <= 10; ++t) {
    double yy = py0 - ph * t / 10;
    v.line(px0, yy, px0 + pw, yy, "#e4e4ea");
    v.text(px0 - 8, yy + 4, std::to_string(t / 10.0).substr(0, 4), "tick", "end");
  }
  // diagonal ideal
  v.line(px0, py0, px0 + pw, py0 - ph, "#999", 1.2, "5,5");
  v.text(px0 + pw - 110, py0 - ph + 22, "perfect calibration", "tick");
  // bars + mean points
  const char* BLUE = "#3b82f6", *RED = "#ef4444";
  for (int b = 0; b < B; ++b) {
    if (!bin_n[b]) continue;
    double bw = pw / B, xx = px0 + b * bw;
    double acc = bin_acc[b] / bin_n[b], mc = bin_conf[b] / bin_n[b];
    v.rect(xx + 2, py0 - ph * acc, bw - 4, ph * acc, BLUE, 0.28);
    v.text(xx + bw / 2, py0 + 14, std::to_string((int)std::round((b + 0.5) / B * 100)) + "%", "tick", "middle");
    // mean-confidence dot + error tick
    double dx = px0 + mc * pw, dy = py0 - acc * ph;
    double err = acc - mc;
    v.line(dx - 5, dy, dx + 5, dy, err > 0 ? BLUE : RED, 3.0);
    v.text(xx + bw / 2, py0 - ph * acc - 6, std::to_string(bin_n[b]), "tick", "middle");
  }
  v.text(px0 - 40, py0 - ph / 2, "empirical accuracy", "tick", "middle");
  v.text(px0 + pw / 2, py0 + 40, "reported confidence (binned, count on top)", "tick", "middle");
  
    std::ofstream(out) << v.svg_str();
    std::string png = out;
    if (png.size() > 4 && png.substr(png.size()-4) == ".svg")
      png.replace(png.size()-4, 4, ".png");
    if (!v.write_png(png))
      std::fprintf(stderr, "snapjudge-plots: PNG write failed for %s\n", png.c_str());
}

// per-workflow accuracy bars
void fig_workflows(const ordered_json& per_wf, const std::string& title,
                   const std::string& out) {
  Figure v(900, 100 + 46 * (int)per_wf.size() + 40, title.c_str());
  double x0 = 260, pw = 560;
  int i = 0;
  for (auto it = per_wf.begin(); it != per_wf.end(); ++it, ++i) {
    double acc = it.value().value("total", 0)
                     ? (double)it.value().value("right", 0) / it.value().value("total", 1)
                     : 0;
    double y = 60 + 46 * i;
    v.text(x0 - 12, y + 18, it.key(), "label", "end");
    v.rect(x0, y, pw, 34, "#e5e7eb");
    v.rect(x0, y, pw * acc, 34, "#3b82f6");
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.3f  (%d/%d)", acc, it.value().value("right", 0),
                  it.value().value("total", 0));
    v.text(x0 + pw + 8, y + 18, buf, "tick");
  }
  // guide lines at 0.25 / 0.5 / 0.75
  for (double g : {0.25, 0.5, 0.75, 1.0}) {
    double xx = x0 + pw * g;
    v.line(xx, 50, xx, 50 + 46 * per_wf.size(), "#eee");
    v.text(xx, 45 + 46 * (int)per_wf.size() + 14, std::to_string(g), "tick", "middle");
  }
  
    std::ofstream(out) << v.svg_str();
    std::string png = out;
    if (png.size() > 4 && png.substr(png.size()-4) == ".svg")
      png.replace(png.size()-4, 4, ".png");
    if (!v.write_png(png))
      std::fprintf(stderr, "snapjudge-plots: PNG write failed for %s\n", png.c_str());
}

// confusion matrix for a type (choice labels, or true/false for noul, or score levels)
void fig_confusion(const std::string& type, const std::vector<std::string>& labels,
                   const std::map<std::pair<std::string, std::string>, int>& cells,
                   const std::string& title, const std::string& out) {
  int n = (int)labels.size();
  int cell = std::max(30, 420 / std::max(1, n));
  int top = 70, left = 200;
  Figure v(left + n * cell + 60, top + n * cell + 90, title.c_str());
  v.text(left + n * cell / 2.0, 50, title, "title", "middle");
  for (int j = 0; j < n; ++j) {
    v.text(left + j * cell + cell / 2.0, top - 8, labels[j], "tick", "middle");
    std::string lab = labels[j];
    // break long labels
    if (lab.size() > 14) lab = lab.substr(0, 13) + "…";
    v.text(left - 8, top + j * cell + cell / 2.0 + 4, lab, "tick", "end");
  }
  int mx = 0;
  for (const auto& [k, c] : cells) mx = std::max(mx, c);
  for (int g = 0; g < n; ++g)
    for (int p = 0; p < n; ++p) {
      int c = 0;
      auto it = cells.find({labels[g], labels[p]});
      if (it != cells.end()) c = it->second;
      double f = mx ? (double)c / mx : 0;
      v.rect(left + p * cell, top + g * cell, cell - 2, cell - 2,
             g == p ? "#22c55e" : "#ef4444", g == p ? 0.25 + 0.55 * f : 0.06 + 0.30 * f);
      if (c) v.text(left + p * cell + cell / 2.0, top + g * cell + cell / 2.0 + 4,
                    std::to_string(c), "label", "middle");
    }
  v.text(left + n * cell / 2.0, top - 34, "predicted", "tick", "middle");
  
    std::ofstream(out) << v.svg_str();
    std::string png = out;
    if (png.size() > 4 && png.substr(png.size()-4) == ".svg")
      png.replace(png.size()-4, 4, ".png");
    if (!v.write_png(png))
      std::fprintf(stderr, "snapjudge-plots: PNG write failed for %s\n", png.c_str());
}

// ROC curve: confidence -> correctness gate
void fig_roc(const std::vector<snapjudge::MetricPoint>& pts, double auroc_v,
             const std::string& title, const std::string& out) {
  std::vector<std::pair<double, double>> sc;   // (conf, correct)
  for (const auto& p : pts) sc.emplace_back(p.conf, p.correct >= 0.5 ? 1.0 : 0.0);
  std::sort(sc.begin(), sc.end(), [](auto a, auto b) { return a.first > b.first; });
  double pos = 0, neg = 0;
  for (const auto& [c, y] : sc) { pos += y; neg += 1 - y; }
  Figure v(700, 600, title.c_str());
  double px0 = 70, py0 = 500, pw = 560, ph = 400;
  v.rect(px0, py0 - ph, pw, ph, "#f8f8fa");
  for (int t = 0; t <= 10; ++t) {
    double xx = px0 + pw * t / 10.0;
    v.line(xx, py0, xx, py0 - ph, "#e8e8ee");
    v.text(xx, py0 + 15, std::to_string(t / 10.0).substr(0, 4), "tick", "middle");
    double yy = py0 - ph * t / 10.0;
    v.line(px0, yy, px0 + pw, yy, "#e8e8ee");
    v.text(px0 - 10, yy + 4, std::to_string(t / 10.0).substr(0, 4), "tick", "end");
  }
  v.line(px0, py0, px0 + pw, py0 - ph, "#bbb", 1.0, "6,4");  // chance
  v.text(px0 + pw * 0.55, py0 - ph * 0.55 + 4, "chance", "tick");
  // walk the ranked list
  double tp = 0, fp = 0;
  double lx = px0, ly = py0;
  for (const auto& [c, y] : sc) {
    if (y >= 0.5) tp += 1; else fp += 1;
    double nx = px0 + pw * fp / std::max(1.0, neg);
    double ny = py0 - ph * tp / std::max(1.0, pos);
    v.line(lx, ly, nx, ly, "#2563eb", 2.0);
    v.line(nx, ly, nx, ny, "#2563eb", 2.0);
    lx = nx; ly = ny;
  }
  char buf[96];
  std::snprintf(buf, sizeof(buf), "AUROC = %.4f", auroc_v);
  v.text(px0 + pw - 20, py0 - 16, buf, "label", "end");
  v.text(px0 + pw / 2, py0 + 46, "false positive rate", "tick", "middle");
  v.text(px0 - 40, py0 - ph / 2, "true positive rate", "tick", "middle");
  
    std::ofstream(out) << v.svg_str();
    std::string png = out;
    if (png.size() > 4 && png.substr(png.size()-4) == ".svg")
      png.replace(png.size()-4, 4, ".png");
    if (!v.write_png(png))
      std::fprintf(stderr, "snapjudge-plots: PNG write failed for %s\n", png.c_str());
}

// Precision-recall across pooled noul questions (gold = q[1] >= 0.5)
void fig_pr_noul(const std::vector<snapjudge::MetricPoint>& pts,
                 const std::string& title, const std::string& out) {
  std::vector<std::pair<double, double>> pr_pts;  // (p_true_pred, gold_is_true)
  for (const auto& p : pts) {
    if (p.qtype != 2 || p.q.size() < 2) continue;
    pr_pts.emplace_back(p.q[1], p.t[1] >= 0.5 ? 1.0 : 0.0);
  }
  std::sort(pr_pts.begin(), pr_pts.end(), [](auto a, auto b) { return a.first > b.first; });
  double pos = 0;
  for (const auto& [s, y] : pr_pts) pos += y;
  Figure v(700, 600, title.c_str());
  double px0 = 70, py0 = 500, pw = 560, ph = 400;
  v.rect(px0, py0 - ph, pw, ph, "#f8f8fa");
  double base = pos / std::max<size_t>(1, pr_pts.size());
  v.line(px0, py0 - ph * base, px0 + pw, py0 - ph * base, "#bbb", 1.0, "6,4");
  char pb[64];
  std::snprintf(pb, sizeof(pb), "prevalence %.3f", base);
  v.text(px0 + 8, py0 - ph * base - 6, pb, "tick");
  double tp = 0, fp = 0;
  double lx = px0, ly = py0 - ph;  // recall 0, precision 1
  std::map<double, double> best;   // recall -> best precision
  for (const auto& [s, y] : pr_pts) {
    if (y >= 0.5) tp += 1; else fp += 1;
    double recall = tp / std::max(1.0, pos);
    double prec = tp / std::max(0.5, tp + fp);
    best[recall] = std::max(best[recall], prec);
  }
  // monotone hull
  double bprec = 0;
  std::vector<std::pair<double, double>> hull;
  for (auto it = best.rbegin(); it != best.rend(); ++it) {
    bprec = std::max(bprec, it->second);
    hull.emplace_back(it->first, bprec);
  }
  std::reverse(hull.begin(), hull.end());
  lx = px0; ly = py0 - ph;
  for (const auto& [r, pp] : hull) {
    double nx = px0 + pw * r;
    double ny = py0 - ph * pp;
    v.line(lx, ly, nx, ly, "#7c3aed", 2.0);
    v.line(nx, ly, nx, ny, "#7c3aed", 2.0);
    lx = nx; ly = ny;
  }
  v.text(px0 + pw / 2, py0 + 46, "recall (of true answers)", "tick", "middle");
  v.text(px0 - 40, py0 - ph / 2, "precision", "tick", "middle");
  
    std::ofstream(out) << v.svg_str();
    std::string png = out;
    if (png.size() > 4 && png.substr(png.size()-4) == ".svg")
      png.replace(png.size()-4, 4, ".png");
    if (!v.write_png(png))
      std::fprintf(stderr, "snapjudge-plots: PNG write failed for %s\n", png.c_str());
}

// Score spread: expected score, teacher vs predicted (score rows)
void fig_score_spread(const std::vector<snapjudge::MetricPoint>& pts, double spearman_v,
                      const std::string& title, const std::string& out) {
  std::vector<std::pair<double, double>> sc;   // (E[t], E[q])
  double vmax = 0;
  for (const auto& p : pts) {
    if (p.qtype != 1) continue;
    double eq = 0, et = 0;
    for (size_t j = 0; j < p.q.size(); ++j) { eq += j * p.q[j]; et += j * p.t[j]; }
    sc.emplace_back(et, eq);
    vmax = std::max(vmax, std::max(et, eq));
  }
  Figure v(700, 600, title.c_str());
  double px0 = 70, py0 = 500, pw = 560, ph = 400;
  v.rect(px0, py0 - ph, pw, ph, "#f8f8fa");
  v.line(px0, py0, px0 + pw, py0 - ph, "#bbb", 1.0, "6,4");
  v.text(px0 + pw * 0.6, py0 - ph * 0.6 + 4, "ideal", "tick");
  for (const auto& [et, eq] : sc) {
    double xx = px0 + pw * et / vmax, yy = py0 - ph * eq / vmax;
    v.rect(xx - 4, yy - 4, 8, 8, "#0ea5e9", 0.45);
  }
  char buf[96];
  std::snprintf(buf, sizeof(buf), "spearman %.4f   n=%zd", spearman_v, sc.size());
  v.text(px0 + 8, py0 - ph + 22, buf, "label");
  v.text(px0 + pw / 2, py0 + 46, "teacher expected score", "tick", "middle");
  v.text(px0 - 40, py0 - ph / 2, "predicted expected score", "tick", "middle");
  
    std::ofstream(out) << v.svg_str();
    std::string png = out;
    if (png.size() > 4 && png.substr(png.size()-4) == ".svg")
      png.replace(png.size()-4, 4, ".png");
    if (!v.write_png(png))
      std::fprintf(stderr, "snapjudge-plots: PNG write failed for %s\n", png.c_str());
}

}  // namespace

int main(int argc, char** argv) {
  std::string json_in, out_dir = "figures";
  for (int i = 1; i < argc; ++i) {
    std::string t = argv[i];
    if (t == "--report" && i + 1 < argc) json_in = argv[++i];
    else if (t == "--out" && i + 1 < argc) out_dir = argv[++i];
    else if (t == "--help" || t == "-h") {
      std::puts("snapjudge-plots --report eval.json [--out dir]\n"
                "  figures: calibration.svg  workflows.svg  confusion-<type>.svg  report.md");
      return 0;
    }
  }
  if (json_in.empty()) {
    std::fprintf(stderr, "snapjudge-plots: --report is required\n");
    return 2;
  }
  ordered_json rep;
  std::ifstream(json_in) >> rep;

  // full metric set (adds nll/mce/tv/score_mae/spearman/auroc_conf)
  std::vector<snapjudge::MetricPoint> mpts;
  for (const auto& p : rep["points"]) {
    snapjudge::MetricPoint mp;
    mp.qtype = p.value("type", std::string()) == "choice" ? 0
               : p.value("type", std::string()) == "score" ? 1 : 2;
    mp.conf = p.value("conf", 0.0);
    mp.correct = p.value("correct", 0.0);
    mp.q = p["q"].get<std::vector<double>>();
    mp.t = p["t"].get<std::vector<double>>();
    mpts.push_back(std::move(mp));
  }
  snapjudge::Metrics mm = snapjudge::compute_metrics(mpts);

  std::filesystem::create_directories(out_dir);

  // confidence / correctness arrays
  std::vector<double> conf, corr;
  for (const auto& p : rep["points"]) {
    conf.push_back(p.value("conf", 0.0));
    corr.push_back(p.value("correct", 0.0));
  }
  fig_calibration(conf, corr,
                  "snapjudge reliability — " + rep.value("checkpoint", std::string()),
                  out_dir + "/calibration.svg");
  fig_workflows(rep["per_workflow"],
                "per-workflow accuracy — " + rep.value("checkpoint", std::string()),
                out_dir + "/workflows.svg");

  // confusion per type
  std::map<std::string, std::map<std::pair<std::string, std::string>, int>> cells;
  std::map<std::string, std::vector<std::string>> labels_of;
  for (const auto& p : rep["points"]) {
    std::string ty = p.value("type", "choice");
    const std::string& g = p["label_gold"].get_ref<const std::string&>();
    const std::string& pr = p["label_pred"].get_ref<const std::string&>();
    ++cells[ty][{g, pr}];
    auto& lbs = labels_of[ty];
    for (const auto* l : {&g, &pr})
      if (std::find(lbs.begin(), lbs.end(), *l) == lbs.end()) lbs.push_back(*l);
  }
  for (auto& [ty, cellmap] : cells) {
    auto lbs = labels_of[ty];
    std::sort(lbs.begin(), lbs.end());
    fig_confusion(ty, lbs, cellmap, "confusion — " + ty + " — " +
                      rep.value("checkpoint", std::string()),
                  out_dir + "/confusion-" + ty + ".svg");
  }

  // publication figures
  fig_roc(mpts, mm.auroc_conf,
          "ROC: confidence → correctness — " + rep.value("checkpoint", std::string()),
          out_dir + "/roc.svg");
  fig_pr_noul(mpts, "precision–recall (noul, pooled) — " +
                rep.value("checkpoint", std::string()),
              out_dir + "/pr-noul.svg");
  fig_score_spread(mpts, mm.spearman,
                   "score spread — " + rep.value("checkpoint", std::string()),
                   out_dir + "/score-spread.svg");

  // metrics: extended set incl. nll/mce/tv/spearman/auroc/score_mae
  {
    std::ofstream md(out_dir + "/report.md");
    md << "# snapjudge eval report\n\n"
       << "checkpoint: `" << rep.value("checkpoint", std::string()) << "`\n"
       << "data: `" << rep.value("data", std::string()) << "`\n"
       << "rows: " << rep.value("rows", 0) << "  questions: " << rep.value("questions", 0)
       << "\n\n"
       << "| metric | value |\n|---|---|\n"
       << "| accuracy ↑ | " << mm.accuracy << " |\n"
       << "| soft score (q·t) ↑ | " << mm.soft << " |\n"
       << "| tv distance ↓ | " << mm.tv << " |\n"
       << "| nll ↓ | " << mm.nll << " |\n"
       << "| brier ↓ | " << mm.brier << " |\n"
       << "| ece ↓ | " << mm.ece << " |\n"
       << "| mce ↓ | " << mm.mce << " |\n"
       << "| auroc (confidence→correct) ↑ | " << mm.auroc_conf << " |\n"
       << "| score mae ↓ | " << (std::isnan(mm.score_mae) ? -0.0 : mm.score_mae) << " |\n"
       << "| score spearman ↑ | " << (std::isnan(mm.spearman) ? -0.0 : mm.spearman) << " |\n"
       << "\n";
    // per-type accuracy table
    auto by = snapjudge::accuracy_by_type(mpts);
    md << "| primitive | accuracy | n |\n|---|---|---|\n";
    for (const auto& [ty, rc] : by)
      md << "| " << ty << " | " << (rc.second ? (double)rc.first / rc.second : 0.0)
         << " | " << rc.second << " |\n";
  }
  // LaTeX table for the paper
  {
    std::ofstream tex(out_dir + "/metrics.tex");
    std::string ck = rep.value("checkpoint", std::string(""));
    // shorten long local paths for print
    if (ck.size() > 60) ck = "…" + ck.substr(ck.size() - 58);
    tex << snapjudge::metrics_latex(mm, ck);
  }
  std::printf("figures -> %s/{calibration.svg, workflows.svg, confusion-*.svg, report.md}\n",
              out_dir.c_str());
  return 0;
}
