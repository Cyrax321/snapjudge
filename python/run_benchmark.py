#!/usr/bin/env python3
"""snapjudge benchmark harness: eval a checkpoint across typed-decisions datasets.

Usage:
  python3 python/run_benchmark.py --ckpt out/typed-ft --datasets data/benchmarks.json \
      [--eval-bin build/snapjudge-eval] [--n N] [--out report.md]

`--datasets` is a JSON file mapping a benchmark name to a typed-decisions JSONL
(produced by python/export_classification.py). The harness runs snapjudge-eval
over each dataset and prints a table of the checkpoint's own metrics.
"""
import argparse
import json
import os
import subprocess
import sys


def run_eval(eval_bin, ckpt, jsonl, n, shortlist_k=0):
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out = f.name
    cmd = [eval_bin, "--ckpt", ckpt, "--data", jsonl, "--json", out]
    if n:
        cmd += ["--n", str(n)]
    if shortlist_k:
        cmd += ["--shortlist-k", str(shortlist_k)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        raise RuntimeError(f"snapjudge-eval failed for {jsonl}")
    with open(out) as f:
        report = json.load(f)
    os.unlink(out)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--datasets", required=True,
                    help="JSON map of benchmark name -> typed-decisions JSONL path")
    ap.add_argument("--eval-bin", default="build/snapjudge-eval")
    ap.add_argument("--n", type=int, default=0, help="cap rows per dataset")
    ap.add_argument("--shortlist-k", type=int, default=0,
                    help="coarse-to-fine choice k (0 = single-pass)")
    ap.add_argument("--out", default="", help="write a markdown report here")
    args = ap.parse_args()

    with open(args.datasets) as f:
        datasets = json.load(f)

    results = {}
    for name, jsonl in datasets.items():
        rep = run_eval(args.eval_bin, args.ckpt, jsonl, args.n, args.shortlist_k)
        results[name] = {
            "accuracy": rep.get("accuracy"),
            "soft": rep.get("soft"),
            "ece": rep.get("ece"),
            "rows": rep.get("rows"),
            "questions": rep.get("questions"),
        }

    hdr = ["benchmark", "accuracy", "soft", "ece", "rows"]
    rows = []
    for name, r in results.items():
        def fmt(x):
            return f"{x:.3f}" if isinstance(x, (int, float)) else "-"
        rows.append([name, fmt(r["accuracy"]), fmt(r["soft"]), fmt(r["ece"]),
                     str(r["rows"])])

    widths = [max(len(str(r[i])) for r in [hdr] + rows) for i in range(len(hdr))]
    def line(cells):
        return "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells)) + " |"
    print(line(hdr))
    print(line(["-" * widths[i] for i in range(len(hdr))]))
    for r in rows:
        print(line(r))

    if args.out:
        with open(args.out, "w") as f:
            f.write("# snapjudge benchmark\n\n")
            f.write(f"checkpoint: `{args.ckpt}`\n\n")
            f.write(line(hdr) + "\n")
            f.write(line(["-" * widths[i] for i in range(len(hdr))]) + "\n")
            for r in rows:
                f.write(line(r) + "\n")
        print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
