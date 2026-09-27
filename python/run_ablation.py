#!/usr/bin/env python3
"""snapjudge ablation study: isolate the effect of each training mechanism.

Trains a checkpoint under a grid of flag variants (baseline + each mechanism
alone + all combined), evals each, and emits a delta table showing the
isolated contribution of loss weighting, label smoothing, temperature
annealing, and LoRA — against accuracy / soft-acc / ECE.

Usage (in Colab, after building):
  python3 python/run_ablation.py \
      --base /content/base \
      --data data/train.jsonl --val data/val.jsonl \
      --train-bin build/snapjudge-train --eval-bin build/snapjudge-eval \
      --out out/ablation

Each run is 1 epoch (~2h CPU on a T4 box); 6 runs ~ 12h. Run overnight.
"""
import argparse
import json
import os
import subprocess
import sys

CONFIGS = [
    ("baseline",       {}),
    ("distill",        {"--w-nll": "2.0", "--w-sph": "0.2"}),
    ("smoothing",      {"--label-smoothing": "0.05"}),
    ("anneal",         {"--temp-start": "2.0", "--temp-end": "1.0"}),
    ("lora",           {"--lora-r": "8"}),
    ("combined",       {"--w-nll": "2.0", "--w-sph": "0.2",
                        "--label-smoothing": "0.05",
                        "--temp-start": "2.0", "--temp-end": "1.0",
                        "--lora-r": "8"}),
]

METRICS = ["accuracy", "soft", "ece", "brier"]


def run(cmd):
    print("  $ " + " ".join(cmd), file=sys.stderr)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        raise RuntimeError("command failed: " + " ".join(cmd))
    return r


def eval_ckpt(eval_bin, ckpt, val, n=0):
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        rep = f.name
    cmd = [eval_bin, "--ckpt", ckpt, "--data", val, "--json", rep]
    if n:
        cmd += ["--n", str(n)]
    run(cmd)
    with open(rep) as f:
        d = json.load(f)
    os.unlink(rep)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--train-bin", default="build/snapjudge-train")
    ap.add_argument("--eval-bin", default="build/snapjudge-eval")
    ap.add_argument("--out", default="out/ablation")
    ap.add_argument("--epochs", default="1")
    ap.add_argument("--n", type=int, default=0, help="cap eval rows")
    ap.add_argument("--only", default="", help="comma list of config names to run")
    args = ap.parse_args()

    only = set(x.strip() for x in args.only.split(",") if x.strip())
    results = {}

    for name, flags in CONFIGS:
        if only and name not in only:
            continue
        ckpt = os.path.join(args.out, name)
        cmd = [args.train_bin, "--base", args.base, "--data", args.data,
               "--val", args.val, "--out", ckpt, "--epochs", args.epochs]
        for k, v in flags.items():
            cmd += [k, v]
        print(f"\n=== training [{name}] ===", file=sys.stderr)
        run(cmd)
        print(f"=== eval [{name}] ===", file=sys.stderr)
        rep = eval_ckpt(args.eval_bin, ckpt, args.val, args.n)
        results[name] = {m: rep.get(m) for m in METRICS}
        results[name]["questions"] = rep.get("questions")

    if not results:
        print("no configs matched", file=sys.stderr)
        return 1

    base = results.get("baseline", {})
    hdr = ["config"] + METRICS + [f"d({m})" for m in METRICS]
    rows = []
    for name, r in results.items():
        def f(v):
            return f"{v:.4f}" if isinstance(v, (int, float)) else "-"
        row = [name] + [f(r.get(m)) for m in METRICS]
        if name == "baseline":
            row += ["-"] * len(METRICS)
        else:
            row += [f(r.get(m, 0) - base.get(m, 0)) if isinstance(r.get(m), (int, float)) else "-"
                    for m in METRICS]
        rows.append(row)

    widths = [max(len(r[i]) for r in [hdr] + rows) for i in range(len(hdr))]
    def line(cells):
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"
    print()
    print(line(hdr))
    print(line(["-" * widths[i] for i in range(len(hdr))]))
    for r in rows:
        print(line(r))

    out_path = os.path.join(args.out, "ablation.md")
    with open(out_path, "w") as f:
        f.write("# snapjudge ablation\n\n")
        f.write("Delta = (variant) - (baseline). Higher accuracy/soft is better; "
                "lower ECE/brier is better.\n\n")
        f.write(line(hdr) + "\n")
        f.write(line(["-" * widths[i] for i in range(len(hdr))]) + "\n")
        for r in rows:
            f.write(line(r) + "\n")
    print(f"\nwrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
