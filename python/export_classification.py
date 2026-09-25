#!/usr/bin/env python3
"""snapjudge dev tool: convert a HF classification dataset to typed-decisions JSONL.

Turns a plain (text, label) classification corpus into the schema the C++
trainer consumes, so snapjudge can be fine-tuned on the same benchmarks the
reference model publishes (AG News, DAIR Emotion, Banking77, SST-5, ...).

Each output line:
  {"state": {"text": "..."},
   "questions": {qid: {"type": "choice"|"score", "instructions": "...",
                       "criteria": {...or[...]}}},
   "gold": {qid: {"type": ..., "probabilities": {label: p, ...}}},
   "workflow": "<dataset>"}

Hard labels become one-hot gold distributions (the trainer softens them via
--label-smoothing when you want calibration). Ordinal datasets (score type)
emit an ordered criteria list with probabilities keyed "0".."k-1".

Usage:
  python3 snapjudge/python/export_classification.py \
      --dataset fancyzhx/ag_news --label text --labels 'World,Sports,Business,Sci/Tech' \
      --question "What news category is this article?" --out data/ag_news \
      --max-per-split 10000

  # ordinal (score) example
  python3 snapjudge/python/export_classification.py \
      --dataset SetFit/sst5 --label text \
      --labels 'very negative,negative,neutral,positive,very positive' \
      --qtype score --question "What is the sentiment of this sentence?" \
      --out data/sst5

The `--labels` list must be ordered to match the dataset's ClassLabel names
(or the raw int order when the dataset stores plain ints).
"""
import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--label", default="text", help="column holding the input text")
    ap.add_argument("--labels", required=True,
                    help="comma-separated class labels, in label-id order")
    ap.add_argument("--question", default="Classify the input.")
    ap.add_argument("--qtype", default="choice", choices=["choice", "score"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-split", type=int, default=0,
                    help="cap rows per split (0 = all)")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="test")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    from datasets import load_dataset

    labels = [l.strip() for l in args.labels.split(",")]
    ds = load_dataset(args.dataset)
    print(f"loaded {args.dataset}: {list(ds.keys())}", file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)

    def split_name(name):
        # datasets with only a test split use it as validation; a 'validation'
        # split is preferred over 'test' when present.
        if name in ds:
            return name
        if name == "test" and "validation" in ds:
            return "validation"
        return name

    def convert(split, out_name):
        if split not in ds:
            print(f"  skip {out_name}: split '{split}' not found", file=sys.stderr)
            return
        path = os.path.join(args.out, out_name)
        kept = 0
        with open(path, "w") as f:
            for raw in ds[split]:
                text = raw[args.label]
                if not isinstance(text, str) or not text.strip():
                    continue
                state = {"text": text.strip()}
                # resolve the label id (ClassLabel or raw int)
                feat = ds[split].features.get("label")
                if feat is not None and hasattr(feat, "int2str"):
                    lab = feat.int2str(int(raw["label"]))
                else:
                    lab = labels[int(raw["label"])]
                lab_idx = labels.index(lab)

                if args.qtype == "score":
                    criteria = list(labels)
                    qdef = {"type": "score", "instructions": args.question,
                            "criteria": criteria}
                    # score gold is keyed by ordinal index "0".."k-1"
                    gold_probs = {str(i): 0.0 for i in range(len(labels))}
                    gold_probs[str(lab_idx)] = 1.0
                else:
                    criteria = {l: l for l in labels}
                    qdef = {"type": "choice", "instructions": args.question,
                            "criteria": criteria}
                    gold_probs = {l: 0.0 for l in labels}
                    gold_probs[lab] = 1.0

                gold = {"q0": {"type": args.qtype, "probabilities": gold_probs}}
                questions = {"q0": qdef}
                f.write(json.dumps({"state": state, "questions": questions,
                                    "gold": gold, "workflow": args.dataset},
                                   ensure_ascii=False) + "\n")
                kept += 1
                if args.max_per_split and kept >= args.max_per_split:
                    break
        print(f"  {out_name}: {kept} rows -> {path}", file=sys.stderr)

    convert(split_name(args.train_split), "train.jsonl")
    convert(split_name(args.val_split), "val.jsonl")


if __name__ == "__main__":
    main()
