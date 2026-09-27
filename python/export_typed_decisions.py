#!/usr/bin/env python3
"""snapjudge dev tool: export LocalLLaMA/typed-decisions to JSONL for the
C++ trainer.

Each line:
  {"state": <json>, "questions": {qid: {type, instructions, criteria|labels}},
   "gold": {qid: {"type":..., "probabilities": {label: p, ...}}},
   "workflow": "..."}

Usage: python3 snapjudge/python/export_typed_decisions.py OUT_DIR [config]
Writes OUT_DIR/train.jsonl + OUT_DIR/val.jsonl (from the test split).
"""
import json
import os
import sys


def main():
    out_dir = sys.argv[1]
    config = sys.argv[2] if len(sys.argv) > 2 else "all"
    os.makedirs(out_dir, exist_ok=True)
    from datasets import load_dataset

    ds = load_dataset("LocalLLaMA/typed-decisions", config)
    for split, out_name in (("train", "train.jsonl"), ("test", "val.jsonl")):
        ds_split = ds[split]
        path = os.path.join(out_dir, out_name)
        kept = 0
        with open(path, "w") as f:
            for raw in ds_split:
                questions = json.loads(raw["questions"])
                gold = json.loads(raw["gold"])
                state = json.loads(raw["state"])
                # keep every question with full gold distributions
                q2, g2 = {}, {}
                ok = True
                for qid, qdef in questions.items():
                    entry = gold.get(qid)
                    if entry is None or "probabilities" not in entry:
                        ok = False
                        continue  # drop question but keep the state
                    q2[qid] = qdef
                    g2[qid] = {"type": entry["type"],
                               "probabilities": entry["probabilities"]}
                if not q2:
                    continue
                f.write(json.dumps({"state": state, "questions": q2, "gold": g2,
                                    "workflow": raw["workflow"]},
                                   ensure_ascii=False) + "\n")
                kept += 1
        print(f"{out_name}: {kept} rows -> {path}")


if __name__ == "__main__":
    main()
