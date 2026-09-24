# Training snapjudge checkpoints (100% C++)

Train the typed decision stack of a snapjudge checkpoint on gold-distribution
data and push the result to the Hugging Face hub. No Python at training time.

## Pipeline

```bash
# 1. export your data as JSONL in the typed-decisions schema (see below)
#    (example export of a public corpus: python3 python/export_typed_decisions.py data all)

# 2. train
build/snapjudge-train \
  --base  <ckpt-dir-or-hub-id> \
  --data  data/train.jsonl \
  --val   data/val.jsonl \
  --out   out/typed-ft \
  --epochs 1 --rows-per-step 2 --accum 8 --lr 2e-5

# 3. evaluate
build/snapjudge-eval --ckpt out/typed-ft --data data/val.jsonl --json out/report.json
build/snapjudge-plots --report out/report.json --out out/figures

# 4. push to the hub
export HF_TOKEN=hf_...
build/snapjudge-train-push --ckpt out/typed-ft --repo you/snapjudge-typed-ft
```

## What's being optimized

The trainer's loss is the offline form of the proper scoring rule snapjudge's
checkpoints are built on. For every question:

    reward = Σ_i t_i·log q_i + 0.5·(t·q)/|q|
    score-type questions also subtract  RPS = Σ (cdf_q − cdf_t)² / (k−1)

and the training loss is `-reward`. The strictly proper property means the
policy is rewarded for reporting *honest* probabilities — which is what makes
snapjudge confidence calibrated.

After training, per-`(qtype, options-count)` temperatures are fitted on the
validation rows (minimising teacher-NLL) and written into the saved
checkpoint's config.

## What's being updated

- **Frozen encoder** — gradients flow only through `type_emb`, the head
  transformer layers, and the scorer (~26M params on the 421M backbone
  checkpoints). This is what carries domain specialisation and fits a T4's
  16GB comfortably in fp32.
- `--rows-per-step` is the micro-batch (each state produces one row per
  question); `--accum` accumulates over micro-steps before the AdamW update.
- The act head is frozen during training (the gold distributions carry no act
  targets).

## Data format

One JSON per line:

```json
{"state": {"document": "...", "any": "json state the model sees"},
 "questions": {"qid": {"type": "choice|score|noul", "instructions": "...",
               "criteria": {"label": "description"}}},
 "gold": {"qid": {"type": "choice", "probabilities": {"label": 0.7, "other": 0.3}}},
 "workflow": "optional tag"}
```

Gold may be soft distributions (teacher labels as full distributions) or hard
labels (probabilities summing to 1, e.g. `{"yes": 1, "no": 0}` for noul).

## Evaluating

`snapjudge-eval` runs the real deployment decode path (an `Agent` with the
checkpoint's fitted temperatures applied) over a validation JSONL and prints:

| metric | meaning | direction |
|---|---|---|
| accuracy | argmax(pred) == argmax(gold) | higher |
| soft score | mean inner product of predicted and teacher dists | higher |
| TV distance | mean total-variation distance to teacher | lower |
| NLL | cross-entropy | lower |
| Brier | mean Σ (q − t)² | lower |
| ECE | confidence-bin calibration error | lower |
| MCE | worst-bin calibration error | lower |
| AUROC(conf→correct) | does confidence know when it's right | higher |
| score MAE | ordinal drift on score questions | lower |
| score Spearman | rank correlation with teacher | higher |

plus per-workflow and per-primitive accuracy tables. `--n N` caps rows for a
quick read. `--json` writes a report for figure generation.

## Figures

`snapjudge-plots` renders an eval report to SVG **and PNG** (pure C++ raster
path, no image dependencies):

```
calibration.svg/png      reliability diagram (binned, counts on top)
workflows.svg/png        per-workflow accuracy bars
confusion-*.svg/png      per-primitive label confusion matrices
roc.svg/png              confidence → correctness ROC curve with AUROC
pr-noul.svg/png          pooled precision–recall for yes/no questions
score-spread.svg/png     teacher vs predicted expected score scatter
metrics.tex              LaTeX booktabs table of the full metric set
report.md                markdown report
```

## Verification

The trainer is gradient-checked: `tests/test_train.cpp` finite-differences
every trainable tensor of a tiny synthetic checkpoint (worst relative error
~2e-3 — floating-point floor), overfits one row, and roundtrips a saved
checkpoint through the inference engine. Run `ctest --test-dir build`.

## Efficiency notes

- The frozen encoder means the forward is the expensive part; on CPU it runs
  at ~200ms per state with Accelerate/OpenBLAS. On GPU builds use
  `-DSNAPJUDGE_CUDA=ON`.
- A full epoch over a 1,200-row corpus is a few hours on CPU, much faster on
  a T4 GPU with the CUDA fast path.
