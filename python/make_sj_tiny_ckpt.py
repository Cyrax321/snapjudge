#!/usr/bin/env python3
"""snapjudge dev tool: synthesize a tiny native snapjudge (SjModel) checkpoint.

This is snapjudge's OWN model: classic BERT-style encoder (learned absolute
positions, GELU, biases, full attention, separate q/k/v/o) plus the typed
decision head, in the `sj.*` tensor namespace. No ModernBERT, no mmBERT, no
external tokenizer — just a toy byte-level vocab like the tiny fixture.

Produces sj_config.json + model.safetensors in OUT.
"""
import json
import os
import sys

import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/snapjudge-sj-tiny"
V, D, H, F, L, LAYERS = 64, 16, 2, 32, 64, 2
HEAD_LAYERS = 1
N_TYPES = 3
HEAD_INTER = 32
rng = np.random.default_rng(7)


def n(scale, *shape):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


os.makedirs(OUT, exist_ok=True)
t = {}

# embeddings
t["sj.embeddings.word.weight"] = n(0.05, V, D)
t["sj.embeddings.position.weight"] = n(0.05, L, D)
t["sj.embeddings.norm.weight"] = np.ones(D, np.float32)
t["sj.embeddings.norm.bias"] = np.zeros(D, np.float32)

for i in range(LAYERS):
    p = f"sj.encoder.layers.{i}."
    t[p + "attn_norm.weight"] = np.ones(D, np.float32)
    t[p + "attn_norm.bias"] = np.zeros(D, np.float32)
    for s in ("q", "k", "v", "o"):
        t[p + f"attn.{s}.weight"] = n(0.05, D, D)
        t[p + f"attn.{s}.bias"] = np.zeros(D, np.float32)
    t[p + "ffn_norm.weight"] = np.ones(D, np.float32)
    t[p + "ffn_norm.bias"] = np.zeros(D, np.float32)
    t[p + "ffn.w1.weight"] = n(0.05, F, D)
    t[p + "ffn.w1.bias"] = np.zeros(F, np.float32)
    t[p + "ffn.w2.weight"] = n(0.05, D, F)
    t[p + "ffn.w2.bias"] = np.zeros(D, np.float32)

t["sj.encoder.final_norm.weight"] = np.ones(D, np.float32)
t["sj.encoder.final_norm.bias"] = np.zeros(D, np.float32)
t["sj.type_emb.weight"] = n(0.02, N_TYPES, D)

for i in range(HEAD_LAYERS):
    p = f"sj.head.layers.{i}."
    t[p + "norm1.weight"] = np.ones(D, np.float32)
    t[p + "norm1.bias"] = np.zeros(D, np.float32)
    t[p + "norm2.weight"] = np.ones(D, np.float32)
    t[p + "norm2.bias"] = np.zeros(D, np.float32)
    for s in ("q", "k", "v", "o"):
        t[p + f"attn.{s}.weight"] = n(0.05, D, D)
        t[p + f"attn.{s}.bias"] = np.zeros(D, np.float32)
    t[p + "ffn.w1.weight"] = n(0.05, HEAD_INTER, D)
    t[p + "ffn.w1.bias"] = np.zeros(HEAD_INTER, np.float32)
    t[p + "ffn.w2.weight"] = n(0.05, D, HEAD_INTER)
    t[p + "ffn.w2.bias"] = np.zeros(D, np.float32)

t["sj.scorer.norm.weight"] = np.ones(D, np.float32)
t["sj.scorer.norm.bias"] = np.zeros(D, np.float32)
t["sj.scorer.w1.weight"] = n(0.05, D, D)
t["sj.scorer.w1.bias"] = np.zeros(D, np.float32)
t["sj.scorer.w2.weight"] = n(0.05, 1, D)
t["sj.scorer.w2.bias"] = np.zeros(1, np.float32)

# safetensors: fp16 storage
parts = {}
off = 0
for k, v in t.items():
    data = v.astype(np.float16).tobytes()
    parts[k] = (off, off + len(data))
    off += len(data)

header = {}
for k, v in t.items():
    o0, o1 = parts[k]
    header[k] = {"dtype": "F16", "shape": list(v.shape), "data_offsets": [o0, o1]}

hj = json.dumps(header).encode()
hj += b" " * ((8 - (8 + len(hj)) % 8) % 8)
with open(os.path.join(OUT, "model.safetensors"), "wb") as f:
    f.write(len(hj).to_bytes(8, "little"))
    f.write(hj)
    for k in t:
        f.write(t[k].astype(np.float16).tobytes())

cfg = {
    "vocab_size": V,
    "hidden_size": D,
    "num_layers": LAYERS,
    "num_heads": H,
    "intermediate_size": F,
    "max_positions": L,
    "norm_eps": 1e-5,
    "head_layers": HEAD_LAYERS,
    "head_intermediate": HEAD_INTER,
    "max_len": 64,
    "head_max_len": 32,
    "n_types": N_TYPES,
    "cls_id": 1,
    "sep_id": 2,
    "mask_id": 4,
    "pad_id": 3,
}
with open(os.path.join(OUT, "sj_config.json"), "w") as f:
    json.dump(cfg, f, indent=1)

print("wrote", OUT)
