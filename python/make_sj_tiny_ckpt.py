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
V, D, H, F, L, LAYERS = 512, 16, 2, 32, 64, 2
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

# toy byte-level tokenizer, matching the vocab_size above (512 entries: 256
# byte-symbols in the GPT-2 space plus the specials + a few extra slots).
def _b2u(b):
    if (0x21 <= b <= 0x7E) or (0xA1 <= b <= 0xAC) or (0xAE <= b <= 0xFF):
        return chr(b)
    return chr(0x100 + b)

vocab = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3, "[MASK]": 4}
for b in range(256):
    u = _b2u(b)
    if u not in vocab:
        vocab[u] = len(vocab)
# ensure vocab size == V
while len(vocab) < V:
    vocab["\u0100extra" + str(len(vocab))] = len(vocab)
tok = {
    "model": {"type": "BPE", "vocab": vocab, "merges": []},
    "normalizer": {"type": "NFC"},
    "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False,
                      "trim_offsets": True, "use_regex": True},
    "added_tokens": [
        {"id": 0, "content": "[UNK]", "special": True, "normalized": False},
        {"id": 1, "content": "[CLS]", "special": True, "normalized": False},
        {"id": 2, "content": "[SEP]", "special": True, "normalized": False},
        {"id": 3, "content": "[PAD]", "special": True, "normalized": False},
        {"id": 4, "content": "[MASK]", "special": True, "normalized": False},
    ],
}
os.makedirs(os.path.join(OUT, "tokenizer"), exist_ok=True)
with open(os.path.join(OUT, "tokenizer", "tokenizer.json"), "w") as f:
    json.dump(tok, f)
with open(os.path.join(OUT, "tokenizer", "tokenizer_config.json"), "w") as f:
    json.dump({"tokenizer_class": "TokenizersBackend"}, f)

print("wrote", OUT)
