#!/usr/bin/env python3
"""snapjudge dev tool: synthesize a tiny snapjudge-format checkpoint for tests.

A 2-layer encoder (D=16, H=2, F=8), 1 head layer, vocab from a toy byte-level
tokenizer. Produces a directory loadable by snapjudge::DecisionModel AND
snapjudge::TrainModel. Pure shapes+dtypes fidelity with real checkpoints.
"""
import json
import os
import string
import sys

import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/snapjudge-tiny"
D, H, F, V, LMAX = 16, 2, 8, 512, 128
rng = np.random.default_rng(7)


def n(scale=0.02, *shape):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


os.makedirs(OUT, exist_ok=True)

tensors = {}
tensors["encoder.embeddings.tok_embeddings.weight"] = n(0.05, V, D)
tensors["encoder.embeddings.norm.weight"] = np.ones(D, dtype=np.float32)
for i in range(2):  # two encoder layers; sliding on layer 1
    p = f"encoder.layers.{i}."
    if i > 0:
        tensors[p + "attn_norm.weight"] = np.ones(D, dtype=np.float32)
    tensors[p + "attn.Wqkv.weight"] = n(0.05, 3 * D, D)
    tensors[p + "attn.Wo.weight"] = n(0.05, D, D)
    tensors[p + "mlp_norm.weight"] = np.ones(D, dtype=np.float32)
    tensors[p + "mlp.Wi.weight"] = n(0.05, 2 * F, D)
    tensors[p + "mlp.Wo.weight"] = n(0.05, D, F)
tensors["encoder.final_norm.weight"] = np.ones(D, dtype=np.float32)
tensors["type_emb.weight"] = n(0.02, 3, D)
tensors["head.layers.0.norm1.weight"] = np.ones(D, dtype=np.float32)
tensors["head.layers.0.norm1.bias"] = np.zeros(D, dtype=np.float32)
tensors["head.layers.0.norm2.weight"] = np.ones(D, dtype=np.float32)
tensors["head.layers.0.norm2.bias"] = np.zeros(D, dtype=np.float32)
tensors["head.layers.0.self_attn.in_proj_weight"] = n(0.05, 3 * D, D)
tensors["head.layers.0.self_attn.in_proj_bias"] = np.zeros(3 * D, dtype=np.float32)
tensors["head.layers.0.self_attn.out_proj.weight"] = n(0.05, D, D)
tensors["head.layers.0.self_attn.out_proj.bias"] = np.zeros(D, dtype=np.float32)
tensors["head.layers.0.linear1.weight"] = n(0.05, 4 * D, D)
tensors["head.layers.0.linear1.bias"] = np.zeros(4 * D, dtype=np.float32)
tensors["head.layers.0.linear2.weight"] = n(0.05, D, 4 * D)
tensors["head.layers.0.linear2.bias"] = np.zeros(D, dtype=np.float32)
tensors["scorer.0.weight"] = np.ones(D, dtype=np.float32)
tensors["scorer.1.weight"] = n(0.05, D, D)
tensors["scorer.3.weight"] = n(0.05, 1, D)
tensors["act_head.0.weight"] = n(0.05, 256, D + 4)
tensors["act_head.0.bias"] = np.zeros(256, dtype=np.float32)
tensors["act_head.2.weight"] = n(0.05, 1, 256)
tensors["act_head.2.bias"] = np.zeros(1, dtype=np.float32)
tensors["temperature"] = np.ones(3, dtype=np.float32)

# fp16 on disk (same as real checkpoints), temperature stays fp32.
parts = {}
off = 0
for k, v in tensors.items():
    data = v.tobytes() if k == "temperature" else v.astype(np.float16).tobytes()
    parts[k] = (off, off + len(data), v)
    off += len(data)

header = {
    k: {"dtype": "F32" if k == "temperature" else "F16",
        "shape": list(v.shape), "data_offsets": [o0, o1]}
    for k, (o0, o1, v) in parts.items()
}
hj = json.dumps(header).encode()
hj += b" " * ((8 - (8 + len(hj)) % 8) % 8)
with open(os.path.join(OUT, "model.safetensors"), "wb") as f:
    f.write(len(hj).to_bytes(8, "little"))
    f.write(hj)
    for k, (o0, o1, v) in parts.items():
        data = v.tobytes() if k == "temperature" else v.astype(np.float16).tobytes()
        f.write(data)

# encoder config — ModernBERT-shaped but tiny
enc = {
    "hidden_size": D, "num_hidden_layers": 2, "num_attention_heads": H,
    "intermediate_size": F, "max_position_embeddings": LMAX, "vocab_size": V,
    "norm_eps": 1e-5, "norm_bias": False, "attention_bias": False,
    "mlp_bias": False, "local_attention": 8,
    "layer_types": ["full_attention", "sliding_attention"],
    "rope_parameters": {
        "full_attention": {"rope_theta": 10000.0, "rope_type": "default"},
        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
    },
}
os.makedirs(os.path.join(OUT, "encoder"), exist_ok=True)
with open(os.path.join(OUT, "encoder", "config.json"), "w") as f:
    json.dump(enc, f, indent=1)

# toy tokenizer.json: one token per ASCII char + specials, byte-level BPE with
# no merges (every char maps to its own id). Vocab keys are in the GPT-2
# byte->unicode space (same as real tokenizers: ' ' appears as 'Ġ', 0x21).
def _b2u(b):
    if (0x21 <= b <= 0x7E) or (0xA1 <= b <= 0xAC) or (0xAE <= b <= 0xFF):
        return chr(b)
    return chr(0x100 + b)  # unused slots only matter for id stability

vocab = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3, "[MASK]": 4}
for b in range(256):
    u = _b2u(b)
    if u not in vocab:
        vocab[u] = len(vocab)
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

cfg = {
    "encoder": "tiny-synthetic", "head_layers": 1, "max_len": LMAX,
    "head_max_len": 48, "act_costs": {}, "cost_wrong_act": 3.0,
    "amp_dtype": "fp32", "model_name": "snapjudge-tiny",
    "temperature": [1.0, 1.0, 1.0], "temperature_by_options": {},
}
with open(os.path.join(OUT, "rl_agent_config.json"), "w") as f:
    json.dump(cfg, f, indent=1)

print(f"wrote tiny checkpoint: {OUT} ({len(tensors)} tensors, D={D})")
