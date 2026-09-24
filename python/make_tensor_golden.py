#!/usr/bin/env python3
"""snapjudge golden generator: tensor op outputs from numpy (fp32 reference).

The C++ parity target for numerics is the Python CPU fp32 path (torch CPU does
fp32 compute from fp16-upcast weights). For unit ops numpy fp32 is the same
math; tolerances here are tight (1e-5) so only order-of-accumulation differs.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "..", "tests", "golden")
os.makedirs(GOLDEN, exist_ok=True)

rng = np.random.default_rng(20260923)

A = rng.standard_normal((7, 13), dtype=np.float32)
W = rng.standard_normal((11, 13), dtype=np.float32) * 0.05
B = rng.standard_normal((11,), dtype=np.float32) * 0.01


def gelu(x):
    from math import erf
    return 0.5 * x * (1.0 + np.vectorize(erf)(x * 0.70710678118654757))


cases = {}
cases["gemm"] = {"a": A.tolist(), "w": W.tolist(), "bias": B.tolist(),
                 "out": (A @ W.T + B).tolist()}
cases["gemm_gelu"] = {"out": gelu(A @ W.T + B).tolist()}
cases["gemm_relu"] = {"out": np.maximum(A @ W.T + B, 0).tolist()}

F = 9
Wi = rng.standard_normal((2 * F, 13), dtype=np.float32) * 0.05
z = A @ Wi.T
cases["geglu"] = {"wi": Wi.tolist(), "out": (gelu(z[:, :F]) * z[:, F:]).tolist()}

# layer norm with eps 1e-5, with and without bias
x = rng.standard_normal((5, 17), dtype=np.float32) * [1, 10, 0.1, 3, 7, 2, 1, 4, 9, 1, 1, 1, 2, 2, 3, 5, 8]
w = rng.standard_normal((17,), dtype=np.float32)
b = rng.standard_normal((17,), dtype=np.float32)
def ln(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + (0 if b is None else b)
cases["layer_norm"] = {"x": x.tolist(), "w": w.tolist(), "b": b.tolist(),
                       "out": ln(x, w, b).tolist(), "out_nobias": ln(x, w, None).tolist()}

# softmax rows incl. large negatives (masked_fill -1e4 territory)
z = np.array([[0.0, 1.0, -1e4, 2.0], [3.0, 3.0, 3.0, 3.0], [1e4, 0.0, -1.0, -100.0]],
             dtype=np.float32)
p = np.exp(z - z.max(-1, keepdims=True))
cases["softmax"] = {"z": z.tolist(), "out": (p / p.sum(-1, keepdims=True)).tolist()}

# attention: B=2, H=3, L=6, Dh=4, one batch shorter than the other, window test
B_, H_, L_, Dh_ = 2, 3, 6, 4
q = rng.standard_normal((B_, H_, L_, Dh_), dtype=np.float32) * 0.5
k = rng.standard_normal((B_, H_, L_, Dh_), dtype=np.float32) * 0.5
v = rng.standard_normal((B_, H_, L_, Dh_), dtype=np.float32)
lens = [6, 4]
def attn(q, k, v, lens, window):
    B, H, L, Dh = q.shape
    out = np.zeros_like(q)
    for b in range(B):
        n = lens[b]
        for h in range(H):
            for i in range(L):
                idx = [j for j in range(L)
                       if j < n and (window == 0 or abs(i - j) <= window)]
                s = q[b, h, i] @ k[b, h, idx].T / np.sqrt(Dh)
                p = np.exp(s - s.max())
                p /= p.sum()
                out[b, h, i] = p @ v[b, h, idx]
    return out
cases["attn_full"] = {"q": q.tolist(), "k": k.tolist(), "v": v.tolist(),
                      "lens": lens, "window": 0, "out": attn(q, k, v, lens, 0).tolist()}
cases["attn_window"] = {"window": 2, "out": attn(q, k, v, lens, 2).tolist()}

# rope rotate-half on packed qkv: M=8 rows, L=4, H=2, Dh=6
M_, Lr, Hr, Dhr = 8, 4, 2, 6
qkv = rng.standard_normal((M_, 3 * Hr * Dhr), dtype=np.float32)
theta = 10000.0
half = Dhr // 2
inv_freq = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) / half * 2 / 2))
# HF: inv_freq = theta^(-arange(0, Dh, 2)/Dh)
inv_freq = 1.0 / (theta ** (np.arange(0, Dhr, 2, dtype=np.float64) / Dhr))
pos = np.arange(Lr)
freqs = np.outer(pos, inv_freq)
cos_t, sin_t = np.cos(freqs), np.sin(freqs)
out = qkv.copy().astype(np.float64)
for r in range(M_):
    p_ = r % Lr
    for hh in range(2 * Hr):
        base = hh * Dhr
        x0 = out[r, base:base + half].copy()
        x1 = out[r, base + half:base + Dhr].copy()
        out[r, base:base + half] = x0 * cos_t[p_] - x1 * sin_t[p_]
        out[r, base + half:base + Dhr] = x1 * cos_t[p_] + x0 * sin_t[p_]
cases["rope"] = {"qkv": qkv.tolist(), "L": Lr, "H": Hr, "Dh": Dhr,
                 "theta": theta, "out": out.tolist(),
                 "cos": cos_t.tolist(), "sin": sin_t.tolist()}

with open(os.path.join(GOLDEN, "tensor_ops.json"), "w") as f:
    json.dump(cases, f)
print("wrote tensor_ops.json:", ", ".join(cases))
