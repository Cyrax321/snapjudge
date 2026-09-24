#pragma once
// snapjudge tensor.hpp: minimal inference-only fp32 runtime.
// Row-major, cblas-accelerated GEMM, no autograd. Only what the
// DecisionModel forward pass needs (see common.py + fast.py).

#include <cstdint>
#include <memory>
#include <vector>

namespace snapjudge {

struct Tensor {
  std::vector<int64_t> shape;
  std::shared_ptr<float[]> data;      // row-major
  bool owner = true;                  // false for views into safetensors mmaps

  Tensor() = default;
  explicit Tensor(std::vector<int64_t> s);
  static Tensor zeros(std::vector<int64_t> s);
  static Tensor wrap(std::vector<int64_t> s, std::shared_ptr<float[]> p) {
    Tensor t;
    t.shape = std::move(s);
    t.data = std::move(p);
    t.owner = false;
    return t;
  }

  int64_t numel() const;
  int64_t dim(int i) const { return shape.at(static_cast<size_t>(i)); }
  int ndim() const { return static_cast<int>(shape.size()); }
  float* mutable_data() { return data.get(); }
  const float* data_ptr() const { return data.get(); }
  // row accessors for matrices
  float* row(int64_t r) { return data.get() + r * dim(ndim() - 1); }
  const float* row(int64_t r) const { return data.get() + r * dim(ndim() - 1); }
};

// ---- elementwise / reductions ----------------------------------------------
Tensor add(const Tensor& a, const Tensor& b);           // same shape
void add_(Tensor& a, const Tensor& b);                  // in place
Tensor layer_norm(const Tensor& x, const float* w, const float* b, float eps);
// Stable softmax over last dim of [N, K].
void softmax_rows(Tensor& x);
float gelu_erf(float x);                                // exact erf GELU (HF "gelu")

// ---- GEMMs ------------------------------------------------------------------
// out[M,N] = act(op_bias + a[M,K] @ w[N,K]^T).  `bias` may be null.
// act: 0 none, 1 erf-GELU, 2 relu.
Tensor gemm_nt(const Tensor& a, const Tensor& w, const float* bias, int act);
// GEGLU: wi is [2F, K]; returns [M, F] = gelu(a @ wi[:F]^T) * (a @ wi[F:]^T).
Tensor gemm_geglu(const Tensor& a, const Tensor& wi);

// ---- attention ---------------------------------------------------------------
// Multi-head attention without scaling surprises:
//   q/k/v: [B, H, L, Dh] separate contiguous tensors.
//   att_mask: per-batch valid length (lens) — keys >= len masked; window > 0
//             additionally requires |i - j| <= window (bidirectional sliding).
// Returns [B, H, L, Dh].
Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v,
                 const std::vector<int64_t>& lens, int window);

// ---- misc ---------------------------------------------------------------------
// Gather rows: h [B, L, D], pos [B, K] int64 -> [B, K, D]
Tensor gather_rows(const Tensor& h, const int64_t* pos, int64_t B, int64_t K);
// Embedding lookup: table [V, D] x ids -> [n, D]
Tensor embedding(const Tensor& table, const int64_t* ids, int64_t n);
// RoPE rotate-half applied in place to q and k packed in qkv [M, 3*H*Dh] as
// (q|k|v)(h)(d); position of row r is r % L. cos/sin are [L, Dh/2].
void rope_inplace(Tensor& qkv, const float* cos_t, const float* sin_t, int64_t L,
                  int64_t H, int64_t Dh);

}  // namespace snapjudge
