#pragma once
// snapjudge backward.hpp: gradient kernels for every op in the training
// forward graph. Each *_backward writes into preallocated grad tensors
// (accumulating, so callers can gradient-accumulate across micro-steps).
// All fp32; numerics checked by tests/test_train.cpp finite differences.

#include <cstdint>
#include <vector>

namespace snapjudge {

// y = x @ W^T (+ b) with act epilogue.  x [M,K], W [N,K], dy [M,N].
// act: 0 none, 1 erf-GELU, 2 relu.  y_pre is the pre-activation values [M,N]
// (needed because GELU's derivative needs z, relu needs the sign).
void gemm_nt_backward(const float* dy, const float* y_pre, const float* x,
                      const float* W, int64_t M, int64_t N, int64_t K, int act,
                      float* dx, float* dW, float* db);

// y = LN(x)*w (+b), rows [M, D].  eps.  Produces dx, dw, db.
void layer_norm_backward(const float* dy, const float* x, const float* w,
                         int64_t M, int64_t D, float eps, bool has_bias,
                         float* dx, float* dw, float* db);

// geglu forward was C = gelu(A@Wi[:F]^T) * (A@Wi[F:]^T). Input A [M,K],
// Wi [2F,K].  dC [M,F].  Returns dA, dWi.
void gemm_geglu_backward(const float* dC, const float* A, const float* Wi,
                         int64_t M, int64_t F, int64_t K, float* dA, float* dWi);

// attention forward (used to recompute scores for backward):
// q,k,v [H, L, Dh] (single batch row), len = valid key count, window.
// Fills out o, probs p [H, L, Lmax_keys].
void attention_forward_one(const float* q, const float* k, const float* v,
                           int64_t H, int64_t L, int64_t Dh, int64_t len,
                           int window, float* o, float* p /*H*L*L*/);

// attention backward: dq, dk, dv from do.
void attention_backward_one(const float* dout, const float* q, const float* k,
                            const float* v, const float* p, int64_t H, int64_t L,
                            int64_t Dh, int64_t len, int window,
                            float* dq, float* dk, float* dv);

// rope backward: given dqkv grads w.r.t. rotated q/k, inverse-rotate them.
void rope_backward(float* dqkv, const float* cos_t, const float* sin_t, int64_t M,
                   int64_t L, int64_t H, int64_t Dh);

// embedding backward: scatter-add dY rows into dW at ids.
void embedding_backward(const float* dY, const int64_t* ids, int64_t n, int64_t D,
                        float* dW);

// softmax row backward: given grad w.r.t. probabilities and the probs,
// compute grad w.r.t. logits.  grad_logit[i] = q_i * (gq_i - sum_j gq_j q_j).
void softmax_row_backward(const float* gq, const float* q, int64_t k, float* gz);

// scalar ops derivative helpers:
float gelu_erf_grad(float x);

}  // namespace snapjudge
