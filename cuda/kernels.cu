// snapjudge/cuda/kernels.cu — CUDA port of tl_kernels.py.
//
// Same contracts as the TileLang kernels:
//   * activations bf16 (CUDA __nv_bfloat16), accumulation fp32
//   * gemm:            C[M,N] = act(A[M,K] @ W[N,K]^T + b)
//   * gemm_geglu:      C[M,F] = gelu(A @ Wi[:F]^T) * (A @ Wi[F:]^T)
//   * add_ln:          X(fp32) += R(bf16); Y(bf16) = LN(X)*w (+b)
//   * rope:            in-place rotate-half on packed qkv [M, 3*H*Dh], pos = r % L
//   * attn:            online-softmax flash attention, lens mask + sliding window
//
// Verification: bench/parity_fast.py on a CUDA machine compares these against
// both the TileLang path and the CPU fp32 reference within fast-path tolerance.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace snapjudge { namespace cuda_kernels {
using kbf16 = __nv_bfloat16;


using bf16 = __nv_bfloat16;

#define FULL_MASK 0xffffffffu

// ------------------------------------------------------------------ GEMM
// C[M,N] = ACT(A[M,K] @ W[N,K]^T (+Bv)). Blocked, fp32 accumulate, bf16 io.
// act: 0 none, 1 erf-GELU, 2 relu — the exact same functions as tl_kernels.
template <int BM, int BN, int BK, int BIAS, int ACT>
__global__ void gemm_kernel(const bf16* __restrict__ A,   // [M, K]
                            const bf16* __restrict__ W,   // [N, K]
                            const float* __restrict__ Bv, // [N] or null
                            bf16* __restrict__ C,         // [M, N]
                            int M, int N, int K) {
  __shared__ float As[BM][BK];
  __shared__ float Ws[BN][BK];

  const int bx = blockIdx.x;  // N tile
  const int by = blockIdx.y;  // M tile
  const int ty = threadIdx.y; // [0, BM/4)
  const int tx = threadIdx.x; // [0, BN/4)

  float acc[4][4] = {};

  const int row0 = by * BM;
  const int col0 = bx * BN;

  for (int k0 = 0; k0 < K; k0 += BK) {
    // load A tile (BM x BK), zero-padded
    for (int i = ty * 4; i < ty * 4 + 4 && i < BM; ++i) {
      int gr = row0 + i;
      for (int dk = tx; dk < BK; dk += blockDim.x) {
        int gk = k0 + dk;
        As[i][dk] = (gr < M && gk < K) ? __bfloat162float(A[gr * K + gk]) : 0.f;
      }
    }
    for (int i = ty * 4; i < ty * 4 + 4 && i < BN; ++i) {
      int gr = col0 + i;
      for (int dk = tx; dk < BK; dk += blockDim.x) {
        int gk = k0 + dk;
        Ws[i][dk] = (gr < N && gk < K) ? __bfloat162float(W[gr * K + gk]) : 0.f;
      }
    }
    __syncthreads();
    for (int dk = 0; dk < BK; ++dk)
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
          acc[i][j] += As[ty * 4 + i][dk] * Ws[tx * 4 + j][dk];
    __syncthreads();
  }
  for (int i = 0; i < 4; ++i) {
    int gr = row0 + ty * 4 + i;
    if (gr >= M) break;
    for (int j = 0; j < 4; ++j) {
      int gc = col0 + tx * 4 + j;
      if (gc >= N) continue;
      float v = acc[i][j];
      if (BIAS) v += Bv[gc];
      if (ACT == 1) v = 0.5f * v * (1.0f + erff(v * 0.70710678118654757f));
      else if (ACT == 2) v = fmaxf(v, 0.f);
      C[gr * N + gc] = __float2bfloat16(v);
    }
  }
}

extern "C" void sj_gemm(const bf16* A, const bf16* W, const float* Bv, bf16* C,
                        int M, int N, int K, bool bias, int act, cudaStream_t st) {
  constexpr int BM = 64, BN = 128, BK = 32;
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 block(BN / 4, BM / 4);
  if (bias) {
    if (act == 1)
      gemm_kernel<BM, BN, BK, 1, 1><<<grid, block, 0, st>>>(A, W, Bv, C, M, N, K);
    else if (act == 2)
      gemm_kernel<BM, BN, BK, 1, 2><<<grid, block, 0, st>>>(A, W, Bv, C, M, N, K);
    else
      gemm_kernel<BM, BN, BK, 1, 0><<<grid, block, 0, st>>>(A, W, Bv, C, M, N, K);
  } else {
    if (act == 1)
      gemm_kernel<BM, BN, BK, 0, 1><<<grid, block, 0, st>>>(A, W, Bv, C, M, N, K);
    else if (act == 2)
      gemm_kernel<BM, BN, BK, 0, 2><<<grid, block, 0, st>>>(A, W, Bv, C, M, N, K);
    else
      gemm_kernel<BM, BN, BK, 0, 0><<<grid, block, 0, st>>>(A, W, Bv, C, M, N, K);
  }
}

// ------------------------------------------------------------------ GEGLU fused
// C[M,F] = gelu(A @ Wi[:F]^T) * (A @ Wi[F:]^T)   (Wi is [2F, K])
template <int BM, int BN, int BK>
__global__ void gemm_geglu_kernel(const bf16* __restrict__ A,
                                  const bf16* __restrict__ Wi,
                                  bf16* __restrict__ C, int M, int F, int K) {
  __shared__ float As[BM][BK];
  __shared__ float Ws[BN][BK];
  __shared__ float Wg[BN][BK];

  const int bx = blockIdx.x, by = blockIdx.y;
  const int ty = threadIdx.y, tx = threadIdx.x;
  float acc_i[4][4] = {}, acc_g[4][4] = {};

  const int row0 = by * BM, col0 = bx * BN;

  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int i = ty * 4; i < ty * 4 + 4 && i < BM; ++i) {
      int gr = row0 + i;
      for (int dk = tx; dk < BK; dk += blockDim.x) {
        int gk = k0 + dk;
        As[i][dk] = (gr < M && gk < K) ? __bfloat162float(A[gr * K + gk]) : 0.f;
      }
    }
    for (int i = ty * 4; i < ty * 4 + 4 && i < BN; ++i) {
      int gr = col0 + i;  // rows of Wi are [0, F) for gate-in and [F, 2F) for gate
      for (int dk = tx; dk < BK; dk += blockDim.x) {
        int gk = k0 + dk;
        float a = (gr < F && gk < K) ? __bfloat162float(Wi[gr * K + gk]) : 0.f;
        float g = (gr < F && gk < K) ? __bfloat162float(Wi[(F + gr) * K + gk]) : 0.f;
        Ws[i][dk] = a;
        Wg[i][dk] = g;
      }
    }
    __syncthreads();
    for (int dk = 0; dk < BK; ++dk)
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
          float a = As[ty * 4 + i][dk];
          acc_i[i][j] += a * Ws[tx * 4 + j][dk];
          acc_g[i][j] += a * Wg[tx * 4 + j][dk];
        }
    __syncthreads();
  }
  for (int i = 0; i < 4; ++i) {
    int gr = row0 + ty * 4 + i;
    if (gr >= M) break;
    for (int j = 0; j < 4; ++j) {
      int gc = col0 + tx * 4 + j;
      if (gc >= F) continue;
      float in = acc_i[i][j];
      float g = acc_g[i][j];
      C[gr * F + gc] =
          __float2bfloat16(0.5f * in * (1.0f + erff(in * 0.70710678118654757f)) * g);
    }
  }
}

extern "C" void sj_gemm_geglu(const bf16* A, const bf16* Wi, bf16* C, int M, int F,
                              int K, cudaStream_t st) {
  constexpr int BM = 64, BN = 64, BK = 32;
  dim3 grid((F + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 block(BN / 4, BM / 4);
  gemm_geglu_kernel<BM, BN, BK><<<grid, block, 0, st>>>(A, Wi, C, M, F, K);
}

// ------------------------------------------------------------------ add + LayerNorm
// X (fp32, in place) [+ R (bf16)]; Y (bf16) = LN(X) * Wv (+ Bv).
// BIAS: whether Bv is applied; EPS as template-friendly runtime param.
template <int BM, int BIAS>
__global__ void add_ln_kernel(float* __restrict__ X, const bf16* __restrict__ R,
                              const float* __restrict__ Wv,
                              const float* __restrict__ Bv, bf16* __restrict__ Y,
                              int M, int D, float eps, bool has_residual) {
  const int bx = blockIdx.x;
  const int row0 = bx * BM;
  extern __shared__ float smem[];  // BM*D x fp32

  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  for (int i = 0; i < BM; ++i) {
    int r = row0 + i;
    if (r >= M) break;
    for (int j = tid; j < D; j += nthreads) {
      float xv = X[static_cast<int64_t>(r) * D + j];
      if (has_residual) xv += __bfloat162float(R[static_cast<int64_t>(r) * D + j]);
      X[static_cast<int64_t>(r) * D + j] = xv;
      smem[i * D + j] = xv;
    }
  }
  __syncthreads();

  for (int i = 0; i < BM; ++i) {
    int r = row0 + i;
    if (r >= M) break;
    // mean
    float mean = 0.f;
    for (int j = tid; j < D; j += nthreads) mean += smem[i * D + j];
    // block reduce
    __shared__ float red[128];
    red[tid] = mean;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
      if (tid < s) red[tid] += red[tid + s];
      __syncthreads();
    }
    mean = red[0] / D;
    __syncthreads();
    if (tid == 0) red[tid] = 0.f;
    __syncthreads();
    float var = 0.f;
    for (int j = tid; j < D; j += nthreads) {
      float d = smem[i * D + j] - mean;
      var += d * d;
    }
    red[tid] = var;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
      if (tid < s) red[tid] += red[tid + s];
      __syncthreads();
    }
    float inv = rsqrtf(red[0] / D + eps);
    __syncthreads();
    for (int j = tid; j < D; j += nthreads) {
      float v = (smem[i * D + j] - mean) * inv * Wv[j];
      if (BIAS) v += Bv[j];
      Y[static_cast<int64_t>(r) * D + j] = __float2bfloat16(v);
    }
    __syncthreads();
  }
}

extern "C" void sj_add_ln(float* X, const bf16* R, const float* W, const float* B,
                          bf16* Y, int M, int D, float eps, bool residual, bool bias,
                          cudaStream_t st) {
  constexpr int BM = 4;
  dim3 grid((M + BM - 1) / BM);
  dim3 block(64);
  size_t sh = BM * D * sizeof(float);
  if (bias)
    add_ln_kernel<BM, 1><<<grid, block, sh, st>>>(X, R, W, B, Y, M, D, eps, residual);
  else
    add_ln_kernel<BM, 0><<<grid, block, sh, st>>>(X, R, W, B, Y, M, D, eps, residual);
}

// ------------------------------------------------------------------ RoPE in place
// qkv [M, 3*H*Dh], layout (q|k|v)(head)(dim); row r has position r % L.
__global__ void rope_kernel(bf16* __restrict__ qkv, const float* __restrict__ cos_t,
                            const float* __restrict__ sin_t, int64_t M, int L, int H,
                            int Dh) {
  int64_t r = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (r >= M) return;
  int half = Dh / 2;
  int c = blockIdx.y * blockDim.x + threadIdx.x;
  if (c >= H * Dh) return;  // H*Dh/2 pairs? c iterates 0..H*Dh/2-1 across heads
  // Wait: index space is (head, d) pairs: total 2*H * half. c in [0, 2*H*half).
  int hh = c / half;
  int d = c % half;
  if (hh >= 2 * H) return;
  int pos = static_cast<int>(r % L);
  float cs = cos_t[pos * half + d];
  float sn = sin_t[pos * half + d];
  bf16* row = qkv + r * 3 * H * Dh;
  int c0 = hh * Dh + d;
  int c1 = c0 + half;
  float x0 = __bfloat162float(row[c0]);
  float x1 = __bfloat162float(row[c1]);
  row[c0] = __float2bfloat16(x0 * cs - x1 * sn);
  row[c1] = __float2bfloat16(x1 * cs + x0 * sn);
}

extern "C" void sj_rope(bf16* qkv, const float* cos_t, const float* sin_t, int64_t M,
                        int L, int H, int Dh, cudaStream_t st) {
  int half = Dh / 2;
  dim3 block(32, 4);
  dim3 grid((M + block.y - 1) / block.y, (2 * H * half + block.x - 1) / block.x);
  rope_kernel<<<grid, block, 0, st>>>(qkv, cos_t, sin_t, M, L, H, Dh);
}

// ------------------------------------------------------------------ flash attention
// qkv: [B, L, 3, H, Dh] bf16 view of packed [M, 3*H*Dh].  lens: [B] int32.
// O: [B, L, H*Dh] bf16. window>0 => bidirectional sliding |i-j| <= window.
// Online softmax with exp2 formulation and large-finite masking, matching
// tl_kernels.py::attn_kernel (NEG=-1e9, max(l,1e-30)).
// One block = one (bz, by=head, query row). blockDim.x threads share the row.
__global__ void attn_kernel(const bf16* __restrict__ qkv,
                            const int* __restrict__ lens, bf16* __restrict__ O, int B,
                            int L, int H, int Dh, int window) {
  const int qi = blockIdx.x;   // query row
  const int by = blockIdx.y;   // head
  const int bz = blockIdx.z;   // batch
  const int n = lens[bz];
  const float scale = rsqrtf(static_cast<float>(Dh)) * 1.44269504088896340736f;

  extern __shared__ float fs[];  // 2*Dh: q row + o accumulator
  float* qrow = fs;
  float* orow = fs + Dh;

  const bf16* base = qkv + ((static_cast<int64_t>(bz) * L + qi) * 3 + 0) * H * Dh + by * Dh;
  for (int d = threadIdx.x; d < Dh; d += blockDim.x)
    qrow[d] = __bfloat162float(base[d]);
  for (int d = threadIdx.x; d < Dh; d += blockDim.x) orow[d] = 0.f;
  __syncthreads();

  float m = -1e9f;
  float l = 0.f;

  int k_lo = 0, k_hi = n;
  if (window > 0) {
    k_lo = qi - window;
    if (k_lo < 0) k_lo = 0;
    k_hi = qi + window + 1;
    if (k_hi > n) k_hi = n;
  }

  for (int kj = k_lo; kj < k_hi; ++kj) {
    const bf16* krow =
        qkv + ((static_cast<int64_t>(bz) * L + kj) * 3 + 1) * H * Dh + by * Dh;
    const bf16* vrow =
        qkv + ((static_cast<int64_t>(bz) * L + kj) * 3 + 2) * H * Dh + by * Dh;
    float s = 0.f;
    for (int d = threadIdx.x; d < Dh; d += blockDim.x)
      s += qrow[d] * __bfloat162float(krow[d]);
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(FULL_MASK, s, off);
    s = __shfl_sync(FULL_MASK, s, 0);

    float m_new = fmaxf(m, s);
    float sc = exp2f(m * scale - m_new * scale);
    float p = exp2f(s * scale - m_new * scale);
    l = l * sc + p;
    m = m_new;
    for (int d = threadIdx.x; d < Dh; d += blockDim.x)
      orow[d] = orow[d] * sc + p * __bfloat162float(vrow[d]);
  }
  float li = fmaxf(l, 1e-30f);
  bf16* orow_out = O + (static_cast<int64_t>(bz) * L + qi) * H * Dh + by * Dh;
  for (int d = threadIdx.x; d < Dh; d += blockDim.x)
    orow_out[d] = __float2bfloat16(orow[d] / li);
}

extern "C" void sj_attn(const bf16* qkv, const int* lens, bf16* O, int B, int L, int H,
                        int Dh, int window, cudaStream_t st) {
  dim3 grid(L, H, B);
  dim3 block(32);
  size_t sh = 2 * Dh * sizeof(float);
  attn_kernel<<<grid, block, sh, st>>>(qkv, lens, O, B, L, H, Dh, window);
}


// ----------------------------------------------------------- glue kernels -----

// embeddings gather: Y (bf16) = emb_w[ids[m], :] with fp16 table read.
__global__ void emb_gather_kernel(const __half* __restrict__ W,
                                  const int64_t* __restrict__ ids,
                                  bf16* __restrict__ Y, int64_t M, int64_t D) {
  int64_t m = static_cast<int64_t>(blockIdx.x);
  if (m >= M) return;
  int64_t id = ids[m];
  for (int64_t j = threadIdx.x; j < D; j += blockDim.x)
    Y[m * D + j] = __float2bfloat16(__half2float(W[id * D + j]));
}

// embeddings norm: X (fp32) = LN(fp32(Y)). One block per row.
__global__ void emb_ln_kernel(const bf16* __restrict__ Y, const float* __restrict__ W,
                              float* __restrict__ X, int64_t M, int64_t D, float eps) {
  int64_t m = blockIdx.x;
  if (m >= M) return;
  extern __shared__ float row[];
  for (int64_t j = threadIdx.x; j < D; j += blockDim.x)
    row[j] = __bfloat162float(Y[m * D + j]);
  __syncthreads();
  __shared__ float red[64];
  float mean = 0.f;
  for (int64_t j = threadIdx.x; j < D; j += blockDim.x) mean += row[j];
  red[threadIdx.x] = mean;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  mean = red[0] / D;
  __syncthreads();
  float var = 0.f;
  for (int64_t j = threadIdx.x; j < D; j += blockDim.x) {
    float d = row[j] - mean;
    var += d * d;
  }
  red[threadIdx.x] = var;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  float inv = rsqrtf(red[0] / D + eps);
  for (int64_t j = threadIdx.x; j < D; j += blockDim.x)
    X[m * D + j] = (row[j] - mean) * inv * W[j];
}

__global__ void f32_to_bf16_kernel(const float* __restrict__ X, bf16* __restrict__ Y,
                                   int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) Y[i] = __float2bfloat16(X[i]);
}

// head entry: X[m, :] (fp32) = fp32(Yh[m, :]) + type_emb[qt[m], :]
// qtype indexes per batch row (m / L).
__global__ void head_entry_kernel(const bf16* __restrict__ Yh,
                                  const bf16* __restrict__ type_emb,
                                  const int64_t* __restrict__ qt, float* __restrict__ X,
                                  int64_t M, int64_t D, int64_t L) {
  int64_t m = blockIdx.x;
  if (m >= M) return;
  int64_t q = qt[m / L];
  for (int64_t j = threadIdx.x; j < D; j += blockDim.x)
    X[m * D + j] = __bfloat162float(Yh[m * D + j]) + __bfloat162float(type_emb[q * D + j]);
}

__global__ void add_bf16_to_f32_kernel(float* __restrict__ X,
                                       const bf16* __restrict__ Y, int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) X[i] += __bfloat162float(Y[i]);
}

}}  // namespace snapjudge::cuda_kernels
