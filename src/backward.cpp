#include "snapjudge/backward.hpp"

#include <cmath>
#include <cstring>
#include <vector>

extern "C" {
void cblas_sgemm(int Order, int TransA, int TransB, int M, int N, int K,
                 float alpha, const float* A, int lda, const float* B, int ldb,
                 float beta, float* C, int ldc);
}
namespace {
constexpr int ROW_MAJOR = 101, NO_T = 111, T = 112;
}

namespace snapjudge {

float gelu_erf_grad(float x) {
  // d/dx [0.5x(1+erf(x/√2))] = Φ(x) + x φ(x)
  float cdf = 0.5f * (1.0f + std::erf(x * 0.70710678118654757f));
  float pdf = 0.39894228040143267794f * std::exp(-0.5f * x * x);
  return cdf + x * pdf;
}

void gemm_nt_backward(const float* dy, const float* y_pre, const float* x,
                      const float* W, int64_t M, int64_t N, int64_t K, int act,
                      float* dx, float* dW, float* db) {
  std::vector<float> dz(static_cast<size_t>(M * N));
  for (int64_t i = 0; i < M * N; ++i) {
    float g = dy[i];
    if (act == 1) g *= gelu_erf_grad(y_pre[i]);
    else if (act == 2) g *= y_pre[i] > 0 ? 1.0f : 0.0f;
    dz[static_cast<size_t>(i)] = g;
  }
  if (db) {
    for (int64_t j = 0; j < N; ++j) {
      double s = 0;
      for (int64_t i = 0; i < M; ++i) s += dz[static_cast<size_t>(i * N + j)];
      db[j] += static_cast<float>(s);
    }
  }
  // dx = dz @ W   ([M,N]x[N,K] -> [M,K]) ; beta=1 accumulates
  cblas_sgemm(ROW_MAJOR, NO_T, NO_T, (int)M, (int)K, (int)N, 1.0f, dz.data(), (int)N,
              W, (int)K, 1.0f, dx, (int)K);
  // dW = dz^T @ x  ([N,M]x[M,K] -> [N,K])
  cblas_sgemm(ROW_MAJOR, T, NO_T, (int)N, (int)K, (int)M, 1.0f, dz.data(), (int)N,
              x, (int)K, 1.0f, dW, (int)K);
}

void layer_norm_backward(const float* dy, const float* x, const float* w,
                         int64_t M, int64_t D, float eps, bool has_bias,
                         float* dx, float* dw, float* db) {
  for (int64_t r = 0; r < M; ++r) {
    const float* xr = x + r * D;
    const float* dr = dy + r * D;
    float* dxr = dx + r * D;
    double mean = 0, var = 0;
    for (int64_t j = 0; j < D; ++j) mean += xr[j];
    mean /= D;
    for (int64_t j = 0; j < D; ++j) {
      double d = xr[j] - mean;
      var += d * d;
    }
    var /= D;
    float inv = static_cast<float>(1.0 / std::sqrt(var + eps));
    double c1 = 0, c2 = 0;
    std::vector<double> xhat(static_cast<size_t>(D));
    for (int64_t j = 0; j < D; ++j) {
      xhat[static_cast<size_t>(j)] = (xr[j] - mean) * inv;
      if (db) db[j] += dr[j];
      if (dw) dw[j] += dr[j] * static_cast<float>(xhat[static_cast<size_t>(j)]);
    }
    double dj = 0;
    for (int64_t j = 0; j < D; ++j) {
      dj += dr[j] * (double)w[j] * xhat[static_cast<size_t>(j)];
      c2 += dr[j] * (double)w[j];
    }
    c1 = dj;
    for (int64_t j = 0; j < D; ++j) {
      double a = dr[j] * w[j];
      double term = (a - c2 / D - xhat[static_cast<size_t>(j)] * (c1 / D)) * inv;
      dxr[j] = static_cast<float>(term);
    }
    (void)has_bias;
  }
}

void gemm_geglu_backward(const float* dC, const float* A, const float* Wi,
                         int64_t M, int64_t F, int64_t K, float* dA, float* dWi) {
  // recompute pre-activations z = A @ Wi^T ([M, 2F])
  std::vector<float> z(static_cast<size_t>(M * 2 * F));
  cblas_sgemm(ROW_MAJOR, NO_T, T, (int)M, (int)(2 * F), (int)K, 1.0f, A, (int)K,
              Wi, (int)K, 0.0f, z.data(), (int)(2 * F));
  std::vector<float> dz(static_cast<size_t>(M * 2 * F));
  for (int64_t i = 0; i < M; ++i) {
    const float* dcr = dC + i * F;
    for (int64_t j = 0; j < F; ++j) {
      float zi = z[static_cast<size_t>(i * 2 * F + j)];
      float zg = z[static_cast<size_t>(i * 2 * F + F + j)];
      float g = gelu_erf_grad(zi);
      // C = gelu(zi) * zg  ==>  dC/dzi = gelu'(zi)*zg ; dC/dzg = gelu(zi)
      dz[static_cast<size_t>(i * 2 * F + j)] = dcr[j] * zg * g;
      dz[static_cast<size_t>(i * 2 * F + F + j)] =
          dcr[j] * (0.5f * zi * (1.0f + std::erf(zi * 0.70710678118654757f)));
    }
  }
  // dA = dz @ Wi  ; dWi = dz^T @ A
  cblas_sgemm(ROW_MAJOR, NO_T, NO_T, (int)M, (int)K, (int)(2 * F), 1.0f, dz.data(),
              (int)(2 * F), Wi, (int)K, 1.0f, dA, (int)K);
  cblas_sgemm(ROW_MAJOR, T, NO_T, (int)(2 * F), (int)K, (int)M, 1.0f, dz.data(),
              (int)(2 * F), A, (int)K, 1.0f, dWi, (int)K);
}

void attention_forward_one(const float* q, const float* k, const float* v,
                           int64_t H, int64_t L, int64_t Dh, int64_t len,
                           int window, float* o, float* p) {
  const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));
  std::vector<float> scores(static_cast<size_t>(L));
  for (int64_t h = 0; h < H; ++h) {
    const float* qh = q + h * L * Dh;
    const float* kh = k + h * L * Dh;
    const float* vh = v + h * L * Dh;
    float* oh = o + h * L * Dh;
    float* ph = p + h * L * L;
    for (int64_t i = 0; i < L; ++i) {
      const float* qi = qh + i * Dh;
      int64_t j_lo = 0, j_hi = len;
      if (window > 0) {
        j_lo = std::max<int64_t>(0, i - window);
        j_hi = std::min(len, i + window + 1);
      }
      float m = -INFINITY;
      for (int64_t j = j_lo; j < j_hi; ++j) {
        const float* kj = kh + j * Dh;
        float s = 0;
        for (int64_t d = 0; d < Dh; ++d) s += qi[d] * kj[d];
        scores[static_cast<size_t>(j)] = s * scale;
        m = std::max(m, scores[static_cast<size_t>(j)]);
      }
      double l = 0;
      for (int64_t j = j_lo; j < j_hi; ++j) {
        float e = std::exp(scores[static_cast<size_t>(j)] - m);
        ph[i * L + j] = e;
        l += e;
      }
      float linv = static_cast<float>(1.0 / l);
      float* oi = oh + i * Dh;
      for (int64_t d = 0; d < Dh; ++d) oi[d] = 0.0f;
      for (int64_t j = j_lo; j < j_hi; ++j) {
        float w = ph[i * L + j] * linv;
        ph[i * L + j] = w;
        const float* vj = vh + j * Dh;
        for (int64_t d = 0; d < Dh; ++d) oi[d] += w * vj[d];
      }
    }
  }
}

void attention_backward_one(const float* dout, const float* q, const float* k,
                            const float* v, const float* p, int64_t H, int64_t L,
                            int64_t Dh, int64_t len, int window,
                            float* dq, float* dk, float* dv) {
  const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));
  std::vector<float> ds(static_cast<size_t>(L));
  for (int64_t h = 0; h < H; ++h) {
    const float* qh = q + h * L * Dh;
    const float* kh = k + h * L * Dh;
    const float* vh = v + h * L * Dh;
    const float* ph = p + h * L * L;
    const float* doh = dout + h * L * Dh;
    float* dqh = dq + h * L * Dh;
    float* dkh = dk + h * L * Dh;
    float* dvh = dv + h * L * Dh;
    for (int64_t i = 0; i < L; ++i) {
      int64_t j_lo = 0, j_hi = len;
      if (window > 0) {
        j_lo = std::max<int64_t>(0, i - window);
        j_hi = std::min(len, i + window + 1);
      }
      // dV_j += p_ij * dO_i
      for (int64_t j = j_lo; j < j_hi; ++j) {
        float w = ph[i * L + j];
        const float* doi = doh + i * Dh;
        float* dvj = dvh + j * Dh;
        for (int64_t d = 0; d < Dh; ++d) dvj[d] += w * doi[d];
      }
      // dP_ij = dO_i . V_j ; softmax backward -> dS
      double dot = 0;
      for (int64_t j = j_lo; j < j_hi; ++j) {
        const float* vj = vh + j * Dh;
        const float* doi = doh + i * Dh;
        double s = 0;
        for (int64_t d = 0; d < Dh; ++d) s += (double)doi[d] * vj[d];
        ds[static_cast<size_t>(j)] = static_cast<float>(s);
        dot += s * ph[i * L + j];
      }
      for (int64_t j = j_lo; j < j_hi; ++j)
        ds[static_cast<size_t>(j)] =
            ph[i * L + j] * static_cast<float>(ds[static_cast<size_t>(j)] - dot);
      // dQ_i = scale * dS @ K ; dK_j += scale * ds_j * Q_i
      for (int64_t d = 0; d < Dh; ++d) {
        double s = 0;
        for (int64_t j = j_lo; j < j_hi; ++j)
          s += (double)ds[static_cast<size_t>(j)] * kh[j * Dh + d];
        dqh[i * Dh + d] += static_cast<float>(s * scale);
      }
      for (int64_t j = j_lo; j < j_hi; ++j) {
        float* dkj = dkh + j * Dh;
        float dsj = ds[static_cast<size_t>(j)] * scale;
        const float* qi = qh + i * Dh;
        for (int64_t d = 0; d < Dh; ++d) dkj[d] += dsj * qi[d];
      }
    }
  }
}

void rope_backward(float* dqkv, const float* cos_t, const float* sin_t, int64_t M,
                   int64_t L, int64_t H, int64_t Dh) {
  // Forward: y0 = x0 c - x1 s ; y1 = x1 c + x0 s  (orthonormal rotation)
  // Backward: dx0 = dy0 c + dy1 s ; dx1 = dy1 c - dy0 s
  int64_t half = Dh / 2;
  for (int64_t r = 0; r < M; ++r) {
    int64_t pos = r % L;
    const float* cr = cos_t + pos * half;
    const float* sr = sin_t + pos * half;
    float* row = dqkv + r * 3 * H * Dh;
    for (int64_t hh = 0; hh < 2 * H; ++hh) {
      float* qk = row + hh * Dh;
      for (int64_t d = 0; d < half; ++d) {
        float g0 = qk[d], g1 = qk[d + half];
        float cs = cr[d], sn = sr[d];
        qk[d] = g0 * cs + g1 * sn;
        qk[d + half] = g1 * cs - g0 * sn;
      }
    }
  }
}

void embedding_backward(const float* dY, const int64_t* ids, int64_t n, int64_t D,
                        float* dW) {
  for (int64_t r = 0; r < n; ++r) {
    float* g = dW + ids[r] * D;
    const float* dy = dY + r * D;
    for (int64_t d = 0; d < D; ++d) g[d] += dy[d];
  }
}

void softmax_row_backward(const float* gq, const float* q, int64_t k, float* gz) {
  double dot = 0;
  for (int64_t i = 0; i < k; ++i) dot += (double)gq[i] * q[i];
  for (int64_t i = 0; i < k; ++i)
    gz[i] = q[i] * static_cast<float>(gq[i] - dot);
}

}  // namespace snapjudge
