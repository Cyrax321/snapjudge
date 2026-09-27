#include "snapjudge/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <atomic>
#include <stdexcept>
#include <thread>

// cblas: Apple Accelerate or OpenBLAS both export C symbol names.
extern "C" {
void cblas_sgemm(int Order, int TransA, int TransB, int M, int N, int K,
                 float alpha, const float* A, int lda, const float* B, int ldb,
                 float beta, float* C, int ldc);
}
namespace {
constexpr int COL_MAJOR_ROW_MAJOR = 101;  // CblasRowMajor
constexpr int NO_TRANS = 111;             // CblasNoTrans
constexpr int TRANS = 112;                // CblasTrans
}  // namespace

namespace snapjudge {

Tensor::Tensor(std::vector<int64_t> s) : shape(std::move(s)) {
  data.reset(new float[static_cast<size_t>(numel())]);
}

Tensor Tensor::zeros(std::vector<int64_t> s) {
  Tensor t(std::move(s));
  std::memset(t.data.get(), 0, sizeof(float) * static_cast<size_t>(t.numel()));
  return t;
}

int64_t Tensor::numel() const {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

// ---- elementwise --------------------------------------------------------------
Tensor add(const Tensor& a, const Tensor& b) {
  if (a.shape != b.shape) throw std::runtime_error("snapjudge add: shape mismatch");
  Tensor out(a.shape);
  for (int64_t i = 0; i < a.numel(); ++i) out.data[i] = a.data[i] + b.data[i];
  return out;
}

void add_(Tensor& a, const Tensor& b) {
  if (a.shape != b.shape) throw std::runtime_error("snapjudge add_: shape mismatch");
  for (int64_t i = 0; i < a.numel(); ++i) a.data[i] += b.data[i];
}

Tensor layer_norm(const Tensor& x, const float* w, const float* b, float eps) {
  int64_t D = x.dim(x.ndim() - 1);
  int64_t rows = x.numel() / D;
  Tensor out(x.shape);
  const float* X = x.data_ptr();
  float* O = out.mutable_data();
  for (int64_t r = 0; r < rows; ++r) {
    const float* xr = X + r * D;
    float* orow = O + r * D;
    double mean = 0;
    for (int64_t j = 0; j < D; ++j) mean += xr[j];
    mean /= D;
    double var = 0;
    for (int64_t j = 0; j < D; ++j) {
      double d = xr[j] - mean;
      var += d * d;
    }
    var /= D;
    float inv = static_cast<float>(1.0 / std::sqrt(var + eps));
    for (int64_t j = 0; j < D; ++j) {
      float v = (xr[j] - static_cast<float>(mean)) * inv * (w ? w[j] : 1.0f);
      orow[j] = b ? v + b[j] : v;
    }
  }
  return out;
}

void softmax_rows(Tensor& x) {
  int64_t K = x.dim(x.ndim() - 1);
  int64_t rows = x.numel() / K;
  float* X = x.mutable_data();
  for (int64_t r = 0; r < rows; ++r) {
    float* xr = X + r * K;
    float m = xr[0];
    for (int64_t j = 1; j < K; ++j) m = std::max(m, xr[j]);
    double s = 0;
    for (int64_t j = 0; j < K; ++j) { xr[j] = std::exp(xr[j] - m); s += xr[j]; }
    float inv = static_cast<float>(1.0 / s);
    for (int64_t j = 0; j < K; ++j) xr[j] *= inv;
  }
}

float gelu_erf(float x) {
  return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654757f));
}

// ---- GEMM ----------------------------------------------------------------------
static void sgemm(bool ta, bool tb, int M, int N, int K, float alpha,
                  const float* A, int lda, const float* B, int ldb,
                  float beta, float* C, int ldc) {
  cblas_sgemm(COL_MAJOR_ROW_MAJOR, ta ? TRANS : NO_TRANS, tb ? TRANS : NO_TRANS,
              M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

Tensor gemm_nt(const Tensor& a, const Tensor& w, const float* bias, int act) {
  int64_t K = a.dim(a.ndim() - 1);
  int64_t M = a.numel() / K;
  int64_t N = w.dim(0);
  if (w.dim(1) != K) throw std::runtime_error("snapjudge gemm: K mismatch");
  Tensor out({M, N});
  // C = A @ W^T
  sgemm(false, true, static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
        1.0f, a.data_ptr(), static_cast<int>(K), w.data_ptr(), static_cast<int>(K),
        0.0f, out.mutable_data(), static_cast<int>(N));
  if (bias || act) {
    for (int64_t r = 0; r < M; ++r) {
      float* orow = out.row(r);
      for (int64_t j = 0; j < N; ++j) {
        float v = orow[j];
        if (bias) v += bias[j];
        if (act == 1) v = gelu_erf(v);
        else if (act == 2) v = std::max(v, 0.0f);
        orow[j] = v;
      }
    }
  }
  return out;
}

Tensor gemm_geglu(const Tensor& a, const Tensor& wi) {
  int64_t K = a.dim(a.ndim() - 1);
  int64_t M = a.numel() / K;
  int64_t F2 = wi.dim(0);
  int64_t F = F2 / 2;
  Tensor tmp({M, F2});
  sgemm(false, true, static_cast<int>(M), static_cast<int>(F2), static_cast<int>(K),
        1.0f, a.data_ptr(), static_cast<int>(K), wi.data_ptr(), static_cast<int>(K),
        0.0f, tmp.mutable_data(), static_cast<int>(F2));
  Tensor out({M, F});
  for (int64_t r = 0; r < M; ++r) {
    const float* tin = tmp.row(r);
    float* orow = out.row(r);
    for (int64_t j = 0; j < F; ++j) {
      orow[j] = gelu_erf(tin[j]) * tin[F + j];
    }
  }
  return out;
}

// ---- attention (fp32, blocked; correctness-first CPU path) ---------------------
// (b, h) pairs are fully independent: parallelize over them with a small
// thread team, since per-(b,h) cost scales with lens[b]^2.
Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v,
                 const std::vector<int64_t>& lens, int window) {
  int64_t B = q.dim(0), H = q.dim(1), L = q.dim(2), Dh = q.dim(3);
  Tensor out({B, H, L, Dh});
  const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));

  unsigned nthreads = std::thread::hardware_concurrency();
  const char* env = std::getenv("SNAPJUDGE_THREADS");
  if (env && *env) {
    int n = std::atoi(env);
    if (n > 0) nthreads = static_cast<unsigned>(n);
  }
  nthreads = std::max(1u, std::min<unsigned>(nthreads, static_cast<unsigned>(B * H)));

  const int64_t work_total = B * H;
  std::vector<std::thread> pool;
  std::atomic<int64_t> next{0};

  auto worker = [&]() {
    std::vector<float> scores(static_cast<size_t>(L));
    while (true) {
      int64_t idx = next.fetch_add(1);
      if (idx >= work_total) break;
      int64_t b = idx / H;
      int64_t h = idx % H;
      int64_t len = lens.empty() ? L : lens[static_cast<size_t>(b)];
      const float* qb = q.data_ptr() + ((b * H + h) * L) * Dh;
      const float* kb = k.data_ptr() + ((b * H + h) * L) * Dh;
      const float* vb = v.data_ptr() + ((b * H + h) * L) * Dh;
      float* ob = out.mutable_data() + ((b * H + h) * L) * Dh;
      for (int64_t i = 0; i < L; ++i) {
        const float* qi = qb + i * Dh;
        int64_t j_lo = 0, j_hi = len;
        if (window > 0) {
          j_lo = std::max<int64_t>(0, i - window);
          j_hi = std::min(len, i + window + 1);
        }
        float m = -INFINITY;
        for (int64_t j = j_lo; j < j_hi; ++j) {
          const float* kj = kb + j * Dh;
          float s = 0;
          for (int64_t d = 0; d < Dh; ++d) s += qi[d] * kj[d];
          scores[static_cast<size_t>(j)] = s * scale;
          m = std::max(m, scores[static_cast<size_t>(j)]);
        }
        double l = 0;
        for (int64_t j = j_lo; j < j_hi; ++j) {
          float e = std::exp(scores[static_cast<size_t>(j)] - m);
          scores[static_cast<size_t>(j)] = e;
          l += e;
        }
        float linv = static_cast<float>(1.0 / l);
        float* oi = ob + i * Dh;
        for (int64_t d = 0; d < Dh; ++d) oi[d] = 0.0f;
        for (int64_t j = j_lo; j < j_hi; ++j) {
          float w = scores[static_cast<size_t>(j)] * linv;
          const float* vj = vb + j * Dh;
          for (int64_t d = 0; d < Dh; ++d) oi[d] += w * vj[d];
        }
      }
    }
  };
  for (unsigned t = 0; t < nthreads; ++t) pool.emplace_back(worker);
  for (auto& th : pool) th.join();
  return out;
}

// ---- gather / embedding / rope -------------------------------------------------
Tensor gather_rows(const Tensor& h, const int64_t* pos, int64_t B, int64_t K) {
  int64_t D = h.dim(2);
  Tensor out({B, K, D});
  int64_t L = h.dim(1);
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t kk = 0; kk < K; ++kk) {
      int64_t p = pos[b * K + kk];
      if (p < 0) p = 0;
      if (p >= L) p = L - 1;
      std::memcpy(out.mutable_data() + (b * K + kk) * D,
                  h.data_ptr() + (b * L + p) * D, sizeof(float) * static_cast<size_t>(D));
    }
  }
  return out;
}

Tensor embedding(const Tensor& table, const int64_t* ids, int64_t n) {
  int64_t D = table.dim(1);
  Tensor out({n, D});
  for (int64_t r = 0; r < n; ++r) {
    std::memcpy(out.mutable_data() + r * D,
                table.data_ptr() + ids[r] * D, sizeof(float) * static_cast<size_t>(D));
  }
  return out;
}

void rope_inplace(Tensor& qkv, const float* cos_t, const float* sin_t, int64_t L,
                  int64_t H, int64_t Dh) {
  // qkv: [M, 3*H*Dh], layout (q|k|v)(head)(dim); row r has position r % L.
  // Rotate-half: pairs (d, d + Dh/2) within each head, fp32 math (HF convention).
  int64_t M = qkv.dim(0);
  int64_t half = Dh / 2;
  float* X = qkv.mutable_data();
  for (int64_t r = 0; r < M; ++r) {
    int64_t pos = r % L;
    const float* cr = cos_t + pos * half;
    const float* sr = sin_t + pos * half;
    float* row = X + r * 3 * H * Dh;
    for (int hh = 0; hh < 2 * H; ++hh) {  // q heads then k heads
      float* qk = row + hh * Dh;
      for (int64_t d = 0; d < half; ++d) {
        float x0 = qk[d], x1 = qk[d + half];
        float cs = cr[d], sn = sr[d];
        qk[d] = x0 * cs - x1 * sn;
        qk[d + half] = x1 * cs + x0 * sn;
      }
    }
  }
}

}  // namespace snapjudge
