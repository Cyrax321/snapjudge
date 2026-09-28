// snapjudge cblas_fallback.cpp: pure-C++ cblas_sgemm for builds without a BLAS
// (SNAPJUDGE_NO_BLAS — WASM, bare-metal, or any environment where Apple
// Accelerate / OpenBLAS is unavailable).
//
// All internal callers use CblasRowMajor (101) with NoTrans (111) / Trans (112)
// flags, so this fallback implements exactly that subset:
//
//   C = alpha * op(A) @ op(B) + beta * C,  row-major
//
// where op(X) is X (NoTrans) or X^T (Trans). This is a correctness-first
// reference path (plain dot products), not an optimized kernel; it exists only
// so the engine links and runs where a BLAS is not present.

#ifdef SNAPJUDGE_NO_BLAS

namespace {
constexpr int ROW_MAJOR = 101;
constexpr int NO_TRANS = 111;
constexpr int TRANS = 112;
}  // namespace

extern "C" void cblas_sgemm(int Order, int TransA, int TransB, int M, int N, int K,
                            float alpha, const float* A, int lda, const float* B, int ldb,
                            float beta, float* C, int ldc) {
  if (Order != ROW_MAJOR) {
    // Column-major is unused by snapjudge; a host with a real BLAS takes that
    // path. Treating it as row-major would silently corrupt results, so do
    // nothing rather than return a wrong answer.
    return;
  }
  const bool ta = (TransA == TRANS);
  const bool tb = (TransB == TRANS);
  const bool add_c = (beta != 0.0f);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        float a = ta ? A[k * lda + i] : A[i * lda + k];
        float b = tb ? B[j * ldb + k] : B[k * ldb + j];
        acc += a * b;
      }
      float v = alpha * acc;
      if (add_c) v += beta * C[i * ldc + j];
      C[i * ldc + j] = v;
    }
  }
}

#endif  // SNAPJUDGE_NO_BLAS
