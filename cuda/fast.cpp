// snapjudge/cuda/fast.cpp — host glue for the CUDA fast path.
// Port of fast engine: weight staging (fp16 embeddings, bf16 GEMM
// weights, fp32 norms/biases — the same mixed residency as fast.py), RoPE
// table upload, encoder via fused kernels on a padded [B,L] bucket, CUDA
// graphs per (B,L) bucket, fp32 scorer/act-head tail.
//
// Numerics contract: matches the stock bf16-autocast GPU path (verified by
// benchmarks/bench_fast.py on snapjudge; the snapjudge equivalent parity check lives
// in bench/parity_fast.py).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

#include "snapjudge/fast.hpp"
#include "snapjudge/model.hpp"

extern "C" {
void sj_gemm(const __nv_bfloat16*, const __nv_bfloat16*, const float*,
             __nv_bfloat16*, int, int, int, bool, int, cudaStream_t);
void sj_gemm_geglu(const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int,
                   int, int, cudaStream_t);
void sj_add_ln(float*, const __nv_bfloat16*, const float*, const float*,
               __nv_bfloat16*, int, int, float, bool, bool, cudaStream_t);
void sj_rope(__nv_bfloat16*, const float*, const float*, int64_t, int, int, int,
             cudaStream_t);
void sj_attn(const __nv_bfloat16*, const int*, __nv_bfloat16*, int, int, int, int,
             int, cudaStream_t);
}

namespace snapjudge {
namespace cuda_kernels {
// Glue kernels defined in kernels.cu. Declared at namespace scope (not inside
// functions) so CUDA's namespace mangling matches on both sides.
__global__ void emb_gather_kernel(const __half*, const int64_t*, __nv_bfloat16*,
                                  int64_t, int64_t);
__global__ void emb_ln_kernel(const __nv_bfloat16*, const float*, float*, int64_t,
                              int64_t, float);
__global__ void f32_to_bf16_kernel(const float*, __nv_bfloat16*, int64_t);
__global__ void head_entry_kernel(const __nv_bfloat16*, const __nv_bfloat16*,
                                  const int64_t*, float*, int64_t, int64_t, int64_t);
__global__ void add_bf16_to_f32_kernel(float*, const __nv_bfloat16*, int64_t);
}  // namespace cuda_kernels

using bf16 = __nv_bfloat16;

namespace {

void cu_check(cudaError_t rc, const char* what) {
  if (rc != cudaSuccess)
    throw std::runtime_error(std::string("snapjudge CUDA: ") + what + ": " +
                             cudaGetErrorString(rc));
}

// ---- device buffers -----------------------------------------------------------
struct DevBuf {
  void* p = nullptr;
  size_t bytes = 0;
  DevBuf() = default;
  explicit DevBuf(size_t b) : bytes(b) { cu_check(cudaMalloc(&p, b), "cudaMalloc"); }
  ~DevBuf() { if (p) cudaFree(p); }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  DevBuf(DevBuf&& o) noexcept : p(o.p), bytes(o.bytes) { o.p = nullptr; o.bytes = 0; }
  DevBuf& operator=(DevBuf&& o) noexcept {
    if (p) cudaFree(p);
    p = o.p;
    bytes = o.bytes;
    o.p = nullptr;
    return *this;
  }
};

// fp32 host -> bf16 device (fast.py b16())
DevBuf to_bf16(const float* src, int64_t n) {
  std::vector<bf16> tmp(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) tmp[static_cast<size_t>(i)] = __float2bfloat16(src[i]);
  DevBuf out(static_cast<size_t>(n) * sizeof(bf16));
  cu_check(cudaMemcpy(out.p, tmp.data(), out.bytes, cudaMemcpyHostToDevice),
           "memcpy bf16");
  return out;
}

// exact fp16 round-trip of checkpoint values then embed-gather fp32-upcast —
// fast.py: emb_w = weight.to(torch.float16).contiguous(), gather upcasts fp32
DevBuf to_fp16(const float* src, int64_t n) {
  std::vector<__half> tmp(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) tmp[static_cast<size_t>(i)] = __float2half(src[i]);
  DevBuf out(static_cast<size_t>(n) * sizeof(__half));
  cu_check(cudaMemcpy(out.p, tmp.data(), out.bytes, cudaMemcpyHostToDevice),
           "memcpy fp16");
  return out;
}

DevBuf to_f32(const float* src, int64_t n) {
  DevBuf out(static_cast<size_t>(n) * sizeof(float));
  cu_check(cudaMemcpy(out.p, src, out.bytes, cudaMemcpyHostToDevice), "memcpy f32");
  return out;
}

struct LayerDev {
  DevBuf attn_ln;          // fp32 [D] or empty (layer 0: no attn_norm)
  DevBuf wqkv, wo, wi, wo2; // bf16
  DevBuf mlp_ln;           // fp32 [D]
  int window = 0;
  int ltype_slide = 0;
};

struct HeadDev {
  DevBuf n1w, n1b, n2w, n2b;        // fp32
  DevBuf in_w, out_w, l1w, l2w;     // bf16
  DevBuf in_b, out_b, l1b, l2b;     // fp32
};

struct GraphEntry {
  cudaGraph_t g = nullptr;
  cudaGraphExec_t exec = nullptr;
  ~GraphEntry() {
    if (exec) cudaGraphExecDestroy(exec);
    if (g) cudaGraphDestroy(g);
  }
};

int next_pow2(int n) {
  int b = 1;
  while (b < n) b <<= 1;
  return b;
}

}  // namespace

struct FastSnapjudge::Impl {
  DecisionModel* m = nullptr;
  int D = 0, H = 0, Dh = 0, F = 0, nlayers = 0, nact = 0;
  int max_len = 512;
  float eps = 1e-5f;
  int win = 0;
  bool use_graphs = true;

  DevBuf emb_w;        // fp16 [V, D]
  DevBuf emb_ln;       // fp32 [D]
  DevBuf final_ln;     // fp32
  DevBuf type_emb;     // bf16 [3, D]
  DevBuf rope_cos_full, rope_sin_full, rope_cos_slide, rope_sin_slide;  // fp32
  std::vector<LayerDev> layers;
  std::vector<HeadDev> heads;
  // fp32 scorer / act weights (tiny; kept on device for a device-side tail)
  DevBuf sc_ln, sc_w1, sc_b1, sc_w2, sc_b2;
  DevBuf act_w1, act_b1, act_w2, act_b2;

  // graph-captured per-shape state
  struct ShapeState {
    int B = 0, L = 0;
    DevBuf ids, lens, qt;
    DevBuf qkv, O, G, X, Y, F1, hcrop, pooled_feats;
    GraphEntry entry;
  };
  std::map<std::pair<int, int>, std::unique_ptr<ShapeState>> graphs;
};

// ------------------------------------------------------------------ encoder ----
namespace {

// fast.py::_encode body: ids [B,L] int64, lens [B] int32, qt [B] int64.
// Scratch layout in ShapeState: qkv, O, G bf16; X fp32; Y bf16; F1 bf16.
void encode_impl(FastSnapjudge::Impl::ShapeState& ss, const FastSnapjudge::Impl& im,
                 cudaStream_t st) {
  const int64_t B = ss.B, L = ss.L, D = im.D, F = im.F, M = B * L;
  bf16* qkv = static_cast<bf16*>(ss.qkv.p);
  bf16* O = static_cast<bf16*>(ss.O.p);
  bf16* G = static_cast<bf16*>(ss.G.p);
  float* X = static_cast<float*>(ss.X.p);
  bf16* Y = static_cast<bf16*>(ss.Y.p);
  bf16* F1 = static_cast<bf16*>(ss.F1.p);
  // emb gather + emb norm run on host side for the pad bucket (id table is
  // device-resident; we do the embedding expand in CUDA with a tiny kernel
  // written here so the whole path stays graph-capturable.
  // --- embeddings: Y_emb[bf16] = emb_w[ids]; X = layer_norm(Y_emb fp32)
  cuda_kernels::emb_gather_kernel<<<(int)((M + 255) / 256), 256, 0, st>>>(
      static_cast<const __half*>(im.emb_w.p),
      static_cast<const int64_t*>(ss.ids.p), Y, M, D);
  cuda_kernels::emb_ln_kernel<<<(int)((M + 3) / 4), 64, 4 * D * sizeof(float), st>>>(
      Y, static_cast<const float*>(im.emb_ln.p), X, M, D, im.eps);
  // Y (bf16) = X (layer 0 attends to X directly: attn_norm is Identity)
  cuda_kernels::f32_to_bf16_kernel<<<(int)((M * D + 255) / 256), 256, 0, st>>>(X, Y, M * D);

  const float* cos_full = static_cast<const float*>(im.rope_cos_full.p);
  const float* sin_full = static_cast<const float*>(im.rope_sin_full.p);
  const float* cos_slide = static_cast<const float*>(im.rope_cos_slide.p);
  const float* sin_slide = static_cast<const float*>(im.rope_sin_slide.p);
  const int* lens = static_cast<const int*>(ss.lens.p);

  for (int i = 0; i < im.nlayers; ++i) {
    const LayerDev& ly = im.layers[i];
    // Y = attn_norm(X) — layer 0: Y already = bf16(X); else add_ln without residual
    if (i > 0)
      sj_add_ln(X, nullptr, static_cast<const float*>(ly.attn_ln.p), nullptr, Y,
                static_cast<int>(M), im.D, im.eps, false, false, st);
    // qkv = Y @ wqkv^T
    sj_gemm(Y, static_cast<const bf16*>(ly.wqkv.p), nullptr, qkv,
            static_cast<int>(M), 3 * im.D, im.D, false, 0, st);
    // rope
    const float* ct = ly.ltype_slide ? cos_slide : cos_full;
    const float* stab = ly.ltype_slide ? sin_slide : sin_full;
    sj_rope(qkv, ct, stab, M, static_cast<int>(L), im.H, im.Dh, st);
    // attention (padding mask via lens + sliding window)
    sj_attn(qkv, lens, O, static_cast<int>(B), static_cast<int>(L), im.H, im.Dh,
            ly.window, st);
    // Y = O @ wo^T
    sj_gemm(O, static_cast<const bf16*>(ly.wo.p), nullptr, Y, static_cast<int>(M),
            im.D, im.D, false, 0, st);
    // X += Y ; Y = mlp_norm(X)
    sj_add_ln(X, Y, static_cast<const float*>(ly.mlp_ln.p), nullptr, Y,
              static_cast<int>(M), im.D, im.eps, true, false, st);
    // G = geglu(Y @ wi^T) ; Y = G @ wo2^T
    sj_gemm_geglu(Y, static_cast<const bf16*>(ly.wi.p), G, static_cast<int>(M), im.F,
                  im.D, st);
    sj_gemm(G, static_cast<const bf16*>(ly.wo2.p), nullptr, Y, static_cast<int>(M),
            im.D, im.F, false, 0, st);
    // X += Y ; Y = next norm(X)
    if (i + 1 < im.nlayers) {
      const LayerDev& nxt = im.layers[i + 1];
      if (nxt.attn_ln.p)
        sj_add_ln(X, Y, static_cast<const float*>(nxt.attn_ln.p), nullptr, Y,
                  static_cast<int>(M), im.D, im.eps, true, false, st);
      else {
        // next is layer 0 attn_norm (never hits at i+1>0 in practice, but the
        // branch mirrors fast.py): Y = bf16(X) after the residual add
        sj_add_ln(X, Y, nullptr, nullptr, Y, static_cast<int>(M), im.D, true, false,
                  st);
      }
    } else {
      sj_add_ln(X, Y, static_cast<const float*>(im.final_ln.p), nullptr, Y,
                static_cast<int>(M), im.D, im.eps, true, false, st);
    }
  }
  // ---- decision head: X = Y.float() + type_emb[qt] (fp32 stream) ----
  cuda_kernels::head_entry_kernel<<<(int)((M + 255) / 256), 256, 0, st>>>(
      Y, static_cast<const bf16*>(im.type_emb.p),
      static_cast<const int64_t*>(ss.qt.p), X, M, D);
  for (const HeadDev& hd : im.heads) {
    // ln1: Y = LN(X)*w+b -> bf16 for GEMM input
    sj_add_ln(X, nullptr, static_cast<const float*>(hd.n1w.p),
              static_cast<const float*>(hd.n1b.p), Y, static_cast<int>(M), im.D,
              1e-5f, false, true, st);
    sj_gemm(Y, static_cast<const bf16*>(hd.in_w.p), static_cast<const float*>(hd.in_b.p),
            qkv, static_cast<int>(M), 3 * im.D, im.D, true, 0, st);
    sj_attn(qkv, lens, O, static_cast<int>(B), static_cast<int>(L), im.H, im.Dh, 0, st);
    sj_gemm(O, static_cast<const bf16*>(hd.out_w.p),
            static_cast<const float*>(hd.out_b.p), Y, static_cast<int>(M), im.D, im.D,
            true, 0, st);
    sj_add_ln(X, Y, static_cast<const float*>(hd.n2w.p),
              static_cast<const float*>(hd.n2b.p), Y, static_cast<int>(M), im.D, 1e-5f,
              true, true, st);
    sj_gemm(Y, static_cast<const bf16*>(hd.l1w.p), static_cast<const float*>(hd.l1b.p),
            F1, static_cast<int>(M), 4 * im.D, im.D, true, 2, st);
    sj_gemm(F1, static_cast<const bf16*>(hd.l2w.p),
            static_cast<const float*>(hd.l2b.p), Y, static_cast<int>(M), im.D,
            4 * im.D, true, 0, st);
    // X += Y (fp32 residual)
    cuda_kernels::add_bf16_to_f32_kernel<<<(int)((M * D + 255) / 256), 256, 0, st>>>(X, Y, M * D);
  }
}

}  // namespace

// ------------------------------------------------------------------ ctor --------
FastSnapjudge::FastSnapjudge(DecisionModel* model, int max_len, bool use_graphs)
    : impl_(std::make_unique<Impl>()) {
  Impl& im = *impl_;
  im.m = model;
  im.max_len = max_len;
  im.use_graphs = use_graphs;

  const ModelConfig& cfg = model->cfg();
  im.D = cfg.hidden_size;
  im.H = cfg.num_attention_heads;
  im.Dh = im.D / im.H;
  im.F = cfg.intermediate_size;
  im.nlayers = cfg.num_hidden_layers;
  im.nact = cfg.n_act;
  im.eps = cfg.norm_eps;
  im.win = cfg.local_attention > 0 ? cfg.local_attention / 2 : 0;

  const int64_t D = im.D;
  // embeddings: exact fp16 residency, like fast.py
  {
    auto shape_v = std::vector<int64_t>{cfg.vocab_size, D};
    int64_t n = cfg.vocab_size * D;
    im.emb_w = to_fp16(model->emb_w_.get(), n);
  }
  im.emb_ln = to_f32(model->emb_norm_.get(), D);
  im.final_ln = to_f32(model->final_norm_.get(), D);
  im.type_emb = to_bf16(model->type_emb_.get(), 3 * D);

  // rope tables: recomputed on device side from cfg thetas — but simpler and
  // exact: upload the fp64-computed tables the CPU path already built.
  {
    int64_t L = cfg.max_position_embeddings, half = im.Dh / 2;
    im.rope_cos_full = to_f32(model->rope_cos_full_.get(), L * half);
    im.rope_sin_full = to_f32(model->rope_sin_full_.get(), L * half);
    im.rope_cos_slide = to_f32(model->rope_cos_slide_.get(), L * half);
    im.rope_sin_slide = to_f32(model->rope_sin_slide_.get(), L * half);
  }

  im.layers.resize(static_cast<size_t>(im.nlayers));
  for (size_t i = 0; i < im.layers.size(); ++i) {
    const LayerWeights& lw = model->layers_[i];
    LayerDev& d = im.layers[i];
    if (lw.has_attn_norm) d.attn_ln = to_f32(lw.attn_norm.d.get(), D);
    d.wqkv = to_bf16(lw.wqkv.d.get(), 3 * D * D);
    d.wo = to_bf16(lw.wo.d.get(), D * D);
    d.mlp_ln = to_f32(lw.mlp_norm.d.get(), D);
    d.wi = to_bf16(lw.wi.d.get(), 2 * im.F * D);
    d.wo2 = to_bf16(lw.wo2.d.get(), D * im.F);
    d.window = lw.window;
    d.ltype_slide = lw.attention_type == "sliding_attention" ? 1 : 0;
  }

  im.heads.resize(model->head_.size());
  for (size_t i = 0; i < im.heads.size(); ++i) {
    const DecisionModel::HeadLayer& hw = model->head_[i];
    HeadDev& d = im.heads[i];
    d.n1w = to_f32(hw.n1w.get(), D);
    d.n1b = to_f32(hw.n1b.get(), D);
    d.n2w = to_f32(hw.n2w.get(), D);
    d.n2b = to_f32(hw.n2b.get(), D);
    d.in_w = to_bf16(hw.in_w.get(), 3 * D * D);
    d.in_b = to_f32(hw.in_b.get(), 3 * D);
    d.out_w = to_bf16(hw.out_w.get(), D * D);
    d.out_b = to_f32(hw.out_b.get(), D);
    d.l1w = to_bf16(hw.l1w.get(), 4 * D * D);
    d.l1b = to_f32(hw.l1b.get(), 4 * D);
    d.l2w = to_bf16(hw.l2w.get(), D * 4 * D);
    d.l2b = to_f32(hw.l2b.get(), D);
  }

  im.sc_ln = to_f32(model->scorer_ln_w_.get(), D);
  if (model->scorer_ln_b_) im.sc_b1 = to_f32(model->scorer_ln_b_.get(), D);
  im.sc_w1 = to_f32(model->scorer_w1_.get(), D * D);
  if (model->scorer_b1_) im.sc_b1 = to_f32(model->scorer_b1_.get(), D);
  im.sc_w2 = to_f32(model->scorer_w2_.get(), D);
  if (model->scorer_b2_) im.sc_b2 = to_f32(model->scorer_b2_.get(), 1);
  im.act_w1 = to_f32(model->act_w1_.get(), 256 * (D + 4));
  im.act_b1 = to_f32(model->act_b1_.get(), 256);
  im.act_w2 = to_f32(model->act_w2_.get(), im.nact * 256);
  im.act_b2 = to_f32(model->act_b2_.get(), im.nact);
}

FastSnapjudge::~FastSnapjudge() = default;

// ------------------------------------------------------------------ forward -----
void FastSnapjudge::forward(const std::vector<std::vector<int64_t>>& input_ids,
                            const std::vector<std::vector<int64_t>>& attention_mask,
                            const std::vector<std::vector<int64_t>>& marker_pos,
                            const std::vector<std::vector<uint8_t>>& marker_mask,
                            const std::vector<int64_t>& qtype,
                            std::vector<std::vector<float>>& logits,
                            std::vector<std::vector<float>>& act) {
  Impl& im = *impl_;
  const int N = static_cast<int>(input_ids.size());
  const int L0 = static_cast<int>(input_ids[0].size());
  const int K = static_cast<int>(marker_pos[0].size());
  const int64_t D = im.D;

  // bucket padding, identical to fast.py::forward
  const int DYNAMIC_MAX_L = 256, LONG_BUCKET = 64;
  int g = L0 <= DYNAMIC_MAX_L ? 16 : LONG_BUCKET;
  int L = std::min(im.max_len, ((L0 + g - 1) / g) * g);
  int B = next_pow2(N);

  auto key = std::make_pair(B, L);
  auto& ss = im.graphs[key];
  if (!ss) {
    ss = std::make_unique<Impl::ShapeState>();
    ss->B = B;
    ss->L = L;
    const int64_t M = static_cast<int64_t>(B) * L;
    ss->ids = DevBuf(M * sizeof(int64_t));
    ss->lens = DevBuf(B * sizeof(int));
    ss->qt = DevBuf(B * sizeof(int64_t));
    ss->qkv = DevBuf(M * 3 * D * sizeof(bf16));
    ss->O = DevBuf(M * D * sizeof(bf16));
    ss->G = DevBuf(M * im.F * sizeof(bf16));
    ss->X = DevBuf(M * D * sizeof(float));
    ss->Y = DevBuf(M * D * sizeof(bf16));
    ss->F1 = DevBuf(M * 4 * D * sizeof(bf16));
    ss->hcrop = DevBuf(static_cast<int64_t>(B) * L0 * D * sizeof(float));
  }

  // upload padded inputs
  std::vector<int64_t> ids(static_cast<size_t>(B) * L, 0);
  for (int r = 0; r < N; ++r)
    for (int c = 0; c < L0; ++c)
      ids[static_cast<size_t>(r) * L + c] = input_ids[static_cast<size_t>(r)][c];
  cu_check(cudaMemcpy(ss->ids.p, ids.data(), ids.size() * sizeof(int64_t),
                      cudaMemcpyHostToDevice),
           "ids upload");
  std::vector<int> lens(static_cast<size_t>(B), 0);
  for (int r = 0; r < N; ++r) {
    int s = 0;
    for (int64_t v : attention_mask[static_cast<size_t>(r)]) s += static_cast<int>(v);
    lens[static_cast<size_t>(r)] = s;
  }
  cu_check(cudaMemcpy(ss->lens.p, lens.data(), lens.size() * sizeof(int),
                      cudaMemcpyHostToDevice),
           "lens upload");
  std::vector<int64_t> qt(static_cast<size_t>(B), 0);
  for (int r = 0; r < N; ++r) qt[static_cast<size_t>(r)] = qtype[static_cast<size_t>(r)];
  cu_check(cudaMemcpy(ss->qt.p, qt.data(), qt.size() * sizeof(int64_t),
                      cudaMemcpyHostToDevice),
           "qt upload");

  if (im.use_graphs) {
    if (!ss->entry.exec) {
      // capture: run once on a side stream, then capture.
      cudaStream_t s;
      cu_check(cudaStreamCreate(&s), "stream create");
      cudaGraph_t graph;
      encode_impl(*ss, im, s);  // warm-up (also for capture legality)
      cu_check(cudaStreamSynchronize(s), "warmup sync");
      cu_check(cudaGraphCreate(&graph, 0), "graph create");
      cu_check(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal),
               "begin capture");
      encode_impl(*ss, im, s);
      cu_check(cudaStreamEndCapture(s, &graph), "end capture");
      cu_check(cudaGraphInstantiate(&ss->entry.exec, graph, nullptr, nullptr, 0),
               "instantiate");
      ss->entry.g = graph;
      cu_check(cudaStreamSynchronize(s), "capture sync");
      cu_check(cudaStreamDestroy(s), "stream destroy");
    }
    cu_check(cudaGraphLaunch(ss->entry.exec, 0), "graph launch");
    cu_check(cudaDeviceSynchronize(), "graph sync");
  } else {
    encode_impl(*ss, im, 0);
    cu_check(cudaDeviceSynchronize(), "encode sync");
  }

  // crop h [:N, :L0] — X already fp32 [M, D] on device
  std::vector<float> h(static_cast<size_t>(N) * L0 * D);
  const float* X = static_cast<const float*>(ss->X.p);
  std::vector<float> full(static_cast<size_t>(B) * ss->L * D);
  cu_check(cudaMemcpy(full.data(), X, full.size() * sizeof(float),
                      cudaMemcpyDeviceToHost),
           "h download");
  for (int r = 0; r < N; ++r)
    for (int64_t c = 0; c < L0; ++c)
      std::memcpy(h.data() + (static_cast<size_t>(r) * L0 + c) * D,
                  full.data() + (static_cast<size_t>(r) * ss->L + c) * D,
                  D * sizeof(float));

  // ---- scorer / act head tail, fp32 host math (identical ops to the model
  // forward — these are a few hundred FLOPs per row, faster on host than the
  // download/upload round trip)
  logits.assign(static_cast<size_t>(N), std::vector<float>(static_cast<size_t>(K)));
  act.assign(static_cast<size_t>(N), std::vector<float>(static_cast<size_t>(im.nact)));
  std::vector<float> sln(static_cast<size_t>(D));
  {
    const float* lnw = static_cast<const float*>(im.sc_ln.p);
    std::vector<float> host(static_cast<size_t>(D));
    cu_check(cudaMemcpy(host.data(), lnw, D * sizeof(float), cudaMemcpyDeviceToHost),
             "scorer ln download");
    // (Scorer weights are small; all scorer math runs host-side.)
  }
  // NOTE: scorer/act tails are computed exactly as in model.cpp on host —
  // duplicated here, operating on the cropped [N, L0, D] hidden states.
  {
    auto ln = [&](const float* xr, int64_t DD, const std::shared_ptr<float[]>& w,
                  const std::shared_ptr<float[]>& b, float eps,
                  std::vector<float>& out) {
      out.resize(static_cast<size_t>(DD));
      double mean = 0;
      for (int64_t j = 0; j < DD; ++j) mean += xr[j];
      mean /= DD;
      double var = 0;
      for (int64_t j = 0; j < DD; ++j) {
        double d = xr[j] - mean;
        var += d * d;
      }
      var /= DD;
      float inv = static_cast<float>(1.0 / std::sqrt(var + eps));
      for (int64_t j = 0; j < DD; ++j) {
        float v = (xr[j] - static_cast<float>(mean)) * inv * (w ? w[j] : 1.0f);
        out[static_cast<size_t>(j)] = b ? v + b[j] : v;
      }
    };
    auto gelu = [](float x) {
      return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654757f));
    };
    DecisionModel& m = *im.m;
    for (int b = 0; b < N; ++b) {
      for (int kk = 0; kk < K; ++kk) {
        int64_t p = marker_pos[static_cast<size_t>(b)][static_cast<size_t>(kk)];
        if (p < 0) p = 0;
        if (p >= L0) p = L0 - 1;
        const float* row = h.data() + (static_cast<size_t>(b) * L0 + p) * D;
        std::vector<float> z;
        ln(row, D, m.scorer_ln_w_, m.scorer_ln_b_, m.cfg().norm_eps, z);
        // two linears
        std::vector<float> u(static_cast<size_t>(D));
        for (int64_t j = 0; j < D; ++j) {
          double s = 0;
          for (int64_t i = 0; i < D; ++i) s += m.scorer_w1_[j * D + i] * z[i];
          u[static_cast<size_t>(j)] = gelu(static_cast<float>(s) +
                                           (m.scorer_b1_ ? m.scorer_b1_[j] : 0.f));
        }
        double s = m.scorer_b2_ ? m.scorer_b2_[0] : 0.0;
        for (int64_t i = 0; i < D; ++i) s += m.scorer_w2_[i] * u[i];
        float lg = static_cast<float>(s);
        if (!marker_mask[static_cast<size_t>(b)][static_cast<size_t>(kk)]) lg = -1e4f;
        logits[static_cast<size_t>(b)][static_cast<size_t>(kk)] = lg;
      }
      // act head features
      const std::vector<float>& lg = logits[static_cast<size_t>(b)];
      float mx = *std::max_element(lg.begin(), lg.end());
      std::vector<double> p(lg.size());
      double s = 0;
      for (size_t j = 0; j < lg.size(); ++j) { p[j] = std::exp(lg[j] - mx); s += p[j]; }
      for (double& v : p) v /= s;
      double top1 = 0, top2v = 0;
      for (double v : p) {
        if (v > top1) { top2v = top1; top1 = v; }
        else if (v > top2v) top2v = v;
      }
      double ksum = 0;
      for (int64_t kk = 0; kk < K; ++kk)
        ksum += marker_mask[static_cast<size_t>(b)][static_cast<size_t>(kk)];
      double kf = std::max<double>(2.0, ksum);
      double ent = 0;
      for (double v : p) ent -= v * std::log(std::max(v, 1e-9));
      ent /= std::log(kf);

      const float* pooled = h.data() + (static_cast<size_t>(b) * L0) * D;
      std::vector<float> feats(static_cast<size_t>(D + 4));
      std::memcpy(feats.data(), pooled, D * sizeof(float));
      feats[D + 0] = static_cast<float>(top1);
      feats[D + 1] = static_cast<float>(top1 - top2v);
      feats[D + 2] = static_cast<float>(ent);
      feats[D + 3] = static_cast<float>(kf / 255.0);
      std::vector<float> u(256);
      for (int j = 0; j < 256; ++j) {
        double acc = 0;
        for (int64_t i = 0; i < D + 4; ++i) acc += m.act_w1_[j * (D + 4) + i] * feats[i];
        u[static_cast<size_t>(j)] =
            gelu(static_cast<float>(acc) + m.act_b1_[j]);
      }
      std::vector<double> lg2(static_cast<size_t>(im.nact));
      for (int j = 0; j < im.nact; ++j) {
        double acc = m.act_b2_[j];
        for (int i = 0; i < 256; ++i) acc += m.act_w2_[j * 256 + i] * u[i];
        lg2[static_cast<size_t>(j)] = acc;
      }
      double amx = *std::max_element(lg2.begin(), lg2.end());
      double as = 0;
      for (int j = 0; j < im.nact; ++j) {
        lg2[static_cast<size_t>(j)] = std::exp(lg2[static_cast<size_t>(j)] - amx);
        as += lg2[static_cast<size_t>(j)];
      }
      for (int j = 0; j < im.nact; ++j)
        act[static_cast<size_t>(b)][static_cast<size_t>(j)] =
            static_cast<float>(lg2[static_cast<size_t>(j)] / as);
    }
  }
}

}  // namespace snapjudge
