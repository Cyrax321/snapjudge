#pragma once
// snapjudge fast.hpp: CUDA fast path (port of fast engine).
// Available only in SNAPJUDGE_CUDA builds on a CUDA host.

#include <cstdint>
#include <memory>
#include <vector>

namespace snapjudge {

class DecisionModel;

// Mirrors fast.py: bf16 resident weights, fused kernels, CUDA graphs per
// shape bucket. Numerics match the stock bf16 autocast path (benchmarked by
// benchmarks/bench_fast.py on GPU).
class FastSnapjudge {
 public:
  FastSnapjudge(DecisionModel* model, int max_len, bool use_graphs = true);
  ~FastSnapjudge();

  void forward(const std::vector<std::vector<int64_t>>& input_ids,
               const std::vector<std::vector<int64_t>>& attention_mask,
               const std::vector<std::vector<int64_t>>& marker_pos,
               const std::vector<std::vector<uint8_t>>& marker_mask,
               const std::vector<int64_t>& qtype,
               std::vector<std::vector<float>>& logits,
               std::vector<std::vector<float>>& act);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace snapjudge
