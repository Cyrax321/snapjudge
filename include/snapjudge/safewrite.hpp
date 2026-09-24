#pragma once
// snapjudge safewrite.hpp: safetensors writer (fp16 storage weights,
// fp32 for the temperature row — the same dtype mix shipped checkpoints use).

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace snapjudge {

struct TensorOut {
  std::string name;
  std::vector<int64_t> shape;
  std::shared_ptr<float[]> data;    // fp32 source
  bool as_f16 = true;               // false -> stored F32
};

// Write <path> as a valid safetensors file; header JSON then raw data.
void save_safetensors(const std::string& path, const std::vector<TensorOut>& tensors);

}  // namespace snapjudge
