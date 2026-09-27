#pragma once
// snapjudge safetensors.hpp: mmap-based safetensors reader.
// Converts fp16/bf16/fp32 to fp32 at materialization; int64/int32 kept as int64.

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace snapjudge {

class SafeTensors {
 public:
  static SafeTensors load(const std::string& path);   // throws std::runtime_error

  bool has(const std::string& name) const;
  const std::vector<std::string>& names() const { return names_; }
  std::vector<int64_t> shape_of(const std::string& name) const;

  // Materialize as fp32, converting F16/BF16/F32 -> F32. Throws on int dtypes.
  std::shared_ptr<float[]> data_f32(const std::string& name) const;
  int64_t numel_of(const std::string& name) const;

 private:
  struct Entry {
    std::string dtype;
    std::vector<int64_t> shape;
    size_t begin = 0, end = 0;   // absolute offsets into the file buffer
    int64_t numel = 0;
  };
  std::shared_ptr<uint8_t[]> mmap_buf_;
  size_t mmap_size_ = 0;
  std::unordered_map<std::string, Entry> entries_;
  std::vector<std::string> names_;
};

}  // namespace snapjudge
