#include "snapjudge/safewrite.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace snapjudge {

static uint16_t f32_to_f16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000;
  uint32_t exp = (x >> 23) & 0xFF;
  uint32_t frac = x & 0x7FFFFF;
  uint16_t out;
  if (exp == 255) {
    out = static_cast<uint16_t>(sign | 0x7C00 | (frac ? 0x200 : 0));
  } else if (exp == 0) {
    out = static_cast<uint16_t>(sign);  // flush denormals to zero (safe for weights)
  } else if (exp > 142) {
    out = static_cast<uint16_t>(sign | 0x7C00);  // inf
  } else if (exp < 113) {
    // subnormal f16
    if (exp < 103) {
      out = static_cast<uint16_t>(sign);
    } else {
      uint32_t f32 = frac | 0x800000;
      int shift = 113 - static_cast<int>(exp) + 1;
      uint16_t f16 = static_cast<uint16_t>(f32 >> (shift + 12));
      if ((f32 >> (shift + 11)) & 1) f16++;
      out = static_cast<uint16_t>(sign | f16);
    }
  } else {
    uint16_t f16 = static_cast<uint16_t>(sign | ((exp - 112) << 10) | (frac >> 13));
    if (frac & 0x1000) f16++;  // round-to-nearest-even-ish (round up)
    out = f16;
  }
  return out;
}

void save_safetensors(const std::string& path, const std::vector<TensorOut>& tensors) {
  nlohmann::json hdr = nlohmann::json::object();
  std::vector<std::pair<size_t, size_t>> spans;
  size_t off = 0;
  for (const auto& t : tensors) {
    int64_t n = 1;
    for (int64_t d : t.shape) n *= d;
    size_t bytes = static_cast<size_t>(n) * (t.as_f16 ? 2 : 4);
    hdr[t.name] = {{"dtype", t.as_f16 ? "F16" : "F32"},
                   {"shape", t.shape},
                   {"data_offsets", {off, off + bytes}}};
    spans.emplace_back(off, off + bytes);
    off += bytes;
  }
  std::string h = hdr.dump();
  // pad header to 8-byte alignment
  while ((8 + h.size()) % 8) h.push_back(' ');
  uint64_t hlen = h.size();

  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("snapjudge: cannot write " + path);
  f.write(reinterpret_cast<const char*>(&hlen), 8);
  f.write(h.data(), static_cast<std::streamsize>(h.size()));
  for (const auto& t : tensors) {
    int64_t n = 1;
    for (int64_t d : t.shape) n *= d;
    if (t.as_f16) {
      std::vector<uint16_t> buf(static_cast<size_t>(n));
      for (int64_t i = 0; i < n; ++i)
        buf[static_cast<size_t>(i)] = f32_to_f16(t.data[static_cast<size_t>(i)]);
      f.write(reinterpret_cast<const char*>(buf.data()),
              static_cast<std::streamsize>(n * 2));
    } else {
      f.write(reinterpret_cast<const char*>(t.data.get()),
              static_cast<std::streamsize>(n * 4));
    }
  }
}

}  // namespace snapjudge
