#include "snapjudge/safetensors.hpp"

#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "nlohmann/json.hpp"

namespace snapjudge {

using nlohmann::json;

SafeTensors SafeTensors::load(const std::string& path) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("snapjudge: cannot open " + path);
  struct stat st;
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    throw std::runtime_error("snapjudge: cannot stat " + path);
  }
  size_t size = static_cast<size_t>(st.st_size);
  if (size < 8) {
    ::close(fd);
    throw std::runtime_error("snapjudge: not a safetensors file (too small): " + path);
  }
  void* m = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (m == MAP_FAILED) throw std::runtime_error("snapjudge: mmap failed for " + path);

  SafeTensors s;
  s.mmap_size_ = size;
  s.mmap_buf_.reset(static_cast<uint8_t*>(m), [size](uint8_t* p) { ::munmap(p, size); });

  uint64_t hlen;
  std::memcpy(&hlen, m, 8);
  if (8 + hlen > size) throw std::runtime_error("snapjudge: bad header length in " + path);
  json hdr;
  try {
    hdr = json::parse(static_cast<char*>(m) + 8, static_cast<char*>(m) + 8 + hlen);
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("snapjudge: bad safetensors header JSON: ") + e.what());
  }
  size_t base = 8 + hlen;
  for (auto it = hdr.begin(); it != hdr.end(); ++it) {
    if (it.key() == "__metadata__") continue;
    const json& v = it.value();
    Entry e;
    e.dtype = v.value("dtype", std::string());
    for (const auto& d : v["shape"]) e.shape.push_back(d.get<int64_t>());
    e.numel = 1;
    for (int64_t d : e.shape) e.numel *= d;
    e.begin = base + v["data_offsets"][0].get<uint64_t>();
    e.end = base + v["data_offsets"][1].get<uint64_t>();
    if (e.end > size) throw std::runtime_error("snapjudge: tensor out of bounds in " + path);
    s.names_.push_back(it.key());
    s.entries_[it.key()] = std::move(e);
  }
  return s;
}

bool SafeTensors::has(const std::string& name) const { return entries_.count(name) > 0; }

std::vector<int64_t> SafeTensors::shape_of(const std::string& name) const {
  auto it = entries_.find(name);
  if (it == entries_.end()) throw std::runtime_error("snapjudge: no tensor named " + name);
  return it->second.shape;
}

int64_t SafeTensors::numel_of(const std::string& name) const {
  auto it = entries_.find(name);
  if (it == entries_.end()) throw std::runtime_error("snapjudge: no tensor named " + name);
  return it->second.numel;
}

static float f16_to_f32(uint16_t h) {
  uint32_t sign = (h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t frac = h & 0x3FF;
  uint32_t out;
  if (exp == 0) {
    if (frac == 0) { out = sign; }
    else {
      // subnormal
      exp = 1;
      while ((frac & 0x400) == 0) { frac <<= 1; exp--; }
      frac &= 0x3FF;
      out = sign | ((exp + 112) << 23) | (frac << 13);
    }
  } else if (exp == 0x1F) {
    out = sign | 0x7F800000 | (frac << 13);
  } else {
    out = sign | ((exp + 112) << 23) | (frac << 13);
  }
  float f;
  std::memcpy(&f, &out, 4);
  return f;
}

static float bf16_to_f32(uint16_t h) {
  uint32_t out = static_cast<uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &out, 4);
  return f;
}

std::shared_ptr<float[]> SafeTensors::data_f32(const std::string& name) const {
  auto it = entries_.find(name);
  if (it == entries_.end()) throw std::runtime_error("snapjudge: no tensor named " + name);
  const Entry& e = it->second;
  size_t n = static_cast<size_t>(e.numel);
  const uint8_t* src = mmap_buf_.get() + e.begin;
  auto out = std::shared_ptr<float[]>(new float[n]);
  if (e.dtype == "F32") {
    if (e.end - e.begin != n * 4) throw std::runtime_error("snapjudge: size mismatch " + name);
    std::memcpy(out.get(), src, n * 4);
  } else if (e.dtype == "F16") {
    for (size_t i = 0; i < n; ++i) {
      uint16_t h;
      std::memcpy(&h, src + 2 * i, 2);
      out[static_cast<size_t>(i)] = f16_to_f32(h);
    }
  } else if (e.dtype == "BF16") {
    for (size_t i = 0; i < n; ++i) {
      uint16_t h;
      std::memcpy(&h, src + 2 * i, 2);
      out[static_cast<size_t>(i)] = bf16_to_f32(h);
    }
  } else {
    throw std::runtime_error("snapjudge: unsupported dtype " + e.dtype + " for " + name);
  }
  return out;
}

}  // namespace snapjudge
