// Deterministic K-quant test inputs and the hash that pins ggml's outputs on
// them (tests/kernels/kquant_test.cpp). Only a hand-written xorshift is used:
// the standard library's distributions differ between MSVC and libc++, and
// these inputs must be the same bytes on every compiler, because the expected
// hashes were computed once by ggml's own reference code (ggml-quants.c at
// the pinned llama.cpp commit).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace golden {

inline constexpr int kBlocks = 16;  // Q4_K and Q6_K blocks
inline constexpr int kRows8 = 32;   // Q8_K blocks of 256 activations

struct Xorshift {
  std::uint32_t s = 2463534242u;
  std::uint32_t next() noexcept {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
  }
};

inline std::uint64_t fnv(const void* p, std::size_t n, std::uint64_t h = 14695981039346656037ull) noexcept {
  const auto* b = static_cast<const std::uint8_t*>(p);
  for (std::size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
  return h;
}

// fp16 scales of every kind but infinity and NaN: normal, subnormal,
// negative, and exactly 1.
inline std::uint16_t scale_bits(Xorshift& r, int i) noexcept {
  const std::uint32_t v = r.next();
  switch (i % 4) {
    case 0:
      return static_cast<std::uint16_t>(0x1000 + (v & 0x0FFF));
    case 1:
      return static_cast<std::uint16_t>(v & 0x03FF);
    case 2:
      return static_cast<std::uint16_t>((0x1400 + (v & 0x0FFF)) | 0x8000);
    default:
      return 0x3C00;
  }
}

// kBlocks Q4_K blocks: random scale/min and nibble bytes, scales as above.
inline std::vector<std::uint8_t> q4_k_blocks() {
  Xorshift r;
  std::vector<std::uint8_t> out(kBlocks * 144);
  for (int i = 0; i < kBlocks; ++i) {
    std::uint8_t* b = out.data() + i * 144;
    for (int j = 4; j < 144; ++j) b[j] = static_cast<std::uint8_t>(r.next() >> 24);
    const std::uint16_t d = scale_bits(r, i), dmin = scale_bits(r, i + 1);
    std::memcpy(b, &d, 2);
    std::memcpy(b + 2, &dmin, 2);
  }
  return out;
}

// kBlocks Q6_K blocks: random ql, qh and int8 scales; d as above.
inline std::vector<std::uint8_t> q6_k_blocks() {
  Xorshift r;
  r.s = 88675123u;
  std::vector<std::uint8_t> out(kBlocks * 210);
  for (int i = 0; i < kBlocks; ++i) {
    std::uint8_t* b = out.data() + i * 210;
    for (int j = 0; j < 208; ++j) b[j] = static_cast<std::uint8_t>(r.next() >> 24);
    const std::uint16_t d = scale_bits(r, i);
    std::memcpy(b + 208, &d, 2);
  }
  return out;
}

// kRows8 rows of 256 activations: plain, tiny, large, an all-zero row,
// rounding ties (integers + 0.5 with a maximum of 127, so iscale = -1 or 1),
// and both signs of the largest magnitude, in either order.
inline std::vector<float> q8_k_inputs() {
  Xorshift r;
  r.s = 521288629u;
  std::vector<float> x(kRows8 * 256);
  for (int row = 0; row < kRows8; ++row) {
    float* v = x.data() + row * 256;
    const int kind = row % 8;
    for (int j = 0; j < 256; ++j) {
      const float u = static_cast<float>(static_cast<std::int32_t>(r.next())) * (1.0f / 2147483648.0f);  // [-1, 1)
      switch (kind) {
        case 1:
          v[j] = u * 1e-3f;
          break;
        case 2:
          v[j] = u * 300.0f;
          break;
        case 3:
          v[j] = 0.0f;
          break;
        case 4:
          v[j] = static_cast<float>(static_cast<int>(u * 100.0f)) + 0.5f;
          break;
        default:
          v[j] = u * 4.0f;
          break;
      }
    }
    if (kind == 4) v[37] = (row % 16 == 4) ? 127.0f : -127.0f;
    if (kind == 5) {
      v[10] = 9.0f;
      v[200] = -9.0f;
    }
    if (kind == 6) {
      v[10] = -9.0f;
      v[200] = 9.0f;
    }
    if (kind == 7) v[255] = -0.0f;
  }
  return x;
}

}  // namespace golden
