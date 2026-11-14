// Simple Q4_K and Q6_K encoders for test models: valid blocks in ggml's
// layout (core/kquant.h), with straightforward scales instead of ggml's
// search for the best ones. Tests only need realistic, decodable weights.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "core/fp16.h"
#include "core/kquant.h"

namespace hearth::test {

// Writes code q (0..15) of value v of a Q4_K block (inverse of quant::q4_k_code).
inline void set_q4_k_code(quant::BlockQ4_K& b, int v, int q) {
  const int chunk = v / 64, in_chunk = v % 64;
  std::uint8_t& byte = b.qs[32 * chunk + in_chunk % 32];
  byte = static_cast<std::uint8_t>(in_chunk < 32 ? (byte & 0xF0) | (q & 0x0F) : (byte & 0x0F) | ((q & 0x0F) << 4));
}

// Writes the 6-bit scale and min of sub-block j (inverse of quant::q4_k_scale_min).
inline void set_q4_k_scale_min(quant::BlockQ4_K& b, int j, int sc, int m) {
  std::uint8_t* q = b.scales;
  if (j < 4) {
    q[j] = static_cast<std::uint8_t>((q[j] & 0xC0) | (sc & 63));
    q[j + 4] = static_cast<std::uint8_t>((q[j + 4] & 0xC0) | (m & 63));
  } else {
    q[j + 4] = static_cast<std::uint8_t>((sc & 0x0F) | ((m & 0x0F) << 4));
    q[j - 4] = static_cast<std::uint8_t>((q[j - 4] & 0x3F) | ((sc >> 4) << 6));
    q[j] = static_cast<std::uint8_t>((q[j] & 0x3F) | ((m >> 4) << 6));
  }
}

// Writes code q (0..63) of value v of a Q6_K block (inverse of quant::q6_k_code).
inline void set_q6_k_code(quant::BlockQ6_K& b, int v, int q) {
  const int half = v / 128, qq = (v % 128) / 32, l = v % 32;
  std::uint8_t& lo = b.ql[64 * half + l + 32 * (qq & 1)];
  lo = static_cast<std::uint8_t>(qq < 2 ? (lo & 0xF0) | (q & 0x0F) : (lo & 0x0F) | ((q & 0x0F) << 4));
  std::uint8_t& hi = b.qh[32 * half + l];
  hi = static_cast<std::uint8_t>((hi & ~(3 << (2 * qq))) | (((q >> 4) & 3) << (2 * qq)));
}

// 256 values per block: value = (d * sc_j) * q - (dmin * m_j), sub-blocks of 32.
inline void encode_q4_k(const float* x, quant::BlockQ4_K& b) {
  b = {};
  float a[8], lo[8];
  for (int j = 0; j < 8; ++j) {
    const float* v = x + 32 * j;
    const float mn = std::min(0.0f, *std::min_element(v, v + 32));
    const float mx = *std::max_element(v, v + 32);
    a[j] = (mx - mn) / 15.0f;
    lo[j] = -mn;
  }
  const float amax = *std::max_element(a, a + 8), lmax = *std::max_element(lo, lo + 8);
  const float d = fp16_to_fp32(fp32_to_fp16(amax / 63.0f)), dmin = fp16_to_fp32(fp32_to_fp16(lmax / 63.0f));
  b.d = fp32_to_fp16(d);
  b.dmin = fp32_to_fp16(dmin);
  for (int j = 0; j < 8; ++j) {
    const int sc = d > 0.0f ? std::clamp(static_cast<int>(std::lround(a[j] / d)), 1, 63) : 0;
    const int m = dmin > 0.0f ? std::clamp(static_cast<int>(std::lround(lo[j] / dmin)), 0, 63) : 0;
    set_q4_k_scale_min(b, j, sc, m);
    const float step = d * static_cast<float>(sc), off = dmin * static_cast<float>(m);
    for (int i = 0; i < 32; ++i) {
      const int q = step > 0.0f ? std::clamp(static_cast<int>(std::lround((x[32 * j + i] + off) / step)), 0, 15) : 0;
      set_q4_k_code(b, 32 * j + i, q);
    }
  }
}

// 256 values per block: value = (d * sc_j) * (q - 32), sub-blocks of 16.
inline void encode_q6_k(const float* x, quant::BlockQ6_K& b) {
  b = {};
  float s[16];
  for (int j = 0; j < 16; ++j) {
    float amax = 0.0f;
    for (int i = 0; i < 16; ++i) amax = std::max(amax, std::fabs(x[16 * j + i]));
    s[j] = amax / 31.0f;
  }
  const float smax = *std::max_element(s, s + 16);
  const float d = fp16_to_fp32(fp32_to_fp16(smax / 127.0f));
  b.d = fp32_to_fp16(d);
  for (int j = 0; j < 16; ++j) {
    const int sc = d > 0.0f ? std::clamp(static_cast<int>(std::lround(s[j] / d)), 1, 127) : 0;
    b.scales[j] = static_cast<std::int8_t>(sc);
    const float step = d * static_cast<float>(sc);
    for (int i = 0; i < 16; ++i) {
      const int q = step > 0.0f ? std::clamp(static_cast<int>(std::lround(x[16 * j + i] / step)) + 32, 0, 63) : 32;
      set_q6_k_code(b, 16 * j + i, q);
    }
  }
}

}  // namespace hearth::test
