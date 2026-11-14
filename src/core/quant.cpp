#include "core/quant.h"

#include <cmath>

#include "core/fp16.h"

namespace hearth::quant {
namespace {

// Round half to even, as _mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT) does.
// std::nearbyint uses the current rounding mode, which is round-to-nearest-
// even unless someone changed it; this project never does.
float round_even(float v) { return std::nearbyint(v); }

void quantize_block(const float* x, std::int8_t* q, float& d_out, std::int32_t& s_out, std::uint16_t& d_bits) {
  float amax = 0.0f;
  for (int j = 0; j < kBlock; ++j) amax = std::fmax(amax, std::fabs(x[j]));
  const float d = amax / 127.0f;
  d_bits = fp32_to_fp16(d);
  d_out = fp16_to_fp32(d_bits);
  const float m = amax != 0.0f ? 127.0f / amax : 0.0f;
  std::int32_t s = 0;
  for (int j = 0; j < kBlock; ++j) {
    const auto v = static_cast<std::int32_t>(round_even(x[j] * m));
    q[j] = static_cast<std::int8_t>(v);
    s += v;
  }
  s_out = s;
}

}  // namespace

void quantize_act_q8_0(const float* x, std::int64_t k, std::int8_t* q, float* d, std::int32_t* s) {
  for (std::int64_t b = 0; b < k / kBlock; ++b) {
    std::uint16_t bits = 0;
    quantize_block(x + b * kBlock, q + b * kBlock, d[b], s[b], bits);
  }
}

void quantize_row_q8_0(const float* x, std::int64_t k, BlockQ8_0* out) {
  for (std::int64_t b = 0; b < k / kBlock; ++b) {
    float d = 0;
    std::int32_t s = 0;
    quantize_block(x + b * kBlock, out[b].q, d, s, out[b].d);
  }
}

std::int32_t block_dot_q8_0(const BlockQ8_0& w, const std::int8_t* xq) {
  std::int32_t s = 0;
  for (int j = 0; j < kBlock; ++j) s += static_cast<std::int32_t>(w.q[j]) * xq[j];
  return s;
}

std::int32_t block_dot_q4_0(const BlockQ4_0& w, const std::int8_t* xq) {
  std::int32_t s = 0;
  for (int j = 0; j < kBlock / 2; ++j) {
    s += (static_cast<std::int32_t>(w.q[j] & 0x0F) - 8) * xq[j];
    s += (static_cast<std::int32_t>(w.q[j] >> 4) - 8) * xq[j + kBlock / 2];
  }
  return s;
}

float dequant(const BlockQ8_0& b, int j) noexcept { return fp16_to_fp32(b.d) * static_cast<float>(b.q[j]); }

float dequant(const BlockQ4_0& b, int j) noexcept {
  const int n = j < kBlock / 2 ? (b.q[j] & 0x0F) : (b.q[j - kBlock / 2] >> 4);
  return fp16_to_fp32(b.d) * static_cast<float>(n - 8);
}

}  // namespace hearth::quant
