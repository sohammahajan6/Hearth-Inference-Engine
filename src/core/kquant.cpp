#include "core/kquant.h"

#include <cmath>
#include <cstring>

#include "core/fp16.h"

namespace hearth::quant {
namespace {

// ggml's nearest_int: adding 1.5 * 2^23 leaves the integer part of v in the
// low mantissa bits, rounded half to even by the add (valid for |v| < 2^22).
int nearest_int(float v) noexcept {
  const float shifted = v + 12582912.0f;
  std::int32_t i = 0;
  std::memcpy(&i, &shifted, sizeof(i));
  return (i & 0x007fffff) - 0x00400000;
}

}  // namespace

void q4_k_scale_min(const BlockQ4_K& b, int j, std::uint8_t& sc, std::uint8_t& m) noexcept {
  const std::uint8_t* q = b.scales;
  if (j < 4) {
    sc = static_cast<std::uint8_t>(q[j] & 63);
    m = static_cast<std::uint8_t>(q[j + 4] & 63);
  } else {
    sc = static_cast<std::uint8_t>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
    m = static_cast<std::uint8_t>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
  }
}

int q4_k_code(const BlockQ4_K& b, int v) noexcept {
  const int chunk = v / 64, in_chunk = v % 64;
  const std::uint8_t byte = b.qs[32 * chunk + in_chunk % 32];
  return in_chunk < 32 ? (byte & 0x0F) : (byte >> 4);
}

int q6_k_code(const BlockQ6_K& b, int v) noexcept {
  // ggml's dequantize_row_q6_K: each half of 128 values uses 64 bytes of ql
  // and 32 of qh; quarter qq of a half takes the low (qq < 2) or high nibble
  // of ql[l + 32 * (qq & 1)] and bits 2qq, 2qq+1 of qh[l], l = value % 32.
  const int half = v / 128, qq = (v % 128) / 32, l = v % 32;
  const std::uint8_t lo = b.ql[64 * half + l + 32 * (qq & 1)];
  const std::uint8_t hi = b.qh[32 * half + l];
  return ((qq < 2 ? lo : lo >> 4) & 0x0F) | (((hi >> (2 * qq)) & 3) << 4);
}

void dequant_q4_k(const BlockQ4_K& b, float* out) noexcept {
  const float d = fp16_to_fp32(b.d);
  const float dmin = fp16_to_fp32(b.dmin);
  for (int j = 0; j < 8; ++j) {
    std::uint8_t sc = 0, m = 0;
    q4_k_scale_min(b, j, sc, m);
    const float d1 = d * static_cast<float>(sc);
    const float m1 = dmin * static_cast<float>(m);
    for (int i = 0; i < 32; ++i) {
      const int v = 32 * j + i;
      out[v] = d1 * static_cast<float>(q4_k_code(b, v)) - m1;
    }
  }
}

void dequant_q6_k(const BlockQ6_K& b, float* out) noexcept {
  const float d = fp16_to_fp32(b.d);
  for (int v = 0; v < kSuperBlock; ++v) {
    const int unit = v / 16;  // the sub-block of 16 values that v belongs to
    const float ds = d * static_cast<float>(b.scales[unit]);
    out[v] = ds * static_cast<float>(q6_k_code(b, v) - 32);
  }
}

void quantize_row_q8_k(const float* x, std::int64_t k, BlockQ8_K* out) noexcept {
  for (std::int64_t i = 0; i < k / kSuperBlock; ++i, x += kSuperBlock) {
    BlockQ8_K& y = out[i];
    float max = 0.0f, amax = 0.0f;
    for (int j = 0; j < kSuperBlock; ++j) {
      const float ax = std::fabs(x[j]);
      if (ax > amax) {
        amax = ax;
        max = x[j];
      }
    }
    if (amax == 0.0f) {
      y.d = 0.0f;
      std::memset(y.qs, 0, sizeof(y.qs));
      std::memset(y.bsums, 0, sizeof(y.bsums));  // ggml leaves them unset; nothing reads them
      continue;
    }
    const float iscale = -127.0f / max;
    for (int j = 0; j < kSuperBlock; ++j) {
      const int v = nearest_int(iscale * x[j]);
      y.qs[j] = static_cast<std::int8_t>(v < 127 ? v : 127);
    }
    for (int j = 0; j < kSuperBlock / 16; ++j) {
      int sum = 0;
      for (int ii = 0; ii < 16; ++ii) sum += y.qs[j * 16 + ii];
      y.bsums[j] = static_cast<std::int16_t>(sum);
    }
    y.d = 1.0f / iscale;
  }
}

}  // namespace hearth::quant
