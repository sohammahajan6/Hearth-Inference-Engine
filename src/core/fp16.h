// Half-precision (IEEE fp16) and bfloat16 conversions, bit-exact.
//
// fp32_to_fp16 rounds to nearest, ties to even, exactly like the F16C
// instruction VCVTPS2PH with rounding mode 0, which is what ggml uses on x86.
// Matching it bit for bit matters: quantization scales are stored as fp16, so
// a one-ulp difference in a scale changes every value in its block. The unit
// test checks this function against the hardware instruction for every class
// of input (zeros, subnormals, rounding ties, overflow, infinities, NaNs).
#pragma once

#include <bit>
#include <cstdint>

namespace hearth {

inline float fp16_to_fp32(std::uint16_t h) noexcept {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
  const std::uint32_t exp = (h >> 10) & 0x1fu;
  const std::uint32_t mant = h & 0x3ffu;
  if (exp == 0) {
    // Zero or subnormal: value = mant * 2^-24, exact in fp32.
    const float mag = static_cast<float>(mant) * 0x1p-24f;
    return std::bit_cast<float>(std::bit_cast<std::uint32_t>(mag) | sign);
  }
  if (exp == 31) {
    if (mant == 0) return std::bit_cast<float>(sign | 0x7f800000u);  // infinity
    // NaN: keep the payload and set the quiet bit, as VCVTPH2PS does
    // (a signalling half NaN becomes a quiet float NaN).
    return std::bit_cast<float>(sign | 0x7fc00000u | (mant << 13));
  }
  return std::bit_cast<float>(sign | ((exp + 112u) << 23) | (mant << 13));
}

inline std::uint16_t fp32_to_fp16(float f) noexcept {
  std::uint32_t x = std::bit_cast<std::uint32_t>(f);
  const std::uint32_t sign = (x >> 16) & 0x8000u;
  x &= 0x7fffffffu;
  std::uint32_t out;
  if (x > 0x7f800000u) {
    out = 0x7e00u | ((x >> 13) & 0x3ffu);  // NaN: quiet it, keep the top payload bits (as F16C does)
  } else if (x >= 0x47800000u) {
    out = 0x7c00u;  // >= 65536 or inf: infinity
  } else if (x < 0x38800000u) {
    // Result is subnormal or zero. Adding this magic constant aligns the 10
    // result bits at the bottom of the fp32 mantissa, and the hardware add
    // performs the round-to-nearest-even for us.
    const float magic = std::bit_cast<float>(0x3f000000u);  // 0.5f = 2^-1: exponent 126
    const float sum = std::bit_cast<float>(x) + magic;
    out = std::bit_cast<std::uint32_t>(sum) - 0x3f000000u;
  } else {
    const std::uint32_t mant_odd = (x >> 13) & 1u;
    x += 0xc8000fffu + mant_odd;  // rebias exponent (127 -> 15) and add the rounding bias
    out = x >> 13;                // an overflow here correctly produces infinity (0x7c00)
  }
  return static_cast<std::uint16_t>(out | sign);
}

inline float bf16_to_fp32(std::uint16_t b) noexcept {
  return std::bit_cast<float>(static_cast<std::uint32_t>(b) << 16);
}

}  // namespace hearth
