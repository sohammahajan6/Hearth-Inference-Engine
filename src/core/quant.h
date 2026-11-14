// Q8_0 / Q4_0 block formats (identical to ggml's) and the reference
// arithmetic every kernel in this project is checked against.
//
//   Q8_0 block: fp16 scale d, then 32 signed bytes q;     value_j = d * q_j
//   Q4_0 block: fp16 scale d, then 16 bytes of nibbles;   value_j = d * (n_j - 8)
//               n_j = low nibble of byte j for j < 16, high nibble of byte j-16 otherwise
#pragma once

#include <cstdint>

namespace hearth::quant {

inline constexpr int kBlock = 32;  // values per block, both formats

struct BlockQ8_0 {
  std::uint16_t d;  // fp16 bits
  std::int8_t q[kBlock];
};
struct BlockQ4_0 {
  std::uint16_t d;  // fp16 bits
  std::uint8_t q[kBlock / 2];
};
static_assert(sizeof(BlockQ8_0) == 34 && alignof(BlockQ8_0) == 2);
static_assert(sizeof(BlockQ4_0) == 18 && alignof(BlockQ4_0) == 2);

// Activations quantized for a matvec, stored as separate arrays (SoA):
// `k` values, `k / 32` blocks. `d` holds each block's fp16 scale already
// decoded to float; `s` holds each block's sum of quants (Q4_0 needs it).
struct ActQ8 {
  const std::int8_t* q;
  const float* d;
  const std::int32_t* s;
  std::int64_t k;
};

// Quantizes `k` floats (k a multiple of 32) exactly as ggml's x86 path
// (quantize_row_q8_0, AVX2 branch) does:
//   amax = max |x_j|;  d = fp16(amax / 127);  m = amax != 0 ? 127 / amax : 0;
//   q_j = round-half-to-even(x_j * m)
// Writes the quants, decoded scales and per-block sums.
void quantize_act_q8_0(const float* x, std::int64_t k, std::int8_t* q, float* d, std::int32_t* s);

// The same rule producing ggml's AoS blocks (used to cross-check layouts).
void quantize_row_q8_0(const float* x, std::int64_t k, BlockQ8_0* out);

// Exact integer block sums.
std::int32_t block_dot_q8_0(const BlockQ8_0& w, const std::int8_t* xq);  // sum w_j * x_j
std::int32_t block_dot_q4_0(const BlockQ4_0& w, const std::int8_t* xq);  // sum (n_j - 8) * x_j

float dequant(const BlockQ8_0& b, int j) noexcept;
float dequant(const BlockQ4_0& b, int j) noexcept;

}  // namespace hearth::quant
