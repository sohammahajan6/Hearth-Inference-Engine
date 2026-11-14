// K-quant block formats Q4_K and Q6_K (identical to ggml's), the Q8_K
// activation format ggml multiplies them with, and the reference arithmetic
// every K-quant kernel in this project is checked against
//.
//
// A super-block holds 256 values (ggml's QK_K).
//   Q4_K: fp16 d, fp16 dmin, 12 bytes holding a 6-bit scale sc_j and a 6-bit
//         min m_j for each of 8 sub-blocks of 32 values, 128 bytes of nibbles.
//           value = (d * sc_j) * q - (dmin * m_j)            q = 0..15
//         Values 64c .. 64c+63 use bytes 32c .. 32c+31: sub-block 2c is their
//         low nibbles, sub-block 2c+1 their high nibbles.
//   Q6_K: 128 bytes ql (low 4 bits), 64 bytes qh (high 2 bits), 16 int8
//         scales sc_j for sub-blocks of 16 values, fp16 d.
//           value = (d * sc_j) * (q - 32)                     q = 0..63
//   Q8_K: float d, 256 int8 codes, 16 int16 sums of 16 consecutive codes.
//           value = d * code
#pragma once

#include <cstdint>

namespace hearth::quant {

inline constexpr int kSuperBlock = 256;  // values per K-quant block, all three formats

struct BlockQ4_K {
  std::uint16_t d;     // fp16 bits: scale of the sub-block scales
  std::uint16_t dmin;  // fp16 bits: scale of the sub-block mins
  std::uint8_t scales[12];
  std::uint8_t qs[kSuperBlock / 2];
};
struct BlockQ6_K {
  std::uint8_t ql[kSuperBlock / 2];
  std::uint8_t qh[kSuperBlock / 4];
  std::int8_t scales[kSuperBlock / 16];
  std::uint16_t d;  // fp16 bits
};
struct BlockQ8_K {
  float d;
  std::int8_t qs[kSuperBlock];
  std::int16_t bsums[kSuperBlock / 16];
};
static_assert(sizeof(BlockQ4_K) == 144 && alignof(BlockQ4_K) == 2);
static_assert(sizeof(BlockQ6_K) == 210 && alignof(BlockQ6_K) == 2);
static_assert(sizeof(BlockQ8_K) == 292 && alignof(BlockQ8_K) == 4);

// The 6-bit scale and min of sub-block j (0..7) of a Q4_K block (ggml's
// get_scale_min_k4).
void q4_k_scale_min(const BlockQ4_K& b, int j, std::uint8_t& sc, std::uint8_t& m) noexcept;

// The code of value v (0..255): 0..15 for Q4_K, 0..63 for Q6_K.
int q4_k_code(const BlockQ4_K& b, int v) noexcept;
int q6_k_code(const BlockQ6_K& b, int v) noexcept;

// The 256 values of a block, with ggml's operations in ggml's order
// (dequantize_row_q4_K, dequantize_row_q6_K):
//   Q4_K: d1 = d * sc; m1 = dmin * m; value = d1 * q - m1
//         (d1 * q is exact, so a fused multiply-add gives the same bits)
//   Q6_K: value = (d * sc) * (q - 32)
void dequant_q4_k(const BlockQ4_K& b, float* out) noexcept;
void dequant_q6_k(const BlockQ6_K& b, float* out) noexcept;

// Quantizes `k` floats (k a multiple of 256) exactly as ggml's
// quantize_row_q8_K_ref, which is what ggml's x86 build uses for Q8_K rows:
//   max = the first value of the largest magnitude amax; amax = 0 gives d = 0
//   and all codes 0; otherwise
//   iscale = -127 / max;  code_j = min(127, round-half-to-even(iscale * x_j));
//   d = 1 / iscale;       bsums = sums of 16 consecutive codes.
// (ggml's 4-row AVX2 variant, used next to its repacked Q4_K kernels, differs
// only when +amax and -amax both occur in a block: it then takes +amax.)
void quantize_row_q8_k(const float* x, std::int64_t k, BlockQ8_K* out) noexcept;

}  // namespace hearth::quant
