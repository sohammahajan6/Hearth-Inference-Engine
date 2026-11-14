// The AVX2/F16C reference sequences of hw_reference.h. Compiled with AVX2
// flags; includes nothing but <immintrin.h> and <cstdint> (see the header).
#include <immintrin.h>

#include "hw_reference.h"

namespace hearth::test {

std::uint16_t hw_fp32_to_fp16(float f) {
  return static_cast<std::uint16_t>(_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(f), _MM_FROUND_TO_NEAREST_INT), 0));
}

float hw_fp16_to_fp32(std::uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h))); }

// ggml's sequence for one block: max |x| with andnot(sign); d = max / 127;
// multiplier 127 / max (0 for an all-zero block); x * m rounded to nearest
// even; converted to int32 (the int8 packing cannot saturate: |q| <= 127).
void ggml_quantize_block(const float* x, std::uint16_t& d_bits, std::int8_t* q) {
  __m256 v0 = _mm256_loadu_ps(x), v1 = _mm256_loadu_ps(x + 8), v2 = _mm256_loadu_ps(x + 16),
         v3 = _mm256_loadu_ps(x + 24);
  const __m256 sign = _mm256_set1_ps(-0.0f);
  __m256 m = _mm256_andnot_ps(sign, v0);
  m = _mm256_max_ps(m, _mm256_andnot_ps(sign, v1));
  m = _mm256_max_ps(m, _mm256_andnot_ps(sign, v2));
  m = _mm256_max_ps(m, _mm256_andnot_ps(sign, v3));
  __m128 m4 = _mm_max_ps(_mm256_extractf128_ps(m, 1), _mm256_castps256_ps128(m));
  m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
  m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
  const float max_scalar = _mm_cvtss_f32(m4);
  const float d = max_scalar / 127.f;
  d_bits = static_cast<std::uint16_t>(_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(d), _MM_FROUND_TO_NEAREST_INT), 0));
  const float id = (max_scalar != 0.0f) ? 127.f / max_scalar : 0.0f;
  const __m256 mul = _mm256_set1_ps(id);
  const __m256 r[4] = {_mm256_round_ps(_mm256_mul_ps(v0, mul), _MM_ROUND_NEAREST),
                       _mm256_round_ps(_mm256_mul_ps(v1, mul), _MM_ROUND_NEAREST),
                       _mm256_round_ps(_mm256_mul_ps(v2, mul), _MM_ROUND_NEAREST),
                       _mm256_round_ps(_mm256_mul_ps(v3, mul), _MM_ROUND_NEAREST)};
  for (int g = 0; g < 4; ++g) {
    alignas(32) std::int32_t lanes[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), _mm256_cvtps_epi32(r[g]));
    for (int j = 0; j < 8; ++j) q[8 * g + j] = static_cast<std::int8_t>(lanes[j]);
  }
}

}  // namespace hearth::test
