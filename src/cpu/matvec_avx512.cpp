// AVX-512 VNNI matvec kernels. Compiled with AVX-512 flags; called only after
// dispatch confirmed AVX-512 F/BW/VL + VNNI (EVEX) support.
//
// Each step handles two blocks in one 512-bit register: 64 int8 products
// summed four at a time by VPDPBUSD into 16 int32 lanes (8 lanes per block,
// exact), converted to float (exact: |lane| <= 4*128*127 < 2^24) and
// accumulated with that block's scale d_w * d_x. AoS and SoA kernels perform
// the same operations in the same order, so they produce identical results.
#include <immintrin.h>

#include "cpu/matvec.h"

namespace hearth::cpu::detail {
namespace {

using quant::kBlock;

// The F16C conversion, which core/fp16.h's fp16_to_fp32 is tested against bit
// for bit. That inline function must not be compiled into this AVX-512 file:
// the linker keeps one copy of it for the whole program and may pick this
// one, which a CPU without AVX-512 cannot run (RISKS D2,
// tools/check_isa_symbols.py).
inline float half_to_float(std::uint16_t h) {
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(h))));
}

// Lanes 0-7 get s0 (first block of the pair), lanes 8-15 get s1.
inline __m512 pair_scale(float s0, float s1) {
  return _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_set1_ps(s0)), _mm256_set1_ps(s1), 1);
}

// Q8_0 x Q8_0 on 64 bytes (two blocks). VPDPBUSD multiplies unsigned by
// signed bytes, so the weights' signs move onto the activations:
// w * x = |w| * (sign(w) x). Exact because activation quants lie in
// [-127, 127] (so negating one never overflows) and |w| <= 128 fits a u8.
inline __m512i dot64_q8(__m512i w, __m512i x) {
  const __m512i zero = _mm512_setzero_si512();
  const __mmask64 neg = _mm512_movepi8_mask(w);
  const __m512i sx = _mm512_mask_sub_epi8(x, neg, zero, x);
  return _mm512_dpbusd_epi32(zero, _mm512_abs_epi8(w), sx);
}

inline __m256i dot32_q8(__m256i w, __m256i x) {
  const __m256i zero = _mm256_setzero_si256();
  const __mmask32 neg = _mm256_movepi8_mask(w);
  const __m256i sx = _mm256_mask_sub_epi8(x, neg, zero, x);
  return _mm256_dpbusd_epi32(zero, _mm256_abs_epi8(w), sx);
}

// 16 Q4_0 bytes -> the block's 32 unsigned nibbles in element order
// (low nibbles are elements 0-15, high nibbles 16-31).
inline __m256i expand_nibbles(__m128i q) {
  const __m128i m4 = _mm_set1_epi8(0x0F);
  return _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q, 4), m4), _mm_and_si128(q, m4));
}

// Q4_0 x Q8_0 on two blocks: sum (n - 8) x = sum n x - sum 8 x, both exact int32.
inline __m512i dot64_q4(__m512i n, __m512i x) {
  const __m512i zero = _mm512_setzero_si512();
  return _mm512_sub_epi32(_mm512_dpbusd_epi32(zero, n, x), _mm512_dpbusd_epi32(zero, _mm512_set1_epi8(8), x));
}

inline __m256i dot32_q4(__m256i n, __m256i x) {
  const __m256i zero = _mm256_setzero_si256();
  return _mm256_sub_epi32(_mm256_dpbusd_epi32(zero, n, x), _mm256_dpbusd_epi32(zero, _mm256_set1_epi8(8), x));
}

inline float tail_block(__m256i isum, float scale) {
  const __m256 v = _mm256_mul_ps(_mm256_cvtepi32_ps(isum), _mm256_set1_ps(scale));
  const __m128 h = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  const __m128 q = _mm_add_ps(h, _mm_movehl_ps(h, h));
  return _mm_cvtss_f32(_mm_add_ss(q, _mm_movehdup_ps(q)));
}

// The shared row loop: `pair(b)` returns the int32 lanes of blocks b and b+1,
// `single(b)` those of block b alone, `scale(b)` its d_w * d_x.
template <class Pair, class Single, class Scale>
inline float row(std::int64_t nb, Pair pair, Single single, Scale scale) {
  __m512 acc = _mm512_setzero_ps();
  std::int64_t b = 0;
  for (; b + 2 <= nb; b += 2) {
    acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(pair(b)), pair_scale(scale(b), scale(b + 1)), acc);
  }
  const float tail = b < nb ? tail_block(single(b), scale(b)) : 0.0f;
  return _mm512_reduce_add_ps(acc) + tail;
}

inline __m512i load_pair(const void* lo, const void* hi) {
  return _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256(static_cast<const __m256i*>(lo))),
                            _mm256_loadu_si256(static_cast<const __m256i*>(hi)), 1);
}

}  // namespace

void matvec_avx512(const Q8_0Aos& w, const quant::ActQ8& x, float* y, std::int64_t r0, std::int64_t r1) {
  const std::int64_t nb = w.k / kBlock;
  for (std::int64_t r = r0; r < r1; ++r) {
    const quant::BlockQ8_0* rw = w.blocks + r * nb;
    y[r] = row(
        nb,
        [&](std::int64_t b) { return dot64_q8(load_pair(rw[b].q, rw[b + 1].q), _mm512_loadu_si512(x.q + b * kBlock)); },
        [&](std::int64_t b) {
          return dot32_q8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(rw[b].q)),
                          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x.q + b * kBlock)));
        },
        [&](std::int64_t b) { return half_to_float(rw[b].d) * x.d[b]; });
  }
}

void matvec_avx512(const Q8_0Soa& w, const quant::ActQ8& x, float* y, std::int64_t r0, std::int64_t r1) {
  const std::int64_t nb = w.k / kBlock;
  for (std::int64_t r = r0; r < r1; ++r) {
    const std::int8_t* rq = w.q + r * nb * kBlock;
    const std::uint16_t* rd = w.d + r * nb;
    y[r] = row(
        nb,
        [&](std::int64_t b) {
          return dot64_q8(_mm512_loadu_si512(rq + b * kBlock), _mm512_loadu_si512(x.q + b * kBlock));
        },
        [&](std::int64_t b) {
          return dot32_q8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(rq + b * kBlock)),
                          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x.q + b * kBlock)));
        },
        [&](std::int64_t b) { return half_to_float(rd[b]) * x.d[b]; });
  }
}

void matvec_avx512(const Q4_0Aos& w, const quant::ActQ8& x, float* y, std::int64_t r0, std::int64_t r1) {
  const std::int64_t nb = w.k / kBlock;
  for (std::int64_t r = r0; r < r1; ++r) {
    const quant::BlockQ4_0* rw = w.blocks + r * nb;
    auto nib = [&](std::int64_t b) {
      return expand_nibbles(_mm_loadu_si128(reinterpret_cast<const __m128i*>(rw[b].q)));
    };
    y[r] = row(
        nb,
        [&](std::int64_t b) {
          const __m512i n = _mm512_inserti64x4(_mm512_castsi256_si512(nib(b)), nib(b + 1), 1);
          return dot64_q4(n, _mm512_loadu_si512(x.q + b * kBlock));
        },
        [&](std::int64_t b) {
          return dot32_q4(nib(b), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x.q + b * kBlock)));
        },
        [&](std::int64_t b) { return half_to_float(rw[b].d) * x.d[b]; });
  }
}

void matvec_avx512(const Q4_0Soa& w, const quant::ActQ8& x, float* y, std::int64_t r0, std::int64_t r1) {
  const std::int64_t nb = w.k / kBlock;
  for (std::int64_t r = r0; r < r1; ++r) {
    const std::uint8_t* rq = w.q + r * nb * (kBlock / 2);
    const std::uint16_t* rd = w.d + r * nb;
    auto nib = [&](std::int64_t b) {
      return expand_nibbles(_mm_loadu_si128(reinterpret_cast<const __m128i*>(rq + b * (kBlock / 2))));
    };
    y[r] = row(
        nb,
        [&](std::int64_t b) {
          const __m512i n = _mm512_inserti64x4(_mm512_castsi256_si512(nib(b)), nib(b + 1), 1);
          return dot64_q4(n, _mm512_loadu_si512(x.q + b * kBlock));
        },
        [&](std::int64_t b) {
          return dot32_q4(nib(b), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x.q + b * kBlock)));
        },
        [&](std::int64_t b) { return half_to_float(rd[b]) * x.d[b]; });
  }
}

}  // namespace hearth::cpu::detail
