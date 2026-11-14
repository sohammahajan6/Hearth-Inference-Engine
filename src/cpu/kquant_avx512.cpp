// AVX-512 VNNI kernels for the canonical K-quant order of ADR 0021
// (cpu/kquant.h). Compiled with AVX-512 flags; called only after dispatch
// confirmed AVX-512 F/BW/DQ/VL + VNNI (EVEX) support.
//
// One 512-bit register holds the 16 lanes of a group: lane j is unit j of the
// group. Per group, for each (row, token) pair of a tile:
//   Q4_K: isum = VPDPBUSD of the low nibbles against chunks 0-3 plus (VPDPBUSD
//         of the high nibbles, kept as 16 n, against chunks 4-7) / 16;
//         t = cvt(isum) * (ws * dx) - cvt(xsum) * (wm * dx)
//   Q6_K: isum = -32 * xsum, then VPDPBUSD of the 6-bit codes against the
//         4 chunks; t = cvt(isum) * (ws * dx)
//   acc = acc + t
// and at the end the halving tree. ws and wm are exact products of a fp16
// scale and a small integer (cpu/kquant.h). Every pair takes exactly these
// steps whatever the tile shape, so GEMV (tokens = 1) and GEMM agree bit for
// bit with the scalar kernel.
#include <immintrin.h>

#include <bit>
#include <cstring>

#include "cpu/kquant.h"

namespace hearth::cpu::detail {
namespace {

// lane j + lane j+8, then + 4, + 2, + 1.
inline float reduce_halving(__m512 v) {
  const __m256 h8 = _mm256_add_ps(_mm512_castps512_ps256(v), _mm512_extractf32x8_ps(v, 1));
  const __m128 h4 = _mm_add_ps(_mm256_castps256_ps128(h8), _mm256_extractf128_ps(h8, 1));
  const __m128 h2 = _mm_add_ps(h4, _mm_movehl_ps(h4, h4));
  return _mm_cvtss_f32(_mm_add_ss(h2, _mm_shuffle_ps(h2, h2, 1)));
}

inline __m512i load512(const void* p) { return _mm512_load_si512(p); }

// The two fp16 values at p, as floats in lanes 0-7 (first) and 8-15 (second).
inline __m512 pair_to_lanes(const std::uint16_t* p) {
  std::int32_t two = 0;
  std::memcpy(&two, p, sizeof(two));
  const __m128 f = _mm_cvtph_ps(_mm_cvtsi32_si128(two));
  const __m512i idx = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0);
  return _mm512_permutexvar_ps(idx, _mm512_castps128_ps512(f));
}

inline __m512 u8_lanes(const std::uint8_t* p) {
  return _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p))));
}
inline __m512 i8_lanes(const std::uint8_t* p) {
  return _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p))));
}

inline __m512 term_q4_k(__m512i isum, __m512 xsum, __m512 ws, __m512 wm, __m512 dx) {
  const __m512 p1 = _mm512_mul_ps(_mm512_cvtepi32_ps(isum), _mm512_mul_ps(ws, dx));
  const __m512 p2 = _mm512_mul_ps(xsum, _mm512_mul_ps(wm, dx));
  return _mm512_sub_ps(p1, p2);
}

inline __m512 term_q6_k(__m512i isum, __m512 ws, __m512 dx) {
  return _mm512_mul_ps(_mm512_cvtepi32_ps(isum), _mm512_mul_ps(ws, dx));
}

// ws and wm of row-group `at` (exact products, so computing them here or in
// advance gives the same bits).
inline __m512 ws_q4_k(const KPackedView& w, std::int64_t at) {
  return _mm512_mul_ps(pair_to_lanes(w.d + at * 2), u8_lanes(w.sc + at * kKLanes));
}
inline __m512 wm_q4_k(const KPackedView& w, std::int64_t at) {
  return _mm512_mul_ps(pair_to_lanes(w.dmin + at * 2), u8_lanes(w.mn + at * kKLanes));
}
inline __m512 ws_q6_k(const KPackedView& w, std::int64_t at) {
  return _mm512_mul_ps(_mm512_set1_ps(_mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(w.d[at * 2])))),
                       i8_lanes(w.sc + at * kKLanes));
}

// The scales of rows [r, r + rows) for a GEMM tile, decoded once per block of
// tokens instead of once per tile: pre[(i * groups + g) * 32 + 0..15] = ws,
// + 16..31 = wm (Q4_K only).
inline void decode_scales(const KPackedView& w, std::int64_t r, std::int64_t rows, float* pre) {
  for (std::int64_t i = 0; i < rows; ++i) {
    for (std::int64_t g = 0; g < w.groups; ++g) {
      const std::int64_t at = (r + i) * w.groups + g;
      float* o = pre + (i * w.groups + g) * 32;
      if (w.type == KType::kQ4_K) {
        _mm512_storeu_ps(o, ws_q4_k(w, at));
        _mm512_storeu_ps(o + 16, wm_q4_k(w, at));
      } else {
        _mm512_storeu_ps(o, ws_q6_k(w, at));
      }
    }
  }
}

inline void store_tile(float* y, std::int64_t t, std::int64_t r, std::int64_t ldy, int rows, int cols, __m512 f00,
                       __m512 f10, __m512 f20, __m512 f30, __m512 f01, __m512 f11, __m512 f21, __m512 f31) {
  float* y0 = y + t * ldy + r;
  y0[0] = reduce_halving(f00);
  if (rows > 1) {
    y0[1] = reduce_halving(f10);
    y0[2] = reduce_halving(f20);
    y0[3] = reduce_halving(f30);
  }
  if (cols > 1) {
    float* y1 = y0 + ldy;
    y1[0] = reduce_halving(f01);
    if (rows > 1) {
      y1[1] = reduce_halving(f11);
      y1[2] = reduce_halving(f21);
      y1[3] = reduce_halving(f31);
    }
  }
}

// R rows starting at r (R = 1 or 4), C tokens starting at t (C = 1 or 2).
// Accumulators are named variables (aRC, fRC: row R, token C), not arrays:
// MSVC keeps __m512i arrays in memory. Unused ones are optimized away.
// kPre: the scales come from decode_scales (`pre`, rows r .. r + R - 1), else
// they are computed here (decode, where each row is used for one token).
template <int R, int C, bool kPre>
void tile_q4_k(const KPackedView& w, std::int64_t r, const KActs& x, std::int64_t t, float* y, std::int64_t ldy,
               [[maybe_unused]] const float* pre) {
  static_assert((R == 1 || R == 4) && (C == 1 || C == 2));
  constexpr std::int64_t gq = k_group_quant_bytes(KType::kQ4_K);
  const std::int64_t G = w.groups;
  const std::int64_t rs = R > 1 ? 1 : 0;  // rows 1-3 alias row 0 when R = 1 (never read then)
  const std::int64_t t1 = t + (C > 1 ? 1 : 0);
  const std::int64_t rg0 = r * G;
  [[maybe_unused]] const std::int64_t rg1 = rg0 + rs * G, rg2 = rg1 + rs * G, rg3 = rg2 + rs * G;
  const __m512i m_lo = _mm512_set1_epi8(0x0F), m_hi = _mm512_set1_epi8(static_cast<char>(0xF0));
  [[maybe_unused]] __m512 f00 = _mm512_setzero_ps(), f01 = f00, f10 = f00, f11 = f00, f20 = f00, f21 = f00, f30 = f00,
                          f31 = f00;
  for (std::int64_t g = 0; g < G; ++g) {
    const std::int8_t* xa = x.q(t, g);
    [[maybe_unused]] const std::int8_t* xb = x.q(t1, g);
    [[maybe_unused]] __m512i a00 = _mm512_setzero_si512(), a01 = a00, a10 = a00, a11 = a00, a20 = a00, a21 = a00,
                             a30 = a00, a31 = a00;
    [[maybe_unused]] __m512i h00 = a00, h01 = a00, h10 = a00, h11 = a00, h20 = a00, h21 = a00, h30 = a00, h31 = a00;
#define HEARTH_NIB(i)                                                                     \
  do {                                                                                    \
    const __m512i raw_ = load512(w.q + (rg##i + g) * gq + o);                             \
    const __m512i lo_ = _mm512_and_si512(raw_, m_lo), hi_ = _mm512_and_si512(raw_, m_hi); \
    a##i##0 = _mm512_dpbusd_epi32(a##i##0, lo_, xl0);                                     \
    h##i##0 = _mm512_dpbusd_epi32(h##i##0, hi_, xh0);                                     \
    if constexpr (C > 1) {                                                                \
      a##i##1 = _mm512_dpbusd_epi32(a##i##1, lo_, xl1);                                   \
      h##i##1 = _mm512_dpbusd_epi32(h##i##1, hi_, xh1);                                   \
    }                                                                                     \
  } while (0)
    for (int ch = 0; ch < 4; ++ch) {
      const std::int64_t o = ch * 64;
      const __m512i xl0 = load512(xa + o), xh0 = load512(xa + o + 256);
      [[maybe_unused]] const __m512i xl1 = C > 1 ? load512(xb + o) : xl0;
      [[maybe_unused]] const __m512i xh1 = C > 1 ? load512(xb + o + 256) : xh0;
      HEARTH_NIB(0);
      if constexpr (R > 1) {
        HEARTH_NIB(1);
        HEARTH_NIB(2);
        HEARTH_NIB(3);
      }
    }
#undef HEARTH_NIB
    // High nibbles were multiplied as 16 n: their sum is a multiple of 16, so
    // the shift is exact.
    a00 = _mm512_add_epi32(a00, _mm512_srai_epi32(h00, 4));
    if constexpr (C > 1) a01 = _mm512_add_epi32(a01, _mm512_srai_epi32(h01, 4));
    if constexpr (R > 1) {
      a10 = _mm512_add_epi32(a10, _mm512_srai_epi32(h10, 4));
      a20 = _mm512_add_epi32(a20, _mm512_srai_epi32(h20, 4));
      a30 = _mm512_add_epi32(a30, _mm512_srai_epi32(h30, 4));
      if constexpr (C > 1) {
        a11 = _mm512_add_epi32(a11, _mm512_srai_epi32(h11, 4));
        a21 = _mm512_add_epi32(a21, _mm512_srai_epi32(h21, 4));
        a31 = _mm512_add_epi32(a31, _mm512_srai_epi32(h31, 4));
      }
    }
    const __m512 dx0 = _mm512_load_ps(x.d(t, g));
    const __m512 xs0 = _mm512_cvtepi32_ps(load512(x.s(t, g)));
    [[maybe_unused]] const __m512 dx1 = C > 1 ? _mm512_load_ps(x.d(t1, g)) : dx0;
    [[maybe_unused]] const __m512 xs1 = C > 1 ? _mm512_cvtepi32_ps(load512(x.s(t1, g))) : xs0;
#define HEARTH_TERMS(i)                                                                            \
  do {                                                                                             \
    const std::int64_t at_ = rg##i + g;                                                            \
    const float* p_ = kPre ? pre + ((i) * G + g) * 32 : nullptr;                                   \
    const __m512 ws_ = kPre ? _mm512_loadu_ps(p_) : ws_q4_k(w, at_);                               \
    const __m512 wm_ = kPre ? _mm512_loadu_ps(p_ + 16) : wm_q4_k(w, at_);                          \
    f##i##0 = _mm512_add_ps(f##i##0, term_q4_k(a##i##0, xs0, ws_, wm_, dx0));                      \
    if constexpr (C > 1) f##i##1 = _mm512_add_ps(f##i##1, term_q4_k(a##i##1, xs1, ws_, wm_, dx1)); \
  } while (0)
    HEARTH_TERMS(0);
    if constexpr (R > 1) {
      HEARTH_TERMS(1);
      HEARTH_TERMS(2);
      HEARTH_TERMS(3);
    }
#undef HEARTH_TERMS
  }
  store_tile(y, t, r, ldy, R, C, f00, f10, f20, f30, f01, f11, f21, f31);
}

template <int R, int C, bool kPre>
void tile_q6_k(const KPackedView& w, std::int64_t r, const KActs& x, std::int64_t t, float* y, std::int64_t ldy,
               [[maybe_unused]] const float* pre) {
  static_assert((R == 1 || R == 4) && (C == 1 || C == 2));
  constexpr std::int64_t gq = k_group_quant_bytes(KType::kQ6_K);
  const std::int64_t G = w.groups;
  const std::int64_t rs = R > 1 ? 1 : 0;
  const std::int64_t t1 = t + (C > 1 ? 1 : 0);
  const std::int64_t rg0 = r * G;
  [[maybe_unused]] const std::int64_t rg1 = rg0 + rs * G, rg2 = rg1 + rs * G, rg3 = rg2 + rs * G;
  const __m512i m_lo = _mm512_set1_epi8(0x0F), m_hi = _mm512_set1_epi8(0x30);
  [[maybe_unused]] __m512 f00 = _mm512_setzero_ps(), f01 = f00, f10 = f00, f11 = f00, f20 = f00, f21 = f00, f30 = f00,
                          f31 = f00;
  for (std::int64_t g = 0; g < G; ++g) {
    const std::int8_t* xa = x.q(t, g);
    [[maybe_unused]] const std::int8_t* xb = x.q(t1, g);
    // isum starts at -32 * xsum: the codes are multiplied as q, not q - 32.
    const __m512i s0 = load512(x.s(t, g));
    const __m512i init0 = _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_slli_epi32(s0, 5));
    [[maybe_unused]] const __m512i s1 = C > 1 ? load512(x.s(t1, g)) : s0;
    [[maybe_unused]] const __m512i init1 =
        C > 1 ? _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_slli_epi32(s1, 5)) : init0;
    [[maybe_unused]] __m512i a00 = init0, a01 = init1, a10 = init0, a11 = init1, a20 = init0, a21 = init1, a30 = init0,
                             a31 = init1;
    const __m512i xc0 = load512(xa), xc1 = load512(xa + 64), xc2 = load512(xa + 128), xc3 = load512(xa + 192);
    [[maybe_unused]] const __m512i yc0 = C > 1 ? load512(xb) : xc0;
    [[maybe_unused]] const __m512i yc1 = C > 1 ? load512(xb + 64) : xc1;
    [[maybe_unused]] const __m512i yc2 = C > 1 ? load512(xb + 128) : xc2;
    [[maybe_unused]] const __m512i yc3 = C > 1 ? load512(xb + 192) : xc3;
    // Codes of chunk c: low 4 bits from a nibble of lo0 (c = 0, 2) or lo1
    // (c = 1, 3); bits 4-5 from bits 2c, 2c+1 of hb, moved to bits 4-5 by a
    // 16-bit shift (the mask drops what the shift carries across bytes).
    // ternarylogic 0xF8 = A | (B & C).
#define HEARTH_Q6(i)                                                                                    \
  do {                                                                                                  \
    const std::uint8_t* p_ = w.q + (rg##i + g) * gq;                                                    \
    const __m512i lo0_ = load512(p_), lo1_ = load512(p_ + 64), hb_ = load512(p_ + 128);                 \
    const __m512i w0_ =                                                                                 \
        _mm512_ternarylogic_epi32(_mm512_and_si512(lo0_, m_lo), _mm512_slli_epi16(hb_, 4), m_hi, 0xF8); \
    const __m512i w1_ =                                                                                 \
        _mm512_ternarylogic_epi32(_mm512_and_si512(lo1_, m_lo), _mm512_slli_epi16(hb_, 2), m_hi, 0xF8); \
    const __m512i w2_ =                                                                                 \
        _mm512_ternarylogic_epi32(_mm512_and_si512(_mm512_srli_epi16(lo0_, 4), m_lo), hb_, m_hi, 0xF8); \
    const __m512i w3_ = _mm512_ternarylogic_epi32(_mm512_and_si512(_mm512_srli_epi16(lo1_, 4), m_lo),   \
                                                  _mm512_srli_epi16(hb_, 2), m_hi, 0xF8);               \
    a##i##0 = _mm512_dpbusd_epi32(a##i##0, w0_, xc0);                                                   \
    a##i##0 = _mm512_dpbusd_epi32(a##i##0, w1_, xc1);                                                   \
    a##i##0 = _mm512_dpbusd_epi32(a##i##0, w2_, xc2);                                                   \
    a##i##0 = _mm512_dpbusd_epi32(a##i##0, w3_, xc3);                                                   \
    if constexpr (C > 1) {                                                                              \
      a##i##1 = _mm512_dpbusd_epi32(a##i##1, w0_, yc0);                                                 \
      a##i##1 = _mm512_dpbusd_epi32(a##i##1, w1_, yc1);                                                 \
      a##i##1 = _mm512_dpbusd_epi32(a##i##1, w2_, yc2);                                                 \
      a##i##1 = _mm512_dpbusd_epi32(a##i##1, w3_, yc3);                                                 \
    }                                                                                                   \
  } while (0)
    HEARTH_Q6(0);
    if constexpr (R > 1) {
      HEARTH_Q6(1);
      HEARTH_Q6(2);
      HEARTH_Q6(3);
    }
#undef HEARTH_Q6
    const __m512 dx0 = _mm512_load_ps(x.d(t, g));
    [[maybe_unused]] const __m512 dx1 = C > 1 ? _mm512_load_ps(x.d(t1, g)) : dx0;
#define HEARTH_TERMS(i)                                                                    \
  do {                                                                                     \
    const std::int64_t at_ = rg##i + g;                                                    \
    const __m512 ws_ = kPre ? _mm512_loadu_ps(pre + ((i) * G + g) * 32) : ws_q6_k(w, at_); \
    f##i##0 = _mm512_add_ps(f##i##0, term_q6_k(a##i##0, ws_, dx0));                        \
    if constexpr (C > 1) f##i##1 = _mm512_add_ps(f##i##1, term_q6_k(a##i##1, ws_, dx1));   \
  } while (0)
    HEARTH_TERMS(0);
    if constexpr (R > 1) {
      HEARTH_TERMS(1);
      HEARTH_TERMS(2);
      HEARTH_TERMS(3);
    }
#undef HEARTH_TERMS
  }
  store_tile(y, t, r, ldy, R, C, f00, f10, f20, f30, f01, f11, f21, f31);
}

template <KType kType, int R, int C, bool kPre>
void tile(const KPackedView& w, std::int64_t r, const KActs& x, std::int64_t t, float* y, std::int64_t ldy,
          const float* pre) {
  if constexpr (kType == KType::kQ4_K) {
    tile_q4_k<R, C, kPre>(w, r, x, t, y, ldy, pre);
  } else {
    tile_q6_k<R, C, kPre>(w, r, x, t, y, ldy, pre);
  }
}

template <KType kType, int R>
void tokens(const KPackedView& w, std::int64_t r, const KActs& x, std::int64_t t0, std::int64_t t1, float* y,
            std::int64_t ldy, float* pre) {
  decode_scales(w, r, R, pre);
  std::int64_t t = t0;
  for (; t + 2 <= t1; t += 2) tile<kType, R, 2, true>(w, r, x, t, y, ldy, pre);
  for (; t < t1; ++t) tile<kType, R, 1, true>(w, r, x, t, y, ldy, pre);
}

template <KType kType>
void run(const KPackedView& w, const KActs& x, float* y, std::int64_t ldy, std::int64_t r0, std::int64_t r1,
         float* pre) {
  // Decode: one row at a time, one sequential stream per thread, as the
  // Q8_0/Q4_0 kernels (cpu/qgemm_avx512.cpp).
  if (x.tokens() == 1) {
    for (std::int64_t r = r0; r < r1; ++r) tile<kType, 1, 1, false>(w, r, x, 0, y, ldy, nullptr);
    return;
  }
  // A block of tokens' activations stays in L2 while the rows pass over it.
  constexpr std::int64_t kTokenBlock = 128;
  for (std::int64_t t0 = 0; t0 < x.tokens(); t0 += kTokenBlock) {
    const std::int64_t t1 = x.tokens() < t0 + kTokenBlock ? x.tokens() : t0 + kTokenBlock;
    std::int64_t r = r0;
    for (; r + 4 <= r1; r += 4) tokens<kType, 4>(w, r, x, t0, t1, y, ldy, pre);
    for (; r < r1; ++r) tokens<kType, 1>(w, r, x, t0, t1, y, ldy, pre);
  }
}

}  // namespace

void quantize_super_avx512(const float* x, std::int8_t* codes, float& d) {
  // The steps of quant::quantize_row_q8_k (ggml's quantize_row_q8_K_ref):
  // the largest magnitude is exact in any order, and the first value that
  // has it is found by position, as the reference's strict > finds it.
  // roundscale rounds half to even, as ggml's nearest_int does for these
  // magnitudes. (Only a NaN activation, which no correct model produces, can
  // make the two differ.)
  __m512 amaxv = _mm512_setzero_ps();
  for (int i = 0; i < 16; ++i) amaxv = _mm512_max_ps(amaxv, _mm512_abs_ps(_mm512_loadu_ps(x + 16 * i)));
  const float amax = _mm512_reduce_max_ps(amaxv);
  if (amax == 0.0f) {
    std::memset(codes, 0, quant::kSuperBlock);
    d = 0.0f;
    return;
  }
  const __m512 target = _mm512_set1_ps(amax);
  float max = 0.0f;
  for (int i = 0; i < 16; ++i) {
    const __mmask16 hit = _mm512_cmp_ps_mask(_mm512_abs_ps(_mm512_loadu_ps(x + 16 * i)), target, _CMP_EQ_OQ);
    if (hit != 0) {
      max = x[16 * i + std::countr_zero(static_cast<unsigned>(hit))];
      break;
    }
  }
  const float iscale = -127.0f / max;
  const __m512 sv = _mm512_set1_ps(iscale);
  const __m512i top = _mm512_set1_epi32(127);
  for (int i = 0; i < 16; ++i) {
    const __m512 v = _mm512_mul_ps(_mm512_loadu_ps(x + 16 * i), sv);
    const __m512i c = _mm512_min_epi32(
        _mm512_cvtps_epi32(_mm512_roundscale_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)), top);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(codes + 16 * i), _mm512_cvtepi32_epi8(c));
  }
  d = 1.0f / iscale;
}

void kgemm_avx512(const KPackedView& w, const KActs& x, float* y, std::int64_t ldy, std::int64_t r0, std::int64_t r1,
                  float* scales) {
  if (w.type == KType::kQ4_K) {
    run<KType::kQ4_K>(w, x, y, ldy, r0, r1, scales);
  } else {
    run<KType::kQ6_K>(w, x, y, ldy, r0, r1, scales);
  }
}

}  // namespace hearth::cpu::detail
