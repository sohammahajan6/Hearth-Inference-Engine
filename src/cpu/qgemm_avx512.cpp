// AVX-512 VNNI kernels for the canonical order of ADR 0009 (cpu/qgemm.h).
// Compiled with AVX-512 flags; called only after dispatch confirmed
// AVX-512 F/BW/DQ/VL + VNNI (EVEX) support.
//
// One 512-bit register holds the 16 lanes of a group: lane j is block j of
// the group. Per group, for each (row, token) pair of a tile:
//   isum  = -128 * sum(x) (Q8_0) or -8 * sum(x) (Q4_0), then VPDPBUSD over the
//           group's chunks: exact int32, one block per lane;
//   s     = cvtph(d_w) * d_x;  t = cvt(isum) * s;  acc = acc + t
// and at the end the halving tree. Every pair takes exactly these steps
// whatever the tile shape, so GEMV (tokens = 1) and GEMM agree bit for bit.
#include <immintrin.h>

#include "cpu/qgemm.h"

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

inline __m512i neg_shifted_sums(const std::int32_t* s, int shift) {
  return _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_slli_epi32(load512(s), static_cast<unsigned>(shift)));
}

inline __m512 block_terms(__m512i isum, __m512 dw, __m512 dx) {
  return _mm512_mul_ps(_mm512_cvtepi32_ps(isum), _mm512_mul_ps(dw, dx));
}

// R rows starting at r (R = 1 or 4), C tokens starting at t (C = 1 or 2).
// The accumulators are named variables, not arrays: MSVC keeps __m512i
// arrays in memory, which costs a store and a reload around every VPDPBUSD
// (seen in the disassembly). Unused ones (R = 1 or C = 1) are optimized away.
// Accumulator aRC / fRC: row R, token C.
template <QType kType, int R, int C>
void tile(const PackedView& w, std::int64_t r, const PackedActs& x, std::int64_t t, float* y, std::int64_t ldy) {
  static_assert((R == 1 || R == 4) && (C == 1 || C == 2));
  constexpr std::int64_t gq = group_quant_bytes(kType);
  constexpr int kShift = kType == QType::kQ8_0 ? 7 : 3;  // 128 or 8 times the code sum
  const std::int64_t row_bytes = w.groups * gq, row_scales = w.groups * kGroupBlocks;
  const std::int64_t rs = R > 1 ? 1 : 0;  // row step: rows 1-3 alias row 0 when R = 1 (never read then)
  const std::int64_t t1 = t + (C > 1 ? 1 : 0);
  const std::uint8_t* q0 = w.q + r * row_bytes;
  [[maybe_unused]] const std::uint8_t *q1 = q0 + rs * row_bytes, *q2 = q1 + rs * row_bytes, *q3 = q2 + rs * row_bytes;
  const std::uint16_t* d0 = w.d + r * row_scales;
  [[maybe_unused]] const std::uint16_t *d1 = d0 + rs * row_scales, *d2 = d1 + rs * row_scales,
                                       *d3 = d2 + rs * row_scales;
  [[maybe_unused]] __m512 f00 = _mm512_setzero_ps(), f01 = f00, f10 = f00, f11 = f00, f20 = f00, f21 = f00, f30 = f00,
                          f31 = f00;
  for (std::int64_t g = 0; g < w.groups; ++g) {
    const std::int64_t gb = g * gq;
    const std::int8_t* xa = x.q(t, g);
    [[maybe_unused]] const std::int8_t* xb = x.q(t1, g);
    const __m512i init0 = neg_shifted_sums(x.s(t, g), kShift);
    const __m512i init1 = C > 1 ? neg_shifted_sums(x.s(t1, g), kShift) : init0;
    [[maybe_unused]] __m512i a00 = init0, a01 = init1, a10 = init0, a11 = init1, a20 = init0, a21 = init1, a30 = init0,
                             a31 = init1;
    // One weight chunk of row i against the activation chunks x0 (token t) and x1 (token t + 1).
#define HEARTH_DOT(i, wv)                             \
  do {                                                \
    const __m512i w_ = (wv);                          \
    a##i##0 = _mm512_dpbusd_epi32(a##i##0, w_, x0);   \
    if constexpr (C > 1) {                            \
      a##i##1 = _mm512_dpbusd_epi32(a##i##1, w_, x1); \
    }                                                 \
  } while (0)
    if constexpr (kType == QType::kQ8_0) {
      for (int ch = 0; ch < 8; ++ch) {
        const std::int64_t o = ch * kChunkBytes;
        const __m512i x0 = load512(xa + o);
        [[maybe_unused]] const __m512i x1 = C > 1 ? load512(xb + o) : x0;
        HEARTH_DOT(0, load512(q0 + gb + o));
        if constexpr (R > 1) {
          HEARTH_DOT(1, load512(q1 + gb + o));
          HEARTH_DOT(2, load512(q2 + gb + o));
          HEARTH_DOT(3, load512(q3 + gb + o));
        }
      }
    } else {
      // Low nibbles, raw & 0x0F = n, against activation chunk ch (values
      // 4ch..); high nibbles, raw & 0xF0 = 16 n, against chunk ch + 4 (values
      // 16 + 4ch..), summed apart and divided by 16 once per group: that sum
      // is a multiple of 16, so the shift is exact, and no chunk needs a
      // shift of its own.
      const __m512i m_lo = _mm512_set1_epi8(0x0F), m_hi = _mm512_set1_epi8(static_cast<char>(0xF0));
      [[maybe_unused]] __m512i h00 = _mm512_setzero_si512(), h01 = h00, h10 = h00, h11 = h00, h20 = h00, h21 = h00,
                               h30 = h00, h31 = h00;
#define HEARTH_NIB(i)                                                                     \
  do {                                                                                    \
    const __m512i raw_ = load512(q##i + gb + o);                                          \
    const __m512i lo_ = _mm512_and_si512(raw_, m_lo), hi_ = _mm512_and_si512(raw_, m_hi); \
    a##i##0 = _mm512_dpbusd_epi32(a##i##0, lo_, xl0);                                     \
    h##i##0 = _mm512_dpbusd_epi32(h##i##0, hi_, xh0);                                     \
    if constexpr (C > 1) {                                                                \
      a##i##1 = _mm512_dpbusd_epi32(a##i##1, lo_, xl1);                                   \
      h##i##1 = _mm512_dpbusd_epi32(h##i##1, hi_, xh1);                                   \
    }                                                                                     \
  } while (0)
      for (int ch = 0; ch < 4; ++ch) {
        const std::int64_t o = ch * kChunkBytes;
        const __m512i xl0 = load512(xa + o), xh0 = load512(xa + o + 4 * kChunkBytes);
        [[maybe_unused]] const __m512i xl1 = C > 1 ? load512(xb + o) : xl0;
        [[maybe_unused]] const __m512i xh1 = C > 1 ? load512(xb + o + 4 * kChunkBytes) : xh0;
        HEARTH_NIB(0);
        if constexpr (R > 1) {
          HEARTH_NIB(1);
          HEARTH_NIB(2);
          HEARTH_NIB(3);
        }
      }
#undef HEARTH_NIB
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
    }
#undef HEARTH_DOT
    const std::int64_t gs = g * kGroupBlocks;
    const __m512 dx0 = _mm512_load_ps(x.d(t, g));
    [[maybe_unused]] const __m512 dx1 = C > 1 ? _mm512_load_ps(x.d(t1, g)) : dx0;
    auto scales = [&](const std::uint16_t* d) {
      return _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i*>(d + gs)));
    };
    const __m512 dw0 = scales(d0);
    f00 = _mm512_add_ps(f00, block_terms(a00, dw0, dx0));
    if constexpr (C > 1) f01 = _mm512_add_ps(f01, block_terms(a01, dw0, dx1));
    if constexpr (R > 1) {
      const __m512 dw1 = scales(d1), dw2 = scales(d2), dw3 = scales(d3);
      f10 = _mm512_add_ps(f10, block_terms(a10, dw1, dx0));
      f20 = _mm512_add_ps(f20, block_terms(a20, dw2, dx0));
      f30 = _mm512_add_ps(f30, block_terms(a30, dw3, dx0));
      if constexpr (C > 1) {
        f11 = _mm512_add_ps(f11, block_terms(a11, dw1, dx1));
        f21 = _mm512_add_ps(f21, block_terms(a21, dw2, dx1));
        f31 = _mm512_add_ps(f31, block_terms(a31, dw3, dx1));
      }
    }
  }
  float* y0 = y + t * ldy + r;
  y0[0] = reduce_halving(f00);
  if constexpr (R > 1) {
    y0[1] = reduce_halving(f10);
    y0[2] = reduce_halving(f20);
    y0[3] = reduce_halving(f30);
  }
  if constexpr (C > 1) {
    float* y1 = y0 + ldy;
    y1[0] = reduce_halving(f01);
    if constexpr (R > 1) {
      y1[1] = reduce_halving(f11);
      y1[2] = reduce_halving(f21);
      y1[3] = reduce_halving(f31);
    }
  }
}

template <QType kType, int R>
void tokens(const PackedView& w, std::int64_t r, const PackedActs& x, std::int64_t t0, std::int64_t t1, float* y,
            std::int64_t ldy) {
  std::int64_t t = t0;
  for (; t + 2 <= t1; t += 2) tile<kType, R, 2>(w, r, x, t, y, ldy);
  for (; t < t1; ++t) tile<kType, R, 1>(w, r, x, t, y, ldy);
}

template <QType kType>
void run(const PackedView& w, const PackedActs& x, float* y, std::int64_t ldy, std::int64_t r0, std::int64_t r1) {
  // Decode (one token): one row at a time, so each thread reads its rows as
  // one sequential stream, which the hardware prefetcher follows. Tiles of 4
  // or 8 rows read 4 or 8 short streams at once (a row is 1-5 KB). Measured
  // in interleaved rounds:
  // 8-row tiles 1-7 % slower per round on all four 1B/3B files, 4-row tiles
  // 2-8 % slower on the two files tried.
  if (x.tokens() == 1) {
    for (std::int64_t r = r0; r < r1; ++r) tile<kType, 1, 1>(w, r, x, 0, y, ldy);
    return;
  }
  // A block of tokens' activations stays in L2 while the rows pass over it,
  // so each weight row is read from memory once per block (07 doc, section 3).
  constexpr std::int64_t kTokenBlock = 128;
  for (std::int64_t t0 = 0; t0 < x.tokens(); t0 += kTokenBlock) {
    const std::int64_t t1 =
        x.tokens() < t0 + kTokenBlock ? x.tokens() : t0 + kTokenBlock;  // no std::min here: see quantize
    std::int64_t r = r0;
    for (; r + 4 <= r1; r += 4) tokens<kType, 4>(w, r, x, t0, t1, y, ldy);
    for (; r < r1; ++r) tokens<kType, 1>(w, r, x, t0, t1, y, ldy);
  }
}

}  // namespace

void quantize_group_avx512(const float* x, int blocks, std::int8_t* q, float* d, std::int32_t* s) {
  // Chunk c of the group holds values 4c..4c+3 of every block: dword c of a
  // block's 32 codes goes to byte offset 64c + 4j.
  const __m256i chunk_offsets = _mm256_setr_epi32(0, 64, 128, 192, 256, 320, 384, 448);
  for (int j = 0; j < kGroupBlocks; ++j) {
    if (j >= blocks) {
      d[j] = 0.0f;
      s[j] = 0;
      _mm256_i32scatter_epi32(q + 4 * j, chunk_offsets, _mm256_setzero_si256(), 1);
      continue;
    }
    const float* xb = x + j * quant::kBlock;
    const __m512 v0 = _mm512_loadu_ps(xb), v1 = _mm512_loadu_ps(xb + 16);
    // The same steps as quant::quantize_act_q8_0 (ggml's rule): the maximum is
    // exact in any order, and roundscale rounds half to even as nearbyint does.
    // (Only a NaN activation, which no correct model produces, can make the
    // two differ: std::fmax skips a NaN, VMAXPS does not.)
    const float amax = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(v0), _mm512_abs_ps(v1)));
    const float dq = amax / 127.0f;
    // F16C conversions, which fp16.h's are tested against bit for bit; an
    // inline function from a shared header must not be compiled into this
    // AVX-512 file (the linker may then use this copy everywhere).
    d[j] = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtps_ph(_mm_set_ss(dq), _MM_FROUND_TO_NEAREST_INT)));
    const float m = amax != 0.0f ? 127.0f / amax : 0.0f;
    const __m512 mv = _mm512_set1_ps(m);
    const __m512i c0 =
        _mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(v0, mv), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    const __m512i c1 =
        _mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(v1, mv), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    s[j] = _mm512_reduce_add_epi32(_mm512_add_epi32(c0, c1));
    const __m256i codes = _mm256_set_m128i(_mm512_cvtepi32_epi8(c1), _mm512_cvtepi32_epi8(c0));
    _mm256_i32scatter_epi32(q + 4 * j, chunk_offsets, codes, 1);
  }
}

void qgemm_avx512(const PackedView& w, const PackedActs& x, float* y, std::int64_t ldy, std::int64_t r0,
                  std::int64_t r1) {
  if (w.type == QType::kQ8_0) {
    run<QType::kQ8_0>(w, x, y, ldy, r0, r1);
  } else {
    run<QType::kQ4_0>(w, x, y, ldy, r0, r1);
  }
}

}  // namespace hearth::cpu::detail
