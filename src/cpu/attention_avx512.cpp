// AVX-512 attention kernel (cpu/attention.h). Compiled with AVX-512 flags;
// called only after dispatch confirmed AVX-512 support.
//
// Scores: lane i of one register is position s0 + i, so the loop over d does
// for each position exactly the scalar mul and add, in the same order.
// Output: lane i is dimension d0 + i, accumulated over s in increasing order.
// Separate _mm512_mul_ps and _mm512_add_ps (never FMA) keep every rounding
// identical to the scalar loops.
#include <immintrin.h>

#include "cpu/attention.h"
#include "cpu/ref_ops.h"

namespace hearth::cpu::detail {
namespace {

inline __mmask16 first(std::int64_t n) {
  return n >= 16 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>((1u << n) - 1u);
}

}  // namespace

void attend_avx512(const AttentionArgs& a) {
  const __m512 scale = _mm512_set1_ps(a.scale);
  for (std::int64_t s0 = 0; s0 < a.len; s0 += 16) {
    const __mmask16 m = first(a.len - s0);
    const float* kb = a.kt + k_index(s0, 0, a.head_dim);
    __m512 acc = _mm512_setzero_ps();
    for (std::int64_t d = 0; d < a.head_dim; ++d) {
      const __m512 k = _mm512_maskz_loadu_ps(m, kb + d * kKBlock);
      acc = _mm512_add_ps(acc, _mm512_mul_ps(_mm512_set1_ps(a.q[d]), k));
    }
    _mm512_mask_storeu_ps(a.scores + s0, m, _mm512_mul_ps(acc, scale));
  }
  ref::softmax(a.scores, a.len);
  // Four registers (64 dimensions) per pass over the positions, then single ones.
  std::int64_t d0 = 0;
  for (; d0 + 64 <= a.head_dim; d0 += 64) {
    __m512 o0 = _mm512_setzero_ps(), o1 = _mm512_setzero_ps(), o2 = _mm512_setzero_ps(), o3 = _mm512_setzero_ps();
    for (std::int64_t s = 0; s < a.len; ++s) {
      const __m512 p = _mm512_set1_ps(a.scores[s]);
      const float* vs = a.v + s * a.v_stride + d0;
      o0 = _mm512_add_ps(o0, _mm512_mul_ps(p, _mm512_loadu_ps(vs)));
      o1 = _mm512_add_ps(o1, _mm512_mul_ps(p, _mm512_loadu_ps(vs + 16)));
      o2 = _mm512_add_ps(o2, _mm512_mul_ps(p, _mm512_loadu_ps(vs + 32)));
      o3 = _mm512_add_ps(o3, _mm512_mul_ps(p, _mm512_loadu_ps(vs + 48)));
    }
    _mm512_storeu_ps(a.out + d0, o0);
    _mm512_storeu_ps(a.out + d0 + 16, o1);
    _mm512_storeu_ps(a.out + d0 + 32, o2);
    _mm512_storeu_ps(a.out + d0 + 48, o3);
  }
  for (; d0 < a.head_dim; d0 += 16) {
    const __mmask16 m = first(a.head_dim - d0);
    __m512 o = _mm512_setzero_ps();
    for (std::int64_t s = 0; s < a.len; ++s) {
      const __m512 p = _mm512_set1_ps(a.scores[s]);
      o = _mm512_add_ps(o, _mm512_mul_ps(p, _mm512_maskz_loadu_ps(m, a.v + s * a.v_stride + d0)));
    }
    _mm512_mask_storeu_ps(a.out + d0, m, o);
  }
}

}  // namespace hearth::cpu::detail
