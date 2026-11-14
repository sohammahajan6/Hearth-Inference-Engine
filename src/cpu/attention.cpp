// The scalar attention kernel and dispatch. Compiled for baseline x86-64
// (no FMA), like model/llama_ref.cpp, whose loops this one repeats.
#include "cpu/attention.h"

#include <algorithm>

#include "core/error.h"
#include "cpu/ref_ops.h"

namespace hearth::cpu {

void attend(Isa isa, const AttentionArgs& a) {
  HEARTH_CHECK(a.len >= 1 && a.head_dim >= 1);
  if (isa == Isa::kAvx512) return detail::attend_avx512(a);
  HEARTH_CHECK(isa == Isa::kScalar);
  for (std::int64_t s = 0; s < a.len; ++s) {
    float dot = 0.0f;
    for (std::int64_t d = 0; d < a.head_dim; ++d) dot += a.q[d] * a.kt[k_index(s, d, a.head_dim)];
    a.scores[s] = dot * a.scale;
  }
  ref::softmax(a.scores, a.len);
  std::fill(a.out, a.out + a.head_dim, 0.0f);
  for (std::int64_t s = 0; s < a.len; ++s) {
    const float p = a.scores[s];
    const float* vs = a.v + s * a.v_stride;
    for (std::int64_t d = 0; d < a.head_dim; ++d) a.out[d] += p * vs[d];
  }
}

}  // namespace hearth::cpu
