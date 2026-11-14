// Causal attention of one (token, head) for the `cpu` backend, in exactly
// the arithmetic of the cpu-ref forward pass (model/llama_ref.cpp):
//   score_s = (+0 + q_0 k_s,0 + q_1 k_s,1 + ...) * scale    one rounded mul and add per d, increasing d
//   p       = ref::softmax(score_0 .. score_len-1)
//   out_d   = +0 + p_0 v_0,d + p_1 v_1,d + ...             one rounded mul and add per s, increasing s
// The K cache is stored in blocks of 16 positions: within a block, the 16
// positions of one dimension are contiguous, and the block's dimensions follow
// each other. So the AVX-512 version computes 16 positions' scores per
// instruction without changing any score's order of operations, and one score
// block reads head_dim * 64 contiguous bytes. (A plain transpose, positions
// contiguous per dimension for the whole context, put each of a block's
// head_dim rows on its own page at long contexts: attention was about 20 %
// slower at --ctx 4096 than at 1024; docs/architecture/07-fast-cpu-backend.md.)
// Both versions give the same bits, and the same bits as cpu-ref's attention.
#pragma once

#include <cstdint>

#include "core/cpu_features.h"

namespace hearth::cpu {

// Positions per K block.
inline constexpr std::int64_t kKBlock = 16;

// Where k_s,d of one KV head lies, in floats from the head's start.
constexpr std::int64_t k_index(std::int64_t s, std::int64_t d, std::int64_t head_dim) {
  return (s / kKBlock) * head_dim * kKBlock + d * kKBlock + s % kKBlock;
}

// Floats one KV head's K needs for `n_ctx` positions (whole blocks).
constexpr std::int64_t k_head_size(std::int64_t n_ctx, std::int64_t head_dim) {
  return (n_ctx + kKBlock - 1) / kKBlock * kKBlock * head_dim;
}

struct AttentionArgs {
  const float* q = nullptr;   // head_dim values
  const float* kt = nullptr;  // k_s,d at kt[k_index(s, d, head_dim)]
  const float* v = nullptr;   // v_s,d at v[s * v_stride + d]
  std::int64_t v_stride = 0;
  std::int64_t len = 0;  // positions 0 .. len - 1, len >= 1
  std::int64_t head_dim = 0;
  float scale = 1.0f;
  float* scores = nullptr;  // len floats of scratch
  float* out = nullptr;     // head_dim values
};

// kScalar on any x86-64; kAvx512 needs AVX-512 F (callers check isa_supported()).
void attend(Isa isa, const AttentionArgs& a);

namespace detail {
void attend_avx512(const AttentionArgs& a);
}  // namespace detail

}  // namespace hearth::cpu
