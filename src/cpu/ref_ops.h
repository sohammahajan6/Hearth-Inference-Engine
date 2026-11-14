// The reference (cpu-ref) element-wise operations of the Llama forward pass.
//
// Plain loops in a fixed order, written to be obviously correct. Where ggml
// uses a wider accumulator, so do these (sums of squares and softmax
// normalisers in double), because those choices change results at the
// 1e-6 level the FP32 gates resolve.
#pragma once

#include <cstdint>
#include <vector>

namespace hearth::cpu::ref {

// out = x / sqrt(mean(x^2) + eps) * w      (w may be null: no weight)
void rmsnorm(const float* x, const float* w, float* out, std::int64_t n, float eps);

// Rotates `heads` vectors of `head_dim` floats in place by position `pos`:
// pairs (2i, 2i+1), or (i, i + rope_dim/2) with `neox` (llama.cpp's
// LLAMA_ROPE_TYPE_NEOX, used for Qwen2), angle pos * theta^(-2i/rope_dim) /
// factors[i] (factors may be null). Angles and their cos/sin are computed in
// double; the rotation is ggml's: x0 c - x1 s, x0 s + x1 c.
void rope(float* x, std::int64_t heads, std::int64_t head_dim, std::int64_t rope_dim, std::int64_t pos, float theta,
          const float* factors, bool neox = false);

// The (cos, sin) pairs rope() uses, for positions 0 .. n_pos - 1:
// [pos][rope_dim / 2][2], computed exactly as rope() computes them. The GPU
// forward pass reads this table, so its rotations equal rope()'s bits.
std::vector<float> rope_table(std::int64_t n_pos, std::int64_t rope_dim, float theta, const float* factors);

// In-place softmax of n scores.
void softmax(float* x, std::int64_t n);

// silu(x) = x / (1 + e^-x)
float silu(float x);

}  // namespace hearth::cpu::ref
