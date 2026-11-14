#include "cpu/ref_ops.h"

#include <cmath>

namespace hearth::cpu::ref {

void rmsnorm(const float* x, const float* w, float* out, std::int64_t n, float eps) {
  double sum = 0;
  for (std::int64_t i = 0; i < n; ++i) sum += static_cast<double>(x[i] * x[i]);
  const float mean = static_cast<float>(sum / static_cast<double>(n));
  const float scale = 1.0f / std::sqrt(mean + eps);
  for (std::int64_t i = 0; i < n; ++i) {
    const float v = x[i] * scale;
    out[i] = w != nullptr ? v * w[i] : v;
  }
}

namespace {

// The rotation of pair i at position pos, in double, rounded to float once.
void rope_angle(std::int64_t pos, std::int64_t i, std::int64_t rope_dim, float theta, const float* factors, float& c,
                float& s) {
  double angle = static_cast<double>(pos) *
                 std::pow(static_cast<double>(theta), -2.0 * static_cast<double>(i) / static_cast<double>(rope_dim));
  if (factors != nullptr) angle /= static_cast<double>(factors[i]);
  c = static_cast<float>(std::cos(angle));
  s = static_cast<float>(std::sin(angle));
}

}  // namespace

std::vector<float> rope_table(std::int64_t n_pos, std::int64_t rope_dim, float theta, const float* factors) {
  std::vector<float> t(static_cast<std::size_t>(n_pos * rope_dim));
  for (std::int64_t pos = 0; pos < n_pos; ++pos) {
    for (std::int64_t i = 0; i < rope_dim / 2; ++i) {
      float* cs = &t[static_cast<std::size_t>(pos * rope_dim + 2 * i)];
      rope_angle(pos, i, rope_dim, theta, factors, cs[0], cs[1]);
    }
  }
  return t;
}

void rope(float* x, std::int64_t heads, std::int64_t head_dim, std::int64_t rope_dim, std::int64_t pos, float theta,
          const float* factors, bool neox) {
  const std::int64_t first = neox ? 1 : 2, other = neox ? rope_dim / 2 : 1;  // pair i: (first * i, first * i + other)
  for (std::int64_t i = 0; i < rope_dim / 2; ++i) {
    float c = 0, s = 0;
    rope_angle(pos, i, rope_dim, theta, factors, c, s);
    for (std::int64_t h = 0; h < heads; ++h) {
      float* v = x + h * head_dim + first * i;
      const float x0 = v[0];
      const float x1 = v[other];
      v[0] = x0 * c - x1 * s;
      v[other] = x0 * s + x1 * c;
    }
  }
}

void softmax(float* x, std::int64_t n) {
  float max = x[0];
  for (std::int64_t i = 1; i < n; ++i) max = std::fmax(max, x[i]);
  double sum = 0;
  for (std::int64_t i = 0; i < n; ++i) {
    x[i] = std::exp(x[i] - max);
    sum += static_cast<double>(x[i]);
  }
  const auto inv = static_cast<float>(1.0 / sum);
  for (std::int64_t i = 0; i < n; ++i) x[i] *= inv;
}

float silu(float x) { return x / (1.0f + std::exp(-x)); }

}  // namespace hearth::cpu::ref
