// Adversarial Q8_0 / Q4_0 weight matrices for the GPU kernel tests
// (kernels/gpu_qgemv_test.cpp, kernels/gpu_qgemm_test.cpp): scales that are
// subnormal, zero, negative zero, negative, large and plain; quants at the
// extremes (-128/127, nibbles 0/15) and all-zero blocks.
#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "core/fp16.h"
#include "core/quant.h"

namespace hearth::test {

// Scales: subnormal, zero, negative zero, negative, large and plain.
inline std::uint16_t scale(std::mt19937& rng, std::size_t i) {
  std::uniform_real_distribution<float> u(1e-4f, 0.05f);
  switch (i % 6) {
    case 0:
      return 0x0003;
    case 1:
      return 0x0000;
    case 2:
      return 0x8000;
    case 3:
      return static_cast<std::uint16_t>(fp32_to_fp16(u(rng)) | 0x8000u);
    case 4:
      return fp32_to_fp16(u(rng) * 1000.0f);
    default:
      return fp32_to_fp16(u(rng));
  }
}

inline std::vector<quant::BlockQ8_0> make_q8(std::int64_t rows, std::int64_t k, std::mt19937& rng) {
  std::vector<quant::BlockQ8_0> w(static_cast<std::size_t>(rows * (k / quant::kBlock)));
  std::uniform_int_distribution<int> q(-128, 127);
  for (std::size_t i = 0; i < w.size(); ++i) {
    w[i].d = scale(rng, i);
    for (int j = 0; j < quant::kBlock; ++j) {
      w[i].q[j] = static_cast<std::int8_t>(i % 5 == 1 ? ((j % 2) != 0 ? -128 : 127) : i % 5 == 2 ? 0 : q(rng));
    }
  }
  return w;
}

inline std::vector<quant::BlockQ4_0> make_q4(std::int64_t rows, std::int64_t k, std::mt19937& rng) {
  std::vector<quant::BlockQ4_0> w(static_cast<std::size_t>(rows * (k / quant::kBlock)));
  std::uniform_int_distribution<int> q(0, 255);
  for (std::size_t i = 0; i < w.size(); ++i) {
    w[i].d = scale(rng, i);
    for (int j = 0; j < quant::kBlock / 2; ++j) {
      w[i].q[j] = static_cast<std::uint8_t>(i % 5 == 1 ? 0xF0 : i % 5 == 2 ? 0xFF : i % 5 == 3 ? 0x00 : q(rng));
    }
  }
  return w;
}

}  // namespace hearth::test
