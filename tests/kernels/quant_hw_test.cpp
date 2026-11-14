// Our Q8_0 activation quantizer against ggml's own AVX2 instruction sequence
// (third_party/llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c,
// quantize_row_q8_0, __AVX2__ branch), reproduced here verbatim in intent.
// The result must be identical bit for bit: same scale bits, same quants.
// The sequence is in hw_reference_avx2.cpp; skipped on a CPU without AVX2.
#include <doctest/doctest.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "core/cpu_features.h"
#include "core/fp16.h"
#include "core/quant.h"
#include "hw_reference.h"

using hearth::test::ggml_quantize_block;

TEST_SUITE("quick") {
  TEST_CASE("quant: quantize_act_q8_0 is bit-identical to ggml's AVX2 sequence") {
    if (!hearth::isa_supported(hearth::Isa::kAvx2)) {
      MESSAGE("no AVX2; skipped");
      return;
    }
    std::mt19937 rng(2026);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_int_distribution<int> pick(0, 5);
    constexpr int kBlocks = 20000;
    std::vector<float> x(kBlocks * 32);
    for (int b = 0; b < kBlocks; ++b) {
      float* p = x.data() + b * 32;
      const int mode = pick(rng);
      const float scale = std::ldexp(1.0f, static_cast<int>(rng() % 40) - 20);  // magnitudes 2^-20 .. 2^19
      for (int j = 0; j < 32; ++j) {
        switch (mode) {
          case 0:
            p[j] = normal(rng) * scale;
            break;
          case 1:
            p[j] = 0.0f;
            break;
          case 2:
            p[j] = (static_cast<float>(static_cast<int>(rng() % 255)) - 127.0f + 0.5f) * scale;
            break;  // ties
          case 3:
            p[j] = -0.0f;
            break;
          case 4:
            p[j] = j == 7 ? -scale : 0.5f * scale;
            break;  // negative maximum
          default:
            p[j] = static_cast<float>(j) * scale;
            break;
        }
      }
    }
    std::vector<std::int8_t> q(x.size());
    std::vector<float> d(kBlocks);
    std::vector<std::int32_t> s(kBlocks);
    hearth::quant::quantize_act_q8_0(x.data(), static_cast<std::int64_t>(x.size()), q.data(), d.data(), s.data());
    for (int b = 0; b < kBlocks; ++b) {
      std::uint16_t want_d = 0;
      std::int8_t want_q[32];
      ggml_quantize_block(x.data() + b * 32, want_d, want_q);
      REQUIRE(hearth::fp32_to_fp16(d[b]) == want_d);
      REQUIRE(std::memcmp(q.data() + b * 32, want_q, 32) == 0);
      std::int32_t sum = 0;
      for (int j = 0; j < 32; ++j) sum += want_q[j];
      REQUIRE(s[b] == sum);
    }
  }
}
