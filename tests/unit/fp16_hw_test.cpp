// fp16 conversions checked bit for bit against the CPU's F16C instructions,
// which is what ggml uses on x86 (hw_reference_avx2.cpp); skipped on a CPU
// without F16C.
#include <doctest/doctest.h>

#include <bit>
#include <cstdint>
#include <random>

#include "core/cpu_features.h"
#include "core/fp16.h"
#include "hw_reference.h"

using hearth::test::hw_fp16_to_fp32;
using hearth::test::hw_fp32_to_fp16;

TEST_SUITE("quick") {
  TEST_CASE("fp16: matches F16C for all 65536 halves (fp16 -> fp32)") {
    if (!hearth::cpu_features().f16c) {
      MESSAGE("CPU has no F16C; skipped");
      return;
    }
    for (std::uint32_t h = 0; h < 0x10000u; ++h) {
      const auto half = static_cast<std::uint16_t>(h);
      REQUIRE(std::bit_cast<std::uint32_t>(hearth::fp16_to_fp32(half)) ==
              std::bit_cast<std::uint32_t>(hw_fp16_to_fp32(half)));
    }
  }

  TEST_CASE("fp16: matches F16C on ties, boundaries and 20 million random floats (fp32 -> fp16)") {
    if (!hearth::cpu_features().f16c) {
      MESSAGE("CPU has no F16C; skipped");
      return;
    }
    // Every half value, the midpoints between neighbours (rounding ties) and
    // the floats one ulp either side of each midpoint.
    for (std::uint32_t h = 0; h < 0x7c00u; ++h) {
      for (std::uint32_t sign : {0u, 0x8000u}) {
        const float lo = hearth::fp16_to_fp32(static_cast<std::uint16_t>(h | sign));
        const float hi = hearth::fp16_to_fp32(static_cast<std::uint16_t>((h + 1) | sign));
        const float mid =
            std::bit_cast<float>((std::bit_cast<std::uint32_t>(lo) + std::bit_cast<std::uint32_t>(hi)) / 2);
        for (float f : {lo, mid, std::bit_cast<float>(std::bit_cast<std::uint32_t>(mid) + 1),
                        std::bit_cast<float>(std::bit_cast<std::uint32_t>(mid) - 1)}) {
          REQUIRE(hearth::fp32_to_fp16(f) == hw_fp32_to_fp16(f));
        }
      }
    }
    std::mt19937_64 rng(1234);
    for (int i = 0; i < 20'000'000; ++i) {
      const float f = std::bit_cast<float>(static_cast<std::uint32_t>(rng()));  // every class, NaNs included
      const std::uint16_t got = hearth::fp32_to_fp16(f);
      const std::uint16_t want = hw_fp32_to_fp16(f);
      if (got != want) {
        FAIL("fp32 bits 0x" << std::hex << std::bit_cast<std::uint32_t>(f) << ": got 0x" << got << ", F16C 0x" << want);
      }
    }
  }
}
