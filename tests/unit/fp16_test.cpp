// fp16/bf16 conversions: exact properties that hold without special hardware.
#include <doctest/doctest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

#include "core/fp16.h"

using hearth::bf16_to_fp32;
using hearth::fp16_to_fp32;
using hearth::fp32_to_fp16;

TEST_SUITE("quick") {
  TEST_CASE("fp16: every finite half round-trips exactly through fp32") {
    for (std::uint32_t h = 0; h < 0x10000u; ++h) {
      const auto half = static_cast<std::uint16_t>(h);
      const std::uint32_t exp = (h >> 10) & 0x1fu;
      if (exp == 31 && (h & 0x3ffu) != 0) continue;  // NaN payloads are covered by the hardware test
      const float f = fp16_to_fp32(half);
      REQUIRE(fp32_to_fp16(f) == half);
    }
  }

  TEST_CASE("fp16: known values") {
    CHECK(fp16_to_fp32(0x3c00) == 1.0f);
    CHECK(fp16_to_fp32(0xc000) == -2.0f);
    CHECK(fp16_to_fp32(0x7bff) == 65504.0f);               // largest finite half
    CHECK(fp16_to_fp32(0x0001) == std::ldexp(1.0f, -24));  // smallest subnormal
    CHECK(fp16_to_fp32(0x0400) == std::ldexp(1.0f, -14));  // smallest normal
    CHECK(std::isinf(fp16_to_fp32(0x7c00)));
    CHECK(std::isnan(fp16_to_fp32(0x7e00)));
    CHECK(std::bit_cast<std::uint32_t>(fp16_to_fp32(0x7c01)) == 0x7fc02000u);  // signalling NaN comes out quiet
    CHECK(std::bit_cast<std::uint32_t>(fp16_to_fp32(0x8000)) == 0x80000000u);  // -0 keeps its sign
  }

  TEST_CASE("fp16: rounding is to nearest, ties to even") {
    // 1 + 2^-11 lies exactly halfway between 1 (even mantissa) and 1 + 2^-10.
    CHECK(fp32_to_fp16(1.0f + std::ldexp(1.0f, -11)) == 0x3c00);
    // 1 + 3*2^-11 lies halfway between 1 + 2^-10 (odd) and 1 + 2^-9 (even): rounds up.
    CHECK(fp32_to_fp16(1.0f + 3.0f * std::ldexp(1.0f, -11)) == 0x3c02);
    // Just above the halfway point rounds up.
    CHECK(fp32_to_fp16(std::nextafter(1.0f + std::ldexp(1.0f, -11), 2.0f)) == 0x3c01);
    // 65520 is halfway between 65504 and 65536 (which is out of range): rounds to infinity.
    CHECK(fp32_to_fp16(65519.996f) == 0x7bff);
    CHECK(fp32_to_fp16(65520.0f) == 0x7c00);
    // Halfway between 0 and the smallest subnormal rounds to 0 (even).
    CHECK(fp32_to_fp16(std::ldexp(1.0f, -25)) == 0x0000);
    CHECK(fp32_to_fp16(std::nextafter(std::ldexp(1.0f, -25), 1.0f)) == 0x0001);
    CHECK(fp32_to_fp16(-std::numeric_limits<float>::infinity()) == 0xfc00);
  }

  TEST_CASE("bf16: conversion is the top 16 bits") {
    CHECK(bf16_to_fp32(0x3f80) == 1.0f);
    CHECK(bf16_to_fp32(0xc0a0) == -5.0f);
    CHECK(std::bit_cast<std::uint32_t>(bf16_to_fp32(0x1234)) == 0x12340000u);
  }
}
