// GPU lanes in --trace:
// gpu::ClockMap maps GPU timestamp ticks onto the CPU's clock. The GPU
// pass's events themselves are checked in tests/unit/llama_ref_test.cpp
// ("gpu backend: --trace records every layer ...").
#include <doctest/doctest.h>

#include <cstdint>

#include "gpu/timestamps.h"

using namespace hearth;

TEST_SUITE("quick") {
  TEST_CASE("gpu ClockMap: ticks map onto the CPU clock across the counter's wrap") {
    // Iris Xe: 36 valid bits, 52.083 ns per tick. The reading is 100 ticks
    // before the wrap, at CPU time 1 s.
    const std::uint64_t range = std::uint64_t{1} << 36;
    const double period = 52.083;
    const gpu::ClockMap m(range - 100, 1'000'000'000, 40, period, 36);
    CHECK(m.to_cpu_ns(range - 100) == 1'000'000'000);
    // 300 ticks later the counter has wrapped to 200.
    CHECK(m.to_cpu_ns(200) == 1'000'000'000 + static_cast<std::int64_t>(300 * period));
    // 50 ticks before the reading.
    CHECK(m.to_cpu_ns(range - 150) == 1'000'000'000 - static_cast<std::int64_t>(50 * period));
    // Bits above the valid ones are ignored, on both sides.
    CHECK(m.to_cpu_ns((std::uint64_t{7} << 36) | 200) == m.to_cpu_ns(200));
    const gpu::ClockMap high((std::uint64_t{3} << 40) | (range - 100), 1'000'000'000, 40, period, 36);
    CHECK(high.to_cpu_ns(200) == m.to_cpu_ns(200));
    // The largest distances either way: just under half the range.
    const std::uint64_t half = range / 2;
    const gpu::ClockMap zero(0, 0, 0, 1.0, 36);
    CHECK(zero.to_cpu_ns(half - 1) == static_cast<std::int64_t>(half - 1));
    CHECK(zero.to_cpu_ns(half) == -static_cast<std::int64_t>(half));
    CHECK(zero.deviation_ns() == 0);
    CHECK(m.deviation_ns() == 40);
  }
}
