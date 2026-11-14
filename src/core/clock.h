// Monotonic wall-clock time for timing and traces.
#pragma once

#include <cstdint>

namespace hearth {

// Nanoseconds since an arbitrary fixed point, from QueryPerformanceCounter.
// Resolution on this machine is 100 ns (QPC runs at 10 MHz); docs/HARDWARE.md.
std::int64_t now_ns() noexcept;

// QueryPerformanceCounter ticks per second.
std::int64_t qpc_frequency() noexcept;

}  // namespace hearth
