#include "core/clock.h"

#include <windows.h>

namespace hearth {

std::int64_t qpc_frequency() noexcept {
  static const std::int64_t freq = [] {
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);  // cannot fail on Windows XP and later
    return static_cast<std::int64_t>(f.QuadPart);
  }();
  return freq;
}

std::int64_t now_ns() noexcept {
  LARGE_INTEGER c;
  QueryPerformanceCounter(&c);
  const std::int64_t f = qpc_frequency();
  const std::int64_t ticks = c.QuadPart;
  // Split to avoid overflow of ticks * 1e9 (overflows after ~15 minutes at 10 MHz otherwise).
  return (ticks / f) * 1'000'000'000 + (ticks % f) * 1'000'000'000 / f;
}

}  // namespace hearth
