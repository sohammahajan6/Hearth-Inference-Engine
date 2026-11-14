// Error reporting for the whole library.
//
// Failures are reported by throwing hearth::Error. The kind says who is at
// fault (the user's input, the model file, the GPU, or this program), so the
// CLI can print an actionable message and pick an exit code. Hot loops never
// throw: they are only entered after their inputs were validated.
#pragma once

#include <format>
#include <string>
#include <utility>

#include "hearth/error.h"  // ErrorKind and Error are part of the public API

namespace hearth {

template <class... Args>
[[noreturn]] void fail(ErrorKind kind, std::format_string<Args...> fmt, Args&&... args) {
  throw Error(kind, std::format(fmt, std::forward<Args>(args)...));
}

namespace detail {
[[noreturn]] void check_failed(const char* expr, const char* file, int line);
}  // namespace detail

}  // namespace hearth

// Invariant check, active in every build type. Use it for conditions that can
// only be false if this program has a bug. Input validation uses fail().
#define HEARTH_CHECK(cond)                                                  \
  do {                                                                      \
    if (!(cond)) ::hearth::detail::check_failed(#cond, __FILE__, __LINE__); \
  } while (0)
