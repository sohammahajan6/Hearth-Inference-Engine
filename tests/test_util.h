// Helpers shared by the tests.
#pragma once

#include <cstdlib>
#include <filesystem>
#include <string>

#include "gpu/context.h"

namespace hearth::test {

// Whether this build runs under AddressSanitizer (the clang `asan` preset).
// ASan commits shadow memory, one byte for every 8 bytes of each address range
// the process touches, so process commit is not the engine's commit there.
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define HEARTH_TEST_ASAN 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__)
#define HEARTH_TEST_ASAN 1
#endif
#ifdef HEARTH_TEST_ASAN
inline constexpr bool kAddressSanitizer = true;
#else
inline constexpr bool kAddressSanitizer = false;
#endif

inline std::filesystem::path repo_dir() { return std::filesystem::path(HEARTH_REPO_DIR); }

inline bool env_is_set(const char* name) {
  char* value = nullptr;
  std::size_t len = 0;
  const bool set = _dupenv_s(&value, &len, name) == 0 && value != nullptr && value[0] != '\0' && value[0] != '0';
  std::free(value);
  return set;
}

// scripts/test.ps1 sets HEARTH_TEST_VALIDATION=1 once the validation layer is
// built, so every GPU test then runs under it and must produce no messages.
inline gpu::ContextOptions gpu_test_options() {
  gpu::ContextOptions o;
  o.validation = env_is_set("HEARTH_TEST_VALIDATION");
  return o;
}

}  // namespace hearth::test
