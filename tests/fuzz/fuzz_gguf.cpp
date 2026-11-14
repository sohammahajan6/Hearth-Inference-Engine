// GGUF fuzz target.
//
// clang + ASan/UBSan build: a libFuzzer target (coverage-guided), run by
// scripts\fuzz.ps1. Any build: with HEARTH_LIBFUZZER undefined, a small main()
// replays the files or directories named on the command line through the
// same code, so crashes found by the fuzzer can be reproduced under MSVC too.
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

#include "fuzz/gguf_target.h"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  hearth::fuzz::gguf_one_input(data, size);
  return 0;
}

#if !defined(HEARTH_LIBFUZZER)
int main(int argc, char** argv) {
  std::size_t n = 0;
  auto replay = [&](const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    LLVMFuzzerTestOneInput(bytes.data(), bytes.size());
    ++n;
  };
  for (int i = 1; i < argc; ++i) {
    const std::filesystem::path p(argv[i]);
    if (std::filesystem::is_directory(p)) {
      for (const auto& e : std::filesystem::directory_iterator(p)) {
        if (e.is_regular_file()) replay(e.path());
      }
    } else {
      replay(p);
    }
  }
  std::printf("replayed %zu input(s) without a finding\n", n);
  return 0;
}
#endif
