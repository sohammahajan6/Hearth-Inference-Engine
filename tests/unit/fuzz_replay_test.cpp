// Replays the fuzzers' seeds and every saved regression input through the
// fuzz targets on each test run, under both compilers. An input that once
// crashed the code can never crash it again unnoticed.
#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <vector>

#include "fuzz/gguf_target.h"
#include "fuzz/http_target.h"
#include "fuzz/tokenizer_target.h"
#include "gguf/gguf.h"
#include "test_util.h"

namespace {

std::size_t replay(const char* target, const std::function<void(const std::vector<std::uint8_t>&)>& run) {
  std::size_t n = 0;
  for (const char* kind : {"seeds", "regressions"}) {
    const auto dir = hearth::test::repo_dir() / "tests" / "fuzz" / kind / target;
    if (!std::filesystem::exists(dir)) continue;
    for (const auto& e : std::filesystem::directory_iterator(dir)) {
      if (!e.is_regular_file()) continue;
      std::ifstream f(e.path(), std::ios::binary);
      const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
      CAPTURE(e.path().string());
      run(bytes);
      ++n;
    }
  }
  return n;
}

}  // namespace

TEST_SUITE("quick") {
  TEST_CASE("fuzz: GGUF seeds and regression inputs replay without a finding") {
    const std::size_t n = replay("gguf", [](const std::vector<std::uint8_t>& b) {
      CHECK_NOTHROW(hearth::fuzz::gguf_one_input(b.data(), b.size()));
    });
    CHECK(n > 0);
  }

  TEST_CASE("fuzz: HTTP request seeds and regression inputs replay without a finding") {
    const std::size_t n = replay("http", [](const std::vector<std::uint8_t>& b) {
      CHECK_NOTHROW(hearth::fuzz::http_one_input(b.data(), b.size()));
    });
    CHECK(n >= 5);
  }
}

TEST_SUITE("model") {
  TEST_CASE("fuzz: tokenizer seeds and regression inputs replay without a finding") {
    const auto path = hearth::test::repo_dir() / "models" / "Llama-3.2-1B-Instruct-Q8_0-pure.gguf";
    if (!std::filesystem::exists(path)) {
      MESSAGE("missing (run scripts\\fetch-models.ps1): Llama-3.2-1B-Instruct-Q8_0-pure.gguf");
      return;
    }
    const hearth::gguf::File f = hearth::gguf::File::open(path);
    const auto tok = hearth::tokenizer::Tokenizer::from_gguf(f);
    const std::size_t n = replay("tokenizer", [&](const std::vector<std::uint8_t>& b) {
      CHECK_NOTHROW(hearth::fuzz::tokenizer_one_input(tok, b.data(), b.size()));
    });
    CHECK(n > 0);
  }
}
