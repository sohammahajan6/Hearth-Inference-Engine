// Tokenizer fuzz target.
//
// clang + ASan/UBSan build: a libFuzzer target, run by scripts\fuzz.ps1
// -Target tokenizer. Any build: with HEARTH_LIBFUZZER undefined, main()
// replays files or directories through the same checks.
//
// The vocabulary is the real Llama-3.2 one, loaded once from
// %HEARTH_FUZZ_MODEL% or models\Llama-3.2-1B-Instruct-Q8_0-pure.gguf.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "core/error.h"
#include "core/platform.h"
#include "fuzz/tokenizer_target.h"
#include "gguf/gguf.h"

namespace {

std::filesystem::path model_path() {
  char* env = nullptr;
  std::size_t len = 0;
  std::string p;
  if (_dupenv_s(&env, &len, "HEARTH_FUZZ_MODEL") == 0 && env != nullptr) p = env;
  std::free(env);
  if (!p.empty()) return hearth::path_from_utf8(p);
  return std::filesystem::path(HEARTH_REPO_DIR) / "models" / "Llama-3.2-1B-Instruct-Q8_0-pure.gguf";
}

const hearth::tokenizer::Tokenizer& vocab() {
  static const hearth::gguf::File file = hearth::gguf::File::open(model_path());
  static const hearth::tokenizer::Tokenizer tok = hearth::tokenizer::Tokenizer::from_gguf(file);
  return tok;
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  try {
    hearth::fuzz::tokenizer_one_input(vocab(), data, size);
  } catch (const hearth::Error& e) {
    // tokenizer_one_input handles the expected errors itself: anything that
    // gets here (a failed check, a model that does not load) is a finding.
    std::fprintf(stderr, "finding: %s\n", e.what());
    std::abort();
  }
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
