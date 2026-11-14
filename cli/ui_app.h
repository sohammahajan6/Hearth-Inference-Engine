// `hearth ui`: the app behind the local web page (docs/architecture/12-ui.md,
// ADR 0019). It owns the engine and runs one job at a time: loading a model
// on the CPU or the GPU, a chat reply, or a benchmark (reading and writing
// speed, as `hearth bench` measures them). ui::Server calls handle() on one
// thread per request; the security checks have already passed.
#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "ui/server.h"

namespace hearth::cli {

struct UiOptions {
  int threads = 4;
  std::uint32_t ctx = 4096;
  int idle_minutes = 10;  // quit after this long with no request and no job (0: never)
  // Where to look for *.gguf files, in order (duplicates are removed).
  std::vector<std::filesystem::path> model_dirs;
  // One JSON line per benchmark.
  std::filesystem::path results_file;
  // The model files added with "Add model", one path per line (they stay
  // where they are; only their paths are kept).
  std::filesystem::path models_file;
  // Speculative decoding when the
  // model's folder holds a suitable draft (find_draft).
  bool draft = true;
};

// The draft model for `model`, or an empty path: the smallest Q4_0 .gguf file
// in the same folder that is at most 30 % of the model's size and has exactly
// its vocabulary. Files that cannot be read are skipped. Decode reads every
// byte once, so the size ratio approximates the cost ratio of a draft step to
// a target step. Measured with the 1B Q4_0 as draft (bench/spec/2026-11-12/):
// the 3B Q8_0 (ratio 0.21) gains 1.7-2.0x greedy and 1.58x at temperature
// 0.7, the app's default; the 3B Q4_0 (ratio 0.39) 1.1x greedy but 0.98x at
// temperature 0.7, so it gets no draft; nor does the 1B Q8_0 (0.53).
std::filesystem::path find_draft(const std::filesystem::path& model);

// The folders a beginner's models are likely in: models\ next to hearth.exe,
// the repository's models\ (hearth.exe is in build\<preset>\cli\), and
// models\ in the current folder.
std::vector<std::filesystem::path> default_model_dirs();

class UiApp {
 public:
  explicit UiApp(UiOptions options);
  ~UiApp();
  UiApp(const UiApp&) = delete;
  UiApp& operator=(const UiApp&) = delete;

  void handle(const ui::Request& request, ui::Connection& connection);
  // False once the page asked to quit, the idle time passed, or Ctrl+C was
  // pressed with no job running. Stops the running job when it turns false.
  bool keep_running();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// The embedded page (cli/ui/index.html, see cmake/HearthEmbed.cmake).
std::string_view ui_index_html();

}  // namespace hearth::cli
