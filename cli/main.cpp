// hearth: command-line front end of the hearth_core library.
#include <windows.h>

#include <cstdio>
#include <exception>
#include <new>
#include <string>
#include <string_view>
#include <vector>

#include "commands.h"
#include "core/error.h"
#include "core/log.h"
#include "core/platform.h"
#include "core/trace.h"
#include "hearth/version.h"

namespace {

using hearth::ErrorKind;

struct Command {
  const char* name;
  int (*run)(hearth::cli::Args&);
  const char* summary;
  bool developer;  // a measurement or correctness tool of the test scripts
};

constexpr Command kCommands[] = {
    {"ui", hearth::cli::cmd_ui,
     "the app in its own window: chat, live speed metrics, benchmarks (also opens when hearth.exe is "
     "double-clicked)\n           [--port N] [--threads N] [--ctx N] [--no-browser] [--no-draft]",
     false},
    {"chat", hearth::cli::cmd_chat,
     "a conversation in this terminal, one turn per line: --model F [--system S] [--n-predict N]; /reset, /exit",
     false},
    {"run", hearth::cli::cmd_run,
     "generate text from one prompt: --model F (--prompt T | --file F) [--n-predict N] [--chat [--system S]]", false},
    {"bench", hearth::cli::cmd_bench,
     "reading (prefill) and writing (decode) speed, as llama-bench: --model F [--prompt 512] [--gen 128] "
     "[--reps 5]\n           [--counters]; speculative decoding on real text: --model F --draft D --prompts FILE",
     false},
    {"info", hearth::cli::cmd_info, "this PC's CPU, GPU and memory; or a model file: --model F [--tensors] [--json]",
     false},
    {"dump", hearth::cli::cmd_dump, "run a backend on token IDs and write named tensors (the oracle format)", true},
    {"ppl", hearth::cli::cmd_ppl, "perplexity with llama-perplexity's protocol: --model F --file F [--chunks N]", true},
    {"tokenize", hearth::cli::cmd_tokenize,
     "text to token IDs, as llama-tokenize --ids: --model F (--text T | --file F) [--no-bos] [--chat]", true},
    {"probe", hearth::cli::cmd_probe, "measure memory bandwidth, latency and compute limits", true},
};

void print_usage() {
  std::printf(
      "usage: hearth <command> [options]\n"
      "       hearth --version\n\n"
      "commands:\n");
  for (const Command& c : kCommands) {
    if (!c.developer) std::printf("  %-8s %s\n", c.name, c.summary);
  }
  std::printf("\ndeveloper tools (used by the test scripts):\n");
  for (const Command& c : kCommands) {
    if (c.developer) std::printf("  %-8s %s\n", c.name, c.summary);
  }
  std::printf(
      "\nmodel options (chat, run, bench, dump, ppl):\n"
      "  --model F          a GGUF file: Llama 3.2 or Qwen2.5; Q8_0, Q4_0, Q4_K_M, F16, BF16 or F32\n"
      "  --backend cpu|gpu  where the model runs (default cpu); gpu needs a Q8_0 or Q4_0 file\n"
      "  --threads N        CPU threads (default 4)    --ctx N   context length in tokens\n"
      "  --draft F          chat, run: a smaller model with the same vocabulary drafts words that the main\n"
      "                     model checks in one pass (speculative decoding; cpu backend). Greedy output is\n"
      "                     unchanged, and sampled output has the same distribution\n"
      "sampling (chat, run):\n"
      "  --temp T (0.8; 0 is greedy)  --top-k K (40)  --top-p P (0.95)  --seed N (default: random, printed)\n"
      "developer options:\n"
      "  --backend cpu-ref  the plain FP32 reference path (slow; the yardstick in the tests)\n"
      "  --isa auto|scalar|avx512   the CPU kernels (all give identical results)\n"
      "  --no-fallback      chat, run with gpu: fail if the GPU fails, instead of continuing on the CPU\n"
      "  --date YYYY-MM-DD  chat, run, tokenize: the date in the chat template (default: today)\n"
      "  run --tokens IDS --ids --special   token IDs in and out; control tokens printed as text\n"
      "  ui --idle-minutes N --url-file F --results F   for the app's end-to-end check\n"
      "  --log-level error|warn|info|debug|trace (default info)\n"
      "  --trace F.json     record timed operations in Chrome trace format (https://ui.perfetto.dev)\n");
}

void print_version() {
  const hearth::BuildInfo& b = hearth::build_info();
  std::printf("hearth %s\ncommit:   %s\ncompiler: %s\nbuild:    %s%s\nflags:    %s\n", b.version, b.commit, b.compiler,
              b.build_type, b.sanitized ? " (ASan+UBSan)" : "", b.flags);
}

// Exit codes, so scripts can tell failure classes apart without parsing text.
int exit_code(ErrorKind kind) {
  switch (kind) {
    case ErrorKind::kInvalidArgument:
      return 2;
    case ErrorKind::kBadModel:
      return 3;
    case ErrorKind::kUnsupported:
      return 4;
    case ErrorKind::kIo:
      return 5;
    case ErrorKind::kGpu:
      return 6;
    case ErrorKind::kDeviceLost:
      return 7;
    case ErrorKind::kOutOfMemory:
      return 8;
    case ErrorKind::kCancelled:
      return 130;
    case ErrorKind::kInternal:
      return 70;
  }
  return 1;
}

// The command line after the program name; returns the exit code.
int run(const std::vector<std::string>& tokens);

}  // namespace

int main(int argc, char** argv) {
  hearth::init_process();
  hearth::install_ctrl_c_handler();
  std::vector<std::string> tokens(argv + 1, argv + argc);
  // Double-clicked in Explorer: Windows made a console just for this process,
  // so it is the only one on it. Start the app instead of printing the usage
  // into a window that closes at once. From a terminal nothing changes.
  bool double_clicked = false;
  if (tokens.empty()) {
    DWORD ids[2] = {};
    double_clicked = GetConsoleProcessList(ids, 2) == 1;
    if (double_clicked) tokens.emplace_back("ui");
  }
  const int code = run(tokens);
  if (double_clicked && code != 0) {
    // The console closes with the process: keep the error readable.
    std::printf("\nHearth could not start (the reason is above). Press Enter to close this window.\n");
    std::fflush(stdout);
    static_cast<void>(std::getchar());
  }
  return code;
}

namespace {

int run(const std::vector<std::string>& tokens) {
  if (tokens.empty()) {
    print_usage();
    return 2;
  }
  const std::string& name = tokens.front();
  if (name == "--help" || name == "-h" || name == "help") {
    print_usage();
    return 0;
  }
  if (name == "--version") {
    print_version();
    return 0;
  }
  try {
    hearth::cli::Args args(std::vector<std::string>(tokens.begin() + 1, tokens.end()));
    if (auto level = args.value("log-level")) {
      hearth::LogLevel l;
      if (!hearth::parse_log_level(*level, l)) {
        hearth::fail(ErrorKind::kInvalidArgument, "--log-level must be error, warn, info, debug or trace");
      }
      hearth::set_log_level(l);
    }
    const auto trace_path = args.value("trace");
    if (trace_path) hearth::trace::start();
    // The trace is written whether the command succeeds or fails: a trace of a
    // failing run is often the one that is needed.
    auto write_trace = [&] {
      if (!trace_path) return;
      hearth::trace::write(hearth::path_from_utf8(*trace_path));
      hearth::log_info("trace: wrote {} (open it in https://ui.perfetto.dev)", *trace_path);
    };
    for (const Command& c : kCommands) {
      if (name != c.name) continue;
      int code = 0;
      try {
        code = c.run(args);
      } catch (...) {
        try {
          write_trace();
        } catch (const std::exception& e) {
          hearth::log_error("{}", e.what());  // the command's own error follows
        }
        throw;
      }
      write_trace();
      return code;
    }
    hearth::fail(ErrorKind::kInvalidArgument, "unknown command '{}' (see `hearth help`)", name);
  } catch (const hearth::Error& e) {
    hearth::log_error("{}", e.what());
    return exit_code(e.kind());
  } catch (const std::bad_alloc&) {
    hearth::log_error("out of memory (a smaller --ctx or a shorter prompt needs less)");
    return exit_code(ErrorKind::kOutOfMemory);
  } catch (const std::exception& e) {
    hearth::log_error("unexpected failure: {}", e.what());
    return 70;
  }
}

}  // namespace
