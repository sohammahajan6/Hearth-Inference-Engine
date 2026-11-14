// The hearth library's public API: load a Llama-3.2 or Qwen2.5 GGUF model and
// generate text on the CPU or on the GPU (Vulkan)
//.
//
//   hearth::Engine::Options o;
//   o.model = "models/Llama-3.2-1B-Instruct-Q8_0-pure.gguf";
//   hearth::Engine engine(o);
//   const hearth::ChatMessage m{"user", "What is the capital of France?"};
//   const auto prompt = engine.chat_prompt("", std::span(&m, 1));
//   hearth::GenerateOptions g;
//   g.temperature = 0;  // greedy
//   const auto r = engine.generate(prompt, g, [](std::int32_t, std::string_view text) { std::fputs(...); });
//
// Every failure throws hearth::Error (hearth/error.h). An Engine is not
// thread-safe: use it from one thread at a time.
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "hearth/error.h"

namespace hearth {

namespace detail {
struct EngineAccess;  // the library's own tools (src/api/engine_internal.h)
}  // namespace detail

enum class Backend {
  kCpuRef,  // plain FP32 reference, slow; for checking
  kCpu,     // the fast CPU backend (quantized activations, AVX-512 when available)
  kGpu,     // the whole forward pass on the GPU (Q8_0/Q4_0 files)
};

enum class Kernels { kAuto, kScalar, kAvx512 };  // the CPU kernels; all give the same results

enum class StopReason {
  kEndOfGeneration,  // the model ended its reply
  kLimit,            // max_new_tokens were produced
  kContextFull,      // no position left in the context
  kCancelled,        // Ctrl+C
};

struct ChatMessage {
  std::string role;  // "user" or "assistant"
  std::string content;
};

struct GenerateOptions {
  int max_new_tokens = 128;
  float temperature = 0.8f;  // <= 0: greedy (always the most likely token)
  int top_k = 40;            // <= 0: no top-k
  float top_p = 0.95f;       // >= 1: no top-p
  std::uint64_t seed = 0;    // the same seed and options repeat the same text
  // The random sequence continues across generate() calls with equal sampling
  // options (a multi-turn chat draws on as one sequence); other options start a new one.
  bool special_text = false;  // also pass the text of control tokens to the callback
};

struct GenerateResult {
  int prompt_tokens = 0;
  double prefill_seconds = 0;
  int generated_tokens = 0;  // passed to the callback; an end-of-generation token is not counted
  double decode_seconds = 0;
  StopReason stop = StopReason::kLimit;
  std::int32_t end_token = -1;  // the end-of-generation token, with StopReason::kEndOfGeneration
  std::string trailing_text;    // bytes of an incomplete UTF-8 character still buffered at the end (usually empty)
  int draft_tokens = 0;         // with a draft model: tokens it proposed
  int draft_accepted = 0;       // with a draft model: proposed tokens that were kept
};

class Engine {
 public:
  struct Options {
    std::filesystem::path model;  // a Llama-3.2 or Qwen2.5 GGUF file (F32, F16, BF16, Q8_0, Q4_0, Q4_K_M)
    Backend backend = Backend::kCpu;
    int threads = 4;               // CPU threads, 1..64
    std::uint32_t context = 4096;  // positions, up to the model's trained context
    Kernels kernels = Kernels::kAuto;
    // kGpu: without a Vulkan GPU, or after the GPU fails, continue on the CPU
    // (the kCpu backend's results) instead of throwing. Measurements should
    // leave this off.
    bool gpu_fallback = false;
    // Speculative decoding: a smaller
    // model with the same vocabulary (the 1B for the 3B) drafts up to
    // draft_max tokens, and this model checks them all in one pass. Greedy
    // output is exactly the same as without a draft; sampled output has the
    // same distribution. The draft runs on the CPU (cpu backend, the same
    // threads); the main model's backend must be kCpu or kCpuRef.
    std::filesystem::path draft_model;  // empty: no draft
    int draft_max = 8;                  // draft tokens per pass, at most (1..16)
    // When drafting stops before draft_max. Negative (default): automatic,
    // by the draft's confidence weighed against the measured costs of a draft
    // step and of a verified token. 0: never. 0 < P <= 1: when the draft's
    // most likely token has a probability below P.
    float draft_p_min = -1.0f;
  };

  // Loads the model and prepares the backend. Throws kInvalidArgument for bad
  // options, kBadModel or kUnsupported for the file, kGpu for a GPU problem.
  explicit Engine(const Options& options);
  ~Engine();
  Engine(Engine&&) noexcept;
  Engine& operator=(Engine&&) noexcept;
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // Token IDs of plain text (with the beginning-of-text token if add_bos).
  std::vector<std::int32_t> tokenize(std::string_view text, bool add_bos = true) const;
  // A chat prompt in the model's template: the system block (with `date` as
  // "YYYY-MM-DD", empty: today), every message, and the assistant header.
  // kUnsupported if the model's template is not Llama 3.1/3.2's.
  std::vector<std::int32_t> chat_prompt(std::string_view system, std::span<const ChatMessage> messages,
                                        std::string_view date = {}) const;
  // What a further user turn appends after a reply that generate() produced.
  std::vector<std::int32_t> chat_next_turn(std::string_view user) const;

  // Feeds `prompt` after what the context already holds, then samples up to
  // max_new_tokens tokens. For each one, on_token gets its ID and the text it
  // completes (whole UTF-8 characters; possibly empty). Generated tokens stay
  // in the context, so a later call continues the conversation.
  using TokenCallback = std::function<void(std::int32_t id, std::string_view text)>;
  GenerateResult generate(std::span<const std::int32_t> prompt, const GenerateOptions& options,
                          const TokenCallback& on_token);

  void reset();  // empties the context
  std::uint32_t context_used() const;
  std::uint32_t context_size() const;

  std::string backend_description() const;  // e.g. "cpu (avx512 kernels)" or "gpu"
  std::string gpu_device() const;           // the GPU in use, or empty
  std::string draft_description() const;    // the draft model and its settings, or empty
  bool gpu_lost() const;                    // the GPU failed and the engine continued on the CPU

 private:
  friend struct detail::EngineAccess;
  struct Impl;
  explicit Engine(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace hearth
