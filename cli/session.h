// What `run`, `chat`, `ppl` and `dump` share: the model options, the loaded
// model, and the sampling options.
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include "api/engine_internal.h"
#include "args.h"
#include "core/cpu_features.h"
#include "core/thread_pool.h"
#include "gguf/gguf.h"
#include "hearth/engine.h"
#include "model/llama_ref.h"
#include "sampling/sampler.h"
#include "tokenizer/tokenizer.h"

namespace hearth::cli {

// --model F, --backend cpu|gpu|cpu-ref (default `default_backend`),
// --isa auto|scalar|avx512, --threads N, --ctx N (default `default_ctx`).
struct ModelOptions : api::LoadOptions {
  std::string backend;  // "cpu-ref", "cpu (avx512 kernels)" or "gpu"
};
ModelOptions model_options(Args& args, std::uint32_t default_ctx, const char* default_backend = "cpu");

// The loaded model: a hearth::Engine (src/api/engine.cpp) and references to
// its parts, which the measurement commands use directly.
struct Session {
  explicit Session(const ModelOptions& o);
  Engine engine;
  gguf::File& file;
  tokenizer::Tokenizer& tok;
  ThreadPool& pool;
  model::LlamaRef& llm;
};

// --draft F: speculative decoding with draft model F, with the measured
// defaults of docs/architecture/13-speculative.md (up to 8 draft tokens, the
// length chosen by measured cost, at most 512 new tokens read at once).
void draft_options(Args& args, ModelOptions& o);

// --temp T, --top-k K, --top-p P, --seed N. Without --seed a random seed is
// drawn; `seed_note` says which, so a run can be repeated.
sampling::Params sampling_options(Args& args, std::string& seed_note);

}  // namespace hearth::cli
