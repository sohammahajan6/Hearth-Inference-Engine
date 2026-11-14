// Not public: what this repository's own tools see of an Engine
//. The measurement and correctness
// commands (bench, dump, ppl, info, tokenize) need the forward pass, the
// file and the tokenizer, and options the public API does not offer (the
// draft's reading limit).
#pragma once

#include <cstdint>
#include <string>

#include "core/cpu_features.h"
#include "core/thread_pool.h"
#include "gguf/gguf.h"
#include "hearth/engine.h"
#include "model/llama_ref.h"
#include "model/speculative.h"
#include "tokenizer/tokenizer.h"

namespace hearth::api {

// Everything an Engine can be loaded with; Engine::Options maps onto it.
struct LoadOptions {
  std::string model;  // UTF-8 path
  model::Matmul matmul = model::Matmul::kFp32;
  Isa isa = Isa::kScalar;
  int threads = 4;
  std::uint32_t ctx = 4096;
  bool gpu_fallback = false;
  std::string draft_model;  // UTF-8 path of the draft model; empty: no speculative decoding
  model::SpeculativeOptions speculative;
};

}  // namespace hearth::api

namespace hearth::detail {

struct EngineAccess {
  static Engine load(const api::LoadOptions& o);
  static const api::LoadOptions& options(const Engine& e);
  static gguf::File& file(Engine& e);
  static tokenizer::Tokenizer& tokenizer(Engine& e);
  static ThreadPool& pool(Engine& e);
  static model::LlamaRef& llama(Engine& e);
  // The draft model of speculative decoding, or nullptr.
  static model::LlamaRef* draft(Engine& e);
};

}  // namespace hearth::detail
