// ADR 0016 follow-up (08 doc step 6.7): on the real models, `--backend gpu`
// computes the same logits for a token whether it arrives in a prefill chunk
// or alone in decode, bit for bit. The greedy gate exercises the decode path
// and the KLD and perplexity gates the prefill path; this test ties the two
// together, so a decode-only defect (a KV-cache boundary, a chunk edge) cannot
// hide behind gates that only see one of them. 150 tokens cross the GPU
// pass's chunk boundaries. Qwen2.5 (Step Q2) adds the bias and NEOX paths.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <span>
#include <string>
#include <vector>

#include "core/thread_pool.h"
#include "gguf/gguf.h"
#include "gpu/context.h"
#include "model/llama_gpu.h"
#include "model/llama_ref.h"
#include "test_util.h"

using namespace hearth;

TEST_SUITE("model") {
  TEST_CASE("gpu backend on real models: prefill equals decode bit for bit (150 tokens)") {
    if (!gpu::vulkan_available()) return;
    for (const char* name : {"Llama-3.2-1B-Instruct-Q8_0-pure.gguf", "Llama-3.2-3B-Instruct-Q4_0-pure.gguf",
                             "Qwen2.5-0.5B-Instruct-Q8_0-pure.gguf"}) {
      CAPTURE(name);
      const auto path = test::repo_dir() / "models" / name;
      if (!std::filesystem::exists(path)) {
        const std::string note = std::string("missing (run scripts\\fetch-models.ps1): ") + name;
        MESSAGE(note);
        continue;
      }
      const gguf::File f = gguf::File::open(path);
      constexpr std::size_t kTokens = 150;
      std::vector<std::int32_t> tokens(kTokens);
      tokens[0] = 128000;  // Llama 3's BOS; an ordinary token in Qwen2.5's vocabulary, which adds none
      for (std::size_t i = 1; i < kTokens; ++i) {
        tokens[i] = static_cast<std::int32_t>((static_cast<std::uint64_t>(i) * 2654435761u) % 128000);
      }
      ThreadPool pool(4);
      const gpu::ContextOptions opts = test::gpu_test_options();
      std::vector<float> prefill, decode;
      std::size_t V = 0;
      {
        model::LlamaRef llama(f, 256, pool, model::Matmul::kGpu, Isa::kScalar, &opts);
        V = llama.config().n_vocab;
        prefill.resize(kTokens * V);
        llama.forward(tokens, prefill, true);
      }
      {
        model::LlamaRef llama(f, 256, pool, model::Matmul::kGpu, Isa::kScalar, &opts);
        decode.resize(kTokens * V);
        for (std::size_t i = 0; i < kTokens; ++i) {
          llama.forward(std::span(tokens).subspan(i, 1), std::span(decode).subspan(i * V, V), false);
        }
      }
      std::size_t differ = 0, first = kTokens;
      for (std::size_t t = 0; t < kTokens; ++t) {
        if (std::memcmp(&prefill[t * V], &decode[t * V], V * sizeof(float)) != 0) {
          ++differ;
          if (first == kTokens) first = t;
        }
      }
      CHECK_MESSAGE(differ == 0, std::format("{} of {} positions differ, the first at {}", differ, kTokens, first));
      const std::string note = std::format("gpu prefill = decode, {}: {} positions x {} logits compared, {} differ",
                                           name, kTokens, V, differ);
      MESSAGE(note);
    }
  }
}
