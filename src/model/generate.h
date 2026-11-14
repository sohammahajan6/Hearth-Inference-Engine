// The prefill and sampling loop shared by `hearth run` and `hearth chat`.
#pragma once

#include <cstdint>
#include <functional>
#include <span>

#include "model/llama_ref.h"
#include "sampling/sampler.h"
#include "tokenizer/tokenizer.h"

namespace hearth::model {

enum class Stop {
  kEndOfGeneration,  // the model produced an end-of-generation token
  kLimit,            // max_new tokens were produced
  kContextFull,      // no position left for the next token
  kCancelled,        // Ctrl+C
};

struct GenerateResult {
  int prompt_tokens = 0;
  double prefill_seconds = 0;
  int generated = 0;  // emitted tokens; an end-of-generation token is not one
  double decode_seconds = 0;
  Stop stop = Stop::kLimit;
  std::int32_t eog_token = -1;  // the end-of-generation token, with Stop::kEndOfGeneration
  int drafted = 0;              // generate_speculative: draft tokens proposed to the target
  int accepted = 0;             // generate_speculative: draft tokens the target kept
  int looked_up = 0;            // generate_speculative: of the drafted tokens, those proposed by prompt lookup
};

// Feeds `prompt` after what the context already holds, then samples up to
// `max_new` tokens and calls `emit` for each. Every emitted token is also fed
// to the model, so a later turn can continue the context; the end-of-
// generation token is not. kInvalidArgument if the prompt does not fit.
GenerateResult generate(LlamaRef& llm, const tokenizer::Tokenizer& tok, sampling::Sampler& sampler,
                        std::span<const std::int32_t> prompt, int max_new,
                        const std::function<void(std::int32_t)>& emit);

}  // namespace hearth::model
