#include "model/generate.h"

#include <vector>

#include "core/clock.h"
#include "core/error.h"
#include "core/platform.h"

namespace hearth::model {

GenerateResult generate(LlamaRef& llm, const tokenizer::Tokenizer& tok, sampling::Sampler& sampler,
                        std::span<const std::int32_t> prompt, int max_new,
                        const std::function<void(std::int32_t)>& emit) {
  GenerateResult r;
  r.prompt_tokens = static_cast<int>(prompt.size());
  if (prompt.empty()) fail(ErrorKind::kInvalidArgument, "generate: the prompt is empty");
  const std::uint32_t room = llm.n_ctx() - llm.n_past();
  if (room == 0) {  // a full context is a stop reason, not a bad argument
    r.stop = Stop::kContextFull;
    return r;
  }
  if (prompt.size() > room) {
    fail(ErrorKind::kInvalidArgument, "the prompt needs {} positions but only {} of the {} are free", prompt.size(),
         room, llm.n_ctx());
  }
  std::vector<float> logits(llm.config().n_vocab);
  const std::int64_t t0 = now_ns();
  try {
    llm.forward(prompt, logits, false);
  } catch (const Error& e) {
    // Ctrl+C during a long prompt: forward() left the context as it was.
    if (e.kind() != ErrorKind::kCancelled) throw;
    r.stop = Stop::kCancelled;
    clear_cancel();
    return r;
  }
  const std::int64_t t1 = now_ns();
  r.prefill_seconds = static_cast<double>(t1 - t0) * 1e-9;
  while (r.generated < max_new) {
    if (cancel_requested()) {
      r.stop = Stop::kCancelled;
      clear_cancel();  // consumed by the call it stopped: the next call runs
      break;
    }
    const std::int32_t id = sampler.sample(logits);
    if (tok.is_eog(id)) {
      r.stop = Stop::kEndOfGeneration;
      r.eog_token = id;
      break;
    }
    emit(id);
    ++r.generated;
    if (llm.n_past() >= llm.n_ctx()) {
      r.stop = Stop::kContextFull;
      break;
    }
    llm.forward(std::span(&id, 1), logits, false);
    r.stop = Stop::kLimit;
  }
  r.decode_seconds = static_cast<double>(now_ns() - t1) * 1e-9;
  return r;
}

}  // namespace hearth::model
