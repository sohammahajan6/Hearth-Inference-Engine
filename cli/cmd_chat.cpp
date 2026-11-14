// `hearth chat`: a conversation, one user turn per input line. The KV cache
// is kept between turns, so each turn feeds only its own tokens.
#include <algorithm>
#include <cstdio>
#include <format>
#include <span>
#include <string>
#include <vector>

#include "commands.h"
#include "core/error.h"
#include "core/log.h"
#include "core/platform.h"
#include "hearth/engine.h"
#include "session.h"

namespace hearth::cli {
namespace {

void write_out(std::string_view s) {
  std::fwrite(s.data(), 1, s.size(), stdout);
  std::fflush(stdout);
}

}  // namespace

int cmd_chat(Args& args) {
  ModelOptions mo = model_options(args, 4096);
  // Interactive use: a lost GPU hands its work to the CPU (09 doc section 5).
  // The gate scripts pass --no-fallback: a measurement never falls back.
  mo.gpu_fallback = !args.flag("no-fallback");
  draft_options(args, mo);
  std::string seed_note;
  const sampling::Params sp = sampling_options(args, seed_note);
  const std::string system = args.value_or("system", "");
  const auto date = args.value("date");
  const int n_predict = static_cast<int>(args.int_or("n-predict", 1024, 1, 1 << 20));
  args.finish();

  // From here on only the public API (include/hearth/engine.h).
  Session s(mo);
  Engine& engine = s.engine;
  const std::string day = date ? *date : std::string();
  GenerateOptions g;
  g.max_new_tokens = n_predict;
  g.temperature = sp.temperature;
  g.top_k = sp.top_k;
  g.top_p = sp.top_p;
  g.seed = sp.seed;
  // Fail now, not at the first turn, if the model's chat template is not supported.
  static_cast<void>(engine.chat_next_turn(""));
  const bool console = stdin_is_console();
  log_info(
      "chat: {}, {} threads, {}, context {}. One line per turn; /reset starts over, /exit or end of input "
      "quits, Ctrl+C stops a reply.",
      mo.backend, mo.threads, seed_note, mo.ctx);
  bool fresh = true;
  for (;;) {
    if (console) write_out("> ");
    const auto line = read_line_utf8();
    if (cancel_requested()) fail(ErrorKind::kCancelled, "interrupted by Ctrl+C");  // Ctrl+C at the prompt
    if (!line || *line == "/exit") return 0;
    if (*line == "/reset") {
      engine.reset();
      fresh = true;
      write_out("[the conversation was reset]\n");
      continue;
    }
    if (line->empty()) continue;
    const ChatMessage m{"user", *line};
    const std::vector<std::int32_t> ids =
        fresh ? engine.chat_prompt(system, std::span(&m, 1), day) : engine.chat_next_turn(*line);
    if (ids.size() > engine.context_size() - engine.context_used()) {
      write_out(
          std::format("[the context is full: {} of {} positions are used. Type /reset, or start again with a "
                      "larger --ctx]\n",
                      engine.context_used(), engine.context_size()));
      continue;
    }
    const GenerateResult r = engine.generate(ids, g, [&](std::int32_t, std::string_view piece) { write_out(piece); });
    write_out(r.trailing_text + "\n");
    // A first prompt stopped by Ctrl+C before it entered the context is
    // still the first: the next one keeps the system header.
    fresh = fresh && engine.context_used() == 0;
    if (r.stop == StopReason::kCancelled) {
      clear_cancel();
      write_out("[stopped]\n");
    } else if (r.stop == StopReason::kContextFull) {
      write_out("[the context is full. Type /reset, or start again with a larger --ctx]\n");
    } else if (r.stop == StopReason::kLimit) {
      write_out(std::format("[stopped after {} tokens (--n-predict)]\n", r.generated_tokens));
    }
    log_debug("chat: {} prompt tokens in {:.2f} s, {} generated in {:.2f} s ({:.2f} tok/s)", r.prompt_tokens,
              r.prefill_seconds, r.generated_tokens, r.decode_seconds,
              r.generated_tokens / std::max(r.decode_seconds, 1e-9));
  }
}

}  // namespace hearth::cli
