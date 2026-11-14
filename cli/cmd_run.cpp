// `hearth run`: one prompt in, generated text out (streamed).
#include <algorithm>
#include <charconv>
#include <cstdio>
#include <fstream>
#include <iterator>
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

std::vector<std::int32_t> parse_ids(std::string_view list) {
  std::vector<std::int32_t> ids;
  while (!list.empty()) {
    const auto comma = list.find(',');
    const std::string_view item = list.substr(0, comma);
    std::int32_t v = 0;
    const auto [p, ec] = std::from_chars(item.data(), item.data() + item.size(), v);
    if (item.empty() || ec != std::errc() || p != item.data() + item.size()) {
      fail(ErrorKind::kInvalidArgument, "--tokens: '{}' is not a token ID", item);
    }
    ids.push_back(v);
    if (comma == std::string_view::npos) break;
    list.remove_prefix(comma + 1);
  }
  return ids;
}

const char* stop_name(StopReason s) {
  switch (s) {
    case StopReason::kEndOfGeneration:
      return "end of generation";
    case StopReason::kLimit:
      return "token limit";
    case StopReason::kContextFull:
      return "context full";
    case StopReason::kCancelled:
      return "Ctrl+C";
  }
  return "?";
}

}  // namespace

int cmd_run(Args& args) {
  ModelOptions mo = model_options(args, 4096);
  // Interactive use: a lost GPU hands its work to the CPU (09 doc section 5).
  // The gate scripts pass --no-fallback: a measurement never falls back.
  mo.gpu_fallback = !args.flag("no-fallback");
  draft_options(args, mo);
  std::string seed_note;
  const sampling::Params sp = sampling_options(args, seed_note);
  const auto prompt = args.value("prompt");
  const auto file = args.value("file");
  const auto tokens = args.value("tokens");
  const int n_predict = static_cast<int>(args.int_or("n-predict", 128, 0, 1 << 20));
  const bool chat = args.flag("chat");
  const std::string system = args.value_or("system", "");
  const auto date = args.value("date");
  const bool ids_out = args.flag("ids");
  const bool special = args.flag("special");
  args.finish();
  if (int(prompt.has_value()) + int(file.has_value()) + int(tokens.has_value()) != 1) {
    fail(ErrorKind::kInvalidArgument, "run needs exactly one of --prompt TEXT, --file F and --tokens ID,ID,...");
  }
  if (tokens && chat) fail(ErrorKind::kInvalidArgument, "--tokens are used as given; --chat does not apply");
  if (!chat && (date || !system.empty())) fail(ErrorKind::kInvalidArgument, "--system and --date need --chat");

  // The prompt text (unless --tokens), read before the model is loaded, so a
  // bad path fails at once.
  std::string text;
  if (prompt) {
    text = *prompt;
  } else if (file) {
    std::ifstream in(path_from_utf8(*file), std::ios::binary);
    if (!in) fail(ErrorKind::kIo, "cannot read '{}'", *file);
    text.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  }

  // From here on only the public API (include/hearth/engine.h); the options
  // above are parsed by the CLI layer shared with the measurement commands.
  Session s(mo);
  Engine& engine = s.engine;
  std::vector<std::int32_t> ids;
  if (tokens) {
    ids = parse_ids(*tokens);
  } else if (chat) {
    const ChatMessage m{"user", text};
    ids = engine.chat_prompt(system, std::span(&m, 1), date ? *date : std::string());
  } else {
    ids = engine.tokenize(text, s.tok.add_bos());
  }
  log_info("run: {}, {} threads, {}; prompt {} tokens", mo.backend, mo.threads, seed_note, ids.size());
  if (const std::string d = engine.draft_description(); !d.empty()) log_info("run: speculative decoding, {}", d);

  GenerateOptions g;
  g.max_new_tokens = n_predict;
  g.temperature = sp.temperature;
  g.top_k = sp.top_k;
  g.top_p = sp.top_p;
  g.seed = sp.seed;
  g.special_text = special;
  std::vector<std::int32_t> generated;
  const GenerateResult r = engine.generate(ids, g, [&](std::int32_t id, std::string_view piece) {
    generated.push_back(id);
    if (!ids_out) write_out(piece);
  });
  if (ids_out) {
    // The end-of-generation token closes the list, as in hetero-oracle-dump --generate.
    if (r.stop == StopReason::kEndOfGeneration) generated.push_back(r.end_token);
    std::string out = "[";
    for (std::size_t i = 0; i < generated.size(); ++i) out += (i ? ", " : "") + std::to_string(generated[i]);
    write_out(out + "]\n");
  } else {
    write_out(r.trailing_text + "\n");
  }
  log_info(
      "run: prefill {} tokens in {:.2f} s ({:.1f} tok/s); generated {} tokens in {:.2f} s ({:.2f} tok/s); "
      "stopped: {}",
      r.prompt_tokens, r.prefill_seconds, r.prompt_tokens / std::max(r.prefill_seconds, 1e-9), r.generated_tokens,
      r.decode_seconds, r.generated_tokens / std::max(r.decode_seconds, 1e-9), stop_name(r.stop));
  if (!engine.draft_description().empty()) {
    log_info("run: the draft proposed {} tokens, {} kept ({:.1f} %)", r.draft_tokens, r.draft_accepted,
             100.0 * r.draft_accepted / std::max(r.draft_tokens, 1));
  }
  if (r.stop == StopReason::kCancelled) fail(ErrorKind::kCancelled, "interrupted by Ctrl+C");
  return 0;
}

}  // namespace hearth::cli
