// `hearth tokenize`: text to token IDs, printed like `llama-tokenize --ids`
// ([a, b, c]), so the two can be compared exactly.
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "commands.h"
#include "core/error.h"
#include "core/platform.h"
#include "gguf/gguf.h"
#include "tokenizer/chat.h"
#include "tokenizer/tokenizer.h"

namespace hearth::cli {

int cmd_tokenize(Args& args) {
  const auto model = args.value("model");
  const auto text = args.value("text");
  const auto file = args.value("file");
  const bool no_bos = args.flag("no-bos");
  // --chat: the text is a user message, rendered with the model's chat template.
  const bool chat = args.flag("chat");
  std::string system = args.value_or("system", "");
  const auto system_file = args.value("system-file");
  const auto date = args.value("date");
  args.finish();
  if (system_file) {
    if (!system.empty()) fail(ErrorKind::kInvalidArgument, "give --system or --system-file, not both");
    std::ifstream in(path_from_utf8(*system_file), std::ios::binary);
    if (!in) fail(ErrorKind::kIo, "cannot read '{}'", *system_file);
    system.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  }
  if (!chat && (date || !system.empty())) fail(ErrorKind::kInvalidArgument, "--system and --date need --chat");
  if (chat && no_bos) fail(ErrorKind::kInvalidArgument, "--chat prompts always start with BOS");
  if (!model || text.has_value() == file.has_value()) {
    fail(ErrorKind::kInvalidArgument, "tokenize needs --model F and one of --text T and --file F");
  }
  std::string input;
  if (text) {
    input = *text;
  } else {
    // Binary mode: the bytes are tokenized exactly as they are in the file.
    std::ifstream in(path_from_utf8(*file), std::ios::binary);
    if (!in) fail(ErrorKind::kIo, "cannot read '{}'", *file);
    input.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    if (in.bad()) fail(ErrorKind::kIo, "cannot read '{}'", *file);
  }
  const gguf::File f = gguf::File::open(path_from_utf8(*model));
  const tokenizer::Tokenizer tok = tokenizer::Tokenizer::from_gguf(f);
  std::vector<std::int32_t> ids;
  if (chat) {
    const tokenizer::ChatTemplate tmpl = tokenizer::ChatTemplate::detect(f, tok);
    const std::string d = date ? tokenizer::template_date(*date) : tokenizer::template_date_today();
    ids = tmpl.prompt(tok, system, {{"user", input}}, d);
  } else {
    ids = tok.encode(input, tok.add_bos() && !no_bos);
  }
  std::string out = "[";
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (i > 0) out += ", ";
    out += std::to_string(ids[i]);
  }
  out += "]\n";
  std::fwrite(out.data(), 1, out.size(), stdout);
  return 0;
}

}  // namespace hearth::cli
