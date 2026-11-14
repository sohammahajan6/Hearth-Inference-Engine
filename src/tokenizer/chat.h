// The Llama-3.1/3.2 chat format and Qwen2.5's ChatML, detected from
// tokenizer.chat_template and rendered by token ID
//.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "gguf/gguf.h"
#include "tokenizer/tokenizer.h"

namespace hearth::tokenizer {

struct Message {
  std::string role;  // "user" or "assistant"
  std::string content;
};

class ChatTemplate {
 public:
  // kUnsupported unless the model's template is a Llama 3 one (3.0, or 3.1+
  // with its date lines) or ChatML (<|im_start|>/<|im_end|>, as Qwen2.5 and
  // many other models use), and the vocabulary has its control tokens.
  static ChatTemplate detect(const gguf::File& file, const Tokenizer& tok);

  // Llama 3.1+: BOS, the system block ("Cutting Knowledge Date ...", `date`,
  // `system`), every message, then the assistant header that asks for a reply.
  // Llama 3.0: the same without the date lines, and the system block only if
  // `system` is not empty. ChatML: BOS if the vocabulary adds one, the system
  // message (`system`; when empty Qwen2.5's default, or none for other ChatML
  // templates; `date` is not used), every message, then
  // "<|im_start|>assistant\n".
  std::vector<std::int32_t> prompt(const Tokenizer& tok, std::string_view system, const std::vector<Message>& messages,
                                   std::string_view date) const;
  // What a further user turn appends after the model's reply: the end of the
  // reply (<|eot_id|>, or <|im_end|> and a newline), the user message, and a
  // new header asking for a reply.
  std::vector<std::int32_t> next_turn(const Tokenizer& tok, std::string_view user) const;

 private:
  enum class Kind : std::uint8_t { kLlama3, kChatML };
  void block(const Tokenizer& tok, std::string_view role, std::string_view text, std::vector<std::int32_t>& out) const;
  void header(const Tokenizer& tok, std::string_view role, std::vector<std::int32_t>& out) const;
  void generation_prompt(const Tokenizer& tok, std::vector<std::int32_t>& out) const;

  Kind kind_ = Kind::kLlama3;
  std::int32_t start_header_ = -1, end_header_ = -1, eot_ = -1;  // Llama 3
  std::int32_t im_start_ = -1, im_end_ = -1;                     // ChatML
  std::string default_system_;                                   // ChatML; empty: no default
  bool dated_ = true;  // Llama 3.1+: the system block carries the knowledge and today's date
};

// The template's trim: the bytes C's isspace accepts in the C locale.
std::string_view trim(std::string_view s);
// A date as the template prints it, "%d %b %Y": "04 Nov 2026". kInvalidArgument
// for a date that does not exist.
std::string template_date(int year, int month, int day);
// "YYYY-MM-DD" -> template_date(); kInvalidArgument if malformed.
std::string template_date(std::string_view iso);
// Today's local date, as template_date().
std::string template_date_today();

}  // namespace hearth::tokenizer
