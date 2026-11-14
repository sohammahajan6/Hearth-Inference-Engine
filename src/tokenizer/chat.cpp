#include "tokenizer/chat.h"

#include <charconv>
#include <ctime>
#include <format>

#include "core/error.h"

namespace hearth::tokenizer {
namespace {

bool is_c_space(char c) { return c == ' ' || (c >= '\t' && c <= '\r'); }  // \t \n \v \f \r

constexpr const char* kMonths[] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};

}  // namespace

std::string_view trim(std::string_view s) {
  while (!s.empty() && is_c_space(s.front())) s.remove_prefix(1);
  while (!s.empty() && is_c_space(s.back())) s.remove_suffix(1);
  return s;
}

std::string template_date(int year, int month, int day) {
  static constexpr int kDays[] = {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  const bool leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
  if (year < 1 || year > 9999 || month < 1 || month > 12 || day < 1 || day > kDays[month - 1] ||
      (month == 2 && day == 29 && !leap)) {
    fail(ErrorKind::kInvalidArgument, "{:04}-{:02}-{:02} is not a date", year, month, day);
  }
  return std::format("{:02} {} {:04}", day, kMonths[month - 1], year);
}

std::string template_date(std::string_view iso) {
  int y = 0, m = 0, d = 0;
  const auto num = [&](std::size_t at, std::size_t len, int& out) {
    const auto [p, ec] = std::from_chars(iso.data() + at, iso.data() + at + len, out);
    return ec == std::errc() && p == iso.data() + at + len;
  };
  if (iso.size() != 10 || iso[4] != '-' || iso[7] != '-' || !num(0, 4, y) || !num(5, 2, m) || !num(8, 2, d)) {
    fail(ErrorKind::kInvalidArgument, "date '{}' is not YYYY-MM-DD", iso);
  }
  return template_date(y, m, d);
}

std::string template_date_today() {
  const std::time_t now = std::time(nullptr);
  std::tm local{};
  if (localtime_s(&local, &now) != 0) fail(ErrorKind::kInternal, "localtime_s failed");
  return template_date(local.tm_year + 1900, local.tm_mon + 1, local.tm_mday);
}

ChatTemplate ChatTemplate::detect(const gguf::File& f, const Tokenizer& tok) {
  const gguf::Value* v = f.find("tokenizer.chat_template");
  if (v == nullptr || v->type != gguf::ValueType::kString) {
    fail(ErrorKind::kUnsupported, "the model has no chat template (tokenizer.chat_template)");
  }
  const std::string_view t = v->s;
  ChatTemplate c;
  const auto id = [&](const char* name) {
    const auto s = tok.special(name);
    if (!s) fail(ErrorKind::kUnsupported, "the vocabulary has no {} control token", name);
    return *s;
  };
  // Qwen2.5: ChatML, with this default system prompt when there is none
  // (its tokenizer.chat_template; the tool-calling branch is not used).
  static constexpr std::string_view kQwenSystem =
      "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
  // ChatML: Qwen2.5's has this default system prompt; other ChatML
  // templates (SmolLM2, many fine-tunes) write a system turn only when given.
  if (t.find("<|im_start|>") != std::string_view::npos) {
    for (const std::string_view marker : {std::string_view("<|im_end|>"), std::string_view("assistant\\n")}) {
      if (t.find(marker) == std::string_view::npos) {
        fail(ErrorKind::kUnsupported, "the model's ChatML chat template is not a standard one (no \"{}\")", marker);
      }
    }
    c.kind_ = Kind::kChatML;
    c.im_start_ = id("<|im_start|>");
    c.im_end_ = id("<|im_end|>");
    if (t.find(kQwenSystem) != std::string_view::npos) c.default_system_ = std::string(kQwenSystem);
    return c;
  }
  for (const char* marker : {"<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"}) {
    if (t.find(marker) == std::string_view::npos) {
      fail(ErrorKind::kUnsupported,
           "the model's chat template is neither a Llama 3 one nor ChatML (no \"{}\"); this version supports those two",
           marker);
    }
  }
  // Llama 3.1+ write the knowledge cutoff and today's date; 3.0 does not.
  c.dated_ = t.find("Cutting Knowledge Date: December 2023") != std::string_view::npos &&
             t.find("Today Date: ") != std::string_view::npos;
  c.start_header_ = id("<|start_header_id|>");
  c.end_header_ = id("<|end_header_id|>");
  c.eot_ = id("<|eot_id|>");
  return c;
}

void ChatTemplate::header(const Tokenizer& tok, std::string_view role, std::vector<std::int32_t>& out) const {
  out.push_back(start_header_);
  tok.encode(role, out);
  out.push_back(end_header_);
}

// Llama 3: header, "\n\n" + trimmed text, <|eot_id|>. ChatML:
// <|im_start|>, role + "\n" + text (not trimmed), <|im_end|>, "\n". Text between
// control tokens is encoded on its own, as llama.cpp splits a rendered
// prompt at control tokens before encoding the rest.
void ChatTemplate::block(const Tokenizer& tok, std::string_view role, std::string_view text,
                         std::vector<std::int32_t>& out) const {
  if (kind_ == Kind::kChatML) {
    out.push_back(im_start_);
    tok.encode(std::string(role) + "\n" + std::string(text), out);
    out.push_back(im_end_);
    tok.encode("\n", out);
    return;
  }
  header(tok, role, out);
  tok.encode(std::string("\n\n") + std::string(trim(text)), out);
  out.push_back(eot_);
}

void ChatTemplate::generation_prompt(const Tokenizer& tok, std::vector<std::int32_t>& out) const {
  if (kind_ == Kind::kChatML) {
    out.push_back(im_start_);
    tok.encode("assistant\n", out);
    return;
  }
  header(tok, "assistant", out);
  tok.encode("\n\n", out);
}

std::vector<std::int32_t> ChatTemplate::prompt(const Tokenizer& tok, std::string_view system,
                                               const std::vector<Message>& messages, std::string_view date) const {
  std::vector<std::int32_t> out;
  if (kind_ == Kind::kChatML) {
    // The template has no BOS of its own; llama.cpp adds one only if the
    // vocabulary asks for it (Qwen2.5 does not).
    if (tok.add_bos()) out.push_back(tok.bos());
    const std::string_view sys = system.empty() ? std::string_view(default_system_) : system;
    if (!sys.empty()) block(tok, "system", sys, out);
    for (const Message& m : messages) {
      if (m.role != "user" && m.role != "assistant") {
        fail(ErrorKind::kInvalidArgument, "chat: message role '{}' is not user or assistant", m.role);
      }
      block(tok, m.role, m.content, out);
    }
    generation_prompt(tok, out);
    return out;
  }
  out.push_back(tok.bos());
  if (dated_) {
    header(tok, "system", out);
    tok.encode(std::format("\n\nCutting Knowledge Date: December 2023\nToday Date: {}\n\n{}", date, trim(system)), out);
    out.push_back(eot_);
  } else if (!system.empty()) {
    block(tok, "system", system, out);
  }
  for (const Message& m : messages) {
    if (m.role != "user" && m.role != "assistant") {
      fail(ErrorKind::kInvalidArgument, "chat: message role '{}' is not user or assistant", m.role);
    }
    block(tok, m.role, m.content, out);
  }
  header(tok, "assistant", out);
  tok.encode("\n\n", out);
  return out;
}

std::vector<std::int32_t> ChatTemplate::next_turn(const Tokenizer& tok, std::string_view user) const {
  std::vector<std::int32_t> out;
  if (kind_ == Kind::kChatML) {
    // The reply's "<|im_end|>\n", which the model's end token stopped short of.
    out.push_back(im_end_);
    tok.encode("\n", out);
  } else {
    out.push_back(eot_);
  }
  block(tok, "user", user, out);
  generation_prompt(tok, out);
  return out;
}

}  // namespace hearth::tokenizer
