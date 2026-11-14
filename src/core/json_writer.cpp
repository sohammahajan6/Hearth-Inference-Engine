#include "core/json_writer.h"

#include <cmath>
#include <format>

#include "core/error.h"

namespace hearth {

std::string JsonWriter::escape(std::string_view s) {
  std::string out;
  out.reserve(s.size() + 2);
  for (const char ch : s) {
    const auto c = static_cast<unsigned char>(ch);
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      default:
        if (c < 0x20) {
          out += std::format("\\u{:04x}", static_cast<unsigned>(c));
        } else {
          out += ch;  // bytes >= 0x80 pass through: the writer emits UTF-8
        }
    }
  }
  return out;
}

void JsonWriter::newline() {
  if (!pretty_) return;
  out_ += '\n';
  out_.append(stack_.size() * 2, ' ');
}

void JsonWriter::before_value() {
  if (after_key_) {
    after_key_ = false;
    return;
  }
  if (stack_.empty()) {
    HEARTH_CHECK(out_.empty());  // a JSON document has exactly one top-level value
    return;
  }
  Level& top = stack_.back();
  HEARTH_CHECK(!top.is_object);  // object members need key() first
  if (!top.empty) out_ += ',';
  top.empty = false;
  newline();
}

void JsonWriter::key(std::string_view k) {
  HEARTH_CHECK(!stack_.empty() && stack_.back().is_object && !after_key_);
  Level& top = stack_.back();
  if (!top.empty) out_ += ',';
  top.empty = false;
  newline();
  out_ += '"';
  out_ += escape(k);
  out_ += pretty_ ? "\": " : "\":";
  after_key_ = true;
}

void JsonWriter::begin_object() {
  before_value();
  out_ += '{';
  stack_.push_back({true, true});
}

void JsonWriter::end_object() {
  HEARTH_CHECK(!stack_.empty() && stack_.back().is_object && !after_key_);
  const bool empty = stack_.back().empty;
  stack_.pop_back();
  if (!empty) newline();
  out_ += '}';
}

void JsonWriter::begin_array() {
  before_value();
  out_ += '[';
  stack_.push_back({false, true});
}

void JsonWriter::end_array() {
  HEARTH_CHECK(!stack_.empty() && !stack_.back().is_object);
  const bool empty = stack_.back().empty;
  stack_.pop_back();
  if (!empty) newline();
  out_ += ']';
}

void JsonWriter::value(std::string_view v) {
  before_value();
  out_ += '"';
  out_ += escape(v);
  out_ += '"';
}

void JsonWriter::value(bool v) {
  before_value();
  out_ += v ? "true" : "false";
}

void JsonWriter::value(double v) {
  before_value();
  if (!std::isfinite(v)) {
    out_ += "null";
    return;
  }
  out_ += std::format("{}", v);
}

void JsonWriter::null() {
  before_value();
  out_ += "null";
}

void JsonWriter::write_integer(std::int64_t v) {
  before_value();
  out_ += std::format("{}", v);
}

void JsonWriter::write_unsigned(std::uint64_t v) {
  before_value();
  out_ += std::format("{}", v);
}

}  // namespace hearth
