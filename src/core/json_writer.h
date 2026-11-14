// A small streaming JSON writer for results files and traces.
//
// It tracks nesting itself, so callers never place commas by hand. Doubles
// use the shortest representation that reads back to the same value; NaN and
// infinity (which JSON cannot represent) are written as null.
#pragma once

#include <concepts>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace hearth {

class JsonWriter {
 public:
  explicit JsonWriter(bool pretty = true) : pretty_(pretty) {}

  void begin_object();
  void end_object();
  void begin_array();
  void end_array();
  void key(std::string_view k);

  void value(std::string_view v);
  void value(const char* v) { value(std::string_view(v)); }
  void value(const std::string& v) { value(std::string_view(v)); }
  void value(bool v);
  void value(double v);
  void value(float v) { value(static_cast<double>(v)); }
  template <std::integral T>
    requires(!std::same_as<T, bool>)
  void value(T v) {
    if constexpr (std::is_signed_v<T>) {
      write_integer(static_cast<std::int64_t>(v));
    } else {
      write_unsigned(static_cast<std::uint64_t>(v));
    }
  }
  void null();

  template <class T>
  void field(std::string_view k, const T& v) {
    key(k);
    value(v);
  }

  // The finished document. Valid once every object and array is closed.
  const std::string& str() const noexcept { return out_; }

  static std::string escape(std::string_view s);

 private:
  void before_value();
  void write_integer(std::int64_t v);
  void write_unsigned(std::uint64_t v);
  void newline();

  struct Level {
    bool is_object;
    bool empty;
  };
  bool pretty_;
  bool after_key_ = false;
  std::vector<Level> stack_;
  std::string out_;
};

}  // namespace hearth
