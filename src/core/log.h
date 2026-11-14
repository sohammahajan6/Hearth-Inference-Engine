// Leveled logging to stderr. stdout is reserved for program output (generated
// text, JSON, tables) so it can be piped or captured cleanly.
#pragma once

#include <format>
#include <string>
#include <string_view>
#include <utility>

namespace hearth {

enum class LogLevel { kError = 0, kWarn = 1, kInfo = 2, kDebug = 3, kTrace = 4 };

void set_log_level(LogLevel level) noexcept;
LogLevel log_level() noexcept;
// Parses "error", "warn", "info", "debug", "trace". Returns false if unknown.
bool parse_log_level(std::string_view text, LogLevel& out) noexcept;
void log_write(LogLevel level, std::string_view message);

// Errors and warnings written to stderr so far (lines caught by a LogCapture
// are not counted). The test runner fails the run if this is not zero, so a
// warning nobody expected cannot pass unnoticed.
unsigned long long log_problem_count() noexcept;

// While a LogCapture exists, every log line (from any thread) goes into it
// instead of stderr. Tests use it around code that warns or reports an error
// on purpose, then check the exact text. Captures do not nest.
class LogCapture {
 public:
  LogCapture();
  ~LogCapture();
  LogCapture(const LogCapture&) = delete;
  LogCapture& operator=(const LogCapture&) = delete;

  // Everything captured so far, one "prefix: message\n" line per call.
  std::string text() const;
  // How many lines contain `needle`.
  int count(std::string_view needle) const;
  // How many lines were logged at `level`.
  int count(LogLevel level) const;
  int lines() const;

  // When on, captured lines are also written to stderr, prefixed
  // "(expected) ", for evidence runs and
  // `test.ps1 -Details`. They still do not count in log_problem_count().
  static void set_echo(bool on) noexcept;
};

inline bool log_enabled(LogLevel level) noexcept { return static_cast<int>(level) <= static_cast<int>(log_level()); }

template <class... A>
void log_error(std::format_string<A...> f, A&&... a) {
  if (log_enabled(LogLevel::kError)) log_write(LogLevel::kError, std::format(f, std::forward<A>(a)...));
}
template <class... A>
void log_warn(std::format_string<A...> f, A&&... a) {
  if (log_enabled(LogLevel::kWarn)) log_write(LogLevel::kWarn, std::format(f, std::forward<A>(a)...));
}
template <class... A>
void log_info(std::format_string<A...> f, A&&... a) {
  if (log_enabled(LogLevel::kInfo)) log_write(LogLevel::kInfo, std::format(f, std::forward<A>(a)...));
}
template <class... A>
void log_debug(std::format_string<A...> f, A&&... a) {
  if (log_enabled(LogLevel::kDebug)) log_write(LogLevel::kDebug, std::format(f, std::forward<A>(a)...));
}
template <class... A>
void log_trace(std::format_string<A...> f, A&&... a) {
  if (log_enabled(LogLevel::kTrace)) log_write(LogLevel::kTrace, std::format(f, std::forward<A>(a)...));
}

}  // namespace hearth
