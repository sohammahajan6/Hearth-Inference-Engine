#include "core/log.h"

#include <atomic>
#include <cstdio>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <vector>

namespace hearth {
namespace {
std::atomic<int> g_level{static_cast<int>(LogLevel::kInfo)};
std::mutex g_mutex;
struct Captured {
  LogLevel level;
  std::string line;
};
// Guarded by g_mutex. Engaged while a LogCapture exists.
std::optional<std::vector<Captured>> g_capture;
unsigned long long g_problems = 0;  // guarded by g_mutex
bool g_echo = false;                // guarded by g_mutex
constexpr const char* kPrefix[] = {"error: ", "warning: ", "", "debug: ", "trace: "};

// The lines of the live capture; the caller holds g_mutex. A LogCapture's
// methods only run while it exists, so the capture is always engaged here.
const std::vector<Captured>& captured() {
  if (!g_capture.has_value()) throw std::logic_error("LogCapture: no capture is active");
  return *g_capture;
}
}  // namespace

void set_log_level(LogLevel level) noexcept { g_level.store(static_cast<int>(level), std::memory_order_relaxed); }

LogLevel log_level() noexcept { return static_cast<LogLevel>(g_level.load(std::memory_order_relaxed)); }

bool parse_log_level(std::string_view text, LogLevel& out) noexcept {
  struct Name {
    std::string_view name;
    LogLevel level;
  };
  static constexpr Name kNames[] = {{"error", LogLevel::kError},
                                    {"warn", LogLevel::kWarn},
                                    {"info", LogLevel::kInfo},
                                    {"debug", LogLevel::kDebug},
                                    {"trace", LogLevel::kTrace}};
  for (const Name& n : kNames) {
    if (n.name == text) {
      out = n.level;
      return true;
    }
  }
  return false;
}

void log_write(LogLevel level, std::string_view message) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_capture.has_value()) {
    std::string line = kPrefix[static_cast<int>(level)];
    line.append(message);
    if (g_echo) {
      std::fputs("(expected) ", stderr);
      std::fwrite(line.data(), 1, line.size(), stderr);
      std::fputc('\n', stderr);
    }
    g_capture->push_back({level, std::move(line)});
    return;
  }
  if (level == LogLevel::kError || level == LogLevel::kWarn) ++g_problems;
  std::fputs(kPrefix[static_cast<int>(level)], stderr);
  std::fwrite(message.data(), 1, message.size(), stderr);
  std::fputc('\n', stderr);
}

unsigned long long log_problem_count() noexcept {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_problems;
}

void LogCapture::set_echo(bool on) noexcept {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_echo = on;
}

LogCapture::LogCapture() {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_capture.has_value()) throw std::logic_error("LogCapture: captures do not nest");
  g_capture.emplace();
}

LogCapture::~LogCapture() {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_capture.reset();
}

std::string LogCapture::text() const {
  std::lock_guard<std::mutex> lock(g_mutex);
  std::string out;
  for (const Captured& c : captured()) {
    out += c.line;
    out += '\n';
  }
  return out;
}

int LogCapture::count(std::string_view needle) const {
  std::lock_guard<std::mutex> lock(g_mutex);
  int n = 0;
  for (const Captured& c : captured()) n += c.line.find(needle) != std::string::npos ? 1 : 0;
  return n;
}

int LogCapture::count(LogLevel level) const {
  std::lock_guard<std::mutex> lock(g_mutex);
  int n = 0;
  for (const Captured& c : captured()) n += c.level == level ? 1 : 0;
  return n;
}

int LogCapture::lines() const {
  std::lock_guard<std::mutex> lock(g_mutex);
  return static_cast<int>(captured().size());
}

}  // namespace hearth
