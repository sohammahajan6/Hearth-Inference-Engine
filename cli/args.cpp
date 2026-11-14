#include "args.h"

#include <charconv>
#include <cstdlib>

#include "core/error.h"

namespace hearth::cli {

std::optional<std::string> Args::value(std::string_view name) {
  const std::string flag = "--" + std::string(name);
  for (std::size_t i = 0; i < tokens_.size(); ++i) {
    const std::string& t = tokens_[i];
    if (t == flag) {
      if (i + 1 >= tokens_.size()) fail(ErrorKind::kInvalidArgument, "option {} needs a value", flag);
      std::string v = tokens_[i + 1];
      tokens_.erase(tokens_.begin() + static_cast<std::ptrdiff_t>(i),
                    tokens_.begin() + static_cast<std::ptrdiff_t>(i) + 2);
      return v;
    }
    if (t.size() > flag.size() && t.compare(0, flag.size(), flag) == 0 && t[flag.size()] == '=') {
      std::string v = t.substr(flag.size() + 1);
      tokens_.erase(tokens_.begin() + static_cast<std::ptrdiff_t>(i));
      return v;
    }
  }
  return std::nullopt;
}

std::string Args::value_or(std::string_view name, const std::string& fallback) {
  auto v = value(name);
  return v ? *v : fallback;
}

long long Args::int_or(std::string_view name, long long fallback, long long min, long long max) {
  const auto v = value(name);
  if (!v) return fallback;
  long long out = 0;
  const char* end = v->data() + v->size();
  const auto [ptr, ec] = std::from_chars(v->data(), end, out);
  if (ec != std::errc() || ptr != end) fail(ErrorKind::kInvalidArgument, "--{} expects an integer, got '{}'", name, *v);
  if (out < min || out > max)
    fail(ErrorKind::kInvalidArgument, "--{} must be in [{}, {}], got {}", name, min, max, out);
  return out;
}

double Args::double_or(std::string_view name, double fallback, double min, double max) {
  const auto v = value(name);
  if (!v) return fallback;
  double out = 0;
  const char* end = v->data() + v->size();
  const auto [ptr, ec] = std::from_chars(v->data(), end, out);
  if (ec != std::errc() || ptr != end) fail(ErrorKind::kInvalidArgument, "--{} expects a number, got '{}'", name, *v);
  if (!(out >= min && out <= max))
    fail(ErrorKind::kInvalidArgument, "--{} must be in [{}, {}], got {}", name, min, max, out);
  return out;
}

bool Args::flag(std::string_view name) {
  const std::string flag = "--" + std::string(name);
  for (std::size_t i = 0; i < tokens_.size(); ++i) {
    if (tokens_[i] == flag) {
      tokens_.erase(tokens_.begin() + static_cast<std::ptrdiff_t>(i));
      return true;
    }
  }
  return false;
}

std::vector<std::string> Args::positionals() {
  std::vector<std::string> out;
  for (std::size_t i = 0; i < tokens_.size();) {
    if (tokens_[i].rfind("--", 0) != 0) {
      out.push_back(tokens_[i]);
      tokens_.erase(tokens_.begin() + static_cast<std::ptrdiff_t>(i));
    } else {
      ++i;
    }
  }
  return out;
}

void Args::finish() const {
  if (tokens_.empty()) return;
  const std::string& t = tokens_.front();
  if (t.rfind("--", 0) == 0) fail(ErrorKind::kInvalidArgument, "unknown option '{}' (see `hearth help`)", t);
  fail(ErrorKind::kInvalidArgument, "unexpected argument '{}' (see `hearth help`)", t);
}

}  // namespace hearth::cli
