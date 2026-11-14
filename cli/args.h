// Minimal command-line option parsing for the hearth subcommands.
//
// Each subcommand pulls the options it understands; whatever is left over is
// an error, so a typo such as `--thread 4` fails loudly instead of being
// silently ignored.
#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace hearth::cli {

class Args {
 public:
  explicit Args(std::vector<std::string> tokens) : tokens_(std::move(tokens)) {}

  // "--name VALUE" or "--name=VALUE". Fails if the option is given without a value.
  std::optional<std::string> value(std::string_view name);
  std::string value_or(std::string_view name, const std::string& fallback);
  long long int_or(std::string_view name, long long fallback, long long min, long long max);
  double double_or(std::string_view name, double fallback, double min, double max);
  // "--name" present?
  bool flag(std::string_view name);
  // Remaining arguments that do not start with "--". Call after all options.
  std::vector<std::string> positionals();
  // Fails if any unconsumed token remains.
  void finish() const;

 private:
  std::vector<std::string> tokens_;
};

}  // namespace hearth::cli
