// The error type of the hearth library's public API (and of everything
// inside it). Every failure is thrown as hearth::Error; its kind says who is
// at fault (the caller's input, the model file, the GPU, or the library), so
// a program can react (for example, retry on the CPU after kDeviceLost).
#pragma once

#include <stdexcept>
#include <string>

namespace hearth {

enum class ErrorKind {
  kInvalidArgument,  // bad command-line value or API argument
  kBadModel,         // the model file is malformed or inconsistent
  kUnsupported,      // valid input that this engine does not handle (yet or ever)
  kIo,               // file system or OS call failed
  kGpu,              // Vulkan call failed (device still usable or never created)
  kDeviceLost,       // the GPU was lost (hang/TDR, driver reset); fall back to CPU
  kOutOfMemory,      // allocation failed or the memory budget would be exceeded
  kCancelled,        // the user pressed Ctrl+C
  kInternal,         // a bug in this program: an invariant did not hold
};

const char* to_string(ErrorKind kind) noexcept;

class Error : public std::runtime_error {
 public:
  Error(ErrorKind kind, const std::string& message) : std::runtime_error(message), kind_(kind) {}
  ErrorKind kind() const noexcept { return kind_; }

 private:
  ErrorKind kind_;
};

}  // namespace hearth
