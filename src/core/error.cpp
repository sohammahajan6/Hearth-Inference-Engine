#include "core/error.h"

namespace hearth {

const char* to_string(ErrorKind kind) noexcept {
  switch (kind) {
    case ErrorKind::kInvalidArgument:
      return "invalid argument";
    case ErrorKind::kBadModel:
      return "bad model file";
    case ErrorKind::kUnsupported:
      return "unsupported";
    case ErrorKind::kIo:
      return "I/O error";
    case ErrorKind::kGpu:
      return "GPU error";
    case ErrorKind::kDeviceLost:
      return "GPU device lost";
    case ErrorKind::kOutOfMemory:
      return "out of memory";
    case ErrorKind::kCancelled:
      return "cancelled";
    case ErrorKind::kInternal:
      return "internal error";
  }
  return "unknown error";
}

namespace detail {
void check_failed(const char* expr, const char* file, int line) {
  throw Error(ErrorKind::kInternal, std::format("internal check failed: {} ({}:{})", expr, file, line));
}
}  // namespace detail

}  // namespace hearth
