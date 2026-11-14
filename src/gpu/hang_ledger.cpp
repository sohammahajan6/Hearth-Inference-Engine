#include "gpu/hang_ledger.h"

#include <windows.h>

#include <chrono>
#include <fstream>
#include <mutex>
#include <string>
#include <system_error>

#include "core/log.h"
#include "core/platform.h"

namespace hearth::gpu {
namespace {

std::mutex g_mutex;
std::filesystem::path g_override;

std::filesystem::path default_path() {
  wchar_t exe[MAX_PATH] = {};
  const DWORD n = GetModuleFileNameW(nullptr, exe, MAX_PATH);
  if (n == 0 || n == MAX_PATH) return L"gpu-hang-ledger.txt";
  const std::filesystem::path dir = std::filesystem::path(exe).parent_path();
  std::error_code ec;
  for (std::filesystem::path p = dir; !p.empty(); p = p.parent_path()) {
    if (std::filesystem::exists(p / L"CMakeCache.txt", ec)) return p / L"gpu-hang-ledger.txt";
    if (p == p.root_path()) break;
  }
  return dir / L"gpu-hang-ledger.txt";
}

// Tabs and line breaks in a detail would break the one-entry-per-line format.
std::string one_field(std::string_view s) {
  std::string out(s);
  for (char& c : out) {
    if (c == '\t' || c == '\n' || c == '\r') c = ' ';
  }
  return out;
}

}  // namespace

std::filesystem::path hang_ledger_path() {
  const std::lock_guard lock(g_mutex);
  return g_override.empty() ? default_path() : g_override;
}

void set_hang_ledger_path(const std::filesystem::path& path) {
  const std::lock_guard lock(g_mutex);
  g_override = path;
}

void record_hang(std::string_view kind, std::string_view detail) noexcept {
  try {
    const std::filesystem::path path = hang_ledger_path();
    const auto now = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
    const std::string line = std::format("{:%Y-%m-%dT%H:%M:%SZ}\t{}\t{}\t{}\n", now, now.time_since_epoch().count(),
                                         one_field(kind), one_field(detail));
    const std::lock_guard lock(g_mutex);
    std::ofstream out(path, std::ios::binary | std::ios::app);
    out << line;
    out.close();
    if (!out) {
      log_warn("could not write the GPU hang ledger {}; entry: {}", path_to_utf8(path), line);
    } else {
      log_warn("GPU hang ledger: {} recorded in {}", kind, path_to_utf8(path));
    }
  } catch (...) {  // NOLINT(bugprone-empty-catch): the ledger must never replace the error being reported
  }
}

}  // namespace hearth::gpu
