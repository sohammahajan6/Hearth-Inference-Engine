// The GPU hang ledger: the
// TDR budget must
// hold across the many processes that tests, gates and benchmarks start, so
// every real device loss, GPU timeout and near-miss watchdog trip is written
// to a file the scripts check before GPU work (scripts/lib/common.ps1
// Assert-GpuBudget). Simulated losses (test hooks) are never written.
#pragma once

#include <filesystem>
#include <format>
#include <string_view>

#include "core/error.h"

namespace hearth::gpu {

// `gpu-hang-ledger.txt` in the running executable's build folder: the
// nearest folder up from the exe that holds CMakeCache.txt, else the exe's
// own folder (a packaged hearth.exe).
std::filesystem::path hang_ledger_path();
// Tests write to a file of their own; an empty path restores the default.
void set_hang_ledger_path(const std::filesystem::path& path);

// Appends "<UTC time>\t<unix seconds>\t<kind>\t<detail>". Never throws: a
// ledger that cannot be written is reported as a warning, and the error that
// caused the entry is still the one the caller reports.
void record_hang(std::string_view kind, std::string_view detail) noexcept;

// Runs f(); a kDeviceLost it throws is written to the ledger as "device-lost"
// (with `where`), then rethrown.
template <class F>
decltype(auto) note_device_loss(std::string_view where, F&& f) {
  try {
    return f();
  } catch (const Error& e) {
    if (e.kind() == ErrorKind::kDeviceLost) record_hang("device-lost", std::format("{}: {}", where, e.what()));
    throw;
  }
}

}  // namespace hearth::gpu
