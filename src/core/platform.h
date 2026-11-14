// Process-level setup and OS queries (Windows).
#pragma once

#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace hearth {

// Call first in main(). Turns off every modal dialog Windows or the C runtime
// could show (missing DLL, crash, abort(), assert), so an unattended run fails
// with an exit code instead of hanging on a dialog nobody will click. Also
// switches console output to UTF-8.
void init_process();

// Ctrl+C handling. The first press requests a cooperative stop: long loops
// poll cancel_requested() and unwind cleanly (GPU work finishes, files close).
// A second press terminates the process immediately.
void install_ctrl_c_handler();
bool cancel_requested() noexcept;
void throw_if_cancelled();  // throws Error(kCancelled)
// Forgets a handled press, so the next one is again a cooperative stop
// (`chat` stops a reply on Ctrl+C and then waits for the next turn).
void clear_cancel() noexcept;
// The same as a first Ctrl+C press (tests; a host that stops generation).
void request_cancel() noexcept;

// One line of standard input as UTF-8, without its line ending; nullopt at
// the end of input (or a lone Ctrl+Z line on a console). A console is read as
// UTF-16 (ReadConsoleW), so every typed character arrives intact; a pipe or
// file is read as UTF-8 bytes.
std::optional<std::string> read_line_utf8();
// True if standard input is an interactive console.
bool stdin_is_console() noexcept;

// For the lifetime of this object: keep the machine from sleeping, and opt
// this process out of EcoQoS power throttling (Windows 11 otherwise may run
// background-looking processes on low clocks, which ruins measurements).
class PerformanceMode {
 public:
  PerformanceMode();
  ~PerformanceMode();
  PerformanceMode(const PerformanceMode&) = delete;
  PerformanceMode& operator=(const PerformanceMode&) = delete;
};

enum class PowerSource { kAc, kBattery, kUnknown };
PowerSource power_source() noexcept;
const char* to_string(PowerSource s) noexcept;

// "10.0.26300.7000 (Dev)" style string. Read-only registry access.
std::string os_version_string();

// Physical memory in bytes: total and currently available.
struct MemoryStatus {
  unsigned long long total_bytes = 0;
  unsigned long long available_bytes = 0;
  unsigned long long commit_limit_bytes = 0;
  unsigned long long commit_bytes = 0;       // this process's commit charge
  unsigned long long working_set_bytes = 0;  // this process's pages in RAM (mapped weights included)
};
MemoryStatus memory_status();

// Sleeps for `us` microseconds with ~0.1 ms accuracy (a high-resolution
// waitable timer; plain Sleep() rounds up to the 15.6 ms scheduler tick).
// Below 500 us it spins instead, which is exact but uses the core.
void precise_sleep_us(long long us);

// Takes the pages of `bytes` (for example part of a mapped file) out of this
// process's working set without unmapping them. They move to the standby
// list, still cached, and a later access brings a page back with a soft
// fault. The `cpu` backend calls it for each tensor of the mapped model file
// once its repacked copy exists.
void release_from_working_set(std::span<const std::byte> bytes) noexcept;

// UTF-8 <-> std::filesystem::path without going through the ANSI code page.
std::filesystem::path path_from_utf8(std::string_view utf8);
std::string path_to_utf8(const std::filesystem::path& path);

}  // namespace hearth
