#include "core/platform.h"

// MSVC /analyze reports C28286 ("error in annotation") at consoleapi.h(204)
// for any call of ReadConsoleW: Windows SDK 10.0.22621 annotates its output
// as byteCount(*lpNumberOfCharsRead * sizeof(TCHAR)), which the analyzer
// cannot parse. The defect is in the SDK header and the warning is reported
// at that header's line, so exactly that warning is turned off while the
// header is read.
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#pragma warning(disable : 28286)
#endif
#include <windows.h>
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif
// (separate block: psapi.h needs windows.h first, and the formatter sorts within a block)
#include <psapi.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <format>

#include "core/error.h"

#if defined(_MSC_VER)
#include <crtdbg.h>
#endif

namespace hearth {
namespace {

std::atomic<int> g_ctrl_c_count{0};

BOOL WINAPI ctrl_handler(DWORD type) {
  if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT) {
    if (g_ctrl_c_count.fetch_add(1) >= 1) {
      // Second press: the user wants out now.
      TerminateProcess(GetCurrentProcess(), 130);
    }
    return TRUE;  // handled: do not run the default handler (which kills the process)
  }
  return FALSE;
}

}  // namespace

void init_process() {
  SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
#if defined(_MSC_VER)
  // abort(): no "Debug Error!" dialog, no Windows Error Reporting.
  _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
  // Debug CRT asserts and errors go to stderr instead of a dialog.
  _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
  _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
  _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
  _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
#endif
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);
}

void install_ctrl_c_handler() { SetConsoleCtrlHandler(ctrl_handler, TRUE); }

bool cancel_requested() noexcept { return g_ctrl_c_count.load(std::memory_order_relaxed) > 0; }

void throw_if_cancelled() {
  if (cancel_requested()) throw Error(ErrorKind::kCancelled, "interrupted by Ctrl+C");
}

void clear_cancel() noexcept { g_ctrl_c_count.store(0, std::memory_order_relaxed); }

void request_cancel() noexcept {
  int expected = 0;
  g_ctrl_c_count.compare_exchange_strong(expected, 1, std::memory_order_relaxed);
}

bool stdin_is_console() noexcept {
  DWORD mode = 0;
  const HANDLE in = GetStdHandle(STD_INPUT_HANDLE);
  return in != nullptr && in != INVALID_HANDLE_VALUE && GetConsoleMode(in, &mode) != 0;
}

std::optional<std::string> read_line_utf8() {
  std::string line;
  if (stdin_is_console()) {
    const HANDLE in = GetStdHandle(STD_INPUT_HANDLE);
    std::wstring w;
    while (w.empty() || w.back() != L'\n') {
      wchar_t buf[256];
      DWORD n = 0;
      // Fails or returns nothing when Ctrl+C interrupts the read.
      if (!ReadConsoleW(in, buf, 256, &n, nullptr) || n == 0) return std::nullopt;
      w.append(buf, n);
    }
    while (!w.empty() && (w.back() == L'\n' || w.back() == L'\r')) w.pop_back();
    if (w == L"\x1a") return std::nullopt;  // Ctrl+Z, Enter: end of input
    if (!w.empty()) {
      // Unpaired surrogates become U+FFFD, so the result is always valid UTF-8.
      const int len =
          WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
      line.resize(static_cast<std::size_t>(len));
      WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), line.data(), len, nullptr, nullptr);
    }
    return line;
  }
  int c = 0;
  bool any = false;
  while ((c = std::fgetc(stdin)) != EOF) {
    any = true;
    if (c == '\n') break;
    line += static_cast<char>(c);
  }
  if (!any) return std::nullopt;
  if (!line.empty() && line.back() == '\r') line.pop_back();
  return line;
}

PerformanceMode::PerformanceMode() {
  SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED);
  PROCESS_POWER_THROTTLING_STATE state{};
  state.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
  state.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
  state.StateMask = 0;  // 0 with the bit in ControlMask = never throttle this process
  SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &state, sizeof(state));
}

PerformanceMode::~PerformanceMode() { SetThreadExecutionState(ES_CONTINUOUS); }

PowerSource power_source() noexcept {
  SYSTEM_POWER_STATUS s{};
  if (!GetSystemPowerStatus(&s)) return PowerSource::kUnknown;
  if (s.ACLineStatus == 1) return PowerSource::kAc;
  if (s.ACLineStatus == 0) return PowerSource::kBattery;
  return PowerSource::kUnknown;
}

const char* to_string(PowerSource s) noexcept {
  switch (s) {
    case PowerSource::kAc:
      return "AC";
    case PowerSource::kBattery:
      return "battery";
    case PowerSource::kUnknown:
      return "unknown";
  }
  return "unknown";
}

std::string os_version_string() {
  // RtlGetVersion reports the true version (GetVersionEx lies to unmanifested apps).
  using RtlGetVersionFn = LONG(WINAPI*)(OSVERSIONINFOW*);
  OSVERSIONINFOW v{};
  v.dwOSVersionInfoSize = sizeof(v);
  if (HMODULE ntdll = GetModuleHandleW(L"ntdll.dll")) {
    // Through void*: a direct cast between unrelated function-pointer types
    // trips -Wcast-function-type; this is the documented GetProcAddress idiom.
    // NOLINTNEXTLINE(bugprone-casting-through-void)
    auto fn = reinterpret_cast<RtlGetVersionFn>(reinterpret_cast<void*>(GetProcAddress(ntdll, "RtlGetVersion")));
    if (fn) fn(&v);
  }
  DWORD ubr = 0;
  DWORD size = sizeof(ubr);
  RegGetValueW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion", L"UBR", RRF_RT_REG_DWORD,
               nullptr, &ubr, &size);
  return std::format("{}.{}.{}.{}", v.dwMajorVersion, v.dwMinorVersion, v.dwBuildNumber, ubr);
}

MemoryStatus memory_status() {
  MemoryStatus m;
  MEMORYSTATUSEX ms{};
  ms.dwLength = sizeof(ms);
  if (GlobalMemoryStatusEx(&ms)) {
    m.total_bytes = ms.ullTotalPhys;
    m.available_bytes = ms.ullAvailPhys;
    m.commit_limit_bytes = ms.ullTotalPageFile;
  }
  PROCESS_MEMORY_COUNTERS_EX pmc{};
  if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc))) {
    m.commit_bytes = pmc.PrivateUsage;
    m.working_set_bytes = pmc.WorkingSetSize;
  }
  return m;
}

void release_from_working_set(std::span<const std::byte> bytes) noexcept {
  if (bytes.empty()) return;
  // Documented behaviour of VirtualUnlock on pages that are not locked: it
  // removes them from the working set and fails with ERROR_NOT_LOCKED, which
  // is the expected result here.
  VirtualUnlock(const_cast<std::byte*>(bytes.data()), bytes.size());
}

void precise_sleep_us(long long us) {
  if (us <= 0) return;
  if (us < 500) {
    LARGE_INTEGER f, t0, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t0);
    const long long target = t0.QuadPart + us * f.QuadPart / 1'000'000;
    do {
      YieldProcessor();
      QueryPerformanceCounter(&t);
    } while (t.QuadPart < target);
    return;
  }
  thread_local HANDLE timer =
      CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
  if (timer == nullptr) {
    Sleep(static_cast<DWORD>((us + 999) / 1000));
    return;
  }
  LARGE_INTEGER due;
  due.QuadPart = -us * 10;  // relative time, in 100 ns units
  SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE);
  WaitForSingleObject(timer, INFINITE);
}

std::filesystem::path path_from_utf8(std::string_view utf8) {
  return std::filesystem::path(std::u8string(reinterpret_cast<const char8_t*>(utf8.data()), utf8.size()));
}

std::string path_to_utf8(const std::filesystem::path& path) {
  const std::u8string s = path.u8string();
  return std::string(reinterpret_cast<const char*>(s.data()), s.size());
}

}  // namespace hearth
