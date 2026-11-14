#include "core/environment.h"

#include <windows.h>
// (separate block: tlhelp32.h needs windows.h first, and the formatter sorts within a block)
#include <tlhelp32.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <format>
#include <map>

#include "core/cpu_features.h"
#include "core/json_writer.h"
#include "core/platform.h"
#include "hearth/version.h"

namespace hearth {
namespace {

std::string narrow(const wchar_t* w) {
  const int len = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
  if (len <= 1) return {};
  std::string s(static_cast<std::size_t>(len - 1), '\0');
  WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), len, nullptr, nullptr);
  return s;
}

std::uint64_t filetime_100ns(const FILETIME& t) {
  return (static_cast<std::uint64_t>(t.dwHighDateTime) << 32) | t.dwLowDateTime;
}

// CPU time (user + kernel, in 100 ns units) of every process we may query.
std::map<DWORD, std::pair<std::string, std::uint64_t>> process_cpu_times() {
  std::map<DWORD, std::pair<std::string, std::uint64_t>> out;
  HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
  if (snap == INVALID_HANDLE_VALUE) return out;
  PROCESSENTRY32W pe{};
  pe.dwSize = sizeof(pe);
  const DWORD self = GetCurrentProcessId();
  for (BOOL ok = Process32FirstW(snap, &pe); ok; ok = Process32NextW(snap, &pe)) {
    if (pe.th32ProcessID == 0 || pe.th32ProcessID == self) continue;
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pe.th32ProcessID);
    if (h == nullptr) continue;  // protected/system processes: not visible without admin rights
    FILETIME created, exited, kernel, user;
    if (GetProcessTimes(h, &created, &exited, &kernel, &user)) {
      out[pe.th32ProcessID] = {narrow(pe.szExeFile), filetime_100ns(kernel) + filetime_100ns(user)};
    }
    CloseHandle(h);
  }
  CloseHandle(snap);
  return out;
}

// SMBIOS memory type codes (DMTF DSP0134, table "Memory Device - Type").
const char* smbios_memory_type(unsigned code) {
  switch (code) {
    case 0x18:
      return "DDR3";
    case 0x1A:
      return "DDR4";
    case 0x1B:
      return "LPDDR";
    case 0x1C:
      return "LPDDR2";
    case 0x1D:
      return "LPDDR3";
    case 0x1E:
      return "LPDDR4";
    case 0x22:
      return "DDR5";
    case 0x23:
      return "LPDDR5";
    default:
      return "other";
  }
}

}  // namespace

std::vector<Dimm> memory_modules() {
  std::vector<Dimm> out;
  const UINT size = GetSystemFirmwareTable('RSMB', 0, nullptr, 0);
  if (size < 8) return out;
  std::vector<std::uint8_t> buf(size);
  if (GetSystemFirmwareTable('RSMB', 0, buf.data(), size) != size) return out;
  // RawSMBIOSData: 4 header bytes, a 32-bit table length, then the table.
  std::uint32_t len = 0;
  std::memcpy(&len, buf.data() + 4, 4);
  const std::uint8_t* p = buf.data() + 8;
  const std::uint8_t* end = p + std::min<std::size_t>(len, size - 8);
  while (p + 4 <= end) {
    const std::uint8_t type = p[0];
    const std::uint8_t flen = p[1];
    if (flen < 4 || p + flen > end) break;
    // The formatted area is followed by NUL-terminated strings, ending in "\0\0".
    const std::uint8_t* strings = p + flen;
    const std::uint8_t* next = strings;
    while (next + 1 < end && !(next[0] == 0 && next[1] == 0)) ++next;
    next += 2;
    auto str = [&](std::uint8_t idx) -> std::string {
      if (idx == 0) return {};
      const std::uint8_t* s = strings;
      for (std::uint8_t i = 1; i < idx && s < next; ++i) s += std::strlen(reinterpret_cast<const char*>(s)) + 1;
      return s < next ? std::string(reinterpret_cast<const char*>(s)) : std::string();
    };
    auto u16 = [&](int off) -> unsigned {
      if (off + 2 > flen) return 0;
      return static_cast<unsigned>(p[off] | (p[off + 1] << 8));
    };
    if (type == 17 && flen >= 0x15) {
      unsigned mb = u16(0x0C);
      if (mb == 0x7FFF && flen >= 0x20) {
        std::uint32_t ext = 0;
        std::memcpy(&ext, p + 0x1C, 4);
        mb = ext & 0x7FFFFFFFu;
      } else if ((mb & 0x8000u) != 0) {
        mb = (mb & 0x7FFFu) / 1024;  // the value was in KB
      }
      if (mb != 0 && mb != 0xFFFF) {
        Dimm d;
        d.locator = str(p[0x10]);
        d.size_mb = mb;
        d.type = smbios_memory_type(p[0x12]);
        d.speed_mts = u16(0x15);
        d.configured_mts = flen >= 0x22 ? u16(0x20) : 0;
        out.push_back(d);
      }
    }
    if (type == 127) break;  // end-of-table marker
    p = next;
  }
  return out;
}

std::string power_overlay_name() {
  // PowerGetEffectiveOverlayScheme exists since Windows 10 1709; loaded at run
  // time so an older SDK does not matter.
  using Fn = DWORD(WINAPI*)(GUID*);
  HMODULE lib = LoadLibraryW(L"powrprof.dll");
  if (lib == nullptr) return "unknown";
  // NOLINTNEXTLINE(bugprone-casting-through-void): GetProcAddress idiom, see platform.cpp
  auto fn = reinterpret_cast<Fn>(reinterpret_cast<void*>(GetProcAddress(lib, "PowerGetEffectiveOverlayScheme")));
  std::string name = "unknown";
  GUID g{};
  if (fn != nullptr && fn(&g) == ERROR_SUCCESS) {
    // GUIDs from Microsoft's power-mode documentation.
    static constexpr GUID kEfficiency = {0x961cc777, 0x2547, 0x4f9d, {0x81, 0x74, 0x7d, 0x86, 0x18, 0x1b, 0x8a, 0x7a}};
    static constexpr GUID kBetterPerf = {0x3af9b8d9, 0x7c97, 0x431d, {0xad, 0x78, 0x34, 0xa8, 0xbf, 0xea, 0x43, 0x9f}};
    static constexpr GUID kBestPerf = {0xded574b5, 0x45a0, 0x4f42, {0x87, 0x37, 0x46, 0x34, 0x5c, 0x09, 0xc2, 0x38}};
    static constexpr GUID kNone = {};
    if (IsEqualGUID(g, kNone))
      name = "balanced";
    else if (IsEqualGUID(g, kEfficiency))
      name = "best power efficiency";
    else if (IsEqualGUID(g, kBetterPerf))
      name = "better performance";
    else if (IsEqualGUID(g, kBestPerf))
      name = "best performance";
    else
      name = "other";
  }
  FreeLibrary(lib);
  return name;
}

std::string gpu_hw_scheduling() {
  DWORD v = 0;
  DWORD size = sizeof(v);
  if (RegGetValueW(HKEY_LOCAL_MACHINE, L"SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers", L"HwSchMode",
                   RRF_RT_REG_DWORD, nullptr, &v, &size) != ERROR_SUCCESS) {
    return "default";
  }
  return v == 2 ? "on" : v == 1 ? "off" : "unknown";
}

std::vector<BusyProcess> busy_processes(int sample_ms, double min_percent) {
  std::vector<BusyProcess> out;
  if (sample_ms <= 0) return out;
  const auto before = process_cpu_times();
  Sleep(static_cast<DWORD>(sample_ms));
  const auto after = process_cpu_times();
  const double capacity = static_cast<double>(sample_ms) * 1e4 * cpu_features().logical_cores;  // in 100 ns units
  for (const auto& [pid, now] : after) {
    const auto it = before.find(pid);
    if (it == before.end() || now.second < it->second.second) continue;
    const double pct = 100.0 * static_cast<double>(now.second - it->second.second) / capacity;
    if (pct >= min_percent) out.push_back({now.first, pct});
  }
  std::sort(out.begin(), out.end(),
            [](const BusyProcess& a, const BusyProcess& b) { return a.cpu_percent > b.cpu_percent; });
  return out;
}

Environment capture_environment(int sample_ms) {
  Environment e;
  const auto now = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
  e.time_utc = std::format("{:%Y-%m-%dT%H:%M:%SZ}", now);
  const BuildInfo& b = build_info();
  e.commit = b.commit;
  e.compiler = b.compiler;
  e.build_type = b.build_type;
  e.sanitized = b.sanitized;
  e.os = os_version_string();
  e.power = to_string(power_source());
  e.power_overlay = power_overlay_name();
  e.gpu_hw_scheduling = gpu_hw_scheduling();
  e.dimms = memory_modules();
  const CpuFeatures& f = cpu_features();
  e.cpu = f.brand;
  e.physical_cores = f.physical_cores;
  e.logical_cores = f.logical_cores;
  const MemoryStatus m = memory_status();
  e.ram_total_gib = static_cast<double>(m.total_bytes) / (1024.0 * 1024 * 1024);
  e.ram_available_gib = static_cast<double>(m.available_bytes) / (1024.0 * 1024 * 1024);
  wchar_t buf[256];
  const DWORD n = GetEnvironmentVariableW(L"VK_LOADER_LAYERS_DISABLE", buf, 256);
  if (n > 0 && n < 256) e.vk_loader_layers_disable = narrow(buf);
  e.busy = busy_processes(sample_ms, 2.0);
  return e;
}

void write_json(JsonWriter& w, const Environment& e) {
  w.begin_object();
  w.field("time_utc", e.time_utc);
  w.field("commit", e.commit);
  w.field("compiler", e.compiler);
  w.field("build_type", e.build_type);
  w.field("sanitized", e.sanitized);
  w.field("os", e.os);
  w.field("power", e.power);
  w.field("power_overlay", e.power_overlay);
  w.field("gpu_hw_scheduling", e.gpu_hw_scheduling);
  w.key("dimms");
  w.begin_array();
  for (const Dimm& d : e.dimms) {
    w.begin_object();
    w.field("locator", d.locator);
    w.field("size_mb", d.size_mb);
    w.field("type", d.type);
    w.field("speed_mts", d.speed_mts);
    w.field("configured_mts", d.configured_mts);
    w.end_object();
  }
  w.end_array();
  w.field("cpu", e.cpu);
  w.field("physical_cores", e.physical_cores);
  w.field("logical_cores", e.logical_cores);
  w.field("ram_total_gib", e.ram_total_gib);
  w.field("ram_available_gib", e.ram_available_gib);
  w.field("vk_loader_layers_disable", e.vk_loader_layers_disable);
  w.field("gpu", e.gpu);
  w.field("gpu_driver", e.gpu_driver);
  w.field("gpu_luid", e.gpu_luid);
  w.key("busy_processes");
  w.begin_array();
  for (const BusyProcess& p : e.busy) {
    w.begin_object();
    w.field("name", p.name);
    w.field("cpu_percent", p.cpu_percent);
    w.end_object();
  }
  w.end_array();
  w.end_object();
}

}  // namespace hearth
