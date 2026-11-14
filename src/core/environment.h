// The conditions a measurement was taken under. Every probe
// and benchmark result carries one of these, so a number can always be traced
// back to the build, the machine state and what else was running.
#pragma once

#include <string>
#include <vector>

namespace hearth {

class JsonWriter;

struct BusyProcess {
  std::string name;
  double cpu_percent = 0;  // share of the whole machine (all logical processors)
};

// One populated memory slot, from the SMBIOS type-17 records.
struct Dimm {
  std::string locator;  // e.g. "DIMM A"
  unsigned size_mb = 0;
  std::string type;             // e.g. "DDR4"
  unsigned speed_mts = 0;       // rated
  unsigned configured_mts = 0;  // actually configured by the firmware
};

struct Environment {
  std::string time_utc;  // ISO 8601
  std::string commit;
  std::string compiler;
  std::string build_type;
  bool sanitized = false;
  std::string os;
  std::string power;
  std::string power_overlay;      // Windows power mode slider ("balanced", "best performance", ...)
  std::string gpu_hw_scheduling;  // hardware-accelerated GPU scheduling: on/off/default
  std::vector<Dimm> dimms;
  std::string cpu;
  int physical_cores = 0;
  int logical_cores = 0;
  double ram_total_gib = 0;
  double ram_available_gib = 0;
  std::string vk_loader_layers_disable;  // value of VK_LOADER_LAYERS_DISABLE ("" if unset)
  std::vector<BusyProcess> busy;         // other processes using >= 2 % of the machine
  // GPU fields are filled by callers that use the GPU.
  std::string gpu;
  std::string gpu_driver;
  std::string gpu_luid;
};

// Captures everything except the GPU fields. Samples process CPU use for
// `sample_ms` milliseconds (0 skips the busy-process scan).
Environment capture_environment(int sample_ms);

void write_json(JsonWriter& w, const Environment& env);

// Other processes' CPU use over `sample_ms`, as a share of the whole machine.
std::vector<BusyProcess> busy_processes(int sample_ms, double min_percent);

std::vector<Dimm> memory_modules();
std::string power_overlay_name();
std::string gpu_hw_scheduling();

}  // namespace hearth
