#include "probe/probe.h"

#include <algorithm>
#include <format>
#include <memory>

#include "core/cpu_features.h"
#include "core/error.h"
#include "core/log.h"
#include "core/platform.h"
#include "gpu/adapters.h"
#include "gpu/context.h"
#include "probe/probe_internal.h"

namespace hearth::probe {
namespace {

// Physical upper bounds for this machine. A measurement above one of them is
// wrong (work optimised away, a timer bug), never a fast machine.
//   DRAM: DDR4-3200, 2 channels x 8 bytes = 51.2 GB/s (2 DIMMs on 2 channels, see SMBIOS in the environment).
//   CPU:  per core and clock, 32 FP32 FLOPs or 128 int8 ops (zmm on 1 port or ymm on 2), at most 4.2 GHz.
//   GPU:  80 EUs x 16 FP32 FLOPs per clock at most 1.3 GHz; fp16 2x, int8 dot 4x.
constexpr double kDramPeakGBs = 51.2;
constexpr double kCpuMaxGHz = 4.2;
constexpr double kGpuFp32PeakT = 80 * 16 * 1.3e9 / 1e12;

bool wants(const std::vector<std::string>& sections, const char* name) {
  return std::find(sections.begin(), sections.end(), name) != sections.end();
}

void sanity_checks(Report& r) {
  const double cores = cpu_features().physical_cores;
  for (const Measurement& m : r.measurements) {
    if (m.samples.empty()) continue;
    double bound = 0;
    if (m.unit == "GB/s") bound = kDramPeakGBs;
    if (m.section == "cpu-compute" || (m.section == "throttle" && m.name.rfind("CPU", 0) == 0)) {
      if (m.unit == "TOPS") bound = cores * 128 * kCpuMaxGHz / 1e3;
      if (m.unit == "TFLOPS") bound = cores * 32 * kCpuMaxGHz / 1e3;
    }
    if (m.section == "gpu-compute" || (m.section == "throttle" && m.name.rfind("GPU", 0) == 0)) {
      if (m.unit == "TFLOPS") bound = m.name.find("fp16") != std::string::npos ? 2 * kGpuFp32PeakT : kGpuFp32PeakT;
      if (m.unit == "TOPS") bound = 4 * kGpuFp32PeakT;
    }
    if (bound > 0 && m.max() > bound) {
      r.warnings.push_back(std::format("{} / {}: {:.4g} {} exceeds the physical bound {:.4g}; the measurement is wrong",
                                       m.section, m.name, m.max(), m.unit, bound));
    }
  }
}

}  // namespace

const std::vector<std::string>& default_sections() {
  static const std::vector<std::string> s = {"cpu-bw", "gpu-bw",      "concurrent-bw", "latency",
                                             "clocks", "cpu-compute", "gpu-compute",   "memory-acct"};
  return s;
}

const std::vector<std::string>& all_sections() {
  static const std::vector<std::string> s = [] {
    auto v = default_sections();
    v.push_back("throttle");
    v.push_back("layout");
    v.push_back("qgemm");
    v.push_back("gpu-qgemm");
    v.push_back("gpu-placement");
    return v;
  }();
  return s;
}

Report run(const Options& options) {
  const std::vector<std::string>& sections = options.sections.empty() ? default_sections() : options.sections;
  for (const std::string& s : sections) {
    if (!wants(all_sections(), s.c_str())) fail(ErrorKind::kInvalidArgument, "unknown probe section '{}'", s);
  }
  PerformanceMode perf;
  Report r;
  r.env = capture_environment(1000);

  std::unique_ptr<gpu::Context> gpu;
  if (gpu::vulkan_available()) {
    // No validation layer: it would perturb every timing. The same shaders run
    // under validation in the test suite.
    gpu = std::make_unique<gpu::Context>(gpu::ContextOptions{});
    r.env.gpu = gpu->info().name;
    r.env.gpu_driver = gpu->info().driver_version_text;
    r.env.gpu_luid = gpu::luid_pdh_string(gpu->info().luid);
  }
  auto need_gpu = [&](const char* s) {
    if (gpu) return true;
    r.warnings.push_back(std::format("{} skipped: no Vulkan GPU", s));
    return false;
  };

  for (const std::string& s : sections) {
    log_info("probe: {}", s);
    if (s == "cpu-bw") probe_cpu_bw(r, options, gpu.get());
    if (s == "cpu-compute") probe_cpu_compute(r, options);
    if (s == "qgemm") probe_qgemm(r, options);
    if (s == "gpu-bw" && need_gpu("gpu-bw")) probe_gpu_bw(r, options, *gpu);
    if (s == "concurrent-bw" && need_gpu("concurrent-bw")) probe_concurrent_bw(r, options, *gpu);
    if (s == "latency" && need_gpu("latency")) probe_latency(r, options, *gpu);
    if (s == "gpu-compute" && need_gpu("gpu-compute")) probe_gpu_compute(r, options, *gpu);
    if (s == "throttle" && need_gpu("throttle")) probe_throttle(r, options, *gpu);
    if (s == "clocks" && need_gpu("clocks")) probe_clocks(r, *gpu);
    if (s == "memory-acct" && need_gpu("memory-acct")) probe_memory_accounting(r, *gpu);
    if (s == "layout" && need_gpu("layout")) probe_layout(r, options, *gpu);
    if (s == "gpu-qgemm" && need_gpu("gpu-qgemm")) probe_gpu_qgemm(r, options, *gpu);
    if (s == "gpu-placement" && need_gpu("gpu-placement")) probe_gpu_placement(r, options, *gpu);
  }
  sanity_checks(r);
  return r;
}

}  // namespace hearth::probe
