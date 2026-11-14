// Section entry points shared by the probe's translation units.
#pragma once

#include "probe/probe.h"

namespace hearth::gpu {
class Context;
}

namespace hearth::probe {

// `gpu` may be null (no Vulkan): GPU-memory sources are then skipped.
void probe_cpu_bw(Report& r, const Options& o, gpu::Context* gpu);
void probe_cpu_compute(Report& r, const Options& o);
// The cpu backend's decode kernel on every 1B/3B matrix shape (probe_qgemm.cpp).
void probe_qgemm(Report& r, const Options& o);
void probe_gpu_bw(Report& r, const Options& o, gpu::Context& gpu);
void probe_concurrent_bw(Report& r, const Options& o, gpu::Context& gpu);
void probe_latency(Report& r, const Options& o, gpu::Context& gpu);
void probe_gpu_compute(Report& r, const Options& o, gpu::Context& gpu);
void probe_throttle(Report& r, const Options& o, gpu::Context& gpu);
void probe_clocks(Report& r, gpu::Context& gpu);
void probe_memory_accounting(Report& r, gpu::Context& gpu);
void probe_layout(Report& r, const Options& o, gpu::Context& gpu);
// The canonical GPU GEMM per tile shape on prefill shapes (probe_gpu_qgemm.cpp).
void probe_gpu_qgemm(Report& r, const Options& o, gpu::Context& gpu);
void probe_gpu_placement(Report& r, const Options& o, gpu::Context& gpu);

// Records time(2N)/time(N) for a kernel and warns unless it is 2 +/- 5 %:
// anything else means work was optimised away or the clock moved mid-test.
void check_scaling(Report& r, const std::string& what, double ratio);

}  // namespace hearth::probe
