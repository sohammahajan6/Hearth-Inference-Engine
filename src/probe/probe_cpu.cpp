// CPU sections of `hearth probe`: read bandwidth and compute peaks.
#include <windows.h>
// (separate block: psapi.h needs windows.h first)
#include <psapi.h>

#include <algorithm>
#include <format>

#include "core/clock.h"
#include "core/cpu_features.h"
#include "core/page_buffer.h"
#include "core/platform.h"
#include "core/thread_pool.h"
#include "cpu/probe_kernels.h"
#include "gpu/buffer.h"
#include "gpu/context.h"
#include "probe/probe_internal.h"

namespace hearth::probe {
namespace {

DWORD page_faults() {
  PROCESS_MEMORY_COUNTERS c{};
  c.cb = sizeof(c);
  GetProcessMemoryInfo(GetCurrentProcess(), &c, sizeof(c));
  return c.PageFaultCount;
}

enum class LoadKind { kNormal, kStreaming };

struct ReadSpec {
  const std::byte* data;
  std::size_t bytes;   // bytes read per repetition
  std::size_t stride;  // distance between the start of consecutive repetitions (0 = same range)
  std::size_t span;    // total bytes the rotating start cycles through
  LoadKind load = LoadKind::kNormal;
};

// GB/s samples for the pool's threads XOR-reading `spec.bytes`, one
// contiguous slice per thread. Repetitions that took a page fault are
// discarded (they would time the OS, not the memory); the count is noted.
void measure_cpu_read(Measurement& m, ThreadPool& pool, const ReadSpec& spec, int reps, int warmup) {
  const Isa isa = best_isa();
  std::vector<std::uint64_t> sink(static_cast<std::size_t>(pool.size()));
  int discarded = 0;
  std::size_t offset = 0;
  for (int rep = 0, kept = 0; kept < reps && rep < 4 * (reps + warmup); ++rep) {
    throw_if_cancelled();
    const std::byte* base = spec.data + offset;
    const DWORD f0 = page_faults();
    const std::int64_t t0 = now_ns();
    pool.run([&](int i, int n) {
      std::int64_t b = 0, e = 0;
      // Slices are whole 256-byte units so every thread's loop is exact.
      ThreadPool::split_range(static_cast<std::int64_t>(spec.bytes / 256), n, i, b, e);
      const std::byte* p = base + b * 256;
      const auto len = static_cast<std::size_t>(e - b) * 256;
      sink[static_cast<std::size_t>(i)] ^=
          spec.load == LoadKind::kStreaming ? cpu::probe_read_xor_streaming(p, len) : cpu::probe_read_xor(isa, p, len);
    });
    const std::int64_t t1 = now_ns();
    // Next start: advanced by the stride, wrapped inside the span, and kept on
    // a 4 KiB boundary (the kernels need 64-byte alignment; the modulus is
    // not a multiple of 64, so the raw remainder would not be).
    if (spec.stride != 0) offset = ((offset + spec.stride) % (spec.span - spec.bytes + 1)) & ~std::size_t{4095};
    if (rep < warmup) continue;
    if (page_faults() != f0) {
      ++discarded;
      continue;
    }
    m.samples.push_back(static_cast<double>(spec.bytes) / static_cast<double>(t1 - t0));
    ++kept;
  }
  std::uint64_t x = 0;
  for (std::uint64_t s : sink) x ^= s;
  m.params.push_back({"checksum", std::format("{:016x}", x)});  // keeps the reads observable
  if (discarded > 0) m.note += std::format("{}{} reps discarded (page faults)", m.note.empty() ? "" : "; ", discarded);
}

void fill_pattern(std::byte* p, std::size_t bytes) {
  auto* w = reinterpret_cast<std::uint64_t*>(p);
  for (std::size_t i = 0; i < bytes / 8; ++i) w[i] = 0x9E3779B97F4A7C15ull * (i + 1);
}

std::string plural_threads(int n) { return std::format("{} thread{}", n, n == 1 ? "" : "s"); }

}  // namespace

void probe_cpu_bw(Report& r, const Options& o, gpu::Context* gpu) {
  const std::size_t bytes = std::size_t{o.quick ? 256u : 1024u} << 20;
  const int max_threads = cpu_features().logical_cores;
  std::vector<int> thread_counts = {1, 2, 4};
  if (max_threads >= 8) thread_counts.push_back(8);

  PageBuffer host(bytes);
  fill_pattern(host.data(), bytes);
  for (int threads : thread_counts) {
    ThreadPool pool(threads);
    Measurement& m = r.add("cpu-bw", std::format("read, {}, virtualalloc", plural_threads(threads)), "GB/s");
    m.params = {{"threads", std::to_string(threads)}, {"memory", "virtualalloc"}, {"bytes", std::to_string(bytes)}};
    measure_cpu_read(m, pool, {host.data(), bytes, 0, bytes}, o.reps, 1);
  }

  // Cold size sweep, 4 threads: decode reads matrices of 1-20 MB, where the
  // fork/join cost of the pool matters. Each repetition starts 1 MiB past
  // the previous one's end, so a small read never hits data the previous one
  // (or a prefetcher running ahead) left in the cache.
  {
    ThreadPool pool(std::min(4, max_threads));
    for (std::size_t mib : {1u, 4u, 16u, 64u}) {
      const std::size_t len = mib << 20;
      Measurement& m = r.add("cpu-bw", std::format("cold read of {} MiB, 4 threads (rotating start)", mib), "GB/s");
      m.params = {{"bytes", std::to_string(len)}, {"threads", "4"}};
      measure_cpu_read(m, pool, {host.data(), len, len + (1u << 20), bytes}, std::max(o.reps, 9), 2);
    }
  }

  if (gpu == nullptr) return;
  // GPU-visible memory read by the CPU. The cached type is the candidate home
  // for shared weights. The write-combined type is measured to show why it is
  // not: CPU reads from it bypass the cache. It is read with streaming loads
  // (the fast way for such memory), on a small buffer.
  ThreadPool pool4(std::min(4, max_threads));
  {
    gpu::Buffer cached(*gpu, bytes, gpu::MemoryKind::kHostCached);
    fill_pattern(static_cast<std::byte*>(cached.mapped()), bytes);
    const std::string name = std::format("vulkan type {} (host-cached)", cached.memory_type());
    for (int threads : {1, 4}) {
      ThreadPool pool(threads);
      Measurement& m = r.add("cpu-bw", std::format("read, {}, {}", plural_threads(threads), name), "GB/s");
      m.params = {{"threads", std::to_string(threads)}, {"memory", name}};
      measure_cpu_read(m, pool, {static_cast<std::byte*>(cached.mapped()), bytes, 0, bytes}, o.reps, 1);
    }
  }
  {
    const std::size_t wc_bytes = 64u << 20;
    gpu::Buffer combined(*gpu, wc_bytes, gpu::MemoryKind::kHostCoherent);
    fill_pattern(static_cast<std::byte*>(combined.mapped()), wc_bytes);
    const std::string name = std::format("vulkan type {} (write-combined)", combined.memory_type());
    for (LoadKind load : {LoadKind::kNormal, LoadKind::kStreaming}) {
      Measurement& m =
          r.add("cpu-bw",
                std::format("read, 4 threads, {}, {} loads", name, load == LoadKind::kNormal ? "normal" : "streaming"),
                "GB/s");
      m.params = {{"threads", "4"}, {"memory", name}};
      m.note = "64 MiB buffer";
      measure_cpu_read(m, pool4, {static_cast<std::byte*>(combined.mapped()), wc_bytes, 0, wc_bytes, load},
                       std::min(o.reps, 3), 1);
    }
  }
  if (gpu->external_memory_host_enabled()) {
    PageBuffer import_host(bytes);
    fill_pattern(import_host.data(), bytes);
    const gpu::Buffer imported = gpu::Buffer::import_host(*gpu, import_host.data(), bytes);
    Measurement& m = r.add("cpu-bw", "read, 4 threads, virtualalloc imported into vulkan", "GB/s");
    m.params = {{"threads", "4"}, {"memory", "imported"}};
    measure_cpu_read(m, pool4, {import_host.data(), bytes, 0, bytes}, o.reps, 1);
  }
}

void probe_cpu_compute(Report& r, const Options& o) {
  if (!isa_supported(Isa::kAvx512)) {
    r.warnings.push_back("cpu-compute skipped: needs AVX-512 VNNI");
    return;
  }
  using cpu::VecWidth;
  std::uint32_t sink_u = 0;
  float sink_f = 0;
  for (VecWidth w : {VecWidth::kZmm, VecWidth::kYmm}) {
    const char* wname = w == VecWidth::kZmm ? "zmm" : "ymm";
    const double lanes = cpu::vec_lanes32(w);
    // Calibrate so one sample runs ~200 ms on one thread.
    auto calibrate = [](auto&& loop) {
      std::uint64_t iters = 1u << 16;
      for (;;) {
        const std::int64_t t0 = now_ns();
        loop(iters);
        const std::int64_t dt = now_ns() - t0;
        if (dt > 20'000'000) return iters * 200'000'000ull / static_cast<std::uint64_t>(dt);
        iters *= 4;
      }
    };
    const std::uint64_t vi = calibrate([&](std::uint64_t n) { sink_u ^= cpu::probe_vnni_loop(w, n, 1); });
    const std::uint64_t fi = calibrate([&](std::uint64_t n) { sink_f += cpu::probe_fma_loop(w, n, 1.0f); });

    // Scaling check (one thread): 2x the iterations must take 2x the time,
    // or the compiler removed part of the work.
    auto seconds = [](auto&& f) {
      const std::int64_t t0 = now_ns();
      f();
      return static_cast<double>(now_ns() - t0) / 1e9;
    };
    const double v1 = seconds([&] { sink_u ^= cpu::probe_vnni_loop(w, vi / 2, 5); });
    const double v2 = seconds([&] { sink_u ^= cpu::probe_vnni_loop(w, vi, 5); });
    const double f1 = seconds([&] { sink_f += cpu::probe_fma_loop(w, fi / 2, 5.0f); });
    const double f2 = seconds([&] { sink_f += cpu::probe_fma_loop(w, fi, 5.0f); });
    check_scaling(r, std::format("cpu VNNI {}", wname), v2 / v1);
    check_scaling(r, std::format("cpu FMA {}", wname), f2 / f1);

    for (int threads : {1, 4}) {
      ThreadPool pool(threads);
      std::vector<std::uint32_t> su(static_cast<std::size_t>(threads));
      std::vector<float> sf(static_cast<std::size_t>(threads));
      Measurement& vnni =
          r.add("cpu-compute", std::format("int8 VNNI (vpdpbusd {}), {}", wname, plural_threads(threads)), "TOPS");
      Measurement& fma =
          r.add("cpu-compute", std::format("fp32 FMA (vfmadd {}), {}", wname, plural_threads(threads)), "TFLOPS");
      vnni.params = {{"threads", std::to_string(threads)}, {"width", wname}, {"iters", std::to_string(vi)}};
      fma.params = {{"threads", std::to_string(threads)}, {"width", wname}, {"iters", std::to_string(fi)}};
      for (int rep = 0; rep < 1 + o.reps; ++rep) {
        throw_if_cancelled();
        std::int64_t t0 = now_ns();
        pool.run([&](int i, int) { su[static_cast<std::size_t>(i)] ^= cpu::probe_vnni_loop(w, vi, 3u + i); });
        std::int64_t dt = now_ns() - t0;
        if (rep > 0)
          vnni.samples.push_back(static_cast<double>(vi) * threads * 12 * lanes * 4 * 2 / static_cast<double>(dt) /
                                 1e3);
        t0 = now_ns();
        pool.run([&](int i, int) { sf[static_cast<std::size_t>(i)] += cpu::probe_fma_loop(w, fi, 1.0f + i); });
        dt = now_ns() - t0;
        if (rep > 0)
          fma.samples.push_back(static_cast<double>(fi) * threads * 12 * lanes * 2 / static_cast<double>(dt) / 1e3);
      }
      for (std::uint32_t v : su) sink_u ^= v;
      for (float v : sf) sink_f += v;
    }
  }
  // The sinks keep the loops' results observable; recording them costs nothing.
  r.measurements.back().params.push_back({"sink", std::format("{:08x}/{}", sink_u, sink_f)});
}

void check_scaling(Report& r, const std::string& what, double ratio) {
  Measurement& m = r.add("scaling", what + ": time(2N) / time(N)", "ratio");
  m.samples.push_back(ratio);
  if (ratio < 1.9 || ratio > 2.1) {
    r.warnings.push_back(
        std::format("{}: doubling the work changed the time by {:.3f}x (expected 2 +/- 5%); "
                    "part of the work may have been optimised away or the clock changed",
                    what, ratio));
  }
}

}  // namespace hearth::probe
