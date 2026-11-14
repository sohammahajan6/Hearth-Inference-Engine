// GPU sections of `hearth probe`: GPU read bandwidth, CPU+GPU concurrent
// bandwidth, dispatch/submit/wake latencies, clock checks, compute peaks,
// throttling and memory accounting. Design and review: docs/architecture/02-probe.md.
//
// TDR safety: every dispatch is sized to finish in well under 100 ms (Windows
// resets the GPU after 2 s). Compute dispatches are sized by doubling from a
// tiny one, never by an estimate.
#include <windows.h>
// (separate block: keep windows.h first)
#include <immintrin.h>
#include <psapi.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <format>
#include <functional>
#include <thread>

#include "core/clock.h"
#include "core/cpu_features.h"
#include "core/error.h"
#include "core/page_buffer.h"
#include "core/pdh.h"
#include "core/platform.h"
#include "core/thread_pool.h"
#include "cpu/probe_kernels.h"
#include "gpu/buffer.h"
#include "gpu/context.h"
#include "gpu/pipeline.h"
#include "gpu/sync.h"
#include "gpu/timestamps.h"
#include "probe/probe_internal.h"

namespace hearth::probe {
namespace {

struct ReadPush {
  std::uint32_t n, magic, write_all, base, blocked;
};
struct FmaPush {
  std::uint32_t iters;
  float mul, add, magic;
};
struct DotPush {
  std::uint32_t iters;
  std::int32_t b, magic;
};

constexpr std::uint32_t kInvocationsPerGroup = 256;
constexpr std::uint32_t kMaxReadGroups = 4096;
constexpr std::uint32_t kComputeGroups = 4096;
constexpr std::uint32_t kNeverMagic = 0xDEADBEEFu;  // probe shaders store only if a result equals this

void fill_gpu(gpu::Context& ctx, const gpu::Buffer& b) {
  ctx.run_once([&](VkCommandBuffer cmd) {
    vkCmdFillBuffer(cmd, b.handle(), 0, VK_WHOLE_SIZE, 0x5A5A5A5Au);
    gpu::compute_barrier(cmd);
  });
}

void fill_pattern(std::byte* p, std::size_t bytes) {
  auto* w = reinterpret_cast<std::uint64_t*>(p);
  for (std::size_t i = 0; i < bytes / 8; ++i) w[i] = 0x9E3779B97F4A7C15ull * (i + 1);
}

double sum(const std::vector<double>& v) {
  double s = 0;
  for (double x : v) s += x;
  return s;
}

// Reads buffers on the GPU, timed with GPU timestamps.
class GpuReader {
 public:
  static constexpr std::uint32_t kMaxBatch = 16;

  explicit GpuReader(gpu::Context& ctx)
      : ctx_(ctx),
        pipe_(ctx, {.shader = "probe_read", .buffers = 2, .push_constant_bytes = sizeof(ReadPush)}),
        pool_(ctx, 16, 32),
        dst_(ctx, VkDeviceSize{kMaxReadGroups} * kInvocationsPerGroup * 16, gpu::MemoryKind::kDeviceLocal),
        ts_(ctx, 2 * kMaxBatch) {}

  VkDescriptorSet bind(const gpu::Buffer& src) { return pool_.allocate(pipe_, {&src, &dst_}); }

  // `count` reads of `bytes` each, back to back in one submission, each
  // between its own timestamps. Read i starts at byte offset base + i*stride.
  // Returns the on-GPU nanoseconds of each read.
  std::vector<double> read(VkDescriptorSet set, VkDeviceSize bytes, std::uint32_t groups, std::uint32_t count,
                           VkDeviceSize base = 0, VkDeviceSize stride = 0, bool blocked = false) {
    HEARTH_CHECK(count <= kMaxBatch && groups <= kMaxReadGroups);
    ts_.reset();
    ctx_.run_once([&](VkCommandBuffer cmd) {
      for (std::uint32_t i = 0; i < count; ++i) {
        const ReadPush push{static_cast<std::uint32_t>(bytes / 16), kNeverMagic, 0,
                            static_cast<std::uint32_t>((base + i * stride) / 16), blocked ? 1u : 0u};
        ts_.write(cmd, 2 * i);
        pipe_.dispatch(cmd, set, &push, sizeof(push), groups);
        ts_.write(cmd, 2 * i + 1);
      }
    });
    const auto t = ts_.read(0, 2 * count);
    std::vector<double> ns(count);
    for (std::uint32_t i = 0; i < count; ++i) ns[i] = ts_.elapsed_ns(t[2 * i], t[2 * i + 1]);
    return ns;
  }

  // `count` reads of `bytes`, read i at byte offset base + i*stride, each
  // separated by a barrier (as dependent decode steps are), timed by two
  // timestamps around the whole batch. Returns the mean on-GPU ns per read.
  // Per-read timestamps are avoided: without barriers consecutive dispatches
  // overlap, and the per-dispatch times would then not add up.
  double read_chain(VkDescriptorSet set, VkDeviceSize bytes, std::uint32_t groups, std::uint32_t count,
                    VkDeviceSize base, VkDeviceSize stride, bool blocked = false) {
    HEARTH_CHECK(groups <= kMaxReadGroups);
    ts_.reset();
    ctx_.run_once([&](VkCommandBuffer cmd) {
      ts_.write(cmd, 0);
      for (std::uint32_t i = 0; i < count; ++i) {
        const ReadPush push{static_cast<std::uint32_t>(bytes / 16), kNeverMagic, 0,
                            static_cast<std::uint32_t>((base + i * stride) / 16), blocked ? 1u : 0u};
        pipe_.dispatch(cmd, set, &push, sizeof(push), groups);
        gpu::compute_barrier(cmd);
      }
      ts_.write(cmd, 1);
    });
    const auto t = ts_.read(0, 2);
    return ts_.elapsed_ns(t[0], t[1]) / count;
  }

 private:
  gpu::Context& ctx_;
  gpu::Pipeline pipe_;
  gpu::DescriptorPool pool_;
  gpu::Buffer dst_;
  gpu::TimestampPool ts_;
};

// Runs a compute-peak shader; returns on-GPU ns per dispatch.
class GpuCompute {
 public:
  GpuCompute(gpu::Context& ctx, const char* shader, std::uint32_t push_bytes, std::uint32_t subgroup_size)
      : ctx_(ctx),
        pipe_(ctx, {.shader = shader,
                    .buffers = 1,
                    .push_constant_bytes = push_bytes,
                    .spec_constants = {},
                    .required_subgroup_size = subgroup_size}),
        pool_(ctx, 1, 1),
        dst_(ctx, VkDeviceSize{kComputeGroups} * kInvocationsPerGroup * 16, gpu::MemoryKind::kDeviceLocal),
        ts_(ctx, 2 * 32) {
    set_ = pool_.allocate(pipe_, {&dst_});
  }

  std::vector<double> run(const void* push, std::uint32_t push_bytes, std::uint32_t count) {
    HEARTH_CHECK(count <= 32);
    ts_.reset();
    ctx_.run_once([&](VkCommandBuffer cmd) {
      for (std::uint32_t i = 0; i < count; ++i) {
        ts_.write(cmd, 2 * i);
        pipe_.dispatch(cmd, set_, push, push_bytes, kComputeGroups);
        ts_.write(cmd, 2 * i + 1);
        gpu::compute_barrier(cmd);  // the dispatches (in principle) write the same output
      }
    });
    const auto t = ts_.read(0, 2 * count);
    std::vector<double> ns(count);
    for (std::uint32_t i = 0; i < count; ++i) ns[i] = ts_.elapsed_ns(t[2 * i], t[2 * i + 1]);
    return ns;
  }

 private:
  gpu::Context& ctx_;
  gpu::Pipeline pipe_;
  gpu::DescriptorPool pool_;
  gpu::Buffer dst_;
  gpu::TimestampPool ts_;
  VkDescriptorSet set_ = VK_NULL_HANDLE;
};

// Doubles `push.iters` from 16 until one dispatch takes at least 4 ms, then
// scales to ~target_ms (at most 4x the last measured step). Never guesses a
// large dispatch, so a slow GPU cannot be pushed into a 2 s TDR timeout.
template <class Push>
void size_dispatch(GpuCompute& g, Push& push, double target_ms) {
  push.iters = 16;
  for (;;) {
    const double ns = g.run(&push, sizeof(push), 1)[0];
    if (ns >= 4e6 || push.iters >= (1u << 20)) {
      const double scale = std::min(4.0, target_ms * 1e6 / ns);
      push.iters = static_cast<std::uint32_t>(std::max(16.0, push.iters * scale));
      return;
    }
    push.iters *= 2;
  }
}

std::uint32_t best_read_groups(Report& r, GpuReader& reader, VkDescriptorSet set, VkDeviceSize bytes) {
  std::uint32_t best = 1024;
  double best_rate = 0;
  for (std::uint32_t groups : {256u, 512u, 1024u, 2048u, 4096u}) {
    reader.read(set, bytes, groups, 2);  // warm-up (GPU clock ramp)
    Measurement& m = r.add("gpu-bw", std::format("occupancy sweep: {} workgroups x 256", groups), "GB/s");
    m.params = {{"workgroups", std::to_string(groups)}};
    for (double ns : reader.read(set, bytes, groups, 4)) m.samples.push_back(static_cast<double>(bytes) / ns);
    if (m.median() > best_rate) {
      best_rate = m.median();
      best = groups;
    }
  }
  return best;
}

}  // namespace

// ---------------------------------------------------------------- gpu-bw
void probe_gpu_bw(Report& r, const Options& o, gpu::Context& ctx) {
  const VkDeviceSize bytes = VkDeviceSize{o.quick ? 128u : 512u} << 20;
  GpuReader reader(ctx);
  gpu::Buffer cached(ctx, bytes, gpu::MemoryKind::kHostCached);
  fill_gpu(ctx, cached);
  const VkDescriptorSet cached_set = reader.bind(cached);
  const std::uint32_t groups = best_read_groups(r, reader, cached_set, bytes);

  {  // Read-size scaling, recorded but not judged: for reads, optimised-away
     // work is excluded by the checksum test (gpu_probe_shaders_test), and
     // departures from 2.0 here are physical (DRAM page locality and cache
     // effects differ with the read footprint). Both sizes walk the whole
     // buffer once, so neither re-reads data the cache may have kept.
    reader.read_chain(cached_set, bytes / 8, groups, 8, 0, bytes / 8);
    const double eighth = reader.read_chain(cached_set, bytes / 8, groups, 8, 0, bytes / 8);
    const double quarter = reader.read_chain(cached_set, bytes / 4, groups, 4, 0, bytes / 4);
    Measurement& m = r.add("scaling", "gpu read (grid-stride): time(2N) / time(N), informational", "ratio");
    m.samples.push_back(quarter / eighth);
  }

  gpu::Buffer local(ctx, bytes, gpu::MemoryKind::kDeviceLocal);
  gpu::Buffer coherent(ctx, bytes, gpu::MemoryKind::kHostCoherent);
  fill_gpu(ctx, local);
  fill_gpu(ctx, coherent);
  struct Source {
    std::string name;
    VkDescriptorSet set;
  };
  std::vector<Source> sources = {
      {std::format("vulkan type {} (device-local)", local.memory_type()), reader.bind(local)},
      {std::format("vulkan type {} (host-coherent)", coherent.memory_type()), reader.bind(coherent)},
      {std::format("vulkan type {} (host-cached)", cached.memory_type()), cached_set},
  };
  PageBuffer host;
  gpu::Buffer imported;
  if (ctx.external_memory_host_enabled()) {
    host = PageBuffer(static_cast<std::size_t>(bytes));
    fill_pattern(host.data(), host.size());
    imported = gpu::Buffer::import_host(ctx, host.data(), bytes);
    sources.push_back(
        {std::format("virtualalloc imported (memory type {})", imported.memory_type()), reader.bind(imported)});
  }
  for (const Source& s : sources) {
    throw_if_cancelled();
    reader.read(s.set, bytes, groups, 3);  // warm-up
    Measurement& m = r.add("gpu-bw", std::format("read, {}", s.name), "GB/s");
    m.params = {{"memory", s.name}, {"bytes", std::to_string(bytes)}, {"workgroups", std::to_string(groups)}};
    for (double ns : reader.read(s.set, bytes, groups, static_cast<std::uint32_t>(std::min(o.reps, 16)))) {
      m.samples.push_back(static_cast<double>(bytes) / ns);
    }
    // The access pattern of a matvec kernel: each workgroup reads its own
    // contiguous rows. Swept over workgroup counts because the best count
    // differs from the grid-stride pattern's.
    double best_blocked = 0;
    std::uint32_t best_blocked_groups = 0;
    for (std::uint32_t bg : {256u, 1024u, 4096u}) {
      reader.read(s.set, bytes, bg, 2, 0, 0, true);
      std::vector<double> rates;
      for (double ns : reader.read(s.set, bytes, bg, static_cast<std::uint32_t>(std::min(o.reps, 16)), 0, 0, true)) {
        rates.push_back(static_cast<double>(bytes) / ns);
      }
      std::sort(rates.begin(), rates.end());
      if (rates[rates.size() / 2] > best_blocked) {
        best_blocked = rates[rates.size() / 2];
        best_blocked_groups = bg;
      }
    }
    Measurement& mb = r.add("gpu-bw", std::format("read, {}, blocked per workgroup (matvec-like)", s.name), "GB/s");
    mb.params = {{"memory", s.name}, {"workgroups", std::to_string(best_blocked_groups)}};
    reader.read(s.set, bytes, best_blocked_groups, 2, 0, 0, true);
    for (double ns :
         reader.read(s.set, bytes, best_blocked_groups, static_cast<std::uint32_t>(std::min(o.reps, 16)), 0, 0, true)) {
      mb.samples.push_back(static_cast<double>(bytes) / ns);
    }
    mb.note = std::format("best of 256/1024/4096 workgroups: {}", best_blocked_groups);
  }

  // Cold size sweep: decode reads matrices of 1-20 MB one after another, with
  // a dependency (barrier) between them, so the fixed cost of a dispatch is a
  // large share. Consecutive reads start 1 MiB past the previous one's end,
  // so none hits data a previous read (or a prefetcher) left in the 8 MB LLC.
  for (VkDeviceSize mib : {1u, 4u, 16u, 64u}) {
    const VkDeviceSize len = mib << 20;
    const auto g =
        static_cast<std::uint32_t>(std::clamp<VkDeviceSize>(len / (kInvocationsPerGroup * 16 * 8), 16, groups));
    const VkDeviceSize stride = len + (1u << 20);
    const auto count = static_cast<std::uint32_t>(std::min<VkDeviceSize>(32, bytes / stride));
    if (count == 0) continue;
    for (bool blocked : {false, true}) {
      reader.read_chain(cached_set, len, g, count, 0, stride, blocked);  // warm-up
      Measurement& m = r.add("gpu-bw",
                             std::format("cold {} MiB reads, chained with barriers, {}", mib,
                                         blocked ? "blocked per workgroup" : "grid-stride"),
                             "GB/s");
      m.params = {{"bytes", std::to_string(len)}, {"workgroups", std::to_string(g)}, {"reads", std::to_string(count)}};
      for (int round = 0; round < std::max(3, o.reps); ++round) {
        m.samples.push_back(static_cast<double>(len) /
                            reader.read_chain(cached_set, len, g, count, 0, stride, blocked));
      }
    }
  }
}

// ---------------------------------------------------------------- concurrent-bw
void probe_concurrent_bw(Report& r, const Options& o, gpu::Context& ctx) {
  // CPU and GPU read *different* buffers at the same time, for each candidate
  // home of the shared weights, with 1/2/4 CPU threads. The GPU is fed from a
  // dedicated thread (not a pool worker), which sleeps in fence waits.
  const std::size_t bytes = std::size_t{o.quick ? 256u : 512u} << 20;
  const double window_s = o.quick ? 2.0 : 4.0;
  constexpr std::uint32_t kGroups = 1024;
  const Isa isa = best_isa();
  GpuReader reader(ctx);

  struct Placement {
    std::string name;
    std::byte* cpu_data;
    VkDescriptorSet gpu_set;
  };
  // Placement A: both devices use Vulkan host-cached memory (one buffer each).
  gpu::Buffer cpu_cached(ctx, bytes, gpu::MemoryKind::kHostCached);
  gpu::Buffer gpu_cached(ctx, bytes, gpu::MemoryKind::kHostCached);
  fill_pattern(static_cast<std::byte*>(cpu_cached.mapped()), bytes);
  fill_gpu(ctx, gpu_cached);
  std::vector<Placement> placements = {
      {"vulkan host-cached", static_cast<std::byte*>(cpu_cached.mapped()), reader.bind(gpu_cached)}};
  // Placement B: both use VirtualAlloc pages; the GPU's are imported.
  PageBuffer cpu_pages, gpu_pages;
  gpu::Buffer gpu_imported;
  if (ctx.external_memory_host_enabled()) {
    cpu_pages = PageBuffer(bytes);
    gpu_pages = PageBuffer(bytes);
    fill_pattern(cpu_pages.data(), bytes);
    fill_pattern(gpu_pages.data(), bytes);
    gpu_imported = gpu::Buffer::import_host(ctx, gpu_pages.data(), bytes);
    placements.push_back({"virtualalloc (GPU side imported)", cpu_pages.data(), reader.bind(gpu_imported)});
  }

  for (const Placement& pl : placements) {
    Measurement& gpu_alone = r.add("concurrent-bw", std::format("GPU alone, {}", pl.name), "GB/s");
    reader.read(pl.gpu_set, bytes, kGroups, 3);
    for (double ns : reader.read(pl.gpu_set, bytes, kGroups, 8)) {
      gpu_alone.samples.push_back(static_cast<double>(bytes) / ns);
    }
    for (int threads : {1, 2, 4}) {
      throw_if_cancelled();
      ThreadPool pool(threads);
      std::vector<std::uint64_t> sink(static_cast<std::size_t>(threads));
      auto cpu_read_once = [&] {
        pool.run([&](int i, int n) {
          std::int64_t b = 0, e = 0;
          ThreadPool::split_range(static_cast<std::int64_t>(bytes / 256), n, i, b, e);
          sink[static_cast<std::size_t>(i)] ^=
              cpu::probe_read_xor(isa, pl.cpu_data + b * 256, static_cast<std::size_t>(e - b) * 256);
        });
      };
      Measurement& cpu_alone =
          r.add("concurrent-bw", std::format("CPU alone, {} thread(s), {}", threads, pl.name), "GB/s");
      for (int rep = 0; rep < 1 + o.reps; ++rep) {
        const std::int64_t t0 = now_ns();
        cpu_read_once();
        if (rep > 0) cpu_alone.samples.push_back(static_cast<double>(bytes) / static_cast<double>(now_ns() - t0));
      }

      // Together: the GPU thread keeps the GPU busy from before the CPU window
      // opens until after it closes; only GPU batches entirely inside the
      // window are counted.
      struct Batch {
        std::int64_t start, end;
        std::vector<double> ns;
      };
      std::vector<Batch> batches;
      std::atomic<bool> stop{false};
      std::atomic<int> started{0};
      std::exception_ptr gpu_error;
      std::thread gpu_thread([&] {
        try {
          while (!stop.load(std::memory_order_acquire)) {
            Batch b;
            b.start = now_ns();
            b.ns = reader.read(pl.gpu_set, bytes, kGroups, 4);
            b.end = now_ns();
            batches.push_back(std::move(b));
            started.fetch_add(1, std::memory_order_release);
          }
        } catch (...) {
          gpu_error = std::current_exception();
          started.store(1 << 20);
        }
      });
      while (started.load(std::memory_order_acquire) < 2) std::this_thread::yield();
      Measurement& cpu_conc =
          r.add("concurrent-bw", std::format("CPU while GPU reads, {} thread(s), {}", threads, pl.name), "GB/s");
      const std::int64_t w0 = now_ns();
      while (static_cast<double>(now_ns() - w0) < window_s * 1e9) {
        const std::int64_t t0 = now_ns();
        cpu_read_once();
        cpu_conc.samples.push_back(static_cast<double>(bytes) / static_cast<double>(now_ns() - t0));
      }
      const std::int64_t w1 = now_ns();
      stop.store(true, std::memory_order_release);
      gpu_thread.join();
      if (gpu_error) std::rethrow_exception(gpu_error);
      Measurement& gpu_conc =
          r.add("concurrent-bw", std::format("GPU while CPU reads, {} thread(s), {}", threads, pl.name), "GB/s");
      for (const Batch& b : batches) {
        if (b.start >= w0 && b.end <= w1) {
          for (double ns : b.ns) gpu_conc.samples.push_back(static_cast<double>(bytes) / ns);
        }
      }
      if (gpu_conc.samples.empty()) {
        r.warnings.push_back(std::format("concurrent-bw: no GPU batch inside the CPU window ({})", pl.name));
        continue;
      }
      Measurement& total =
          r.add("concurrent-bw", std::format("together: CPU + GPU, {} thread(s), {}", threads, pl.name), "GB/s");
      total.samples.push_back(cpu_conc.median() + gpu_conc.median());
      total.note = std::format("best device alone {:.1f}", std::max(cpu_alone.median(), gpu_alone.median()));
    }
  }
}

// ---------------------------------------------------------------- latency
void probe_latency(Report& r, const Options& o, gpu::Context& ctx) {
  const int n = o.quick ? 500 : 2000;
  const int warmup = 50;
  gpu::Buffer dummy(ctx, 256, gpu::MemoryKind::kDeviceLocal);
  gpu::Pipeline empty(ctx, {.shader = "probe_empty", .buffers = 1, .push_constant_bytes = 4});
  gpu::DescriptorPool pool(ctx, 1, 1);
  const VkDescriptorSet set = pool.allocate(empty, {&dummy});
  const std::uint32_t magic = 0xFFFFFFFFu;  // no invocation has this id
  auto record_empty = [&](gpu::CommandBuffer& cb) {
    VkCommandBuffer cmd = cb.begin();
    empty.dispatch(cmd, set, &magic, sizeof(magic), 1);
    cb.end();
  };
  gpu::CommandBuffer cb(ctx);
  record_empty(cb);

  auto us = [](std::int64_t a, std::int64_t b) { return static_cast<double>(b - a) / 1e3; };
  auto add = [&](const std::string& name) -> Measurement& { return r.add("latency", name, "us"); };
  std::vector<Measurement*> with_p99;

  Measurement& submit_call = add("vkQueueSubmit2 call (CPU time)");
  Measurement& rt_block = add("submit -> fence, blocking wait");
  for (int i = 0; i < warmup + n; ++i) {
    const std::int64_t t0 = now_ns();
    cb.submit();
    const std::int64_t t1 = now_ns();
    cb.wait();
    const std::int64_t t2 = now_ns();
    if (i >= warmup) {
      submit_call.samples.push_back(us(t0, t1));
      rt_block.samples.push_back(us(t0, t2));
    }
  }
  Measurement& rt_spin = add("submit -> fence, spinning");
  for (int i = 0; i < warmup + n; ++i) {
    const std::int64_t t0 = now_ns();
    cb.submit();
    while (!cb.done()) _mm_pause();
    if (i >= warmup) rt_spin.samples.push_back(us(t0, now_ns()));
  }
  with_p99 = {&submit_call, &rt_block, &rt_spin};

  // After the GPU has been idle: power gating and clock drop may add latency.
  for (long long gap_us : {100LL, 1000LL, 10000LL}) {
    const int k = gap_us >= 10000 ? n / 10 : gap_us >= 1000 ? n / 4 : n / 2;
    Measurement& m = add(std::format("submit -> fence, blocking, after {} us idle", gap_us));
    for (int i = 0; i < k; ++i) {
      precise_sleep_us(gap_us);
      const std::int64_t t0 = now_ns();
      cb.submit();
      cb.wait();
      m.samples.push_back(us(t0, now_ns()));
    }
    with_p99.push_back(&m);
  }

  // While 3 other cores stream memory (as they would during hearth work).
  {
    const std::size_t bytes = 256u << 20;
    PageBuffer load_buf(bytes);
    fill_pattern(load_buf.data(), bytes);
    std::atomic<bool> stop{false};
    std::vector<std::thread> load;
    const int others = std::min(3, cpu_features().physical_cores - 1);
    const DWORD_PTR old_mask = SetThreadAffinityMask(GetCurrentThread(), thread_affinity_mask(0));
    for (int t = 1; t <= others; ++t) {
      load.emplace_back([&, t] {
        SetThreadAffinityMask(GetCurrentThread(), thread_affinity_mask(t));
        std::uint64_t x = 0;
        while (!stop.load(std::memory_order_relaxed)) x ^= cpu::probe_read_xor(best_isa(), load_buf.data(), bytes);
        if (x == 42) std::abort();  // keeps the loop observable; practically never true
      });
    }
    Measurement& m = add(std::format("submit -> fence, blocking, {} other cores streaming memory", others));
    for (int i = 0; i < warmup + n / 2; ++i) {
      const std::int64_t t0 = now_ns();
      cb.submit();
      cb.wait();
      if (i >= warmup) m.samples.push_back(us(t0, now_ns()));
    }
    stop.store(true);
    for (auto& t : load) t.join();
    if (old_mask != 0) SetThreadAffinityMask(GetCurrentThread(), old_mask);
    with_p99.push_back(&m);
  }

  gpu::TimelineSemaphore done(ctx);
  std::uint64_t v = 0;
  Measurement& sem_block = add("submit -> timeline semaphore, blocking wait");
  for (int i = 0; i < warmup + n; ++i) {
    ++v;
    const gpu::SemaphoreValue sig{done.handle(), v};
    const std::int64_t t0 = now_ns();
    cb.submit({}, {&sig, 1});
    done.wait(v);
    if (i >= warmup) sem_block.samples.push_back(us(t0, now_ns()));
    cb.wait();
  }
  Measurement& sem_spin = add("submit -> timeline semaphore, spinning");
  for (int i = 0; i < warmup + n; ++i) {
    ++v;
    const gpu::SemaphoreValue sig{done.handle(), v};
    const std::int64_t t0 = now_ns();
    cb.submit({}, {&sig, 1});
    while (done.value() < v) _mm_pause();
    if (i >= warmup) sem_spin.samples.push_back(us(t0, now_ns()));
    cb.wait();
  }
  with_p99.push_back(&sem_block);
  with_p99.push_back(&sem_spin);

  // Wait-before-signal: GPU work submitted first, released later by the host.
  // (a) Does the submit call itself block? (b) Does a waiting submission hold
  // up later, independent submissions on the same (only) queue?
  gpu::TimelineSemaphore go(ctx);
  {
    gpu::CommandBuffer other(ctx);
    record_empty(other);
    ++v;
    const gpu::SemaphoreValue wait{go.handle(), v};
    const gpu::SemaphoreValue sig{done.handle(), v};
    const std::int64_t t0 = now_ns();
    cb.submit({&wait, 1}, {&sig, 1});
    const std::int64_t t1 = now_ns();
    other.submit();
    bool other_finished = false;
    for (const std::int64_t deadline = now_ns() + 200'000'000; now_ns() < deadline;) {
      if (other.done()) {
        other_finished = true;
        break;
      }
      precise_sleep_us(1000);
    }
    go.signal(v);
    done.wait(v);
    cb.wait();
    other.wait();
    Measurement& call = add("vkQueueSubmit2 call with an unsignalled wait (CPU time)");
    call.samples.push_back(us(t0, t1));
    Measurement& blocks = r.add("latency", "later submit finishes while an earlier one waits (1 = yes)", "bool");
    blocks.samples.push_back(other_finished ? 1.0 : 0.0);
    blocks.note = other_finished ? "independent work is not blocked" : "a waiting submission blocks the queue";
  }

  // Host signal -> GPU work done, observed by the CPU spinning. A helper
  // thread raises `go` (recording the time); the main thread spins on `done`.
  // Signalling from another thread keeps this correct even if the driver
  // blocked inside vkQueueSubmit2 until the wait is met.
  gpu::TimestampPool start_ts(ctx, 1);
  Measurement& wake = add("host signal -> GPU work done -> CPU sees it");
  Measurement* one_way = nullptr;
  if (ctx.calibrated_timestamps_enabled()) one_way = &add("host signal -> GPU starts the released work (one way)");
  {
    gpu::CommandBuffer gated(ctx);
    VkCommandBuffer cmd = gated.begin();
    start_ts.write(cmd, 0);
    empty.dispatch(cmd, set, &magic, sizeof(magic), 1);
    gated.end();
    const int k = n / 4;
    for (int i = 0; i < warmup + k; ++i) {
      ++v;
      start_ts.reset();
      std::atomic<std::int64_t> t_signal{0};
      std::thread signaller([&, value = v] {
        precise_sleep_us(300);
        t_signal.store(now_ns(), std::memory_order_release);
        go.signal(value);
      });
      const gpu::SemaphoreValue wait{go.handle(), v};
      const gpu::SemaphoreValue sig{done.handle(), v};
      gated.submit({&wait, 1}, {&sig, 1});
      const std::int64_t deadline = now_ns() + 5'000'000'000LL;
      while (done.value() < v) {
        _mm_pause();
        if (now_ns() > deadline) {
          signaller.join();
          fail(ErrorKind::kDeviceLost, "GPU did not run work released by a host semaphore signal within 5 s");
        }
      }
      const std::int64_t t1 = now_ns();
      signaller.join();
      gated.wait();
      if (i < warmup) continue;
      const std::int64_t ts_cpu = t_signal.load(std::memory_order_acquire);
      wake.samples.push_back(us(ts_cpu, t1));
      if (one_way != nullptr) {
        const auto pair = ctx.sample_clocks();
        const std::uint64_t start = start_ts.read(0, 1)[0];
        const double gpu_to_now_ns = start_ts.elapsed_ns(start, pair.gpu_ticks);  // GPU start -> pair sample
        const double start_cpu_ns = static_cast<double>(pair.cpu_ns) - gpu_to_now_ns;
        one_way->samples.push_back((start_cpu_ns - static_cast<double>(ts_cpu)) / 1e3);
      }
    }
  }
  with_p99.push_back(&wake);
  if (one_way != nullptr) with_p99.push_back(one_way);

  // On-GPU cost of back-to-back dispatches, with and without a barrier between
  // (two timestamps around N dispatches: per-dispatch timestamps would drain
  // the pipeline on Intel and inflate the result).
  gpu::TimestampPool ts(ctx, 3);
  constexpr int kChain = 200;
  Measurement& with_barrier = add("on-GPU time per empty dispatch, barrier between");
  Measurement& no_barrier = add("on-GPU time per empty dispatch, no barrier");
  for (int rep = 0; rep < 2 + o.reps; ++rep) {
    ts.reset();
    ctx.run_once([&](VkCommandBuffer c) {
      ts.write(c, 0);
      for (int k = 0; k < kChain; ++k) {
        empty.dispatch(c, set, &magic, sizeof(magic), 1);
        gpu::compute_barrier(c);
      }
      ts.write(c, 1);
      for (int k = 0; k < kChain; ++k) empty.dispatch(c, set, &magic, sizeof(magic), 1);
      ts.write(c, 2);
    });
    const auto t = ts.read(0, 3);
    if (rep >= 2) {
      with_barrier.samples.push_back(ts.elapsed_ns(t[0], t[1]) / kChain / 1e3);
      no_barrier.samples.push_back(ts.elapsed_ns(t[1], t[2]) / kChain / 1e3);
    }
  }

  // CPU thread-pool fork/join: hot (back to back) and after the workers have
  // gone to sleep (1 ms idle is past their ~100 us spin window).
  {
    ThreadPool tp(std::min(4, cpu_features().logical_cores));
    Measurement& hot = add(std::format("thread pool fork/join, {} threads, hot", tp.size()));
    Measurement& cold = add(std::format("thread pool fork/join, {} threads, after 1 ms idle", tp.size()));
    std::atomic<int> sink{0};
    for (int i = 0; i < warmup + n; ++i) {
      const std::int64_t t0 = now_ns();
      tp.run([&](int idx, int) { sink.fetch_add(idx, std::memory_order_relaxed); });
      if (i >= warmup) hot.samples.push_back(us(t0, now_ns()));
    }
    for (int i = 0; i < n / 4; ++i) {
      precise_sleep_us(1000);
      const std::int64_t t0 = now_ns();
      tp.run([&](int idx, int) { sink.fetch_add(idx, std::memory_order_relaxed); });
      cold.samples.push_back(us(t0, now_ns()));
    }
    with_p99.push_back(&hot);
    with_p99.push_back(&cold);
  }
  for (Measurement* m : with_p99) m->note = std::format("p99 {:.1f}", m->percentile(99));
}

// ---------------------------------------------------------------- clocks
void probe_clocks(Report& r, gpu::Context& ctx) {
  // Is timestampPeriod right, and does the GPU counter keep ticking while the
  // GPU idles? Compare GPU ticks with QPC over a busy span and an idle span.
  if (!ctx.calibrated_timestamps_enabled()) {
    r.warnings.push_back("clocks: VK_EXT_calibrated_timestamps not available; GPU tick period not verified");
    return;
  }
  const double reported = ctx.info().timestamp_period_ns;
  const std::uint64_t mask =
      ctx.info().timestamp_valid_bits >= 64 ? ~0ull : ((1ull << ctx.info().timestamp_valid_bits) - 1);
  auto period = [&](const gpu::Context::ClockPair& a, const gpu::Context::ClockPair& b) {
    return static_cast<double>(b.cpu_ns - a.cpu_ns) / static_cast<double>((b.gpu_ticks - a.gpu_ticks) & mask);
  };
  Measurement& idle = r.add("clocks", "GPU tick period over 1 s idle (QPC reference)", "ns");
  Measurement& busy = r.add("clocks", "GPU tick period over 1 s busy (QPC reference)", "ns");
  Measurement& dev = r.add("clocks", "calibration max deviation", "us");
  for (int rep = 0; rep < 3; ++rep) {
    const auto a = ctx.sample_clocks();
    precise_sleep_us(1'000'000);
    const auto b = ctx.sample_clocks();
    idle.samples.push_back(period(a, b));
    dev.samples.push_back(static_cast<double>(b.max_deviation_ns) / 1e3);
  }
  GpuCompute g(ctx, "probe_fma32", sizeof(FmaPush), 0);
  FmaPush push{16, 0.9999999f, 1e-7f, -1.0f};
  size_dispatch(g, push, 30.0);
  for (int rep = 0; rep < 3; ++rep) {
    const auto a = ctx.sample_clocks();
    const std::int64_t end = a.cpu_ns + 1'000'000'000;
    while (now_ns() < end) g.run(&push, sizeof(push), 8);
    const auto b = ctx.sample_clocks();
    busy.samples.push_back(period(a, b));
  }
  for (Measurement* m : {&idle, &busy}) {
    m->note = std::format("reported {:.4f}", reported);
    if (std::abs(m->median() - reported) / reported > 0.01) {
      r.warnings.push_back(std::format("clocks: measured GPU tick period {:.4f} ns differs from reported {:.4f} ns",
                                       m->median(), reported));
    }
  }
}

// ---------------------------------------------------------------- gpu-compute
namespace {

template <class Push>
std::vector<double> measure_gpu_peak(GpuCompute& g, Push& push, double ops_per_iter_invocation, int reps) {
  size_dispatch(g, push, 40.0);
  g.run(&push, sizeof(push), 2);  // warm-up at the final size
  std::vector<double> out;
  const double invocations = static_cast<double>(kComputeGroups) * kInvocationsPerGroup;
  for (double t : g.run(&push, sizeof(push), static_cast<std::uint32_t>(std::min(reps, 32)))) {
    out.push_back(invocations * push.iters * ops_per_iter_invocation / t / 1e3);
  }
  return out;
}

template <class Push>
double gpu_scaling_ratio(GpuCompute& g, Push push) {
  const double t1 = sum(g.run(&push, sizeof(push), 4));
  push.iters *= 2;
  const double t2 = sum(g.run(&push, sizeof(push), 4));
  return t2 / t1;
}

}  // namespace

void probe_gpu_compute(Report& r, const Options& o, gpu::Context& ctx) {
  // Each shader at every subgroup size the GPU allows: the compiler's choice
  // of SIMD width changes register pressure, and the best width is part of
  // what this section has to find out.
  std::vector<std::uint32_t> sizes;
  for (std::uint32_t sg = ctx.info().min_subgroup_size; sg <= ctx.info().max_subgroup_size && sg != 0; sg *= 2) {
    sizes.push_back(sg);
  }
  struct Kind {
    const char* label;
    const char* shader;
    const char* unit;
    bool supported;
  };
  const Kind kinds[] = {
      {"fp32 FMA (vec4 fma x8 chains)", "probe_fma32", "TFLOPS", true},
      {"fp16 FMA (f16vec4 fma x8 chains)", "probe_fma16", "TFLOPS", ctx.info().shader_float16},
      {"int8 dot (dotPacked4x8EXT x8 chains)", "probe_dot8", "TOPS", ctx.info().integer_dot_product},
  };
  for (const Kind& k : kinds) {
    if (!k.supported) continue;
    const bool is_dot = std::string_view(k.shader) == "probe_dot8";
    double best = 0;
    std::uint32_t best_sg = 0;
    double best_ratio = 0;
    for (std::uint32_t sg : sizes) {
      throw_if_cancelled();
      Measurement& m = r.add("gpu-compute", std::format("{}, subgroup {}", k.label, sg), k.unit);
      m.params = {{"subgroup", std::to_string(sg)}};
      GpuCompute g(ctx, k.shader, is_dot ? sizeof(DotPush) : sizeof(FmaPush), sg);
      double ratio = 0;
      if (is_dot) {
        DotPush push{0, 0x01FF7F80, -1};
        m.samples = measure_gpu_peak(g, push, 64, o.reps);
        ratio = gpu_scaling_ratio(g, push);
      } else {
        FmaPush push = std::string_view(k.shader) == "probe_fma16" ? FmaPush{0, 0.999f, 1e-3f, -1.0f}
                                                                   : FmaPush{0, 0.9999999f, 1e-7f, -1.0f};
        m.samples = measure_gpu_peak(g, push, 64, o.reps);
        ratio = gpu_scaling_ratio(g, push);
      }
      if (m.median() > best) {
        best = m.median();
        best_sg = sg;
        best_ratio = ratio;
      }
    }
    check_scaling(r, std::format("gpu {} (subgroup {})", k.shader, best_sg), best_ratio);
    Measurement& peak = r.add("gpu-compute", std::format("{}: best", k.label), k.unit);
    peak.samples.push_back(best);
    peak.note = std::format("at subgroup {}", best_sg);
  }
}

// ---------------------------------------------------------------- throttle
void probe_throttle(Report& r, const Options& o, gpu::Context& ctx) {
  // Sustained behaviour, sampled about once a second, for compute and for
  // memory: CPU alone, GPU alone, both together. They share one power budget
  // and one cooler. Each run starts from the same temperature (the hottest
  // thermal zone back within 2 C of the idle baseline, at most 90 s).
  const int seconds = o.quick ? 15 : 45;
  const int threads = std::min(4, cpu_features().physical_cores);
  PdhQuery pdh;
  const int c_perf = pdh.add(L"\\Processor Information(_Total)\\% Processor Performance");
  const int c_gpu = pdh.add(L"\\GPU Engine(*)\\Utilization Percentage");
  const int c_temp = pdh.add(L"\\Thermal Zone Information(*)\\Temperature");
  // Intel RAPL energy counters, exposed by Windows as "Energy Meter" (mW).
  const int c_pkg = pdh.add(L"\\Energy Meter(rapl_package0_pkg)\\Power");
  const int c_cores = pdh.add(L"\\Energy Meter(rapl_package0_pp0)\\Power");
  const int c_gfx = pdh.add(L"\\Energy Meter(rapl_package0_pp1)\\Power");
  pdh.collect();
  const std::string pid_filter = std::format("pid_{}_", GetCurrentProcessId());
  auto temp_c = [&] {
    pdh.collect();
    const double k = pdh.max(c_temp);
    return std::isnan(k) ? k : k - 273.15;
  };
  precise_sleep_us(5'000'000);
  const double baseline = temp_c();
  if (std::isnan(baseline)) {
    r.warnings.push_back(
        "throttle: no thermal zone counter on this machine; temperature not measured, "
        "cool-down is a fixed 30 s");
  }

  ThreadPool pool(threads);
  const Isa isa = best_isa();
  std::vector<std::uint64_t> sink(static_cast<std::size_t>(threads));
  // CPU int8 compute: one sample = ~1 s of VNNI (zmm) on all threads.
  std::uint64_t vnni_iters = 1u << 20;
  {
    const std::int64_t t0 = now_ns();
    (void)cpu::probe_vnni_loop(cpu::VecWidth::kZmm, vnni_iters, 1);
    vnni_iters = static_cast<std::uint64_t>(static_cast<double>(vnni_iters) * 1e9 / static_cast<double>(now_ns() - t0));
  }
  auto cpu_compute = [&] {
    const std::int64_t t0 = now_ns();
    pool.run([&](int i, int) {
      sink[static_cast<std::size_t>(i)] ^= cpu::probe_vnni_loop(cpu::VecWidth::kZmm, vnni_iters, 7u + i);
    });
    return static_cast<double>(vnni_iters) * threads * 12 * 64 * 2 / static_cast<double>(now_ns() - t0) / 1e3;
  };
  // CPU memory: one sample = repeated 4-thread reads of 512 MiB for ~1 s.
  const std::size_t cpu_bytes = 512u << 20;
  PageBuffer cpu_buf(cpu_bytes);
  fill_pattern(cpu_buf.data(), cpu_bytes);
  auto cpu_read = [&] {
    double bytes = 0;
    const std::int64_t t0 = now_ns();
    while (now_ns() - t0 < 1'000'000'000) {
      pool.run([&](int i, int n) {
        std::int64_t b = 0, e = 0;
        ThreadPool::split_range(static_cast<std::int64_t>(cpu_bytes / 256), n, i, b, e);
        sink[static_cast<std::size_t>(i)] ^=
            cpu::probe_read_xor(isa, cpu_buf.data() + b * 256, static_cast<std::size_t>(e - b) * 256);
      });
      bytes += static_cast<double>(cpu_bytes);
    }
    return bytes / static_cast<double>(now_ns() - t0);
  };
  // GPU int8 dot (the arithmetic of ADR 0004), at the narrowest subgroup size.
  GpuCompute g(ctx, "probe_dot8", sizeof(DotPush), ctx.info().min_subgroup_size);
  DotPush dot{16, 0x01FF7F80, -1};
  size_dispatch(g, dot, 40.0);
  const double ops_per_dispatch = static_cast<double>(kComputeGroups) * kInvocationsPerGroup * dot.iters * 64;
  auto gpu_compute = [&] {
    const auto ns = g.run(&dot, sizeof(dot), 25);
    return ops_per_dispatch * 25 / sum(ns) / 1e3;
  };
  // GPU memory: one sample = 16 reads of 512 MiB of host-cached memory.
  GpuReader reader(ctx);
  const VkDeviceSize gpu_bytes = VkDeviceSize{512} << 20;
  gpu::Buffer gpu_buf(ctx, gpu_bytes, gpu::MemoryKind::kHostCached);
  fill_gpu(ctx, gpu_buf);
  const VkDescriptorSet gset = reader.bind(gpu_buf);
  auto gpu_read = [&] { return 16.0 * static_cast<double>(gpu_bytes) / sum(reader.read(gset, gpu_bytes, 1024, 16)); };

  auto cool_down = [&] {
    if (std::isnan(baseline)) {
      precise_sleep_us(30'000'000);
      return;
    }
    for (int s = 0; s < 90; ++s) {
      if (temp_c() <= baseline + 2.0) return;
      precise_sleep_us(1'000'000);
    }
  };
  struct Run {
    std::string name;
    std::string unit;
    std::function<double()> cpu, gpu;
  };
  const std::vector<Run> runs = {
      {"int8 compute", "TOPS", cpu_compute, gpu_compute},
      {"memory read", "GB/s", cpu_read, gpu_read},
  };
  for (const Run& run : runs) {
    for (int mode = 0; mode < 3; ++mode) {  // 0: CPU alone, 1: GPU alone, 2: both
      throw_if_cancelled();
      cool_down();
      const double start_temp = temp_c();
      const char* who = mode == 0 ? "CPU" : mode == 1 ? "GPU" : "both";
      const char* how = mode == 2 ? "together" : "alone";
      Measurement* cm =
          mode != 1 ? &r.add("throttle", std::format("CPU {} {}, {} s", run.name, how, seconds), run.unit) : nullptr;
      Measurement* gm =
          mode != 0 ? &r.add("throttle", std::format("GPU {} {}, {} s", run.name, how, seconds), run.unit) : nullptr;
      Measurement& perf = r.add("throttle", std::format("  {} ({}): CPU % processor performance", run.name, who), "%");
      Measurement& util =
          r.add("throttle", std::format("  {} ({}): GPU engine utilisation, this process", run.name, who), "%");
      Measurement& temp = r.add("throttle", std::format("  {} ({}): hottest thermal zone", run.name, who), "C");
      Measurement& pkg = r.add("throttle", std::format("  {} ({}): package power (RAPL)", run.name, who), "W");
      Measurement& cores = r.add("throttle", std::format("  {} ({}): CPU cores power (RAPL pp0)", run.name, who), "W");
      Measurement& gfx = r.add("throttle", std::format("  {} ({}): graphics power (RAPL pp1)", run.name, who), "W");
      std::atomic<bool> stop{false};
      std::exception_ptr gpu_error;
      std::thread gpu_thread;
      if (mode == 2) {
        gpu_thread = std::thread([&] {
          try {
            while (!stop.load(std::memory_order_acquire)) gm->samples.push_back(run.gpu());
          } catch (...) {
            gpu_error = std::current_exception();
          }
        });
      }
      for (int s = 0; s < seconds; ++s) {
        if (mode != 1) cm->samples.push_back(run.cpu());
        if (mode == 1) gm->samples.push_back(run.gpu());
        pdh.collect();
        perf.samples.push_back(pdh.sum(c_perf, {}));
        util.samples.push_back(pdh.sum(c_gpu, {pid_filter}));
        const double t = pdh.max(c_temp);
        if (!std::isnan(t)) temp.samples.push_back(t - 273.15);
        for (auto [m, c] : {std::pair{&pkg, c_pkg}, std::pair{&cores, c_cores}, std::pair{&gfx, c_gfx}}) {
          if (c >= 0) m->samples.push_back(pdh.sum(c, {}) / 1000.0);
        }
      }
      stop.store(true, std::memory_order_release);
      if (gpu_thread.joinable()) gpu_thread.join();
      if (gpu_error) std::rethrow_exception(gpu_error);
      for (Measurement* m : {cm, gm}) {
        if (m == nullptr || m->samples.size() < 15) continue;
        double burst = 0, sustained = 0;
        for (std::size_t i = 0; i < 5; ++i) burst += m->samples[i];
        for (std::size_t i = m->samples.size() - 10; i < m->samples.size(); ++i) sustained += m->samples[i];
        m->note = std::format("burst (first 5 s) {:.3g}, sustained (last 10 s) {:.3g}", burst / 5, sustained / 10);
      }
      temp.note = std::format("start {:.1f}, baseline {:.1f}", start_temp, baseline);
    }
  }
}

// ---------------------------------------------------------------- memory-accounting
void probe_memory_accounting(Report& r, gpu::Context& ctx) {
  // How Windows counts our memory: process commit (PrivateUsage) and the GPU
  // memory counters for this process, before and after allocating 512 MiB of
  // Vulkan host-cached memory and of imported VirtualAlloc pages. The Phase 6
  // zero-copy gate ("about 1x the weights") has to be read in these terms.
  const std::size_t bytes = 512u << 20;
  const double mib = 1024.0 * 1024.0;
  PdhQuery pdh;
  const int shared = pdh.add(L"\\GPU Process Memory(*)\\Shared Usage");
  const int dedicated = pdh.add(L"\\GPU Process Memory(*)\\Dedicated Usage");
  const std::string filter = std::format("pid_{}_", GetCurrentProcessId());
  struct Snap {
    double commit, shared, dedicated;
  };
  auto snapshot = [&] {
    precise_sleep_us(1'500'000);  // the GPU memory counters update about once a second
    pdh.collect();
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    pmc.cb = sizeof(pmc);
    GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc));
    return Snap{static_cast<double>(pmc.PrivateUsage) / mib, pdh.sum(shared, {filter}) / mib,
                pdh.sum(dedicated, {filter}) / mib};
  };
  const Snap base = snapshot();
  auto record = [&](const char* what, const Snap& after) {
    r.add("memory-acct", std::format("{}: process commit delta", what), "MiB")
        .samples.push_back(after.commit - base.commit);
    r.add("memory-acct", std::format("{}: GPU shared usage delta", what), "MiB")
        .samples.push_back(after.shared - base.shared);
    r.add("memory-acct", std::format("{}: GPU dedicated usage delta", what), "MiB")
        .samples.push_back(after.dedicated - base.dedicated);
  };
  {
    gpu::Buffer cached(ctx, bytes, gpu::MemoryKind::kHostCached);
    std::memset(cached.mapped(), 1, bytes);  // touch every page
    record("512 MiB vulkan host-cached", snapshot());
  }
  if (!ctx.external_memory_host_enabled()) return;
  PageBuffer pages(bytes);
  fill_pattern(pages.data(), bytes);
  record("512 MiB virtualalloc (not imported)", snapshot());
  gpu::Buffer imported = gpu::Buffer::import_host(ctx, pages.data(), bytes);
  record("512 MiB virtualalloc, imported into vulkan", snapshot());

  // CPU -> GPU canary on imported pages: the GPU must see what the CPU wrote.
  gpu::Buffer out(ctx, VkDeviceSize{64} * kInvocationsPerGroup * 16, gpu::MemoryKind::kHostCached);
  gpu::Pipeline pipe(ctx, {.shader = "probe_read", .buffers = 2, .push_constant_bytes = sizeof(ReadPush)});
  gpu::DescriptorPool dp(ctx, 1, 2);
  const VkDescriptorSet set = dp.allocate(pipe, {&imported, &out});
  const ReadPush push{static_cast<std::uint32_t>(bytes / 16), 0, 1, 0, 0};
  ctx.run_once([&](VkCommandBuffer cmd) {
    pipe.dispatch(cmd, set, &push, sizeof(push), 64);
    gpu::compute_to_host_barrier(cmd);
  });
  std::uint64_t gpu_x = 0, cpu_x = 0;
  const auto* w = static_cast<const std::uint64_t*>(out.mapped());
  for (std::size_t i = 0; i < std::size_t{64} * kInvocationsPerGroup * 2; ++i) gpu_x ^= w[i];
  const auto* h = reinterpret_cast<const std::uint64_t*>(pages.data());
  for (std::size_t i = 0; i < bytes / 8; ++i) cpu_x ^= h[i];
  r.add("memory-acct", "CPU->GPU canary on imported pages (1 = GPU saw CPU writes)", "bool")
      .samples.push_back(gpu_x == cpu_x ? 1.0 : 0.0);
  if (gpu_x != cpu_x) r.warnings.push_back("memory-acct: GPU read different data than the CPU wrote to imported pages");
}

}  // namespace hearth::probe
