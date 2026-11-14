// `hearth probe --sections layout`: the weight-layout experiment
//. GGUF array-of-blocks vs
// structure-of-arrays, for Q8_0 and Q4_0, on the CPU, the GPU and both at
// once, in both candidate shared-memory placements. Objective: estimated
// decode time per token over the 1B model's tensor list.
#include <algorithm>
#include <atomic>
#include <bit>
#include <cstring>
#include <format>
#include <memory>
#include <thread>

#include "core/clock.h"
#include "core/error.h"
#include "core/fp16.h"
#include "core/page_buffer.h"
#include "core/platform.h"
#include "core/quant.h"
#include "core/thread_pool.h"
#include "cpu/matvec.h"
#include "gpu/buffer.h"
#include "gpu/context.h"
#include "gpu/matvec.h"
#include "gpu/sync.h"
#include "gpu/timestamps.h"
#include "probe/probe_internal.h"

namespace hearth::probe {
namespace {

using quant::kBlock;

struct Shape {
  const char* name;
  std::int64_t rows, k;
  int per_token;  // how many times one decoded token multiplies by a matrix of this shape (1B model)
};
constexpr Shape kShapes[] = {
    {"q/o 2048x2048", 2048, 2048, 32},  {"k/v 512x2048", 512, 2048, 32},          {"gate/up 8192x2048", 8192, 2048, 32},
    {"down 2048x8192", 2048, 8192, 16}, {"lm_head 128256x2048", 128256, 2048, 1},
};

// Host memory for one placement: either Vulkan host-cached memory, or
// VirtualAlloc pages imported into Vulkan. Both are CPU-readable and GPU-readable.
struct HostGpuBuffer {
  PageBuffer pages;
  gpu::Buffer buffer;
  std::byte* data() const { return static_cast<std::byte*>(buffer.mapped()); }
};

HostGpuBuffer make_buffer(gpu::Context& ctx, std::size_t bytes, bool imported) {
  HostGpuBuffer b;
  const std::size_t rounded = (bytes + 65535) & ~std::size_t{65535};
  if (imported) {
    b.pages = PageBuffer(rounded);
    b.buffer = gpu::Buffer::import_host(ctx, b.pages.data(), rounded);
  } else {
    b.buffer = gpu::Buffer(ctx, rounded, gpu::MemoryKind::kHostCached);
  }
  return b;
}

// Fast deterministic filler (xorshift64*). Weight values do not affect speed,
// but scales must be normal numbers: denormal floats would trigger slow
// microcode paths in the CPU's FMA and distort the timing.
void fill_random(std::byte* p, std::size_t bytes, std::uint64_t seed) {
  std::uint64_t s = seed | 1;
  auto* w = reinterpret_cast<std::uint64_t*>(p);
  for (std::size_t i = 0; i < bytes / 8; ++i) {
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    w[i] = s * 2685821657736338717ull;
  }
}

constexpr std::uint16_t kNormalScale = 0x2000;  // 2^-7 as fp16

struct Pools {
  std::int64_t blocks_per_copy = 0;
  int copies = 0;
  std::size_t block_bytes = 0;  // AoS block size: 34 or 18
  std::size_t quant_bytes = 0;  // SoA quants per block: 32 or 16
  HostGpuBuffer aos, soa_d, soa_q;
};

Pools make_pools(gpu::Context& ctx, gpu::WeightFormat fmt, const Shape& sh, bool imported) {
  Pools p;
  p.block_bytes = fmt == gpu::WeightFormat::kQ8_0 ? sizeof(quant::BlockQ8_0) : sizeof(quant::BlockQ4_0);
  p.quant_bytes = fmt == gpu::WeightFormat::kQ8_0 ? 32 : 16;
  p.blocks_per_copy = sh.rows * (sh.k / kBlock);
  const std::size_t matrix_bytes = static_cast<std::size_t>(p.blocks_per_copy) * p.block_bytes;
  // >= 64 MiB of distinct copies and >= 2 copies: no matvec can reuse
  // weights the 8 MB last-level cache might still hold (review H1).
  p.copies = static_cast<int>(std::max<std::size_t>(2, ((64u << 20) + matrix_bytes - 1) / matrix_bytes));
  const std::size_t n = static_cast<std::size_t>(p.blocks_per_copy) * static_cast<std::size_t>(p.copies);
  p.aos = make_buffer(ctx, n * p.block_bytes, imported);
  p.soa_d = make_buffer(ctx, n * 2, imported);
  p.soa_q = make_buffer(ctx, n * p.quant_bytes, imported);
  fill_random(p.aos.data(), n * p.block_bytes, 11);
  fill_random(p.soa_q.data(), n * p.quant_bytes, 13);
  for (std::size_t i = 0; i < n; ++i) {
    std::memcpy(p.aos.data() + i * p.block_bytes, &kNormalScale, 2);
    std::memcpy(p.soa_d.data() + i * 2, &kNormalScale, 2);
  }
  return p;
}

// One measured configuration: format x shape x placement, both layouts, all modes.
struct Experiment {
  gpu::Context& ctx;
  gpu::MatvecPipelines& pipes;
  gpu::DescriptorPool& dpool;
  ThreadPool& pool;
  gpu::WeightFormat fmt;
  const Shape& sh;
  const Pools& pools;
  const quant::ActQ8& act;
  gpu::Buffer& xq;
  gpu::Buffer& xd;
  gpu::Buffer& xs;
  gpu::Buffer& ygpu;
  std::vector<float>& ycpu;

  std::int64_t nb() const { return sh.k / kBlock; }

  // CPU rows [r0, r1) of copy c, split across the pool.
  void cpu_rows(bool soa, int c, std::int64_t r0, std::int64_t r1) {
    const std::int64_t first = pools.blocks_per_copy * c;
    pool.run([&](int i, int n) {
      std::int64_t b = 0, e = 0;
      ThreadPool::split_range(r1 - r0, n, i, b, e);
      b += r0;
      e += r0;
      if (fmt == gpu::WeightFormat::kQ8_0) {
        if (soa) {
          const cpu::Q8_0Soa w{reinterpret_cast<const std::uint16_t*>(pools.soa_d.data()) + first,
                               reinterpret_cast<const std::int8_t*>(pools.soa_q.data()) + first * 32, sh.rows, sh.k};
          cpu::matvec(Isa::kAvx512, w, act, ycpu.data(), b, e);
        } else {
          const cpu::Q8_0Aos w{reinterpret_cast<const quant::BlockQ8_0*>(pools.aos.data()) + first, sh.rows, sh.k};
          cpu::matvec(Isa::kAvx512, w, act, ycpu.data(), b, e);
        }
      } else {
        if (soa) {
          const cpu::Q4_0Soa w{reinterpret_cast<const std::uint16_t*>(pools.soa_d.data()) + first,
                               reinterpret_cast<const std::uint8_t*>(pools.soa_q.data()) + first * 16, sh.rows, sh.k};
          cpu::matvec(Isa::kAvx512, w, act, ycpu.data(), b, e);
        } else {
          const cpu::Q4_0Aos w{reinterpret_cast<const quant::BlockQ4_0*>(pools.aos.data()) + first, sh.rows, sh.k};
          cpu::matvec(Isa::kAvx512, w, act, ycpu.data(), b, e);
        }
      }
    });
  }

  gpu::MatvecPush push(bool soa, int c, std::int64_t r0, std::int64_t r1) const {
    const std::int64_t first = pools.blocks_per_copy * c;
    const std::uint32_t words = static_cast<std::uint32_t>(pools.block_bytes / 2);
    gpu::MatvecPush p{static_cast<std::uint32_t>(nb()), static_cast<std::uint32_t>(r0), static_cast<std::uint32_t>(r1),
                      0, 0};
    if (soa) {
      p.w_base0 = static_cast<std::uint32_t>(first);                             // scales, in 16-bit units
      p.w_base1 = static_cast<std::uint32_t>(first * (pools.quant_bytes / 16));  // quants, in uvec4 units
    } else {
      p.w_base0 = static_cast<std::uint32_t>(first) * words;  // blocks, in 16-bit units
    }
    return p;
  }

  VkDescriptorSet set(bool soa) {
    const gpu::Pipeline& pl = pipes.get(fmt, soa ? gpu::WeightLayout::kSoa : gpu::WeightLayout::kAos);
    if (soa) return dpool.allocate(pl, {&pools.soa_d.buffer, &pools.soa_q.buffer, &xq, &xd, &xs, &ygpu});
    return dpool.allocate(pl, {&pools.aos.buffer, &pools.aos.buffer, &xq, &xd, &xs, &ygpu});
  }
};

}  // namespace

void probe_layout(Report& r, const Options& o, gpu::Context& ctx) {
  if (!isa_supported(Isa::kAvx512)) {
    r.warnings.push_back("layout skipped: the experiment's CPU kernels need AVX-512 VNNI");
    return;
  }
  const int threads = std::min(4, cpu_features().physical_cores);
  ThreadPool pool(threads);
  gpu::MatvecPipelines pipes(ctx);
  gpu::TimestampPool ts(ctx, 2);
  const int reps = o.quick ? 4 : 8;    // matvecs per timed run
  const int rounds = o.quick ? 1 : 2;  // ABBA rounds

  // One set of activations for all shapes with k <= 8192.
  constexpr std::int64_t kMaxK = 8192;
  std::vector<float> raw(kMaxK);
  fill_random(reinterpret_cast<std::byte*>(raw.data()), raw.size() * 4, 5);
  for (float& v : raw) v = static_cast<float>(std::bit_cast<std::uint32_t>(v) % 2001) / 1000.0f - 1.0f;
  std::vector<std::int8_t> q(kMaxK);
  std::vector<float> d(kMaxK / kBlock);
  std::vector<std::int32_t> s(kMaxK / kBlock);
  quant::quantize_act_q8_0(raw.data(), kMaxK, q.data(), d.data(), s.data());
  gpu::Buffer xq(ctx, kMaxK, gpu::MemoryKind::kHostCoherent);
  gpu::Buffer xd(ctx, kMaxK / kBlock * 4, gpu::MemoryKind::kHostCoherent);
  gpu::Buffer xs(ctx, kMaxK / kBlock * 4, gpu::MemoryKind::kHostCoherent);
  std::memcpy(xq.mapped(), q.data(), q.size());
  std::memcpy(xd.mapped(), d.data(), d.size() * 4);
  std::memcpy(xs.mapped(), s.data(), s.size() * 4);

  for (gpu::WeightFormat fmt : {gpu::WeightFormat::kQ8_0, gpu::WeightFormat::kQ4_0}) {
    const char* fname = fmt == gpu::WeightFormat::kQ8_0 ? "Q8_0" : "Q4_0";
    for (bool imported : {false, true}) {
      if (imported && !ctx.external_memory_host_enabled()) continue;
      const char* place = imported ? "imported pages" : "vulkan host-cached";
      // Estimated decode ms per token, per mode (0 cpu, 1 gpu, 2 both) and layout (0 AoS, 1 SoA).
      double token_ms[3][2] = {};
      double spread[3][2] = {};
      for (const Shape& sh : kShapes) {
        throw_if_cancelled();
        Pools pools = make_pools(ctx, fmt, sh, imported);
        const quant::ActQ8 act{q.data(), d.data(), s.data(), sh.k};
        gpu::Buffer ygpu(ctx, static_cast<VkDeviceSize>(sh.rows) * 4, gpu::MemoryKind::kHostCached);
        std::vector<float> ycpu(static_cast<std::size_t>(sh.rows));
        gpu::DescriptorPool dpool(ctx, 4, 4 * 6);
        Experiment ex{ctx, pipes, dpool, pool, fmt, sh, pools, act, xq, xd, xs, ygpu, ycpu};
        const VkDescriptorSet sets[2] = {ex.set(false), ex.set(true)};
        const double matrix_bytes =
            static_cast<double>(pools.blocks_per_copy * static_cast<std::int64_t>(pools.block_bytes));
        int copy = 0;
        auto next_copy = [&] { return copy = (copy + 1) % pools.copies; };

        auto time_cpu = [&](bool soa) {
          const std::int64_t t0 = now_ns();
          ex.cpu_rows(soa, next_copy(), 0, sh.rows);
          return static_cast<double>(now_ns() - t0);
        };
        auto time_gpu = [&](bool soa) {  // `reps` dependent matvecs on rotating copies, two timestamps
          ts.reset();
          ctx.run_once([&](VkCommandBuffer cmd) {
            ts.write(cmd, 0);
            for (int i = 0; i < reps; ++i) {
              pipes.record(cmd, fmt, soa ? gpu::WeightLayout::kSoa : gpu::WeightLayout::kAos, sets[soa ? 1 : 0],
                           ex.push(soa, next_copy(), 0, sh.rows));
              gpu::compute_barrier(cmd);
            }
            ts.write(cmd, 1);
          });
          const auto t = ts.read(0, 2);
          return ts.elapsed_ns(t[0], t[1]) / reps;
        };
        // Both at once on disjoint rows of the same copy; split by standalone speed.
        gpu::CommandBuffer cb(ctx);
        auto time_both = [&](bool soa, std::int64_t split) {
          const int c = next_copy();
          VkCommandBuffer cmd = cb.begin();
          pipes.record(cmd, fmt, soa ? gpu::WeightLayout::kSoa : gpu::WeightLayout::kAos, sets[soa ? 1 : 0],
                       ex.push(soa, c, split, sh.rows));
          cb.end();
          const std::int64_t t0 = now_ns();
          cb.submit();
          ex.cpu_rows(soa, c, 0, split);
          cb.wait();
          return static_cast<double>(now_ns() - t0);
        };

        // Warm-up both layouts on both devices.
        for (bool soa : {false, true}) {
          time_cpu(soa);
          time_gpu(soa);
        }
        // ABBA interleaving: A B B A per round, `reps` measurements per visit.
        std::vector<double> ns[3][2];
        const bool order[4] = {false, true, true, false};
        for (int round = 0; round < rounds; ++round) {
          for (bool soa : order) {
            for (int i = 0; i < reps; ++i) ns[0][soa ? 1 : 0].push_back(time_cpu(soa));
            ns[1][soa ? 1 : 0].push_back(time_gpu(soa));
          }
        }
        auto median = [](std::vector<double> v) {
          std::sort(v.begin(), v.end());
          return v[v.size() / 2];
        };
        auto iqr_rel = [](std::vector<double> v) {
          std::sort(v.begin(), v.end());
          return (v[v.size() * 3 / 4] - v[v.size() / 4]) / v[v.size() / 2];
        };
        for (bool soa : order) {
          const double cpu_ns = median(ns[0][soa ? 1 : 0]);
          const double gpu_ns = median(ns[1][soa ? 1 : 0]);
          const auto split =
              static_cast<std::int64_t>(static_cast<double>(sh.rows) * gpu_ns / (cpu_ns + gpu_ns)) & ~std::int64_t{63};
          for (int i = 0; i < reps; ++i)
            ns[2][soa ? 1 : 0].push_back(time_both(soa, std::clamp<std::int64_t>(split, 0, sh.rows)));
        }
        for (int mode = 0; mode < 3; ++mode) {
          for (int lay = 0; lay < 2; ++lay) {
            const char* mname = mode == 0 ? "cpu" : mode == 1 ? "gpu" : "both";
            Measurement& m = r.add(
                "layout", std::format("{} {} {} {} {}", fname, sh.name, mname, place, lay ? "SoA" : "AoS"), "GB/s");
            for (double t : ns[mode][lay]) m.samples.push_back(matrix_bytes / t);
            m.params = {{"copies", std::to_string(pools.copies)}, {"threads", std::to_string(threads)}};
            token_ms[mode][lay] += sh.per_token * median(ns[mode][lay]) / 1e6;
            spread[mode][lay] = std::max(spread[mode][lay], iqr_rel(ns[mode][lay]));
          }
        }
      }
      for (int mode = 0; mode < 3; ++mode) {
        const char* mname = mode == 0 ? "cpu" : mode == 1 ? "gpu" : "both";
        for (int lay = 0; lay < 2; ++lay) {
          Measurement& m = r.add(
              "layout",
              std::format("{} {} {} {}: est. decode time per token", fname, mname, place, lay ? "SoA" : "AoS"), "ms");
          m.samples.push_back(token_ms[mode][lay]);
          m.note = std::format("largest IQR/median over shapes {:.3f}", spread[mode][lay]);
        }
      }
    }
  }
}

}  // namespace hearth::probe
