// The `gpu-placement` section of `hearth probe`: how fast the production
// decode kernel (the canonical GEMV, src/gpu/shaders/qgemv_*.comp) reads each
// decode matrix shape of the 1B and 3B models when the weights are
//   imported - the CPU's own page-aligned memory, imported with
//              VK_EXT_external_memory_host (memory type 0; one copy for both
//              devices), or
//   type2    - Vulkan host-cached memory (type 2), written once.
// docs/architecture/08-gpu-backend.md section 1 leaves this choice to step 6.1;
// Phase 1 measured it only with the Phase 1 kernels.
//
// Every timed submission runs the GEMV over copies of the matrix that span at
// least kMinSpan bytes, one dispatch per copy with a barrier between them (as
// decode does, layer after layer), so the weights come from DRAM and not
// from the 8 MiB last-level cache. The two placements are timed alternately
// (A B B A ...), so neither always runs hotter. TDR safety: at most about
// 256 MiB is read per submission (~10-20 ms at the measured read rates).
#include <algorithm>
#include <cstring>
#include <format>
#include <vector>

#include "core/page_buffer.h"
#include "core/platform.h"
#include "core/quant.h"
#include "cpu/qgemm.h"
#include "gpu/buffer.h"
#include "gpu/context.h"
#include "gpu/pipeline.h"
#include "gpu/timestamps.h"
#include "probe/probe_internal.h"

namespace hearth::probe {
namespace {

constexpr std::size_t kMinSpan = std::size_t{256} << 20;
constexpr std::size_t kAlign = 4096;
constexpr std::uint32_t kSubgroups = 8;  // rows per workgroup, as in the GEMV test

struct Shape {
  const char* what;
  std::int64_t rows, k;
};

struct Push {
  std::uint32_t rows, groups, row_begin;
};

std::vector<std::byte> random_blocks(cpu::QType type, std::int64_t rows, std::int64_t k, std::uint64_t seed) {
  const std::int64_t n = rows * k / quant::kBlock;
  const std::size_t size = type == cpu::QType::kQ8_0 ? sizeof(quant::BlockQ8_0) : sizeof(quant::BlockQ4_0);
  std::vector<std::byte> out(static_cast<std::size_t>(n) * size);
  std::uint64_t s = seed;
  for (std::byte& b : out) {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    b = static_cast<std::byte>(s >> 56);
  }
  const std::uint16_t d = 0x2000;  // fp16 2^-7: the values do not change the timing
  for (std::int64_t i = 0; i < n; ++i) std::memcpy(out.data() + static_cast<std::size_t>(i) * size, &d, sizeof d);
  return out;
}

std::size_t align_up(std::size_t x) { return (x + kAlign - 1) / kAlign * kAlign; }

}  // namespace

void probe_gpu_placement(Report& r, const Options& o, gpu::Context& ctx) {
  if (!ctx.external_memory_host_enabled()) {
    r.warnings.push_back("gpu-placement skipped: the GPU cannot import host memory");
    return;
  }
  const std::vector<Shape> shapes =
      o.quick ? std::vector<Shape>{{"1B attn_q/output", 2048, 2048}, {"1B ffn_gate/up", 8192, 2048}}
              : std::vector<Shape>{{"1B attn_q/output", 2048, 2048}, {"1B attn_k/v", 512, 2048},
                                   {"1B ffn_gate/up", 8192, 2048},   {"1B ffn_down", 2048, 8192},
                                   {"1B lm_head", 128256, 2048},     {"3B attn_q/output", 3072, 3072},
                                   {"3B attn_k/v", 1024, 3072},      {"3B ffn_gate/up", 8192, 3072},
                                   {"3B ffn_down", 3072, 8192},      {"3B lm_head", 128256, 3072}};
  const int reps = o.quick ? 3 : 7;
  gpu::TimestampPool ts(ctx, 2);
  for (const cpu::QType type : {cpu::QType::kQ8_0, cpu::QType::kQ4_0}) {
    const bool q8 = type == cpu::QType::kQ8_0;
    const char* tname = q8 ? "Q8_0" : "Q4_0";
    gpu::Pipeline pipe(ctx, {.shader = q8 ? "qgemv_q8_0" : "qgemv_q4_0",
                             .buffers = q8 ? 5u : 6u,
                             .push_constant_bytes = sizeof(Push),
                             .spec_constants = {kSubgroups, 8 * kSubgroups},
                             .required_subgroup_size = 8,
                             .require_full_subgroups = true});
    for (const Shape& sh : shapes) {
      throw_if_cancelled();
      const std::vector<std::byte> blocks = random_blocks(type, sh.rows, sh.k, 0xC2B2AE3D27D4EB4Full + sh.rows);
      const cpu::PackedMatrix pm(type, blocks.data(), sh.rows, sh.k);
      const cpu::PackedView v = pm.view();
      const std::size_t mat = cpu::packed_bytes(type, sh.rows, sh.k);
      const std::size_t quant_bytes = cpu::packed_quant_bytes(type, sh.rows, sh.k);
      const std::size_t stride = align_up(mat);
      const std::size_t copies = std::max<std::size_t>(1, (kMinSpan + mat - 1) / mat);
      const std::size_t total = copies * stride;
      // Placement A: CPU pages, imported. Placement B: type 2, written once.
      PageBuffer host(total);
      for (std::size_t c = 0; c < copies; ++c) std::memcpy(host.data() + c * stride, v.q, mat);
      gpu::Buffer imported = gpu::Buffer::import_host(ctx, host.data(), total);
      gpu::Buffer type2(ctx, total, gpu::MemoryKind::kHostCached);
      std::memcpy(type2.mapped(), host.data(), total);
      // Activations of one token, and the output, in type 2 (exchange buffers, 08 doc section 1).
      const auto groups = static_cast<std::size_t>(v.groups);
      std::vector<float> x(static_cast<std::size_t>(sh.k));
      for (std::size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>(static_cast<int>(i % 23) - 11) * 0.1f;
      cpu::PackedActs acts;
      acts.resize(1, sh.k);
      acts.quantize(Isa::kScalar, x.data(), 0, 1);
      auto upload = [&](const void* p, std::size_t bytes) {
        gpu::Buffer b(ctx, bytes, gpu::MemoryKind::kHostCached);
        std::memcpy(b.mapped(), p, bytes);
        return b;
      };
      gpu::Buffer xq = upload(acts.q(0, 0), groups * 512), xd = upload(acts.d(0, 0), groups * 64),
                  xs = upload(acts.s(0, 0), groups * 64);
      gpu::Buffer y(ctx, static_cast<std::size_t>(sh.rows) * 4, gpu::MemoryKind::kHostCached);
      gpu::DescriptorPool dpool(ctx, static_cast<std::uint32_t>(2 * copies), static_cast<std::uint32_t>(12 * copies));
      std::vector<VkDescriptorSet> sets[2];
      const gpu::Buffer* place[2] = {&imported, &type2};
      for (int p = 0; p < 2; ++p) {
        for (std::size_t c = 0; c < copies; ++c) {
          std::vector<gpu::DescriptorPool::Range> rg = {{place[p], c * stride, quant_bytes},
                                                        {place[p], c * stride + quant_bytes, mat - quant_bytes},
                                                        {&xq, 0, xq.size()},
                                                        {&xd, 0, xd.size()}};
          if (!q8) rg.push_back({&xs, 0, xs.size()});
          rg.push_back({&y, 0, y.size()});
          sets[p].push_back(dpool.allocate(pipe, rg));
        }
      }
      const Push push{static_cast<std::uint32_t>(sh.rows), static_cast<std::uint32_t>(groups), 0};
      const auto wgs = static_cast<std::uint32_t>((sh.rows + kSubgroups - 1) / kSubgroups);
      auto time_ns = [&](int p) {
        ts.reset();
        ctx.run_once([&](VkCommandBuffer cmd) {
          ts.write(cmd, 0);
          for (std::size_t c = 0; c < copies; ++c) {
            if (c > 0) gpu::compute_barrier(cmd);
            pipe.dispatch(cmd, sets[p][c], &push, sizeof(push), wgs);
          }
          ts.write(cmd, 1);
        });
        const auto t = ts.read(0, 2);
        return ts.elapsed_ns(t[0], t[1]);
      };
      time_ns(0);  // warm-up of both placements
      time_ns(1);
      Measurement* m[2];
      for (int p = 0; p < 2; ++p) {
        m[p] = &r.add(
            "gpu-placement",
            std::format("decode GEMV {} {}, {} x {}, {}", tname, sh.what, sh.rows, sh.k, p == 0 ? "imported" : "type2"),
            "GB/s");
        m[p]->params = {{"type", tname},
                        {"rows", std::to_string(sh.rows)},
                        {"k", std::to_string(sh.k)},
                        {"placement", p == 0 ? "imported host pages (memory type 0)" : "vulkan host-cached (type 2)"},
                        {"matrix_bytes", std::to_string(mat)},
                        {"copies_per_submission", std::to_string(copies)}};
      }
      const double bytes = static_cast<double>(mat) * static_cast<double>(copies);
      for (int i = 0; i < reps; ++i) {
        const int first = i % 2;  // A B, then B A, ...
        for (const int p : {first, 1 - first}) m[p]->samples.push_back(bytes / time_ns(p));
      }
    }
  }
}

}  // namespace hearth::probe
