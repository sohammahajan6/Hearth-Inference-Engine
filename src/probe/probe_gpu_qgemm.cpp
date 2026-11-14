// The `gpu-qgemm` section of `hearth probe`: how fast the canonical GPU GEMM
// (src/gpu/shaders/qgemm_*.comp) runs prefill-shaped products for each tile
// shape R rows x C tokens, so the tile is chosen by measurement
//.
//
// TDR safety: each configuration first runs one token tile alone; the token
// count of the timed dispatches is then capped so that one submission is
// predicted to take at most kBudgetNs. The token count used is recorded.
// Weights sit in Vulkan host-cached memory (type 2); placement is decided
// separately (step 6.1).
#include <algorithm>
#include <cstring>
#include <format>
#include <vector>

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

constexpr double kBudgetNs = 50e6;  // per submission, half the 100 ms target of the 08 doc
constexpr std::uint32_t kSubgroups = 4;

struct Shape {
  const char* what;
  std::int64_t rows, k, tokens;  // tokens: the most a timed dispatch uses
};

struct Push {
  std::uint32_t rows, groups, row_begin, tokens, ldy;
};

// GGUF blocks with random codes and a small fixed scale (the values do not
// change the timing; they only must not be all zero).
std::vector<std::byte> random_blocks(cpu::QType type, std::int64_t rows, std::int64_t k, std::uint64_t seed) {
  const std::int64_t n = rows * k / quant::kBlock;
  const std::size_t size = type == cpu::QType::kQ8_0 ? sizeof(quant::BlockQ8_0) : sizeof(quant::BlockQ4_0);
  std::vector<std::byte> out(static_cast<std::size_t>(n) * size);
  std::uint64_t s = seed;
  for (std::byte& b : out) {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    b = static_cast<std::byte>(s >> 56);
  }
  const std::uint16_t d = 0x2000;  // fp16 2^-7
  for (std::int64_t i = 0; i < n; ++i) std::memcpy(out.data() + static_cast<std::size_t>(i) * size, &d, sizeof d);
  return out;
}

gpu::Buffer upload(gpu::Context& ctx, const void* data, std::size_t bytes, gpu::MemoryKind kind) {
  gpu::Buffer b(ctx, std::max<std::size_t>(bytes, 16), kind);
  std::memcpy(b.mapped(), data, bytes);
  return b;
}

}  // namespace

void probe_gpu_qgemm(Report& r, const Options& o, gpu::Context& ctx) {
  const std::vector<Shape> shapes = {
      {"1B attn_q/output", 2048, 2048, 512}, {"1B ffn_gate/up", 8192, 2048, 512}, {"1B ffn_down", 2048, 8192, 512},
      {"3B ffn_gate/up", 8192, 3072, 512},   {"1B lm_head", 128256, 2048, 64},
  };
  struct Tile {
    std::uint32_t r, c;
  };
  const std::vector<Tile> tiles = o.quick ? std::vector<Tile>{{1, 1}, {2, 2}, {4, 2}}
                                          : std::vector<Tile>{{1, 1}, {1, 2}, {1, 4}, {2, 1}, {2, 2}, {2, 4}, {4, 1},
                                                              {4, 2}, {4, 4}, {3, 3}, {4, 6}, {6, 4}, {4, 8}, {8, 4}};
  const int reps = o.quick ? 3 : 7;
  gpu::TimestampPool ts(ctx, 2);
  for (const cpu::QType type : {cpu::QType::kQ8_0, cpu::QType::kQ4_0}) {
    const bool q8 = type == cpu::QType::kQ8_0;
    const char* tname = q8 ? "Q8_0" : "Q4_0";
    for (const Shape& sh : shapes) {
      throw_if_cancelled();
      if (o.quick && sh.rows == 128256) continue;
      const std::vector<std::byte> blocks = random_blocks(type, sh.rows, sh.k, 0x9E3779B97F4A7C15ull + sh.rows);
      const cpu::PackedMatrix pm(type, blocks.data(), sh.rows, sh.k);
      const cpu::PackedView v = pm.view();
      const auto groups = static_cast<std::size_t>(v.groups);
      std::vector<float> x(static_cast<std::size_t>(sh.tokens * sh.k));
      for (std::size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>(static_cast<int>(i % 23) - 11) * 0.1f;
      cpu::PackedActs acts;
      acts.resize(sh.tokens, sh.k);
      acts.quantize(Isa::kScalar, x.data(), 0, sh.tokens);
      const auto rows_u = static_cast<std::size_t>(sh.rows);
      const auto tok_u = static_cast<std::size_t>(sh.tokens);
      gpu::Buffer wq = upload(ctx, v.q, rows_u * groups * static_cast<std::size_t>(cpu::group_quant_bytes(type)),
                              gpu::MemoryKind::kHostCached);
      gpu::Buffer wd = upload(ctx, v.d, rows_u * groups * 32, gpu::MemoryKind::kHostCached);
      gpu::Buffer xq = upload(ctx, acts.q(0, 0), tok_u * groups * 512, gpu::MemoryKind::kHostCached);
      gpu::Buffer xd = upload(ctx, acts.d(0, 0), tok_u * groups * 64, gpu::MemoryKind::kHostCached);
      gpu::Buffer xs = upload(ctx, acts.s(0, 0), tok_u * groups * 64, gpu::MemoryKind::kHostCached);
      gpu::Buffer y(ctx, tok_u * rows_u * 4, gpu::MemoryKind::kHostCached);
      gpu::DescriptorPool dpool(ctx, static_cast<std::uint32_t>(tiles.size()),
                                static_cast<std::uint32_t>(tiles.size()) * 6);
      for (const Tile& tile : tiles) {
        gpu::Pipeline pipe(ctx, {.shader = q8 ? "qgemm_q8_0" : "qgemm_q4_0",
                                 .buffers = q8 ? 5u : 6u,
                                 .push_constant_bytes = sizeof(Push),
                                 .spec_constants = {kSubgroups, 8 * kSubgroups, tile.r, tile.c},
                                 .required_subgroup_size = 8,
                                 .require_full_subgroups = true});
        const VkDescriptorSet set =
            q8 ? dpool.allocate(pipe, {&wq, &wd, &xq, &xd, &y}) : dpool.allocate(pipe, {&wq, &wd, &xq, &xd, &xs, &y});
        const std::uint32_t rows_per_wg = kSubgroups * tile.r;
        const auto wg_rows = static_cast<std::uint32_t>((sh.rows + rows_per_wg - 1) / rows_per_wg);
        auto time_ns = [&](std::uint32_t tokens) {
          const Push push{static_cast<std::uint32_t>(sh.rows), static_cast<std::uint32_t>(groups), 0, tokens,
                          static_cast<std::uint32_t>(sh.rows)};
          ts.reset();
          ctx.run_once([&](VkCommandBuffer cmd) {
            ts.write(cmd, 0);
            pipe.dispatch(cmd, set, &push, sizeof(push), (tokens + tile.c - 1) / tile.c, wg_rows);
            ts.write(cmd, 1);
          });
          const auto t = ts.read(0, 2);
          return ts.elapsed_ns(t[0], t[1]);
        };
        // One token tile first (also the warm-up), then the largest token
        // count, in whole tiles, that fits the budget.
        const double one = time_ns(tile.c);
        const auto tiles_fit = static_cast<std::int64_t>(kBudgetNs / std::max(one, 1.0));
        const auto tokens = static_cast<std::uint32_t>(
            std::clamp<std::int64_t>(tiles_fit * tile.c, tile.c, sh.tokens / tile.c * tile.c));
        time_ns(tokens);  // warm-up at the measured size
        Measurement& m = r.add(
            "gpu-qgemm",
            std::format("prefill {} {}, {} x {}, tile {}x{}", tname, sh.what, sh.rows, sh.k, tile.r, tile.c), "GMAC/s");
        m.params = {{"type", tname},
                    {"rows", std::to_string(sh.rows)},
                    {"k", std::to_string(sh.k)},
                    {"tile", std::format("{}x{}", tile.r, tile.c)},
                    {"tokens", std::to_string(tokens)},
                    {"subgroups_per_workgroup", std::to_string(kSubgroups)},
                    {"one_tile_ms", std::format("{:.3f}", one / 1e6)},
                    {"memory", "vulkan host-cached (type 2)"}};
        const double macs = static_cast<double>(sh.rows) * static_cast<double>(sh.k) * tokens;
        for (int i = 0; i < reps; ++i) m.samples.push_back(macs / time_ns(tokens));
      }
    }
  }
}

}  // namespace hearth::probe
