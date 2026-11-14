// The GPU's canonical GEMM (src/gpu/shaders/qgemm_*.comp) on the CPU's packed
// bytes must give cpu::canonical_dot's bits for every (row, token) pair, for
// every tile shape R x C, with 0 validation messages (RISKS R15, ADR 0009).
// Shapes and data are adversarial: partial groups, row and token counts that
// leave partial tiles, negative-zero and subnormal scales, extreme quants,
// all-zero activation blocks.
#include <doctest/doctest.h>

#include <algorithm>
#include <cstring>
#include <format>
#include <limits>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "core/quant.h"
#include "cpu/qgemm.h"
#include "gpu/buffer.h"
#include "gpu/context.h"
#include "gpu/pipeline.h"
#include "qweights.h"
#include "test_util.h"

using namespace hearth;
using quant::BlockQ4_0;
using quant::BlockQ8_0;
using quant::kBlock;

namespace {

struct Push {
  std::uint32_t rows, groups, row_begin, tokens, ldy;
};

constexpr std::uint32_t kSubgroups = 4;  // tiles per workgroup

struct Tile {
  std::uint32_t r, c;
};

// Returns the number of (row, token) pairs compared bit for bit.
template <class Block>
std::int64_t check(gpu::Context& ctx, gpu::DescriptorPool& pool, cpu::QType type, Tile tile, std::mt19937& rng) {
  std::int64_t compared = 0;
  const bool q8 = type == cpu::QType::kQ8_0;
  gpu::Pipeline pipe(ctx, {.shader = q8 ? "qgemm_q8_0" : "qgemm_q4_0",
                           .buffers = q8 ? 5u : 6u,
                           .push_constant_bytes = sizeof(Push),
                           .spec_constants = {kSubgroups, 8 * kSubgroups, tile.r, tile.c},
                           .required_subgroup_size = 8,
                           .require_full_subgroups = true});
  std::normal_distribution<float> n(0.0f, 1.5f);
  auto upload = [&](const void* data, std::size_t bytes) {
    gpu::Buffer b(ctx, std::max<std::size_t>(bytes, 16), gpu::MemoryKind::kHostCoherent);
    std::memcpy(b.mapped(), data, bytes);
    return b;
  };
  // Tiny shapes first (TDR safety), then the models' widths, partial groups and partial tiles.
  for (const auto& [rows, k, tokens] : {std::tuple<std::int64_t, std::int64_t, std::int64_t>{1, 32, 1},
                                        {3, 480, 2},
                                        {9, 512, 5},
                                        {13, 2048, 7},
                                        {67, 544, 3},
                                        {5, 8192, 4},
                                        {130, 3072, 11},
                                        {256, 2048, 64}}) {
    CAPTURE(rows);
    CAPTURE(k);
    CAPTURE(tokens);
    const std::int64_t nb = k / kBlock;
    std::vector<Block> w;
    if constexpr (std::is_same_v<Block, BlockQ8_0>) {
      w = test::make_q8(rows, k, rng);
    } else {
      w = test::make_q4(rows, k, rng);
    }
    std::vector<float> x(static_cast<std::size_t>(tokens * k));
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = (i / kBlock) % 4 == 1 ? 0.0f : n(rng);
    // The specification, pair by pair, from the unpacked activations of each token.
    std::vector<float> want(static_cast<std::size_t>(tokens * rows));
    {
      std::vector<std::int8_t> xq(static_cast<std::size_t>(k));
      std::vector<float> xd(static_cast<std::size_t>(nb));
      std::vector<std::int32_t> xs(static_cast<std::size_t>(nb));
      for (std::int64_t t = 0; t < tokens; ++t) {
        quant::quantize_act_q8_0(x.data() + t * k, k, xq.data(), xd.data(), xs.data());
        for (std::int64_t r = 0; r < rows; ++r) {
          want[static_cast<std::size_t>(t * rows + r)] =
              cpu::canonical_dot(w.data() + r * nb, {xq.data(), xd.data(), xs.data(), k});
        }
      }
    }
    const cpu::PackedMatrix pm(type, w.data(), rows, k);
    cpu::PackedActs a;
    a.resize(tokens, k);
    a.quantize(Isa::kScalar, x.data(), 0, tokens);
    const cpu::PackedView v = pm.view();
    const auto groups = static_cast<std::size_t>(v.groups);
    const auto rows_u = static_cast<std::size_t>(rows);
    const auto tok_u = static_cast<std::size_t>(tokens);
    gpu::Buffer wq = upload(v.q, rows_u * groups * static_cast<std::size_t>(cpu::group_quant_bytes(type)));
    gpu::Buffer wd = upload(v.d, rows_u * groups * 32);
    gpu::Buffer gxq = upload(a.q(0, 0), tok_u * groups * 512);
    gpu::Buffer gxd = upload(a.d(0, 0), tok_u * groups * 64);
    gpu::Buffer gxs = upload(a.s(0, 0), tok_u * groups * 64);
    gpu::Buffer y(ctx, tok_u * rows_u * 4, gpu::MemoryKind::kHostCached);
    std::fill_n(static_cast<float*>(y.mapped()), tok_u * rows_u, std::numeric_limits<float>::quiet_NaN());
    const VkDescriptorSet set =
        q8 ? pool.allocate(pipe, {&wq, &wd, &gxq, &gxd, &y}) : pool.allocate(pipe, {&wq, &wd, &gxq, &gxd, &gxs, &y});
    const Push push{static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(groups), 0,
                    static_cast<std::uint32_t>(tokens), static_cast<std::uint32_t>(rows)};
    const std::uint32_t rows_per_wg = kSubgroups * tile.r;
    ctx.run_once([&](VkCommandBuffer cmd) {
      pipe.dispatch(cmd, set, &push, sizeof(push), static_cast<std::uint32_t>((tokens + tile.c - 1) / tile.c),
                    static_cast<std::uint32_t>((rows + rows_per_wg - 1) / rows_per_wg));
      gpu::compute_to_host_barrier(cmd);
    });
    const auto* got = static_cast<const float*>(y.mapped());
    for (std::int64_t t = 0; t < tokens; ++t) {
      for (std::int64_t r = 0; r < rows; ++r) {
        const auto i = static_cast<std::size_t>(t * rows + r);
        if (std::memcmp(&got[i], &want[i], sizeof(float)) != 0) {
          FAIL(std::format("{} tile {}x{}, row {} token {}: gpu {:a}, canonical {:a}", q8 ? "Q8_0" : "Q4_0", tile.r,
                           tile.c, r, t, got[i], want[i]));
        }
        ++compared;
      }
    }
  }
  return compared;
}

}  // namespace

TEST_SUITE("quick") {
  TEST_CASE("gpu qgemm: every tile shape gives canonical_dot's bits for every (row, token) (R15; validation on)") {
    if (!gpu::vulkan_available()) return;
    gpu::Context ctx(test::gpu_test_options());
    gpu::DescriptorPool pool(ctx, 256, 256 * 6);
    std::mt19937 rng(23);
    // Pairs per shape list: 1 + 6 + 45 + 91 + 201 + 20 + 1430 + 16384.
    constexpr std::int64_t kPairs = 18178;
    for (const Tile tile :
         {Tile{1, 1}, Tile{2, 2}, Tile{4, 1}, Tile{1, 4}, Tile{4, 2}, Tile{2, 4}, Tile{3, 3}, Tile{4, 4}, Tile{4, 8}}) {
      CAPTURE(tile.r);
      CAPTURE(tile.c);
      CHECK(check<BlockQ8_0>(ctx, pool, cpu::QType::kQ8_0, tile, rng) == kPairs);
      CHECK(check<BlockQ4_0>(ctx, pool, cpu::QType::kQ4_0, tile, rng) == kPairs);
    }
    const std::string note =
        std::format("9 tile shapes x 2 formats x {} pairs bit-identical to canonical_dot; validation layer {}", kPairs,
                    test::gpu_test_options().validation ? "on" : "off");
    MESSAGE(note);
    CHECK(ctx.validation_message_count() == 0);
  }
}
