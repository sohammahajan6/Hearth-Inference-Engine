// The GPU's canonical GEMV (src/gpu/shaders/qgemv_*.comp) on the CPU's packed
// bytes must give cpu::canonical_dot's bits for every row (RISKS R15, ADR
// 0009), with 0 validation messages; and `precise` must really keep a * b + c
// unfused on this driver (probe_fma_detect.comp).
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/fp16.h"
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
  std::uint32_t rows, groups, row_begin;
};

constexpr std::uint32_t kSubgroups = 8;  // rows per workgroup

// Returns the number of rows compared bit for bit.
template <class Block>
std::int64_t check(gpu::Context& ctx, gpu::DescriptorPool& pool, cpu::QType type, std::mt19937& rng) {
  std::int64_t compared = 0;
  const bool q8 = type == cpu::QType::kQ8_0;
  gpu::Pipeline pipe(ctx, {.shader = q8 ? "qgemv_q8_0" : "qgemv_q4_0",
                           .buffers = q8 ? 5u : 6u,
                           .push_constant_bytes = sizeof(Push),
                           .spec_constants = {kSubgroups, 8 * kSubgroups},
                           .required_subgroup_size = 8,
                           .require_full_subgroups = true});
  std::normal_distribution<float> n(0.0f, 1.5f);
  auto upload = [&](const void* data, std::size_t bytes) {
    gpu::Buffer b(ctx, std::max<std::size_t>(bytes, 16), gpu::MemoryKind::kHostCoherent);
    std::memcpy(b.mapped(), data, bytes);
    return b;
  };
  // Tiny shapes first (TDR safety), then the model's widths and a partial group.
  for (const auto& [rows, k] : {std::pair<std::int64_t, std::int64_t>{1, 32},
                                {3, 480},
                                {9, 512},
                                {13, 2048},
                                {67, 544},
                                {5, 8192},
                                {2048, 2048}}) {
    CAPTURE(rows);
    CAPTURE(k);
    const std::int64_t nb = k / kBlock;
    std::vector<Block> w;
    if constexpr (std::is_same_v<Block, BlockQ8_0>) {
      w = test::make_q8(rows, k, rng);
    } else {
      w = test::make_q4(rows, k, rng);
    }
    std::vector<float> x(static_cast<std::size_t>(k));
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = (i / kBlock) % 4 == 1 ? 0.0f : n(rng);
    std::vector<std::int8_t> xq(static_cast<std::size_t>(k));
    std::vector<float> xd(static_cast<std::size_t>(nb));
    std::vector<std::int32_t> xs(static_cast<std::size_t>(nb));
    quant::quantize_act_q8_0(x.data(), k, xq.data(), xd.data(), xs.data());
    std::vector<float> want(static_cast<std::size_t>(rows));
    for (std::int64_t r = 0; r < rows; ++r) {
      want[static_cast<std::size_t>(r)] = cpu::canonical_dot(w.data() + r * nb, {xq.data(), xd.data(), xs.data(), k});
    }
    const cpu::PackedMatrix pm(type, w.data(), rows, k);
    cpu::PackedActs a;
    a.resize(1, k);
    a.quantize(Isa::kScalar, x.data(), 0, 1);
    const cpu::PackedView v = pm.view();
    const auto groups = static_cast<std::size_t>(v.groups);
    const auto rows_u = static_cast<std::size_t>(rows);
    gpu::Buffer wq = upload(v.q, rows_u * groups * static_cast<std::size_t>(cpu::group_quant_bytes(type)));
    gpu::Buffer wd = upload(v.d, rows_u * groups * 32);
    gpu::Buffer gxq = upload(a.q(0, 0), groups * 512);
    gpu::Buffer gxd = upload(a.d(0, 0), groups * 64);
    gpu::Buffer gxs = upload(a.s(0, 0), groups * 64);
    gpu::Buffer y(ctx, rows_u * 4, gpu::MemoryKind::kHostCached);
    std::fill_n(static_cast<float*>(y.mapped()), rows_u, std::numeric_limits<float>::quiet_NaN());
    const VkDescriptorSet set =
        q8 ? pool.allocate(pipe, {&wq, &wd, &gxq, &gxd, &y}) : pool.allocate(pipe, {&wq, &wd, &gxq, &gxd, &gxs, &y});
    const Push push{static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(groups), 0};
    ctx.run_once([&](VkCommandBuffer cmd) {
      pipe.dispatch(cmd, set, &push, sizeof(push), static_cast<std::uint32_t>((rows + kSubgroups - 1) / kSubgroups));
      gpu::compute_to_host_barrier(cmd);
    });
    const auto* got = static_cast<const float*>(y.mapped());
    for (std::int64_t r = 0; r < rows; ++r) {
      if (std::memcmp(&got[r], &want[static_cast<std::size_t>(r)], sizeof(float)) != 0) {
        FAIL(std::format("{} row {}: gpu {:a}, canonical {:a}", q8 ? "Q8_0" : "Q4_0", r, got[r],
                         want[static_cast<std::size_t>(r)]));
      }
      ++compared;
    }
  }
  return compared;
}

}  // namespace

TEST_SUITE("quick") {
  TEST_CASE("gpu fma detector: `precise` keeps a * b + c unfused (validation on)") {
    if (!gpu::vulkan_available()) return;
    gpu::Context ctx(test::gpu_test_options());
    gpu::Pipeline pipe(ctx, {.shader = "probe_fma_detect", .buffers = 2});
    gpu::DescriptorPool pool(ctx, 1, 2);
    // a = b = 1 + 2^-12, c = -(1 + 2^-11): unfused 0, fused 2^-24.
    const float in[4] = {1.0f + 0x1p-12f, 1.0f + 0x1p-12f, -(1.0f + 0x1p-11f), 0.0f};
    gpu::Buffer bi(ctx, sizeof(in), gpu::MemoryKind::kHostCoherent);
    std::memcpy(bi.mapped(), in, sizeof(in));
    gpu::Buffer bo(ctx, 16, gpu::MemoryKind::kHostCached);
    std::fill_n(static_cast<float*>(bo.mapped()), 4, std::numeric_limits<float>::quiet_NaN());
    const VkDescriptorSet set = pool.allocate(pipe, {&bi, &bo});
    ctx.run_once([&](VkCommandBuffer cmd) {
      pipe.dispatch(cmd, set, nullptr, 0, 1);
      gpu::compute_to_host_barrier(cmd);
    });
    const auto* o = static_cast<const float*>(bo.mapped());
    CHECK(o[0] == 0.0f);  // precise: unfused
    const std::string note =
        std::format("without precise the driver computes {:a} ({})", o[1], o[1] == 0.0f ? "unfused" : "fused");
    MESSAGE(note);
    CHECK(ctx.validation_message_count() == 0);
  }

  TEST_CASE("gpu qgemv: Q8_0 and Q4_0 rows equal canonical_dot bit for bit (R15; validation on)") {
    if (!gpu::vulkan_available()) return;
    gpu::Context ctx(test::gpu_test_options());
    gpu::DescriptorPool pool(ctx, 32, 32 * 6);
    std::mt19937 rng(11);
    const std::int64_t q8 = check<BlockQ8_0>(ctx, pool, cpu::QType::kQ8_0, rng);
    const std::int64_t q4 = check<BlockQ4_0>(ctx, pool, cpu::QType::kQ4_0, rng);
    // Every row of every shape was compared (1 + 3 + 9 + 13 + 67 + 5 + 2048 per format).
    CHECK(q8 == 2146);
    CHECK(q4 == 2146);
    const std::string note = std::format("{} Q8_0 and {} Q4_0 rows bit-identical to canonical_dot; validation layer {}",
                                         q8, q4, test::gpu_test_options().validation ? "on" : "off");
    MESSAGE(note);
    CHECK(ctx.validation_message_count() == 0);
  }
}
