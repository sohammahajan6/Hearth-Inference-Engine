// The probe shaders must compute what we think they compute, on every memory
// type, before any of their timings mean anything. Runs under validation.
#include <doctest/doctest.h>

#include <bit>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "core/page_buffer.h"
#include "gpu/buffer.h"
#include "gpu/context.h"
#include "gpu/pipeline.h"
#include "gpu/timestamps.h"
#include "test_util.h"

using namespace hearth;

namespace {

struct ReadPush {
  std::uint32_t n, magic, write_all, base, blocked;
};

constexpr std::uint32_t kGroups = 64;
constexpr std::uint32_t kInvocations = kGroups * 256;

// XOR-reduce `src` on the GPU and return the XOR of all partial results.
std::uint64_t gpu_xor(gpu::Context& ctx, const gpu::Buffer& src, std::uint32_t n_uvec4, bool blocked = false) {
  gpu::Buffer dst(ctx, kInvocations * 16, gpu::MemoryKind::kHostCached);
  gpu::Pipeline pipe(ctx, {.shader = "probe_read", .buffers = 2, .push_constant_bytes = sizeof(ReadPush)});
  gpu::DescriptorPool pool(ctx, 1, 2);
  VkDescriptorSet set = pool.allocate(pipe, {&src, &dst});
  const ReadPush push{n_uvec4, 0, 1, 0, blocked ? 1u : 0u};
  ctx.run_once([&](VkCommandBuffer cmd) {
    pipe.dispatch(cmd, set, &push, sizeof(push), kGroups);
    gpu::compute_to_host_barrier(cmd);
  });
  const auto* w = static_cast<const std::uint64_t*>(dst.mapped());
  std::uint64_t x = 0;
  for (std::uint32_t i = 0; i < kInvocations * 2; ++i) x ^= w[i];
  return x;
}

void fill_random(void* p, std::size_t bytes, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  auto* w = static_cast<std::uint64_t*>(p);
  for (std::size_t i = 0; i < bytes / 8; ++i) w[i] = rng();
}

std::uint64_t cpu_xor(const void* p, std::size_t bytes) {
  const auto* w = static_cast<const std::uint64_t*>(p);
  std::uint64_t x = 0;
  for (std::size_t i = 0; i < bytes / 8; ++i) x ^= w[i];
  return x;
}

}  // namespace

TEST_SUITE("quick") {
  TEST_CASE("gpu probe_read: XOR checksum matches the CPU on every memory kind and on imported memory") {
    if (!gpu::vulkan_available()) return;
    gpu::Context ctx(test::gpu_test_options());
    constexpr std::size_t bytes = 1 << 20;  // 1 MiB: 64 Ki uvec4, several per invocation
    constexpr auto n = static_cast<std::uint32_t>(bytes / 16);

    for (gpu::MemoryKind kind : {gpu::MemoryKind::kHostCoherent, gpu::MemoryKind::kHostCached}) {
      gpu::Buffer src(ctx, bytes, kind);
      fill_random(src.mapped(), bytes, 7);
      CHECK_MESSAGE(gpu_xor(ctx, src, n) == cpu_xor(src.mapped(), bytes), gpu::to_string(kind));
      CHECK_MESSAGE(gpu_xor(ctx, src, n, true) == cpu_xor(src.mapped(), bytes), "blocked, " << gpu::to_string(kind));
    }
    {  // device-local: filled by a GPU copy from a host-visible staging buffer
      gpu::Buffer staging(ctx, bytes, gpu::MemoryKind::kHostCoherent);
      fill_random(staging.mapped(), bytes, 8);
      gpu::Buffer src(ctx, bytes, gpu::MemoryKind::kDeviceLocal);
      ctx.run_once([&](VkCommandBuffer cmd) {
        const VkBufferCopy region{0, 0, bytes};
        vkCmdCopyBuffer(cmd, staging.handle(), src.handle(), 1, &region);
        gpu::compute_barrier(cmd);
      });
      CHECK(gpu_xor(ctx, src, n) == cpu_xor(staging.mapped(), bytes));
    }
    if (ctx.external_memory_host_enabled()) {
      PageBuffer host(bytes);
      fill_random(host.data(), bytes, 9);
      gpu::Buffer src = gpu::Buffer::import_host(ctx, host.data(), bytes);
      CHECK(gpu_xor(ctx, src, n) == cpu_xor(host.data(), bytes));
      MESSAGE("imported host memory uses memory type " << src.memory_type());
    }
    CHECK(ctx.validation_message_count() == 0);
  }

  TEST_CASE("gpu probe_dot8: dotPacked4x8EXT matches signed int8 dot products computed on the CPU") {
    if (!gpu::vulkan_available()) return;
    gpu::Context ctx(test::gpu_test_options());
    // Reproduce invocation 0's arithmetic on the CPU, then ask the GPU to
    // store its result only if it equals ours: dst[0] is written iff they agree.
    struct Push {
      std::uint32_t iters;
      std::int32_t b, magic;
    };
    auto dot = [](std::int32_t a, std::int32_t b) {
      std::int32_t s = 0;
      for (int k = 0; k < 4; ++k) {
        s += static_cast<std::int8_t>(static_cast<std::uint32_t>(a) >> (8 * k)) *
             static_cast<std::int8_t>(static_cast<std::uint32_t>(b) >> (8 * k));
      }
      return s;
    };
    const std::int32_t b = static_cast<std::int32_t>(0x80FF7F01u);  // -128, -1, 127, 1: sign handling
    const std::uint32_t iters = 37;
    std::int32_t acc[8];
    const std::int32_t xors[8] = {0,          0x11111111, 0x22222222, 0x33333333,
                                  0x44444444, 0x55555555, 0x66666666, 0x77777777};
    for (int k = 0; k < 8; ++k) acc[k] = 0 ^ xors[k];  // invocation 0: base = 0
    for (std::uint32_t i = 0; i < iters; ++i) {
      for (auto& a : acc)
        a = static_cast<std::int32_t>(static_cast<std::uint32_t>(a) + static_cast<std::uint32_t>(dot(a, b)));
    }
    std::int32_t expected = 0;
    for (std::int32_t a : acc) expected ^= a;

    gpu::Buffer dst(ctx, 256 * 4, gpu::MemoryKind::kHostCached);
    std::memset(dst.mapped(), 0, 256 * 4);
    gpu::Pipeline pipe(ctx, {.shader = "probe_dot8", .buffers = 1, .push_constant_bytes = sizeof(Push)});
    gpu::DescriptorPool pool(ctx, 1, 1);
    VkDescriptorSet set = pool.allocate(pipe, {&dst});
    const Push push{iters, b, expected};
    ctx.run_once([&](VkCommandBuffer cmd) {
      pipe.dispatch(cmd, set, &push, sizeof(push), 1);
      gpu::compute_to_host_barrier(cmd);
    });
    CHECK(static_cast<const std::int32_t*>(dst.mapped())[0] == expected);
    CHECK(ctx.validation_message_count() == 0);
  }

  TEST_CASE("gpu timestamps: monotonic, and a longer dispatch takes longer") {
    if (!gpu::vulkan_available()) return;
    gpu::Context ctx(test::gpu_test_options());
    struct Push {
      std::uint32_t iters;
      float mul, add, magic;
    };
    gpu::Buffer dst(ctx, 1024 * 256 * 16, gpu::MemoryKind::kDeviceLocal);
    gpu::Pipeline pipe(ctx, {.shader = "probe_fma32", .buffers = 1, .push_constant_bytes = sizeof(Push)});
    gpu::DescriptorPool pool(ctx, 1, 1);
    VkDescriptorSet set = pool.allocate(pipe, {&dst});
    gpu::TimestampPool ts(ctx, 4);
    const Push short_run{16, 0.9999999f, 1e-7f, -1.0f};
    const Push long_run{1024, 0.9999999f, 1e-7f, -1.0f};
    ctx.run_once([&](VkCommandBuffer cmd) {
      ts.write(cmd, 0);
      pipe.dispatch(cmd, set, &short_run, sizeof(short_run), 1024);
      ts.write(cmd, 1);
      gpu::compute_barrier(cmd);
      ts.write(cmd, 2);
      pipe.dispatch(cmd, set, &long_run, sizeof(long_run), 1024);
      ts.write(cmd, 3);
    });
    const auto t = ts.read(0, 4);
    const double short_ns = ts.elapsed_ns(t[0], t[1]);
    const double long_ns = ts.elapsed_ns(t[2], t[3]);
    CHECK(short_ns > 0.0);
    CHECK(long_ns > 4.0 * short_ns);
    CHECK(ctx.validation_message_count() == 0);
    MESSAGE("fma32 dispatch: 16 iters " << short_ns / 1e3 << " us, 1024 iters " << long_ns / 1e3 << " us");
  }
}
