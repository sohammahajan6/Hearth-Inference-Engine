// The GPU activation quantizer (src/gpu/shaders/quantize_act.comp) must write
// exactly the bytes cpu::PackedActs::quantize writes (ACCEPTANCE "Kernels":
// activation codes are bit-exact), and its exact integer division and fp16
// rounding (quant_common.glsl, through probe_div_exact.comp) must equal the
// CPU's correctly rounded a / b and core/fp16.h fp32_to_fp16, with 0
// validation messages.
#include <doctest/doctest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "core/fp16.h"
#include "cpu/qgemm.h"
#include "gpu/buffer.h"
#include "gpu/context.h"
#include "gpu/pipeline.h"
#include "test_util.h"

using namespace hearth;

namespace {

gpu::Buffer upload(gpu::Context& ctx, const void* data, std::size_t bytes) {
  gpu::Buffer b(ctx, std::max<std::size_t>(bytes, 16), gpu::MemoryKind::kHostCoherent);
  std::memcpy(b.mapped(), data, bytes);
  return b;
}

gpu::Buffer output(gpu::Context& ctx, std::size_t bytes) {
  gpu::Buffer b(ctx, std::max<std::size_t>(bytes, 16), gpu::MemoryKind::kHostCached);
  std::memset(b.mapped(), 0xA5, std::max<std::size_t>(bytes, 16));  // garbage that must be overwritten
  return b;
}

// A positive float with a uniformly random exponent in [lo, hi] and random significand.
float random_float(std::mt19937& rng, int lo, int hi) {
  const auto e = static_cast<std::uint32_t>(lo + static_cast<int>(rng() % static_cast<unsigned>(hi - lo + 1)) + 127);
  return std::bit_cast<float>((e << 23) | (rng() & 0x7FFFFFu));
}

// One block of 32 activations; the modes cover the quantizer's edge cases.
void fill_block(float* p, int mode, std::mt19937& rng) {
  std::normal_distribution<float> normal(0.0f, 1.0f);
  const float scale = std::ldexp(1.0f, static_cast<int>(rng() % 40) - 20);  // 2^-20 .. 2^19
  for (int j = 0; j < 32; ++j) {
    switch (mode) {
      case 0:  // ordinary
        p[j] = normal(rng) * scale;
        break;
      case 1:  // all zero
        p[j] = 0.0f;
        break;
      case 2:  // all negative zero
        p[j] = -0.0f;
        break;
      case 3:  // exact rounding ties: amax = 127 * scale makes m = 1 / scale exact, so x * m = k + 0.5
        p[j] = j == 0 ? 127.0f * scale : (static_cast<float>(static_cast<int>(rng() % 253)) - 126.0f + 0.5f) * scale;
        break;
      case 4:  // negative maximum
        p[j] = j == 7 ? -scale : 0.5f * scale;
        break;
      case 5:  // the domain's lower edge: amax = 2^-119, the rest smaller (some subnormal)
        p[j] = j == 3 ? 0x1p-119f : (j % 2 != 0 ? 0x1p-130f : -0x1p-121f * static_cast<float>(j));
        break;
      case 6:  // huge: fp16 scale overflows to infinity, the multiplier is tiny
        p[j] = (j == 5 ? -1.0f : 1.0f) * random_float(rng, 90, 127);
        break;
      case 7:  // any exponent of the domain, tiny entries mixed in
        p[j] = j == 0 ? random_float(rng, -119, 127) : p[0] * (normal(rng) * 0.01f);  // (no overflow: |factor| < 1)
        break;
      default:  // a ramp, as in the CPU test
        p[j] = static_cast<float>(j) * scale;
        break;
    }
  }
}

struct QuantPush {
  std::uint32_t tokens, groups, blocks;
};

}  // namespace

TEST_SUITE("quick") {
  TEST_CASE("gpu quantizer: exact division and fp16 rounding equal the CPU's (validation on)") {
    if (!gpu::vulkan_available()) return;
    gpu::Context ctx(test::gpu_test_options());
    gpu::Pipeline pipe(ctx, {.shader = "probe_div_exact", .buffers = 5, .push_constant_bytes = 4});
    gpu::DescriptorPool pool(ctx, 1, 5);
    std::mt19937 rng(20261106);
    std::vector<float> a, b, c;
    // The quantizer's two quotients over its whole domain, amax in [2^-119, FLT_MAX].
    for (int i = 0; i < 400000; ++i) {
      const float amax = random_float(rng, -119, 127);
      a.push_back(amax);
      b.push_back(127.0f);
      a.push_back(127.0f);
      b.push_back(amax);
    }
    for (const float amax :
         {0x1p-119f, 127.0f, 1.0f, std::numeric_limits<float>::max(), 0x1.fffffep-1f, 0x1.000002p0f}) {
      a.push_back(amax);
      b.push_back(127.0f);
      a.push_back(127.0f);
      b.push_back(amax);
    }
    // General normal quotients, to test the divider beyond the two divisors used.
    while (a.size() < 1000000) {
      a.push_back(random_float(rng, -60, 60));
      b.push_back(random_float(rng, -60, 60));
    }
    // fp16 rounding: every fp16 rounding midpoint (positive) and its two fp32 neighbours,
    // then random values over the full non-negative range (subnormal fp32 included).
    for (std::uint32_t h = 0; h < 0x7BFFu; ++h) {  // (0x7BFF's neighbour is infinity)
      const float lo = fp16_to_fp32(static_cast<std::uint16_t>(h));
      const float hi = fp16_to_fp32(static_cast<std::uint16_t>(h + 1));
      const float mid = std::bit_cast<float>((std::bit_cast<std::uint32_t>(lo) + std::bit_cast<std::uint32_t>(hi)) / 2);
      for (const float f : {mid, std::nextafter(mid, 0.0f), std::nextafter(mid, 1e30f)}) c.push_back(f);
    }
    for (const float f : {65520.0f, 65519.996f, 65504.0f, 0x1p-24f, 0x1p-25f, 0x1.000002p-25f}) c.push_back(f);
    while (c.size() < a.size()) c.push_back(std::bit_cast<float>(rng() % 0x7F800000u));
    c.resize(a.size());
    const auto n = static_cast<std::uint32_t>(a.size());
    gpu::Buffer ba = upload(ctx, a.data(), a.size() * 4), bb = upload(ctx, b.data(), b.size() * 4);
    gpu::Buffer bc = upload(ctx, c.data(), c.size() * 4);
    gpu::Buffer bo = output(ctx, a.size() * 4), bh = output(ctx, a.size() * 4);
    const VkDescriptorSet set = pool.allocate(pipe, {&ba, &bb, &bc, &bo, &bh});
    ctx.run_once([&](VkCommandBuffer cmd) {
      pipe.dispatch(cmd, set, &n, sizeof(n), (n + 63) / 64);
      gpu::compute_to_host_barrier(cmd);
    });
    const auto* o = static_cast<const float*>(bo.mapped());
    const auto* h = static_cast<const std::uint32_t*>(bh.mapped());
    std::size_t div_bad = 0, half_bad = 0;
    std::string first;
    for (std::size_t i = 0; i < a.size(); ++i) {
      const float want = a[i] / b[i];
      if (std::bit_cast<std::uint32_t>(o[i]) != std::bit_cast<std::uint32_t>(want)) {
        if (div_bad++ == 0) first = std::format("{:a} / {:a}: gpu {:a}, cpu {:a}", a[i], b[i], o[i], want);
      }
      if (h[i] != fp32_to_fp16(c[i])) {
        if (half_bad++ == 0) first += std::format(" fp16({:a}): gpu {:#x}, cpu {:#x}", c[i], h[i], fp32_to_fp16(c[i]));
      }
    }
    CHECK_MESSAGE(div_bad == 0, first);
    CHECK_MESSAGE(half_bad == 0, first);
    MESSAGE(std::format("{} quotients and {} fp16 roundings compared", a.size(), c.size()));
    CHECK(ctx.validation_message_count() == 0);
  }

  TEST_CASE("gpu quantizer: packed codes, scales and sums equal cpu::PackedActs bit for bit (validation on)") {
    if (!gpu::vulkan_available()) return;
    gpu::Context ctx(test::gpu_test_options());
    gpu::Pipeline pipe(ctx, {.shader = "quantize_act", .buffers = 4, .push_constant_bytes = sizeof(QuantPush)});
    gpu::DescriptorPool pool(ctx, 16, 64);
    std::mt19937 rng(7);
    std::int64_t blocks_compared = 0;
    // Tiny shapes first (TDR safety), then partial groups, the models' widths and a prefill chunk.
    for (const auto& [tokens, k] : {std::pair<std::int64_t, std::int64_t>{1, 32},
                                    {1, 480},
                                    {3, 544},
                                    {2, 2048},
                                    {5, 3072},
                                    {1, 8192},
                                    {130, 2048},
                                    {64, 8192}}) {
      CAPTURE(tokens);
      CAPTURE(k);
      std::vector<float> x(static_cast<std::size_t>(tokens * k));
      for (std::int64_t blk = 0; blk < tokens * k / 32; ++blk) {
        fill_block(x.data() + blk * 32, static_cast<int>(blk % 9), rng);
      }
      cpu::PackedActs want;
      want.resize(tokens, k);
      want.quantize(Isa::kScalar, x.data(), 0, tokens);
      const auto groups = static_cast<std::size_t>(want.groups());
      const auto tg = static_cast<std::size_t>(tokens) * groups;
      gpu::Buffer bx = upload(ctx, x.data(), x.size() * 4);
      gpu::Buffer bq = output(ctx, tg * 512), bd = output(ctx, tg * 64), bs = output(ctx, tg * 64);
      const VkDescriptorSet set = pool.allocate(pipe, {&bx, &bq, &bd, &bs});
      const QuantPush push{static_cast<std::uint32_t>(tokens), static_cast<std::uint32_t>(groups),
                           static_cast<std::uint32_t>(k / 32)};
      ctx.run_once([&](VkCommandBuffer cmd) {
        pipe.dispatch(cmd, set, &push, sizeof(push), static_cast<std::uint32_t>((tg * 16 + 63) / 64));
        gpu::compute_to_host_barrier(cmd);
      });
      CHECK(std::memcmp(bq.mapped(), want.q(0, 0), tg * 512) == 0);
      // On a mismatch, name the first differing code: its block, mode and input.
      const auto* gq = static_cast<const std::int8_t*>(bq.mapped());
      for (std::size_t i = 0; i < tg * 512; ++i) {
        if (gq[i] == want.q(0, 0)[i]) continue;
        const std::size_t grp = i / 512, word = (i % 512) / 4, chunk = word / 16, slot = word % 16;
        const std::size_t t = grp / groups, g = grp % groups;
        const std::size_t blk = g * 16 + slot, v = chunk * 4 + i % 4;
        const float xv = x[t * static_cast<std::size_t>(k) + blk * 32 + v];
        float amax = 0.0f;
        for (int u = 0; u < 32; ++u)
          amax =
              std::fmax(amax, std::fabs(x[t * static_cast<std::size_t>(k) + blk * 32 + static_cast<std::size_t>(u)]));
        FAIL(std::format("token {} block {} value {} (mode {}): x {:a}, amax {:a}, m {:a}, x*m {:a}: gpu {}, cpu {}", t,
                         blk, v, (t * static_cast<std::size_t>(k) / 32 + blk) % 9, xv, amax, 127.0f / amax,
                         xv * (127.0f / amax), gq[i], want.q(0, 0)[i]));
      }
      CHECK(std::memcmp(bd.mapped(), want.d(0, 0), tg * 64) == 0);
      CHECK(std::memcmp(bs.mapped(), want.s(0, 0), tg * 64) == 0);
      if (std::memcmp(bd.mapped(), want.d(0, 0), tg * 64) != 0) {
        const auto* gd = static_cast<const float*>(bd.mapped());
        for (std::size_t i = 0; i < tg * 16; ++i) {
          if (std::bit_cast<std::uint32_t>(gd[i]) != std::bit_cast<std::uint32_t>(want.d(0, 0)[i])) {
            FAIL(std::format("scale {}: gpu {:a}, cpu {:a}", i, gd[i], want.d(0, 0)[i]));
          }
        }
      }
      blocks_compared += tokens * k / 32;
    }
    MESSAGE(std::format("{} blocks bit-identical to cpu::PackedActs", blocks_compared));
    CHECK(blocks_compared == 1 + 15 + 51 + 128 + 480 + 256 + 8320 + 16384);
    CHECK(ctx.validation_message_count() == 0);
  }
}
