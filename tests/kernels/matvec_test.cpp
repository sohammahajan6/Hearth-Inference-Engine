// Matvec kernels (CPU scalar, CPU AVX-512, GPU; AoS and SoA layouts) against
// an FP64 reference built from exact integer block sums. docs/ACCEPTANCE.md:
// float dot products within 3e-5 * sum|terms|; AoS and SoA kernels of the
// same device must agree bit for bit (same operations, same order).
#include <doctest/doctest.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "core/cpu_features.h"
#include "core/fp16.h"
#include "core/quant.h"
#include "cpu/matvec.h"
#include "gpu/buffer.h"
#include "gpu/context.h"
#include "gpu/matvec.h"
#include "test_util.h"

using namespace hearth;
using quant::BlockQ4_0;
using quant::BlockQ8_0;
using quant::kBlock;

namespace {

struct Activations {
  std::vector<float> raw;
  std::vector<std::int8_t> q;
  std::vector<float> d;
  std::vector<std::int32_t> s;
  quant::ActQ8 view() const { return {q.data(), d.data(), s.data(), static_cast<std::int64_t>(raw.size())}; }
};

Activations make_activations(std::int64_t k, std::mt19937& rng) {
  Activations a;
  a.raw.resize(static_cast<std::size_t>(k));
  std::normal_distribution<float> n(0.0f, 1.5f);
  for (std::int64_t b = 0; b < k / kBlock; ++b) {
    float* x = a.raw.data() + b * kBlock;
    switch (b % 4) {
      case 0:
        for (int j = 0; j < kBlock; ++j) x[j] = n(rng);
        break;
      case 1:
        for (int j = 0; j < kBlock; ++j) x[j] = 0.0f;
        break;  // all-zero block: scale 0
      case 2:   // values that land exactly on .5 after scaling (rounding ties)
        for (int j = 0; j < kBlock; ++j)
          x[j] = (j == 0 ? 127.0f : static_cast<float>(j) + 0.5f) * ((j % 2) != 0 ? -1.0f : 1.0f);
        break;
      default:
        for (int j = 0; j < kBlock; ++j) x[j] = n(rng) * 1e-3f;
        break;
    }
  }
  a.q.resize(a.raw.size());
  a.d.resize(a.raw.size() / kBlock);
  a.s.resize(a.raw.size() / kBlock);
  quant::quantize_act_q8_0(a.raw.data(), k, a.q.data(), a.d.data(), a.s.data());
  return a;
}

std::uint16_t random_scale(std::mt19937& rng, int mode) {
  if (mode == 0) return 0x0003;  // subnormal fp16
  if (mode == 1) return 0x0000;  // zero
  std::uniform_real_distribution<float> u(1e-4f, 0.05f);
  return fp32_to_fp16(u(rng));
}

std::vector<BlockQ8_0> make_q8(std::int64_t rows, std::int64_t k, std::mt19937& rng) {
  std::vector<BlockQ8_0> w(static_cast<std::size_t>(rows * (k / kBlock)));
  std::uniform_int_distribution<int> q(-128, 127);
  for (std::size_t i = 0; i < w.size(); ++i) {
    const int mode = static_cast<int>(i % 7);
    w[i].d = random_scale(rng, mode);
    for (int j = 0; j < kBlock; ++j) {
      // mode 2: the extremes, including -128 (legal in a file, never produced by our own quantizer)
      w[i].q[j] = static_cast<std::int8_t>(mode == 2 ? ((j % 2) != 0 ? -128 : 127) : (mode == 3 ? 0 : q(rng)));
    }
  }
  return w;
}

std::vector<BlockQ4_0> make_q4(std::int64_t rows, std::int64_t k, std::mt19937& rng) {
  std::vector<BlockQ4_0> w(static_cast<std::size_t>(rows * (k / kBlock)));
  std::uniform_int_distribution<int> q(0, 255);
  for (std::size_t i = 0; i < w.size(); ++i) {
    const int mode = static_cast<int>(i % 7);
    w[i].d = random_scale(rng, mode);
    for (int j = 0; j < kBlock / 2; ++j) {
      w[i].q[j] = static_cast<std::uint8_t>(mode == 2 ? 0xF0 : mode == 3 ? 0xFF : mode == 4 ? 0x00 : q(rng));
    }
  }
  return w;
}

void check_against_reference(const std::vector<float>& y, const std::vector<double>& ref,
                             const std::vector<double>& mag, const char* what) {
  for (std::size_t r = 0; r < y.size(); ++r) {
    const double err = std::fabs(static_cast<double>(y[r]) - ref[r]);
    if (!(err <= 3e-5 * mag[r])) {
      FAIL(what << ": row " << r << " got " << y[r] << " reference " << ref[r] << " |terms| " << mag[r]);
    }
  }
}

struct Shape {
  std::int64_t rows, k;
};
constexpr Shape kShapes[] = {{67, 160}, {33, 2048}, {5, 32}, {16, 8192}};

}  // namespace

TEST_SUITE("quick") {
  TEST_CASE("quant: activation quantizer matches ggml's rule on edge cases") {
    // All-zero block: scale 0, quants 0.
    std::vector<float> x(64, 0.0f);
    for (int j = 0; j < 32; ++j) x[32 + j] = static_cast<float>(j - 16);  // max |x| = 16 at j = 0
    std::vector<std::int8_t> q(64);
    std::vector<float> d(2);
    std::vector<std::int32_t> s(2);
    quant::quantize_act_q8_0(x.data(), 64, q.data(), d.data(), s.data());
    CHECK(d[0] == 0.0f);
    CHECK(s[0] == 0);
    for (int j = 0; j < 32; ++j) CHECK(q[j] == 0);
    // Second block: multiplier 127/16 = 7.9375; x = -16 -> -127, x = 8 -> 63.5 -> 64 (ties to even), 15 -> 119.06 ->
    // 119.
    CHECK(q[32] == -127);
    CHECK(q[32 + 24] == 64);
    CHECK(q[32 + 31] == 119);
    CHECK(d[1] == fp16_to_fp32(fp32_to_fp16(16.0f / 127.0f)));
  }

  TEST_CASE("cpu matvec: scalar and AVX-512 kernels, AoS and SoA, match the FP64 reference") {
    std::mt19937 rng(42);
    for (const Shape sh : kShapes) {
      CAPTURE(sh.rows);
      CAPTURE(sh.k);
      const Activations act = make_activations(sh.k, rng);
      const auto rows = static_cast<std::size_t>(sh.rows);
      {  // Q8_0
        const auto w = make_q8(sh.rows, sh.k, rng);
        const cpu::Q8_0Aos aos{w.data(), sh.rows, sh.k};
        std::vector<std::uint16_t> sd(w.size());
        std::vector<std::int8_t> sq(w.size() * kBlock);
        cpu::repack(aos, sd.data(), sq.data());
        const cpu::Q8_0Soa soa{sd.data(), sq.data(), sh.rows, sh.k};
        std::vector<double> ref(rows), mag(rows);
        cpu::matvec_reference(aos, act.view(), ref.data(), mag.data());
        for (Isa isa : {Isa::kScalar, Isa::kAvx512}) {
          if (!isa_supported(isa)) continue;
          std::vector<float> ya(rows), ys(rows);
          cpu::matvec(isa, aos, act.view(), ya.data(), 0, sh.rows);
          cpu::matvec(isa, soa, act.view(), ys.data(), 0, sh.rows);
          CHECK(std::memcmp(ya.data(), ys.data(), rows * sizeof(float)) == 0);
          check_against_reference(ya, ref, mag, isa == Isa::kScalar ? "q8 scalar" : "q8 avx512");
        }
      }
      {  // Q4_0
        const auto w = make_q4(sh.rows, sh.k, rng);
        const cpu::Q4_0Aos aos{w.data(), sh.rows, sh.k};
        std::vector<std::uint16_t> sd(w.size());
        std::vector<std::uint8_t> sq(w.size() * kBlock / 2);
        cpu::repack(aos, sd.data(), sq.data());
        const cpu::Q4_0Soa soa{sd.data(), sq.data(), sh.rows, sh.k};
        std::vector<double> ref(rows), mag(rows);
        cpu::matvec_reference(aos, act.view(), ref.data(), mag.data());
        for (Isa isa : {Isa::kScalar, Isa::kAvx512}) {
          if (!isa_supported(isa)) continue;
          std::vector<float> ya(rows), ys(rows);
          cpu::matvec(isa, aos, act.view(), ya.data(), 0, sh.rows);
          cpu::matvec(isa, soa, act.view(), ys.data(), 0, sh.rows);
          CHECK(std::memcmp(ya.data(), ys.data(), rows * sizeof(float)) == 0);
          check_against_reference(ya, ref, mag, isa == Isa::kScalar ? "q4 scalar" : "q4 avx512");
        }
      }
    }
  }

  TEST_CASE("gpu matvec: AoS and SoA shaders match each other and the FP64 reference (validation on)") {
    if (!gpu::vulkan_available()) return;
    gpu::Context ctx(test::gpu_test_options());
    gpu::MatvecPipelines pipes(ctx);
    gpu::DescriptorPool pool(ctx, 32, 32 * 6);
    std::mt19937 rng(7);
    auto upload = [&](const void* data, std::size_t bytes) {
      gpu::Buffer b(ctx, std::max<std::size_t>(bytes, 16), gpu::MemoryKind::kHostCoherent);
      std::memcpy(b.mapped(), data, bytes);
      return b;
    };
    for (const Shape sh : kShapes) {
      CAPTURE(sh.rows);
      CAPTURE(sh.k);
      const Activations act = make_activations(sh.k, rng);
      const auto rows = static_cast<std::size_t>(sh.rows);
      const auto nb = static_cast<std::uint32_t>(sh.k / kBlock);
      gpu::Buffer xq = upload(act.q.data(), act.q.size());
      gpu::Buffer xd = upload(act.d.data(), act.d.size() * 4);
      gpu::Buffer xs = upload(act.s.data(), act.s.size() * 4);
      for (gpu::WeightFormat fmt : {gpu::WeightFormat::kQ8_0, gpu::WeightFormat::kQ4_0}) {
        std::vector<double> ref(rows), mag(rows);
        std::vector<std::uint8_t> aos_bytes, soa_d, soa_q;
        if (fmt == gpu::WeightFormat::kQ8_0) {
          const auto w = make_q8(sh.rows, sh.k, rng);
          const cpu::Q8_0Aos aos{w.data(), sh.rows, sh.k};
          cpu::matvec_reference(aos, act.view(), ref.data(), mag.data());
          aos_bytes.assign(reinterpret_cast<const std::uint8_t*>(w.data()),
                           reinterpret_cast<const std::uint8_t*>(w.data()) + w.size() * sizeof(BlockQ8_0));
          soa_d.resize(w.size() * 2);
          soa_q.resize(w.size() * kBlock);
          cpu::repack(aos, reinterpret_cast<std::uint16_t*>(soa_d.data()),
                      reinterpret_cast<std::int8_t*>(soa_q.data()));
        } else {
          const auto w = make_q4(sh.rows, sh.k, rng);
          const cpu::Q4_0Aos aos{w.data(), sh.rows, sh.k};
          cpu::matvec_reference(aos, act.view(), ref.data(), mag.data());
          aos_bytes.assign(reinterpret_cast<const std::uint8_t*>(w.data()),
                           reinterpret_cast<const std::uint8_t*>(w.data()) + w.size() * sizeof(BlockQ4_0));
          soa_d.resize(w.size() * 2);
          soa_q.resize(w.size() * kBlock / 2);
          cpu::repack(aos, reinterpret_cast<std::uint16_t*>(soa_d.data()), soa_q.data());
        }
        gpu::Buffer w_aos = upload(aos_bytes.data(), aos_bytes.size());
        gpu::Buffer w_d = upload(soa_d.data(), soa_d.size());
        gpu::Buffer w_q = upload(soa_q.data(), soa_q.size());
        std::vector<float> results[2];
        for (gpu::WeightLayout lay : {gpu::WeightLayout::kAos, gpu::WeightLayout::kSoa}) {
          gpu::Buffer y(ctx, rows * 4, gpu::MemoryKind::kHostCached);
          const bool is_aos = lay == gpu::WeightLayout::kAos;
          const VkDescriptorSet set =
              pool.allocate(pipes.get(fmt, lay), {is_aos ? &w_aos : &w_d, is_aos ? &w_aos : &w_q, &xq, &xd, &xs, &y});
          const gpu::MatvecPush push{nb, 0, static_cast<std::uint32_t>(sh.rows), 0, 0};
          ctx.run_once([&](VkCommandBuffer cmd) {
            pipes.record(cmd, fmt, lay, set, push);
            gpu::compute_to_host_barrier(cmd);
          });
          const auto* p = static_cast<const float*>(y.mapped());
          results[is_aos ? 0 : 1].assign(p, p + rows);
          check_against_reference(results[is_aos ? 0 : 1], ref, mag,
                                  fmt == gpu::WeightFormat::kQ8_0 ? (is_aos ? "gpu q8 aos" : "gpu q8 soa")
                                                                  : (is_aos ? "gpu q4 aos" : "gpu q4 soa"));
        }
        CHECK(std::memcmp(results[0].data(), results[1].data(), rows * sizeof(float)) == 0);
      }
    }
    CHECK(ctx.validation_message_count() == 0);
  }
}
