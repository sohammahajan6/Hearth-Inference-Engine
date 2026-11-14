// The non-matrix shaders of the GPU forward pass (docs/architecture/08-gpu-
// backend.md section 3) against cpu::ref, with ACCEPTANCE's bars:
//   RoPE (Llama's and NEOX pairs), the residual add and the bias add: the
//   CPU reference's bits;
//   RMSNorm and SiLU-up: |d| <= 1e-5 |ref| + 1e-6 per element;
//   attention: |d| <= 3e-5 sum |terms| + 1e-6 per output, against an FP64
//   reference, and the same bits for a chunk of tokens as for one token at a
//   time (prefill = decode).
// All under the validation layer when the test scripts enable it.
#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <format>
#include <random>
#include <string>
#include <vector>

#include "cpu/ref_ops.h"
#include "gpu/buffer.h"
#include "gpu/context.h"
#include "gpu/pipeline.h"
#include "test_util.h"

using namespace hearth;

namespace {

gpu::Buffer upload(gpu::Context& ctx, const std::vector<float>& v,
                   gpu::MemoryKind kind = gpu::MemoryKind::kHostCached) {
  gpu::Buffer b(ctx, std::max<std::size_t>(v.size() * 4, 16), kind);
  std::memcpy(b.mapped(), v.data(), v.size() * 4);
  return b;
}

std::vector<float> download(const gpu::Buffer& b, std::size_t n) {
  std::vector<float> v(n);
  std::memcpy(v.data(), b.mapped(), n * 4);
  return v;
}

std::vector<float> randoms(std::size_t n, std::mt19937& rng, float sigma) {
  std::normal_distribution<float> d(0.0f, sigma);
  std::vector<float> v(n);
  for (float& x : v) x = d(rng);
  return v;
}

void run(gpu::Context& ctx, const gpu::Pipeline& pipe, VkDescriptorSet set, const void* push, std::uint32_t bytes,
         std::uint32_t groups) {
  ctx.run_once([&](VkCommandBuffer cmd) {
    pipe.dispatch(cmd, set, push, bytes, groups);
    gpu::compute_to_host_barrier(cmd);
  });
}

// Worst ratio |gpu - ref| / (rel * |ref| + abs) over all elements (<= 1 passes).
double worst(const std::vector<float>& got, const std::vector<double>& ref, const std::vector<double>& mag, double rel,
             double abs) {
  double w = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    w = std::max(w, std::fabs(static_cast<double>(got[i]) - ref[i]) / (rel * mag[i] + abs));
  }
  return w;
}

}  // namespace

TEST_SUITE("quick") {
  TEST_CASE("gpu ops: RMSNorm, RoPE, SiLU-up, the residual add and the bias add against cpu::ref (validation on)") {
    if (!gpu::vulkan_available()) return;
    gpu::Context ctx(test::gpu_test_options());
    gpu::DescriptorPool pool(ctx, 32, 32 * 4);
    std::mt19937 rng(41);

    // RMSNorm, the models' widths and an odd one, several tokens.
    gpu::Pipeline norm(ctx, {.shader = "rmsnorm", .buffers = 3, .push_constant_bytes = 8});
    for (const std::uint32_t n : {96u, 2048u, 3072u}) {
      CAPTURE(n);
      const std::uint32_t T = 3;
      const auto x = randoms(T * n, rng, 2.0f);
      const auto w = randoms(n, rng, 1.0f);
      gpu::Buffer bx = upload(ctx, x), bw = upload(ctx, w), bo(ctx, T * n * 4, gpu::MemoryKind::kHostCached);
      const struct {
        std::uint32_t n;
        float eps;
      } push{n, 1e-5f};
      run(ctx, norm, pool.allocate(norm, {&bx, &bw, &bo}), &push, sizeof push, T);
      const auto got = download(bo, T * n);
      std::vector<float> want(T * n);
      for (std::uint32_t t = 0; t < T; ++t) cpu::ref::rmsnorm(&x[t * n], w.data(), &want[t * n], n, 1e-5f);
      const std::vector<double> ref(want.begin(), want.end());
      std::vector<double> mag(ref.size());
      for (std::size_t i = 0; i < mag.size(); ++i) mag[i] = std::fabs(ref[i]);
      const double r = worst(got, ref, mag, 1e-5, 1e-6);
      const std::string note1 = std::format("rmsnorm n={}: worst error / bar = {:.3f}", n, r);
      MESSAGE(note1);
      CHECK_MESSAGE(r <= 1.0, note1);
    }

    // RoPE, q (32 or 14 heads) and k (8 or 2 heads) rows, positions 37..40:
    // Llama 3's pairs, theta and rope_freqs-like factors; and NEOX pairs with
    // Qwen2's theta and no factors (15-qwen.md). Control: the two pairings
    // give different values for the same input, so the comparison sees which
    // one the shader used.
    gpu::Pipeline rope(ctx, {.shader = "rope", .buffers = 2, .push_constant_bytes = 28});
    for (const bool neox : {false, true}) {
      CAPTURE(neox);
      const std::uint32_t hd = 64, rd = 64, p0 = 37, T = 4;
      const float theta = neox ? 1000000.0f : 500000.0f;
      std::vector<float> factors(rd / 2);
      for (std::size_t i = 0; i < factors.size(); ++i) factors[i] = 1.0f + static_cast<float>(i % 7);
      const float* f = neox ? nullptr : factors.data();
      const std::vector<float> table = cpu::ref::rope_table(p0 + T, rd, theta, f);
      gpu::Buffer bt = upload(ctx, table);
      for (const std::uint32_t heads :
           neox ? std::vector<std::uint32_t>{14u, 2u} : std::vector<std::uint32_t>{32u, 8u}) {
        CAPTURE(heads);
        const std::uint32_t stride = heads * hd;
        auto x = randoms(T * stride, rng, 1.0f);
        auto other = x;
        gpu::Buffer bx = upload(ctx, x);
        const std::uint32_t push[7] = {T, heads, hd, rd, stride, p0, neox ? 1u : 0u};
        run(ctx, rope, pool.allocate(rope, {&bx, &bt}), push, sizeof push, (T * heads * (rd / 2) + 63) / 64);
        for (std::uint32_t t = 0; t < T; ++t) {
          cpu::ref::rope(&x[t * stride], heads, hd, rd, p0 + t, theta, f, neox);
          cpu::ref::rope(&other[t * stride], heads, hd, rd, p0 + t, theta, f, !neox);
        }
        const auto got = download(bx, x.size());
        CHECK(std::memcmp(got.data(), x.data(), x.size() * 4) == 0);
        CHECK(std::memcmp(got.data(), other.data(), x.size() * 4) != 0);
      }
    }

    // SiLU-up, values across the range where exp under- and overflows.
    gpu::Pipeline silu(ctx, {.shader = "silu_mul", .buffers = 2, .push_constant_bytes = 4});
    {
      const std::uint32_t n = 8192 * 3;
      auto g = randoms(n, rng, 4.0f);
      g[0] = 0.0f;
      g[1] = -0.0f;
      g[2] = 88.0f;
      g[3] = -88.0f;
      g[4] = -100.0f;
      const auto u = randoms(n, rng, 1.0f);
      gpu::Buffer bg = upload(ctx, g), bu = upload(ctx, u);
      run(ctx, silu, pool.allocate(silu, {&bg, &bu}), &n, 4, (n + 255) / 256);
      const auto got = download(bg, n);
      std::vector<double> ref(n), mag(n);
      for (std::uint32_t i = 0; i < n; ++i) {
        ref[i] = cpu::ref::silu(g[i]) * u[i];
        mag[i] = std::fabs(ref[i]);
      }
      const double r = worst(got, ref, mag, 1e-5, 1e-6);
      const std::string note2 = std::format("silu_mul: worst error / bar = {:.3f}", r);
      MESSAGE(note2);
      CHECK_MESSAGE(r <= 1.0, note2);
    }

    // The residual add: one rounded add per element.
    gpu::Pipeline add(ctx, {.shader = "residual_add", .buffers = 2, .push_constant_bytes = 4});
    {
      const std::uint32_t n = 2048 * 5 + 7;
      auto x = randoms(n, rng, 3.0f);
      const auto y = randoms(n, rng, 1e-3f);
      gpu::Buffer bx = upload(ctx, x), by = upload(ctx, y);
      run(ctx, add, pool.allocate(add, {&bx, &by}), &n, 4, (n + 255) / 256);
      for (std::uint32_t i = 0; i < n; ++i) x[i] += y[i];
      CHECK(std::memcmp(download(bx, n).data(), x.data(), n * 4) == 0);
    }

    // The bias add (Qwen2's Q/K/V): one rounded add per element, the bias
    // repeated for every token row; Qwen2.5-0.5B's q and kv widths, 5 tokens.
    gpu::Pipeline bias(ctx, {.shader = "bias_add", .buffers = 2, .push_constant_bytes = 8});
    for (const std::uint32_t width : {896u, 128u}) {
      CAPTURE(width);
      const std::uint32_t T = 5, n = T * width;
      auto x = randoms(n, rng, 3.0f);
      const auto b = randoms(width, rng, 1.0f);
      gpu::Buffer bx = upload(ctx, x), bb = upload(ctx, b);
      const std::uint32_t push[2] = {n, width};
      run(ctx, bias, pool.allocate(bias, {&bx, &bb}), push, sizeof push, (n + 255) / 256);
      for (std::uint32_t t = 0; t < T; ++t) {
        for (std::uint32_t j = 0; j < width; ++j) x[t * width + j] += b[j];
      }
      CHECK(std::memcmp(download(bx, n).data(), x.data(), n * 4) == 0);
    }
    CHECK(ctx.validation_message_count() == 0);
  }

  TEST_CASE("gpu ops: attention against FP64, GQA, and prefill = decode bit for bit (validation on)") {
    if (!gpu::vulkan_available()) return;
    gpu::Context ctx(test::gpu_test_options());
    gpu::DescriptorPool pool(ctx, 64, 64 * 5);
    gpu::Pipeline att(ctx, {.shader = "attention", .buffers = 5, .push_constant_bytes = 24});
    std::mt19937 rng(43);
    struct Push {
      std::uint32_t p0, heads, kv_heads, head_dim, n_ctx;
      float scale;
    };
    // (heads, kv heads, head dim): the 1B's ratio 4 and the 3B's ratio 3 with its head width 128.
    for (const auto& [nh, nkv, hd] :
         {std::array<std::uint32_t, 3>{8, 2, 64}, std::array<std::uint32_t, 3>{6, 2, 128}}) {
      CAPTURE(nh);
      CAPTURE(hd);
      const std::uint32_t n_ctx = 320, p0 = 250, T = 9;  // tokens at positions 250..258
      const std::uint32_t Q = nh * hd, KV = nkv * hd;
      const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
      const auto q = randoms(T * Q, rng, 1.0f);
      auto k = randoms(n_ctx * KV, rng, 1.0f);
      const auto v = randoms(n_ctx * KV, rng, 1.0f);
      for (std::uint32_t i = 0; i < KV; ++i) k[7 * KV + i] = 3.0f;  // one position that dominates
      gpu::Buffer bq = upload(ctx, q), bk = upload(ctx, k), bv = upload(ctx, v);
      gpu::Buffer bs(ctx, T * nh * n_ctx * 4, gpu::MemoryKind::kDeviceLocal);
      gpu::Buffer bo(ctx, T * Q * 4, gpu::MemoryKind::kHostCached);
      const VkDescriptorSet set = pool.allocate(att, {&bq, &bk, &bv, &bs, &bo});
      const Push push{p0, nh, nkv, hd, n_ctx, scale};
      run(ctx, att, set, &push, sizeof push, T * nh);
      const auto got = download(bo, T * Q);
      // FP64 reference, and the bar: 3e-5 * sum |p_s v_sd| + 1e-6.
      std::vector<double> ref(T * Q), mag(T * Q);
      for (std::uint32_t t = 0; t < T; ++t) {
        for (std::uint32_t h = 0; h < nh; ++h) {
          const std::uint32_t kvh = h / (nh / nkv), len = p0 + t + 1;
          std::vector<double> s(len);
          double m = -1e300;
          for (std::uint32_t j = 0; j < len; ++j) {
            double dot = 0;
            for (std::uint32_t d = 0; d < hd; ++d)
              dot += static_cast<double>(q[t * Q + h * hd + d]) * k[j * KV + kvh * hd + d];
            s[j] = dot * scale;
            m = std::max(m, s[j]);
          }
          double sum = 0;
          for (double& x : s) sum += (x = std::exp(x - m));
          for (std::uint32_t d = 0; d < hd; ++d) {
            double o = 0, a = 0;
            for (std::uint32_t j = 0; j < len; ++j) {
              const double term = s[j] / sum * v[j * KV + kvh * hd + d];
              o += term;
              a += std::fabs(term);
            }
            ref[t * Q + h * hd + d] = o;
            mag[t * Q + h * hd + d] = a;
          }
        }
      }
      const double r = worst(got, ref, mag, 3e-5, 1e-6);
      const std::string note3 = std::format("attention {}:{} hd {}: worst error / bar = {:.3f}", nh, nkv, hd, r);
      MESSAGE(note3);
      CHECK_MESSAGE(r <= 1.0, note3);
      // Prefill = decode: token t alone, at position p0 + t, gives the same bits.
      for (std::uint32_t t = 0; t < T; ++t) {
        std::vector<float> qt(q.begin() + t * Q, q.begin() + (t + 1) * Q);
        gpu::Buffer bq1 = upload(ctx, qt);
        gpu::Buffer bo1(ctx, Q * 4, gpu::MemoryKind::kHostCached);
        const Push p1{p0 + t, nh, nkv, hd, n_ctx, scale};
        run(ctx, att, pool.allocate(att, {&bq1, &bk, &bv, &bs, &bo1}), &p1, sizeof p1, nh);
        CHECK(std::memcmp(download(bo1, Q).data(), got.data() + t * Q, Q * 4) == 0);
      }
    }
    CHECK(ctx.validation_message_count() == 0);
  }
}
