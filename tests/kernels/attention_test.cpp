// cpu::attend (scalar and AVX-512) against the cpu-ref attention loop of
// model/llama_ref.cpp, bit for bit: every length remainder of the 16-position
// score blocks and every head width remainder of the 64/16-dimension passes.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "core/cpu_features.h"
#include "cpu/attention.h"
#include "cpu/ref_ops.h"

using namespace hearth;

namespace {

// The cpu-ref loop, reading K row by row (row s at k + s * stride).
void reference(const float* q, const float* k, const float* v, std::int64_t stride, std::int64_t len, std::int64_t hd,
               float scale, float* out) {
  std::vector<float> scores(static_cast<std::size_t>(len));
  for (std::int64_t s = 0; s < len; ++s) {
    const float* ks = k + s * stride;
    float dot = 0.0f;
    for (std::int64_t d = 0; d < hd; ++d) dot += q[d] * ks[d];
    scores[static_cast<std::size_t>(s)] = dot * scale;
  }
  cpu::ref::softmax(scores.data(), len);
  std::fill(out, out + hd, 0.0f);
  for (std::int64_t s = 0; s < len; ++s) {
    const float p = scores[static_cast<std::size_t>(s)];
    const float* vs = v + s * stride;
    for (std::int64_t d = 0; d < hd; ++d) out[d] += p * vs[d];
  }
}

}  // namespace

TEST_SUITE("quick") {
  TEST_CASE("cpu attention: scalar and AVX-512 give the cpu-ref loop's bits") {
    std::mt19937 rng(5);
    std::normal_distribution<float> n(0.0f, 1.0f);
    for (const std::int64_t hd : {8, 48, 64, 80, 128}) {
      for (const std::int64_t len : {1, 2, 15, 16, 17, 33, 100, 513}) {
        CAPTURE(hd);
        CAPTURE(len);
        // Two KV heads side by side, as in the cache (stride 2 * hd); the second is used.
        const std::int64_t stride = 2 * hd, ctx = len + 7;
        std::vector<float> q(static_cast<std::size_t>(hd)), k(static_cast<std::size_t>(ctx * stride)), v(k.size());
        for (auto& x : q) x = n(rng) * 3.0f;
        for (auto& x : k) x = n(rng);
        for (auto& x : v) x = n(rng);
        const float* kh = k.data() + hd;
        const float* vh = v.data() + hd;
        // The blocked cache: [kv_head][cpu::k_index(position, d)], whole 16-position blocks
        // per head. Unwritten floats (the last block's tail) stay -1; the kernels' masked loads
        // zero those lanes, so they cannot reach a stored score.
        const std::int64_t head = cpu::k_head_size(ctx, hd);
        std::vector<float> kt(static_cast<std::size_t>(2 * head), -1.0f);
        for (std::int64_t s = 0; s < ctx; ++s) {
          for (std::int64_t j = 0; j < stride; ++j)
            kt[static_cast<std::size_t>(j / hd * head + cpu::k_index(s, j % hd, hd))] =
                k[static_cast<std::size_t>(s * stride + j)];
        }
        const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
        std::vector<float> want(static_cast<std::size_t>(hd));
        reference(q.data(), kh, vh, stride, len, hd, scale, want.data());
        for (const Isa isa : {Isa::kScalar, Isa::kAvx512}) {
          if (!isa_supported(isa)) continue;
          CAPTURE(to_string(isa));
          std::vector<float> scores(static_cast<std::size_t>(len)), got(static_cast<std::size_t>(hd), -1.0f);
          cpu::attend(isa, {q.data(), kt.data() + head, vh, stride, len, hd, scale, scores.data(), got.data()});
          CHECK(std::memcmp(got.data(), want.data(), want.size() * sizeof(float)) == 0);
        }
      }
    }
  }
}
