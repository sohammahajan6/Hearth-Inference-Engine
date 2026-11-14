// The canonical dot product of ADR 0009: the packed scalar kernel and the
// AVX-512 GEMV/GEMM tiles must give canonical_dot's bits exactly, for every
// tile remainder, partial group and adversarial block; and canonical_dot must
// meet docs/ACCEPTANCE.md's float-dot bound (3e-5 * sum|terms|) against FP64.
#include <doctest/doctest.h>

#include <cmath>
#include <cstring>
#include <format>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "core/cpu_features.h"
#include "core/fp16.h"
#include "core/quant.h"
#include "cpu/qgemm.h"

using namespace hearth;
using quant::BlockQ4_0;
using quant::BlockQ8_0;
using quant::kBlock;

namespace {

// Scales: positive, negative, subnormal, zero and large magnitudes.
std::uint16_t scale(std::mt19937& rng, std::size_t i) {
  std::uniform_real_distribution<float> u(1e-4f, 0.05f);
  switch (i % 9) {
    case 0:
      return 0x0003;  // subnormal fp16
    case 1:
      return 0x0000;  // zero
    case 2:
      return 0x8000;  // negative zero
    case 3:
      return static_cast<std::uint16_t>(fp32_to_fp16(u(rng)) | 0x8000u);  // negative
    case 4:
      return fp32_to_fp16(u(rng) * 1000.0f);
    default:
      return fp32_to_fp16(u(rng));
  }
}

std::vector<BlockQ8_0> make_q8(std::int64_t rows, std::int64_t k, std::mt19937& rng) {
  std::vector<BlockQ8_0> w(static_cast<std::size_t>(rows * (k / kBlock)));
  std::uniform_int_distribution<int> q(-128, 127);
  for (std::size_t i = 0; i < w.size(); ++i) {
    w[i].d = scale(rng, i);
    const int mode = static_cast<int>(i % 5);
    for (int j = 0; j < kBlock; ++j) {
      // mode 1: the extremes, including -128 (legal in a file, never produced by a quantizer)
      w[i].q[j] = static_cast<std::int8_t>(mode == 1 ? ((j % 2) != 0 ? -128 : 127) : mode == 2 ? 0 : q(rng));
    }
  }
  return w;
}

std::vector<BlockQ4_0> make_q4(std::int64_t rows, std::int64_t k, std::mt19937& rng) {
  std::vector<BlockQ4_0> w(static_cast<std::size_t>(rows * (k / kBlock)));
  std::uniform_int_distribution<int> q(0, 255);
  for (std::size_t i = 0; i < w.size(); ++i) {
    w[i].d = scale(rng, i);
    const int mode = static_cast<int>(i % 5);
    for (int j = 0; j < kBlock / 2; ++j) {
      w[i].q[j] = static_cast<std::uint8_t>(mode == 1 ? 0xF0 : mode == 2 ? 0xFF : mode == 3 ? 0x00 : q(rng));
    }
  }
  return w;
}

// `tokens` rows of k activations: normal, all-zero, rounding-tie, tiny and large blocks.
std::vector<float> make_x(std::int64_t tokens, std::int64_t k, std::mt19937& rng) {
  std::vector<float> x(static_cast<std::size_t>(tokens * k));
  std::normal_distribution<float> n(0.0f, 1.5f);
  for (std::size_t b = 0; b < x.size() / kBlock; ++b) {
    float* v = x.data() + b * kBlock;
    const int mode = static_cast<int>((b * 7 + b / 5) % 6);
    for (int j = 0; j < kBlock; ++j) {
      switch (mode) {
        case 1:
          v[j] = 0.0f;
          break;
        case 2:
          v[j] = (j == 0 ? 127.0f : static_cast<float>(j) + 0.5f) * ((j % 2) != 0 ? -1.0f : 1.0f);
          break;
        case 3:
          v[j] = n(rng) * 1e-3f;
          break;
        case 4:
          v[j] = n(rng) * 1e3f;
          break;
        default:
          v[j] = n(rng);
          break;
      }
    }
  }
  return x;
}

struct Unpacked {
  std::vector<std::int8_t> q;
  std::vector<float> d;
  std::vector<std::int32_t> s;
  quant::ActQ8 view(std::int64_t k) const { return {q.data(), d.data(), s.data(), k}; }
};

Unpacked unpacked(const float* x, std::int64_t k) {
  Unpacked u;
  u.q.resize(static_cast<std::size_t>(k));
  u.d.resize(static_cast<std::size_t>(k / kBlock));
  u.s.resize(static_cast<std::size_t>(k / kBlock));
  quant::quantize_act_q8_0(x, k, u.q.data(), u.d.data(), u.s.data());
  return u;
}

std::string first_difference(const std::vector<float>& got, const std::vector<float>& want, std::int64_t rows) {
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (std::memcmp(&got[i], &want[i], sizeof(float)) != 0) {
      return std::format("token {} row {}: got {:a}, canonical {:a}", static_cast<std::int64_t>(i) / rows,
                         static_cast<std::int64_t>(i) % rows, got[i], want[i]);
    }
  }
  return {};
}

struct Shape {
  std::int64_t rows, k, tokens;
};
// Every row remainder of the 4-row tiles (1..9 rows), odd and even token
// counts (2-token tiles plus a 1-token remainder), partial groups (k not a
// multiple of 512) and the real models' widths.
constexpr Shape kShapes[] = {{1, 32, 1},   {2, 64, 2},   {3, 480, 3}, {4, 512, 1},   {5, 544, 4},   {6, 1056, 5},
                             {7, 2048, 2}, {8, 3072, 3}, {9, 160, 7}, {13, 8192, 1}, {4, 2048, 130}};

template <class Block>
void check_type(cpu::QType type, const std::vector<Block>& w, const Shape& sh, const std::vector<float>& x) {
  const auto rows = sh.rows, k = sh.k, T = sh.tokens;
  const std::int64_t nb = k / kBlock;
  // The specification, element by element.
  std::vector<float> want(static_cast<std::size_t>(T * rows));
  for (std::int64_t t = 0; t < T; ++t) {
    const Unpacked u = unpacked(x.data() + t * k, k);
    for (std::int64_t r = 0; r < rows; ++r) {
      want[static_cast<std::size_t>(t * rows + r)] = cpu::canonical_dot(w.data() + r * nb, u.view(k));
      // ACCEPTANCE: float dots within 3e-5 * sum |terms| of the exact value.
      double exact = 0, mag = 0;
      for (std::int64_t b = 0; b < nb; ++b) {
        const Block& blk = w[static_cast<std::size_t>(r * nb + b)];
        std::int32_t isum = 0;
        if constexpr (std::is_same_v<Block, BlockQ8_0>) {
          isum = quant::block_dot_q8_0(blk, u.q.data() + b * kBlock);
        } else {
          isum = quant::block_dot_q4_0(blk, u.q.data() + b * kBlock);
        }
        const double term = static_cast<double>(fp16_to_fp32(blk.d)) * u.d[static_cast<std::size_t>(b)] * isum;
        exact += term;
        mag += std::fabs(term);
      }
      const double err = std::fabs(want[static_cast<std::size_t>(t * rows + r)] - exact);
      if (!(err <= 3e-5 * mag)) FAIL(std::format("row {} token {}: error {} > 3e-5 * {}", r, t, err, mag));
    }
  }
  const cpu::PackedMatrix pm(type, w.data(), rows, k);
  cpu::PackedActs acts;
  acts.resize(T, k);
  acts.quantize(Isa::kScalar, x.data(), 0, T);
  for (const Isa isa : {Isa::kScalar, Isa::kAvx512}) {
    if (!isa_supported(isa)) continue;
    CAPTURE(to_string(isa));
    std::vector<float> got(want.size(), -1.0f);
    cpu::qgemm(isa, pm.view(), acts, got.data(), rows, 0, rows);
    const std::string diff = first_difference(got, want, rows);
    CHECK_MESSAGE(diff.empty(), diff);
    // Any split of the rows (as threads split them) gives the same bits.
    std::vector<float> split(want.size(), -1.0f);
    const std::int64_t cut = rows / 3 + 1;
    cpu::qgemm(isa, pm.view(), acts, split.data(), rows, 0, cut);
    cpu::qgemm(isa, pm.view(), acts, split.data(), rows, cut, rows);
    CHECK(std::memcmp(split.data(), want.data(), want.size() * sizeof(float)) == 0);
  }
}

}  // namespace

TEST_SUITE("quick") {
  TEST_CASE("qgemm: tile_range splits rows in whole 4-row tiles, covering each row once") {
    for (const std::int64_t rows : {1, 3, 4, 5, 13, 2048, 128256}) {
      for (const int parts : {1, 2, 3, 4, 8, 64}) {
        CAPTURE(rows);
        CAPTURE(parts);
        std::int64_t next = 0;
        for (int i = 0; i < parts; ++i) {
          std::int64_t r0 = -1, r1 = -1;
          cpu::tile_range(rows, parts, i, r0, r1);
          REQUIRE(r0 <= r1);
          CHECK(r0 == next);  // contiguous, in order
          CHECK((r0 % 4 == 0 || r0 == rows));
          next = r1;
        }
        CHECK(next == rows);
      }
    }
  }

  TEST_CASE("qgemm: packed activations hold the ggml quantizer's codes, scales and sums, zero-padded") {
    std::mt19937 rng(3);
    // Per ISA, one buffer for every shape, largest first: each quantize must
    // overwrite the previous shape's bytes (real values), padding included.
    for (const Isa isa : {Isa::kScalar, Isa::kAvx512}) {
      if (!isa_supported(isa)) continue;
      cpu::PackedActs a;
      for (const std::int64_t k : {1056, 512, 480, 32}) {
        CAPTURE(k);
        CAPTURE(to_string(isa));
        const std::int64_t T = 3, nb = k / kBlock, groups = cpu::groups_for(k);
        const auto x = make_x(T, k, rng);
        a.resize(T, k);
        a.quantize(isa, x.data(), 0, T);
        REQUIRE(a.groups() == groups);
        for (std::int64_t t = 0; t < T; ++t) {
          const Unpacked u = unpacked(x.data() + t * k, k);
          for (std::int64_t b = 0; b < groups * cpu::kGroupBlocks; ++b) {
            const std::int64_t g = b / cpu::kGroupBlocks, j = b % cpu::kGroupBlocks;
            const bool real = b < nb;
            CHECK(a.d(t, g)[j] == (real ? u.d[static_cast<std::size_t>(b)] : 0.0f));
            CHECK(a.s(t, g)[j] == (real ? u.s[static_cast<std::size_t>(b)] : 0));
            for (int v = 0; v < kBlock; ++v) {
              const std::int8_t code = a.q(t, g)[(v / 4) * cpu::kChunkBytes + j * 4 + v % 4];
              CHECK(code == (real ? u.q[static_cast<std::size_t>(b * kBlock + v)] : 0));
            }
          }
        }
      }
    }
  }

  TEST_CASE("qgemm: scalar and AVX-512 kernels give canonical_dot's bits for Q8_0 and Q4_0") {
    std::mt19937 rng(9);
    for (const Shape& sh : kShapes) {
      CAPTURE(sh.rows);
      CAPTURE(sh.k);
      CAPTURE(sh.tokens);
      const auto x = make_x(sh.tokens, sh.k, rng);
      check_type(cpu::QType::kQ8_0, make_q8(sh.rows, sh.k, rng), sh, x);
      check_type(cpu::QType::kQ4_0, make_q4(sh.rows, sh.k, rng), sh, x);
    }
  }

  TEST_CASE("qgemm: canonical_dot sums in 16 lanes, then by halving") {
    // 32 blocks (k = 1024) whose weights are 1 at value 0 and 0 elsewhere, so
    // isum_b = activation code 0 of block b, and t_b = code * d_x exactly.
    // The values are chosen so that the rounding of the sum exposes the order.
    constexpr std::int64_t k = 1024, nb = k / kBlock;
    std::vector<BlockQ8_0> w(nb);
    for (auto& blk : w) {
      blk = {};
      blk.d = fp32_to_fp16(1.0f);
      blk.q[0] = 1;
    }
    std::vector<std::int8_t> q(k, 0);
    std::vector<float> d(nb, 0.0f);
    std::vector<std::int32_t> s(nb, 0);
    auto set = [&](std::int64_t b, float t) {
      q[static_cast<std::size_t>(b * kBlock)] = 1;
      d[static_cast<std::size_t>(b)] = t;
    };
    set(0, 0x1p24f);  // lane 0
    set(1, 0.5f);     // lane 1
    set(17, 0.5f);    // lane 1 again (second group): lane 1 = 1, exact
    set(9, 1.0f);     // lane 9
    const float got = cpu::canonical_dot(w.data(), {q.data(), d.data(), s.data(), k});
    // Halving: lane 0 + lane 8 = 2^24 and lane 1 + lane 9 = 2, so the sum is
    // 2^24 + 2, exact.
    CHECK(got == 0x1p24f + 2.0f);
    // Adding one block at a time, or one lane at a time, rounds each small
    // term away against 2^24 (2^24 + 1 is a tie and rounds to even).
    float by_block = 0.0f;
    for (std::int64_t b = 0; b < nb; ++b) by_block += d[static_cast<std::size_t>(b)];
    CHECK(by_block == 0x1p24f);
    // 16 lanes, not 8: blocks 8 and 24 share lane 8 (= 2, exact), so the sum
    // is 2^24 + 2. With 8 lanes they would join block 0's lane one at a time,
    // and each 1 would round away against 2^24.
    std::fill(d.begin(), d.end(), 0.0f);
    std::fill(q.begin(), q.end(), std::int8_t{0});
    set(0, 0x1p24f);
    set(8, 1.0f);
    set(24, 1.0f);
    CHECK(cpu::canonical_dot(w.data(), {q.data(), d.data(), s.data(), k}) == 0x1p24f + 2.0f);
    // Signed zeros: negative weight scales times all-zero activations give
    // -0 block terms, and +0 + -0 = +0, so every kernel returns +0.
    for (auto& blk : w) blk.d = fp32_to_fp16(-1.0f);
    std::vector<float> x(k, 0.0f);
    cpu::PackedActs acts;
    acts.resize(1, k);
    acts.quantize(Isa::kScalar, x.data(), 0, 1);
    const cpu::PackedMatrix pm(cpu::QType::kQ8_0, w.data(), 1, k);
    for (const Isa isa : {Isa::kScalar, Isa::kAvx512}) {
      if (!isa_supported(isa)) continue;
      float y = -1.0f;
      cpu::qgemm(isa, pm.view(), acts, &y, 1, 0, 1);
      CHECK(y == 0.0f);
      CHECK(!std::signbit(y));
    }
  }
}
