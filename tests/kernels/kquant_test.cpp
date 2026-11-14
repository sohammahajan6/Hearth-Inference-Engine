// K-quants: the block formats decode and Q8_K activations quantize
// exactly as ggml does (pinned by hashes of ggml's own outputs); the packed
// scalar kernel and the AVX-512 GEMV/GEMM tiles give canonical_dot's bits for
// every tile remainder, partial group and adversarial block; and
// canonical_dot meets docs/ACCEPTANCE.md's float-dot bound against FP64.
#include <doctest/doctest.h>

#include <cmath>
#include <cstring>
#include <format>
#include <random>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include "core/cpu_features.h"
#include "core/fp16.h"
#include "core/kquant.h"
#include "cpu/kquant.h"
#include "gguf/types.h"
#include "kquant_golden.h"
#include "model/dequant.h"

using namespace hearth;
using quant::BlockQ4_K;
using quant::BlockQ6_K;
using quant::BlockQ8_K;
using quant::kSuperBlock;

namespace {

// Weight scales: positive, negative, subnormal, zero, negative zero and large.
std::uint16_t scale(std::mt19937& rng, std::size_t i) {
  std::uniform_real_distribution<float> u(1e-4f, 0.05f);
  switch (i % 9) {
    case 0:
      return 0x0003;  // subnormal fp16
    case 1:
      return 0x0000;
    case 2:
      return 0x8000;
    case 3:
      return static_cast<std::uint16_t>(fp32_to_fp16(u(rng)) | 0x8000u);
    case 4:
      return fp32_to_fp16(u(rng) * 1000.0f);
    default:
      return fp32_to_fp16(u(rng));
  }
}

// Q4_K blocks: random bytes, and the extremes (every nibble 15 or 0, every
// 6-bit scale and min 63).
std::vector<BlockQ4_K> make_q4_k(std::int64_t rows, std::int64_t k, std::mt19937& rng) {
  std::vector<BlockQ4_K> w(static_cast<std::size_t>(rows * (k / kSuperBlock)));
  std::uniform_int_distribution<int> byte(0, 255);
  for (std::size_t i = 0; i < w.size(); ++i) {
    const int mode = static_cast<int>(i % 5);
    for (auto& s : w[i].scales) s = static_cast<std::uint8_t>(mode == 1 ? 0xFF : byte(rng));
    for (auto& q : w[i].qs) q = static_cast<std::uint8_t>(mode == 1 ? 0xFF : mode == 2 ? 0x00 : byte(rng));
    w[i].d = scale(rng, i);
    w[i].dmin = scale(rng, i + 4);
  }
  return w;
}

// Q6_K blocks: random bytes, and the extremes (every code 63 or 0, scales
// -128 and 127).
std::vector<BlockQ6_K> make_q6_k(std::int64_t rows, std::int64_t k, std::mt19937& rng) {
  std::vector<BlockQ6_K> w(static_cast<std::size_t>(rows * (k / kSuperBlock)));
  std::uniform_int_distribution<int> byte(0, 255);
  for (std::size_t i = 0; i < w.size(); ++i) {
    const int mode = static_cast<int>(i % 5);
    for (auto& q : w[i].ql) q = static_cast<std::uint8_t>(mode == 1 ? 0xFF : mode == 2 ? 0x00 : byte(rng));
    for (auto& q : w[i].qh) q = static_cast<std::uint8_t>(mode == 1 ? 0xFF : mode == 2 ? 0x00 : byte(rng));
    for (int j = 0; j < 16; ++j) {
      w[i].scales[j] = static_cast<std::int8_t>(mode == 1 ? ((j % 2) != 0 ? -128 : 127) : byte(rng) - 128);
    }
    w[i].d = scale(rng, i);
  }
  return w;
}

// `tokens` rows of k activations, per super-block: normal, all-zero,
// rounding ties, tiny, large, and both signs of the largest magnitude.
std::vector<float> make_x(std::int64_t tokens, std::int64_t k, std::mt19937& rng) {
  std::vector<float> x(static_cast<std::size_t>(tokens * k));
  std::normal_distribution<float> n(0.0f, 1.5f);
  for (std::size_t b = 0; b < x.size() / kSuperBlock; ++b) {
    float* v = x.data() + b * kSuperBlock;
    const int mode = static_cast<int>((b * 7 + b / 5) % 7);
    for (int j = 0; j < kSuperBlock; ++j) {
      switch (mode) {
        case 1:
          v[j] = 0.0f;
          break;
        case 2:
          v[j] = (j == 0 ? -127.0f : static_cast<float>(j % 100) + 0.5f) * ((j % 2) != 0 ? -1.0f : 1.0f);
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
    if (mode == 5) {
      v[3] = 50.0f;
      v[99] = -50.0f;
    }
  }
  return x;
}

std::vector<BlockQ8_K> q8_k_rows(const float* x, std::int64_t k) {
  std::vector<BlockQ8_K> out(static_cast<std::size_t>(k / kSuperBlock));
  quant::quantize_row_q8_k(x, k, out.data());
  return out;
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

// FP64 value and magnitude of one row's terms, for ACCEPTANCE's bound.
template <class Block>
void exact_dot(const Block* row, const BlockQ8_K* x, std::int64_t k, double& exact, double& mag) {
  exact = 0;
  mag = 0;
  float w[kSuperBlock];
  for (std::int64_t b = 0; b < k / kSuperBlock; ++b) {
    if constexpr (std::is_same_v<Block, BlockQ4_K>) {
      // Both parts of each term (scale and min) count towards the magnitude.
      const double d = fp16_to_fp32(row[b].d), dmin = fp16_to_fp32(row[b].dmin);
      for (int j = 0; j < 8; ++j) {
        std::uint8_t sc = 0, m = 0;
        quant::q4_k_scale_min(row[b], j, sc, m);
        double isum = 0, xsum = 0;
        for (int i = 0; i < 32; ++i) {
          isum += quant::q4_k_code(row[b], 32 * j + i) * static_cast<double>(x[b].qs[32 * j + i]);
          xsum += x[b].qs[32 * j + i];
        }
        const double p1 = isum * d * sc * x[b].d, p2 = xsum * dmin * m * x[b].d;
        exact += p1 - p2;
        mag += std::fabs(p1) + std::fabs(p2);
      }
    } else {
      quant::dequant_q6_k(row[b], w);  // exact up to one rounding of (d * sc) * (q - 32)
      const double d = fp16_to_fp32(row[b].d);
      for (int j = 0; j < 16; ++j) {
        double isum = 0;
        for (int i = 0; i < 16; ++i) {
          isum += (quant::q6_k_code(row[b], 16 * j + i) - 32) * static_cast<double>(x[b].qs[16 * j + i]);
        }
        const double t = isum * d * row[b].scales[j] * x[b].d;
        exact += t;
        mag += std::fabs(t);
      }
    }
  }
}

struct Shape {
  std::int64_t rows, k, tokens;
};
// Every row remainder of the 4-row tiles (1..9 rows), odd and even token
// counts, a half-filled last Q4_K group (k = 256, 768: an odd number of
// super-blocks) and the real models' widths (2048, 3072, 8192).
constexpr Shape kShapes[] = {{1, 256, 1},  {2, 512, 2}, {3, 768, 3}, {4, 1024, 1},  {5, 1280, 4},  {6, 2048, 5},
                             {7, 3072, 2}, {8, 256, 3}, {9, 768, 7}, {13, 8192, 1}, {4, 2048, 130}};

template <class Block>
void check_type(cpu::KType type, const std::vector<Block>& w, const Shape& sh, const std::vector<float>& x) {
  const auto rows = sh.rows, k = sh.k, T = sh.tokens;
  const std::int64_t nb = k / kSuperBlock;
  std::vector<float> want(static_cast<std::size_t>(T * rows));
  for (std::int64_t t = 0; t < T; ++t) {
    const std::vector<BlockQ8_K> xq = q8_k_rows(x.data() + t * k, k);
    for (std::int64_t r = 0; r < rows; ++r) {
      const float c = cpu::canonical_dot(w.data() + r * nb, xq.data(), k);
      want[static_cast<std::size_t>(t * rows + r)] = c;
      double exact = 0, mag = 0;
      exact_dot(w.data() + r * nb, xq.data(), k, exact, mag);
      const double err = std::fabs(c - exact);
      if (!(err <= 3e-5 * mag)) FAIL(std::format("row {} token {}: error {} > 3e-5 * {}", r, t, err, mag));
    }
  }
  const cpu::KMatrix km(type, w.data(), rows, k);
  cpu::KActs acts;
  acts.resize(type, T, k);
  acts.quantize(Isa::kScalar, x.data(), 0, T);
  for (const Isa isa : {Isa::kScalar, Isa::kAvx512}) {
    if (!isa_supported(isa)) continue;
    CAPTURE(to_string(isa));
    std::vector<float> got(want.size(), -1.0f);
    cpu::kgemm(isa, km.view(), acts, got.data(), rows, 0, rows);
    const std::string diff = first_difference(got, want, rows);
    CHECK_MESSAGE(diff.empty(), diff);
    // Any split of the rows (as threads split them) gives the same bits.
    std::vector<float> split(want.size(), -1.0f);
    const std::int64_t cut = rows / 3 + 1;
    cpu::kgemm(isa, km.view(), acts, split.data(), rows, 0, cut);
    cpu::kgemm(isa, km.view(), acts, split.data(), rows, cut, rows);
    CHECK(std::memcmp(split.data(), want.data(), want.size() * sizeof(float)) == 0);
  }
}

}  // namespace

TEST_SUITE("quick") {
  TEST_CASE("kquant: Q4_K and Q6_K decode, and Q8_K quantizes, with ggml's bits") {
    // The expected hashes are of ggml's own outputs (dequantize_row_q4_K,
    // dequantize_row_q6_K, quantize_row_q8_K_ref of ggml-quants.c at the
    // pinned llama.cpp commit) on these inputs; tests/kquant_golden.h.
    const std::vector<std::uint8_t> q4 = golden::q4_k_blocks(), q6 = golden::q6_k_blocks();
    std::vector<float> out(golden::kBlocks * kSuperBlock);
    for (int i = 0; i < golden::kBlocks; ++i) {
      BlockQ4_K b;
      std::memcpy(&b, q4.data() + i * sizeof(b), sizeof(b));
      quant::dequant_q4_k(b, out.data() + i * kSuperBlock);
    }
    CHECK(golden::fnv(out.data(), out.size() * 4) == 0x2ef0bf0e59664a6eull);
    for (int i = 0; i < golden::kBlocks; ++i) {
      BlockQ6_K b;
      std::memcpy(&b, q6.data() + i * sizeof(b), sizeof(b));
      quant::dequant_q6_k(b, out.data() + i * kSuperBlock);
    }
    CHECK(golden::fnv(out.data(), out.size() * 4) == 0xb16577de6d3a0490ull);
    const std::vector<float> x = golden::q8_k_inputs();
    std::vector<BlockQ8_K> q8(golden::kRows8);
    quant::quantize_row_q8_k(x.data(), golden::kRows8 * kSuperBlock, q8.data());
    std::uint64_t h = 14695981039346656037ull;
    for (const BlockQ8_K& b : q8) {
      h = golden::fnv(&b.d, 4, h);
      h = golden::fnv(b.qs, sizeof(b.qs), h);
      if (b.d != 0.0f) h = golden::fnv(b.bsums, sizeof(b.bsums), h);
    }
    CHECK(h == 0x568dbed48a8b75fbull);
  }

  TEST_CASE("kquant: the Q8_K rule takes the sign of the first largest value") {
    std::vector<float> x(kSuperBlock, 1.0f);
    x[7] = -8.0f;
    x[9] = 8.0f;
    BlockQ8_K b;
    quant::quantize_row_q8_k(x.data(), kSuperBlock, &b);
    // max = -8: iscale = -127 / -8 = 15.875, so x = -8 is code -127.
    CHECK(b.d == 1.0f / 15.875f);
    CHECK(b.qs[7] == -127);
    CHECK(b.qs[9] == 127);
    CHECK(b.qs[0] == 16);  // 15.875 rounds to 16
    CHECK(b.bsums[0] == 16 * 14 - 127 + 127);
    x[7] = 8.0f;
    x[9] = -8.0f;
    quant::quantize_row_q8_k(x.data(), kSuperBlock, &b);
    // max = +8: iscale = -15.875 and every code changes sign.
    CHECK(b.d == 1.0f / -15.875f);
    CHECK(b.qs[7] == -127);
    CHECK(b.qs[0] == -16);
  }

  TEST_CASE("kquant: dequant_row decodes Q4_K and Q6_K tensors row by row") {
    std::mt19937 rng(5);
    constexpr std::int64_t k = 512, rows = 3;
    const auto q4 = make_q4_k(rows, k, rng);
    const auto q6 = make_q6_k(rows, k, rng);
    for (const bool is_q4 : {true, false}) {
      gguf::TensorInfo t;
      t.name = "w";
      t.n_dims = 2;
      t.ne = {k, rows, 1, 1};
      t.type = is_q4 ? 12u : 14u;
      t.traits = gguf::type_traits(t.type);
      REQUIRE(t.traits != nullptr);
      CHECK(t.traits->supported);
      const auto* bytes =
          is_q4 ? reinterpret_cast<const std::byte*>(q4.data()) : reinterpret_cast<const std::byte*>(q6.data());
      const std::size_t n = is_q4 ? q4.size() * sizeof(BlockQ4_K) : q6.size() * sizeof(BlockQ6_K);
      std::vector<float> got(k), want(k);
      for (std::int64_t r = 0; r < rows; ++r) {
        model::dequant_row(t, std::span<const std::byte>(bytes, n), r, got.data());
        for (std::int64_t b = 0; b < k / kSuperBlock; ++b) {
          if (is_q4) {
            quant::dequant_q4_k(q4[static_cast<std::size_t>(r * (k / kSuperBlock) + b)], want.data() + b * kSuperBlock);
          } else {
            quant::dequant_q6_k(q6[static_cast<std::size_t>(r * (k / kSuperBlock) + b)], want.data() + b * kSuperBlock);
          }
        }
        CHECK(std::memcmp(got.data(), want.data(), k * sizeof(float)) == 0);
      }
    }
  }

  TEST_CASE("kquant: packed activations hold the Q8_K codes, scales and sums, zero-padded") {
    std::mt19937 rng(3);
    for (const cpu::KType type : {cpu::KType::kQ4_K, cpu::KType::kQ6_K}) {
      // Per ISA, one buffer for every shape, largest first: each quantize must
      // overwrite the previous shape's bytes, padding included.
      cpu::KActs a[2];
      for (const std::int64_t k : {1280, 768, 512, 256}) {
        const std::int64_t T = 3, unit = cpu::k_unit_values(type);
        const auto x = make_x(T, k, rng);
        for (int ia = 0; ia < 2; ++ia) {
          const Isa isa = ia == 0 ? Isa::kScalar : Isa::kAvx512;
          if (!isa_supported(isa)) continue;
          CAPTURE(k);
          CAPTURE(to_string(isa));
          a[ia].resize(type, T, k);
          a[ia].quantize(isa, x.data(), 0, T);
          REQUIRE(a[ia].groups() == cpu::k_groups_for(type, k));
          for (std::int64_t t = 0; t < T; ++t) {
            const std::vector<BlockQ8_K> xq = q8_k_rows(x.data() + t * k, k);
            for (std::int64_t u = 0; u < a[ia].groups() * cpu::kKLanes; ++u) {
              const std::int64_t g = u / cpu::kKLanes, j = u % cpu::kKLanes;
              const std::int64_t b = u * unit / kSuperBlock;  // the unit's super-block
              const bool real = u * unit < k;
              std::int32_t sum = 0;
              bool codes_ok = true;
              for (int v = 0; v < unit; ++v) {
                const std::int8_t code = a[ia].q(t, g)[(v / 4) * 64 + j * 4 + v % 4];
                const std::int8_t want = real ? xq[static_cast<std::size_t>(b)].qs[(u * unit) % kSuperBlock + v] : 0;
                codes_ok = codes_ok && code == want;
                sum += want;
              }
              CHECK(codes_ok);
              CHECK(a[ia].s(t, g)[j] == sum);
              CHECK(a[ia].d(t, g)[j] == (real ? xq[static_cast<std::size_t>(b)].d : 0.0f));
            }
          }
        }
        if (isa_supported(Isa::kAvx512)) {
          // The two ISAs write the same bytes.
          const std::int64_t groups = a[0].groups();
          CHECK(std::memcmp(a[0].q(0, 0), a[1].q(0, 0),
                            static_cast<std::size_t>(T * groups * cpu::k_group_values(type))) == 0);
        }
      }
    }
  }

  TEST_CASE("kquant: scalar and AVX-512 kernels give canonical_dot's bits for Q4_K and Q6_K") {
    std::mt19937 rng(9);
    for (const Shape& sh : kShapes) {
      CAPTURE(sh.rows);
      CAPTURE(sh.k);
      CAPTURE(sh.tokens);
      const auto x = make_x(sh.tokens, sh.k, rng);
      check_type(cpu::KType::kQ4_K, make_q4_k(sh.rows, sh.k, rng), sh, x);
      check_type(cpu::KType::kQ6_K, make_q6_k(sh.rows, sh.k, rng), sh, x);
    }
  }

  TEST_CASE("kquant: canonical_dot sums in 16 lanes, then by halving") {
    // Q6_K, k = 512: 32 units of 16, lane = unit % 16. Super-block 0 (units
    // 0-15, d = 512) has one term, t_0 = 2^24 (scale 32, isum 1024); super-
    // block 1 (units 16-31, d = 1) has t = 1 in every unit (scale 1, isum 1).
    // Lane 0: 2^24 + 1 rounds to 2^24; lanes 1-15 hold 1; the halving tree
    // then adds 2, 4, 8 exactly: 2^24 + 14. Summed unit by unit, every + 1
    // after 2^24 rounds away: 2^24. The exact sum is 2^24 + 16.
    BlockQ6_K w[2] = {};
    BlockQ8_K x[2] = {};
    auto set_code = [](BlockQ6_K& b, int v, int code) {
      const int half = v / 128, qq = (v % 128) / 32, l = v % 32;
      std::uint8_t& lo = b.ql[64 * half + l + 32 * (qq & 1)];
      lo = static_cast<std::uint8_t>(qq < 2 ? (lo & 0xF0) | (code & 0x0F) : (lo & 0x0F) | ((code & 0x0F) << 4));
      std::uint8_t& hi = b.qh[32 * half + l];
      hi = static_cast<std::uint8_t>((hi & ~(3 << (2 * qq))) | (((code >> 4) & 3) << (2 * qq)));
    };
    for (int s = 0; s < 2; ++s) {
      for (int v = 0; v < kSuperBlock; ++v) set_code(w[s], v, 32);  // value 0
      x[s].d = 1.0f;
    }
    w[0].d = fp32_to_fp16(512.0f);
    w[0].scales[0] = 32;
    for (int v = 0; v < 16; ++v) {
      set_code(w[0], v, 33);  // value 1
      x[0].qs[v] = 64;        // isum = 16 * 64 = 1024
    }
    w[1].d = fp32_to_fp16(1.0f);
    for (int u = 0; u < 16; ++u) {
      w[1].scales[u] = 1;
      set_code(w[1], 16 * u, 33);
      x[1].qs[16 * u] = 1;
    }
    for (int v = 0; v < kSuperBlock; ++v) CHECK(quant::q6_k_code(w[0], v) == (v < 16 ? 33 : 32));
    const float got = cpu::canonical_dot(w, x, 512);
    CHECK(got == 16777216.0f + 14.0f);
  }
}
