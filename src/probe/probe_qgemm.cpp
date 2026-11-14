// The `qgemm` section of `hearth probe`: how fast the `cpu` backend's decode
// kernel (cpu::qgemm, one token, rows split across 4 threads as in the
// forward pass) reads the weights of each matrix shape of the 1B and 3B
// models. Compared with "cpu-bw", it shows how close the real kernel gets to
// the read ceiling, and whether that depends on the matrix size.
//
// Matrices smaller than kMinSpan are replicated and visited in turn, so every
// repetition reads from DRAM, as decode does when it moves from layer to layer.
#include <algorithm>
#include <cstring>
#include <format>
#include <vector>

#include "core/clock.h"
#include "core/cpu_features.h"
#include "core/quant.h"
#include "core/thread_pool.h"
#include "cpu/qgemm.h"
#include "probe/probe_internal.h"

namespace hearth::probe {
namespace {

constexpr std::size_t kMinSpan = std::size_t{256} << 20;  // 32 x the 8 MiB L3

struct Shape {
  const char* what;
  std::int64_t rows, k;
};

// GGUF blocks of `rows` x `k` with random codes and a small fixed scale.
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

}  // namespace

void probe_qgemm(Report& r, const Options& o) {
  if (!isa_supported(Isa::kAvx512)) {
    r.warnings.push_back("qgemm skipped: needs AVX-512 VNNI");
    return;
  }
  // (what, rows, k): every distinct decode matrix of the 1B and 3B models.
  const std::vector<Shape> shapes = {
      {"1B attn_k/v", 512, 2048},       {"1B attn_q/output", 2048, 2048}, {"1B ffn_gate/up", 8192, 2048},
      {"1B ffn_down", 2048, 8192},      {"1B lm_head", 128256, 2048},     {"3B attn_k/v", 1024, 3072},
      {"3B attn_q/output", 3072, 3072}, {"3B ffn_gate/up", 8192, 3072},   {"3B ffn_down", 3072, 8192},
      {"3B lm_head", 128256, 3072},
  };
  const int reps = o.quick ? 5 : 20;
  ThreadPool pool(4);
  for (const cpu::QType type : {cpu::QType::kQ8_0, cpu::QType::kQ4_0}) {
    const char* tname = type == cpu::QType::kQ8_0 ? "Q8_0" : "Q4_0";
    for (const Shape& sh : shapes) {
      if (o.quick && sh.rows == 128256 && sh.k == 3072) continue;
      std::vector<cpu::PackedMatrix> copies;
      {
        const std::vector<std::byte> blocks = random_blocks(type, sh.rows, sh.k, 0x9E3779B97F4A7C15ull + sh.rows);
        copies.emplace_back(type, blocks.data(), sh.rows, sh.k);
        const std::size_t n = std::max<std::size_t>(1, (kMinSpan + copies[0].bytes() - 1) / copies[0].bytes());
        while (copies.size() < n) copies.emplace_back(type, blocks.data(), sh.rows, sh.k);
      }
      std::vector<float> x(static_cast<std::size_t>(sh.k));
      for (std::size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>(static_cast<int>(i % 23) - 11) * 0.1f;
      cpu::PackedActs acts;
      acts.resize(1, sh.k);
      acts.quantize(Isa::kAvx512, x.data(), 0, 1);
      std::vector<float> y(static_cast<std::size_t>(sh.rows));
      auto once = [&](const cpu::PackedMatrix& m) {
        const cpu::PackedView v = m.view();
        pool.run([&](int i, int n) {
          std::int64_t r0 = 0, r1 = 0;
          cpu::tile_range(v.rows, n, i, r0, r1);
          cpu::qgemm(Isa::kAvx512, v, acts, y.data(), v.rows, r0, r1);
        });
      };
      for (const auto& m : copies) once(m);  // warm-up: page in, wake the pool
      Measurement& meas = r.add("qgemm", std::format("decode {} {}, {} x {}", tname, sh.what, sh.rows, sh.k), "GB/s");
      meas.params = {
          {"type", tname},  {"rows", std::to_string(sh.rows)},         {"k", std::to_string(sh.k)},
          {"threads", "4"}, {"copies", std::to_string(copies.size())}, {"bytes", std::to_string(copies[0].bytes())}};
      std::size_t next = 0;
      for (int rep = 0; rep < reps; ++rep) {
        const cpu::PackedMatrix& m = copies[next];
        next = (next + 1) % copies.size();
        const std::int64_t t0 = now_ns();
        once(m);
        const double s = static_cast<double>(now_ns() - t0) / 1e9;
        meas.samples.push_back(static_cast<double>(m.bytes()) / s / 1e9);
      }
    }
  }
}

}  // namespace hearth::probe
