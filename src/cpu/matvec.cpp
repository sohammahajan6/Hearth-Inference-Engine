// Scalar matvec kernels (any x86-64), the FP64 reference, repacking and
// dispatch. Compiled for baseline x86-64.
#include "cpu/matvec.h"

#include <cmath>
#include <cstring>

#include "core/error.h"
#include "core/fp16.h"

namespace hearth::cpu {
namespace {

using quant::kBlock;

// Scalar kernels: exact int32 block sum, scaled, accumulated in FP32 in block
// order. (The AVX-512 kernels accumulate per vector lane, so results agree to
// FP32 rounding, not bit for bit.)
template <class BlockDot, class Scale>
float row_scalar(std::int64_t nb, BlockDot dot, Scale scale) {
  float acc = 0.0f;
  for (std::int64_t b = 0; b < nb; ++b) acc += scale(b) * static_cast<float>(dot(b));
  return acc;
}

std::int32_t nibble_dot(const std::uint8_t* q, const std::int8_t* x) {
  std::int32_t s = 0;
  for (int j = 0; j < kBlock / 2; ++j) {
    s += (static_cast<std::int32_t>(q[j] & 0x0F) - 8) * x[j];
    s += (static_cast<std::int32_t>(q[j] >> 4) - 8) * x[j + kBlock / 2];
  }
  return s;
}

std::int32_t byte_dot(const std::int8_t* q, const std::int8_t* x) {
  std::int32_t s = 0;
  for (int j = 0; j < kBlock; ++j) s += static_cast<std::int32_t>(q[j]) * x[j];
  return s;
}

void check_shape(std::int64_t k, const quant::ActQ8& x) { HEARTH_CHECK(k % kBlock == 0 && x.k == k); }

}  // namespace

void matvec(Isa isa, const Q8_0Aos& w, const quant::ActQ8& x, float* y, std::int64_t r0, std::int64_t r1) {
  check_shape(w.k, x);
  if (isa == Isa::kAvx512) return detail::matvec_avx512(w, x, y, r0, r1);
  const std::int64_t nb = w.k / kBlock;
  for (std::int64_t r = r0; r < r1; ++r) {
    const quant::BlockQ8_0* row = w.blocks + r * nb;
    y[r] = row_scalar(
        nb, [&](std::int64_t b) { return byte_dot(row[b].q, x.q + b * kBlock); },
        [&](std::int64_t b) { return fp16_to_fp32(row[b].d) * x.d[b]; });
  }
}

void matvec(Isa isa, const Q4_0Aos& w, const quant::ActQ8& x, float* y, std::int64_t r0, std::int64_t r1) {
  check_shape(w.k, x);
  if (isa == Isa::kAvx512) return detail::matvec_avx512(w, x, y, r0, r1);
  const std::int64_t nb = w.k / kBlock;
  for (std::int64_t r = r0; r < r1; ++r) {
    const quant::BlockQ4_0* row = w.blocks + r * nb;
    y[r] = row_scalar(
        nb, [&](std::int64_t b) { return nibble_dot(row[b].q, x.q + b * kBlock); },
        [&](std::int64_t b) { return fp16_to_fp32(row[b].d) * x.d[b]; });
  }
}

void matvec(Isa isa, const Q8_0Soa& w, const quant::ActQ8& x, float* y, std::int64_t r0, std::int64_t r1) {
  check_shape(w.k, x);
  if (isa == Isa::kAvx512) return detail::matvec_avx512(w, x, y, r0, r1);
  const std::int64_t nb = w.k / kBlock;
  for (std::int64_t r = r0; r < r1; ++r) {
    y[r] = row_scalar(
        nb, [&](std::int64_t b) { return byte_dot(w.q + (r * nb + b) * kBlock, x.q + b * kBlock); },
        [&](std::int64_t b) { return fp16_to_fp32(w.d[r * nb + b]) * x.d[b]; });
  }
}

void matvec(Isa isa, const Q4_0Soa& w, const quant::ActQ8& x, float* y, std::int64_t r0, std::int64_t r1) {
  check_shape(w.k, x);
  if (isa == Isa::kAvx512) return detail::matvec_avx512(w, x, y, r0, r1);
  const std::int64_t nb = w.k / kBlock;
  for (std::int64_t r = r0; r < r1; ++r) {
    y[r] = row_scalar(
        nb, [&](std::int64_t b) { return nibble_dot(w.q + (r * nb + b) * (kBlock / 2), x.q + b * kBlock); },
        [&](std::int64_t b) { return fp16_to_fp32(w.d[r * nb + b]) * x.d[b]; });
  }
}

void matvec_reference(const Q8_0Aos& w, const quant::ActQ8& x, double* y, double* magnitude) {
  const std::int64_t nb = w.k / kBlock;
  for (std::int64_t r = 0; r < w.rows; ++r) {
    double s = 0, m = 0;
    for (std::int64_t b = 0; b < nb; ++b) {
      const quant::BlockQ8_0& blk = w.blocks[r * nb + b];
      const double t = static_cast<double>(fp16_to_fp32(blk.d)) * x.d[b] * quant::block_dot_q8_0(blk, x.q + b * kBlock);
      s += t;
      m += std::fabs(t);
    }
    y[r] = s;
    magnitude[r] = m;
  }
}

void matvec_reference(const Q4_0Aos& w, const quant::ActQ8& x, double* y, double* magnitude) {
  const std::int64_t nb = w.k / kBlock;
  for (std::int64_t r = 0; r < w.rows; ++r) {
    double s = 0, m = 0;
    for (std::int64_t b = 0; b < nb; ++b) {
      const quant::BlockQ4_0& blk = w.blocks[r * nb + b];
      const double t = static_cast<double>(fp16_to_fp32(blk.d)) * x.d[b] * quant::block_dot_q4_0(blk, x.q + b * kBlock);
      s += t;
      m += std::fabs(t);
    }
    y[r] = s;
    magnitude[r] = m;
  }
}

void repack(const Q8_0Aos& w, std::uint16_t* d, std::int8_t* q) {
  const std::int64_t n = w.rows * (w.k / kBlock);
  for (std::int64_t i = 0; i < n; ++i) {
    d[i] = w.blocks[i].d;
    std::memcpy(q + i * kBlock, w.blocks[i].q, kBlock);
  }
}

void repack(const Q4_0Aos& w, std::uint16_t* d, std::uint8_t* q) {
  const std::int64_t n = w.rows * (w.k / kBlock);
  for (std::int64_t i = 0; i < n; ++i) {
    d[i] = w.blocks[i].d;
    std::memcpy(q + i * (kBlock / 2), w.blocks[i].q, kBlock / 2);
  }
}

}  // namespace hearth::cpu
