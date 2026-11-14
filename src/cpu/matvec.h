// Decode matrix-vector products y = W x on quantized weights (ADR 0004
// arithmetic): activations quantized to Q8_0, exact int32 block sums,
// block results scaled by d_w * d_x and accumulated in FP32.
//
// Two weight layouts:
//   AoS - GGUF's own blocks, {fp16 d; quants} back to back along each row.
//   SoA - per tensor, one array of fp16 scales and one array of quants
//         (scale of row r, block b at [r * nb + b]; its quants at [(r * nb + b) * bytes_per_block]).
#pragma once

#include <cstdint>

#include "core/cpu_features.h"
#include "core/quant.h"

namespace hearth::cpu {

struct Q8_0Aos {
  const quant::BlockQ8_0* blocks;
  std::int64_t rows, k;
};
struct Q4_0Aos {
  const quant::BlockQ4_0* blocks;
  std::int64_t rows, k;
};
struct Q8_0Soa {
  const std::uint16_t* d;
  const std::int8_t* q;  // 32 bytes per block
  std::int64_t rows, k;
};
struct Q4_0Soa {
  const std::uint16_t* d;
  const std::uint8_t* q;  // 16 bytes per block, same nibble order as Q4_0
  std::int64_t rows, k;
};

// y[r] for r in [r0, r1). `isa` selects the kernel family; kScalar works on
// any x86-64, kAvx512 needs AVX-512 VNNI (callers check isa_supported()).
void matvec(Isa isa, const Q8_0Aos& w, const quant::ActQ8& x, float* y, std::int64_t r0, std::int64_t r1);
void matvec(Isa isa, const Q4_0Aos& w, const quant::ActQ8& x, float* y, std::int64_t r0, std::int64_t r1);
void matvec(Isa isa, const Q8_0Soa& w, const quant::ActQ8& x, float* y, std::int64_t r0, std::int64_t r1);
void matvec(Isa isa, const Q4_0Soa& w, const quant::ActQ8& x, float* y, std::int64_t r0, std::int64_t r1);

// FP64 reference from the exact integer block sums, plus the bound
// sum_b |d_w d_x isum_b| that docs/ACCEPTANCE.md scales the tolerance by.
void matvec_reference(const Q8_0Aos& w, const quant::ActQ8& x, double* y, double* magnitude);
void matvec_reference(const Q4_0Aos& w, const quant::ActQ8& x, double* y, double* magnitude);

// Repacks AoS weights into the SoA arrays (d: rows * nb, q: rows * nb * bytes_per_block).
void repack(const Q8_0Aos& w, std::uint16_t* d, std::int8_t* q);
void repack(const Q4_0Aos& w, std::uint16_t* d, std::uint8_t* q);

namespace detail {
void matvec_avx512(const Q8_0Aos& w, const quant::ActQ8& x, float* y, std::int64_t r0, std::int64_t r1);
void matvec_avx512(const Q4_0Aos& w, const quant::ActQ8& x, float* y, std::int64_t r0, std::int64_t r1);
void matvec_avx512(const Q8_0Soa& w, const quant::ActQ8& x, float* y, std::int64_t r0, std::int64_t r1);
void matvec_avx512(const Q4_0Soa& w, const quant::ActQ8& x, float* y, std::int64_t r0, std::int64_t r1);
}  // namespace detail

}  // namespace hearth::cpu
