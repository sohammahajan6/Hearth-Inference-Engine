// Quantized matrix products in the canonical order of ADR 0009, on weights
// and activations repacked into the 16-block interleave of
// docs/architecture/07-fast-cpu-backend.md.
//
// For one weight row and one token, over the blocks b of the row:
//   isum_b = exact int32 block sum
//   t_b    = float(isum_b) * (float(d_w,b) * d_x,b)       two rounded multiplies, no FMA
//   lane j = ((+0 + t_j) + t_{j+16}) + t_{j+32} + ...      16 lanes, increasing b
//   y      = halving tree: j + j+8, then + 4, + 2, + 1
// Every kernel (scalar, AVX-512 GEMV and GEMM tiles of any shape) produces
// exactly these bits, so the thread count, the batch size and the ISA never
// change a result, and the GPU kernels of Phase 6 can reproduce them.
//
// Packed layout, per row and per group of 16 consecutive blocks (512 values):
//   Q8_0 quants: 8 chunks of 64 bytes; chunk c holds bytes 4c..4c+3 of each of
//                the 16 blocks (block j at offset 4j), stored as q + 128.
//   Q4_0 quants: 4 chunks of 64 bytes; chunk m holds nibble bytes 4m..4m+3 of
//                each block (low nibbles: values 4m.., high: values 16+4m..).
//   scales:      the 16 fp16 scales of the group, in a separate array.
// Activations use the Q8_0 interleave (codes not offset), with the 16 decoded
// scales and 16 code sums of each group. A last, partial group is padded with
// zero blocks (scale 0, value 0), which leaves the canonical sum unchanged.
#pragma once

#include <cstdint>

#include "core/cpu_features.h"
#include "core/page_buffer.h"
#include "core/quant.h"

namespace hearth::cpu {

inline constexpr int kGroupBlocks = 16;                            // blocks per group = lanes
inline constexpr int kGroupValues = kGroupBlocks * quant::kBlock;  // 512
inline constexpr int kChunkBytes = 64;

enum class QType : std::uint8_t { kQ8_0, kQ4_0 };

inline constexpr std::int64_t group_quant_bytes(QType t) noexcept { return t == QType::kQ8_0 ? 512 : 256; }
inline constexpr std::int64_t groups_for(std::int64_t k) noexcept { return (k + kGroupValues - 1) / kGroupValues; }

// What the kernels read. `q` is 64-byte aligned.
struct PackedView {
  QType type = QType::kQ8_0;
  std::int64_t rows = 0, k = 0, groups = 0;
  const std::uint8_t* q = nullptr;   // [rows][groups][group_quant_bytes]
  const std::uint16_t* d = nullptr;  // [rows][groups][16] fp16 bits
};

// Bytes of a rows x k matrix in the packed layout: the quants, then the
// scales (at offset packed_quant_bytes). k is a multiple of 32.
std::size_t packed_quant_bytes(QType type, std::int64_t rows, std::int64_t k);
std::size_t packed_bytes(QType type, std::int64_t rows, std::int64_t k);

// Repacks rows * k / 32 GGUF blocks (BlockQ8_0 or BlockQ4_0) into `dst`,
// packed_bytes(type, rows, k) bytes that must be zero on entry (the padding
// scales of a partial group stay 0), and returns the view of it. `dst` must be
// 64-byte aligned.
PackedView pack_matrix(QType type, const void* blocks, std::int64_t rows, std::int64_t k, std::byte* dst);

// A weight matrix repacked from GGUF's blocks (one allocation, page aligned).
class PackedMatrix {
 public:
  PackedMatrix() = default;
  // `blocks` holds rows * k / 32 GGUF blocks (BlockQ8_0 or BlockQ4_0); k is a multiple of 32.
  PackedMatrix(QType type, const void* blocks, std::int64_t rows, std::int64_t k);
  PackedView view() const noexcept { return view_; }
  std::size_t bytes() const noexcept { return buf_.size(); }

 private:
  PageBuffer buf_;
  PackedView view_;
};

// Quantized activations of `tokens` rows of `k` values, in the packed layout.
// The storage grows as needed and is reused across calls.
class PackedActs {
 public:
  void resize(std::int64_t tokens, std::int64_t k);
  // Quantizes rows t0 .. t1 - 1 of x (row t at x + t * k) exactly as ggml's
  // quantize_row_q8_0 does (quant::quantize_act_q8_0), into this layout.
  // kScalar and kAvx512 give the same bytes.
  void quantize(Isa isa, const float* x, std::int64_t t0, std::int64_t t1);

  std::int64_t tokens() const noexcept { return tokens_; }
  std::int64_t k() const noexcept { return k_; }
  std::int64_t groups() const noexcept { return groups_; }
  // Token t, group g: 512 interleaved codes, 16 scales, 16 code sums.
  const std::int8_t* q(std::int64_t t, std::int64_t g) const noexcept {
    return reinterpret_cast<const std::int8_t*>(buf_.data()) + (t * groups_ + g) * kGroupValues;
  }
  const float* d(std::int64_t t, std::int64_t g) const noexcept { return d_ + (t * groups_ + g) * kGroupBlocks; }
  const std::int32_t* s(std::int64_t t, std::int64_t g) const noexcept { return s_ + (t * groups_ + g) * kGroupBlocks; }

 private:
  PageBuffer buf_;
  float* d_ = nullptr;
  std::int32_t* s_ = nullptr;
  std::int64_t tokens_ = 0, k_ = 0, groups_ = 0;
};

// y[t * ldy + r] for rows r in [r0, r1) and every token t of x.
// kScalar works on any x86-64; kAvx512 needs AVX-512 VNNI (callers check
// isa_supported()). Both give the canonical bits.
void qgemm(Isa isa, const PackedView& w, const PackedActs& x, float* y, std::int64_t ldy, std::int64_t r0,
           std::int64_t r1);

// Rows [r0, r1) of part `index` of `parts`, split in whole tiles of the
// kernel's 4 rows (the last range ends at `rows`). Parts beyond the number of
// tiles get an empty range.
void tile_range(std::int64_t rows, int parts, int index, std::int64_t& r0, std::int64_t& r1) noexcept;

// The canonical order computed directly from GGUF blocks and unpacked
// activations (quant::ActQ8): the executable form of ADR 0009, written for
// clarity, that every kernel is tested against bit for bit.
float canonical_dot(const quant::BlockQ8_0* row, const quant::ActQ8& x);
float canonical_dot(const quant::BlockQ4_0* row, const quant::ActQ8& x);

namespace detail {
// One group of the packed activations: `blocks` (1..16) real blocks of 32
// values at x; the rest of the group is written as zero padding.
void quantize_group_avx512(const float* x, int blocks, std::int8_t* q, float* d, std::int32_t* s);
void qgemm_avx512(const PackedView& w, const PackedActs& x, float* y, std::int64_t ldy, std::int64_t r0,
                  std::int64_t r1);
}  // namespace detail

}  // namespace hearth::cpu
