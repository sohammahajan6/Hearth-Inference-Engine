// Matrix products for the K-quant weight formats Q4_K and Q6_K, in the
// canonical order of ADR 0021, on weights and Q8_K activations repacked into
// the 16-lane interleave of docs/architecture/14-k-quants.md.
//
// The order's unit is a sub-block u: 32 values for Q4_K, 16 for Q6_K; b(u)
// is its super-block of 256 values and dx_b that super-block's Q8_K
// activation scale. For one weight row and one token:
//   Q4_K: t_u = float(isum_u) * (ws_u * dx_b) - float(xsum_u) * (wm_u * dx_b)
//           isum_u = sum q * x (q = 0..15), xsum_u = sum x,
//           ws_u = float(d_b) * sc_u, wm_u = float(dmin_b) * m_u
//   Q6_K: t_u = float(isum_u) * (ws_u * dx_b)
//           isum_u = sum (q - 32) * x, ws_u = float(d_b) * sc_u
//   lane j = ((+0 + t_j) + t_{j+16}) + t_{j+32} + ...   16 lanes, increasing u
//   y      = halving tree: j + j+8, then + 4, + 2, + 1 
// The integer sums are exact int32, and ws_u, wm_u are exact in FP32 (at most
// 11 + 8 significant bits), so any instructions may compute those. Every
// multiply and subtract after them is one rounded FP32 operation (no FMA).
// So the scalar kernel, every AVX-512 tile, every thread count and every
// batch split give the same bits, and prefill equals decode.
//
// Packed layout, per row and per group of 16 consecutive units:
//   Q4_K (512 values = 2 super-blocks): quants as Q4_0's 16-block interleave
//         (cpu/qgemm.h): 4 chunks of 64 bytes; byte 4j + i of chunk m holds,
//         for unit j, value 4m + i in its low nibble and value 16 + 4m + i
//         in its high nibble. Scales: 16 sc_u and 16 m_u bytes; the fp16 d
//         and dmin of both super-blocks.
//   Q6_K (256 values = 1 super-block): 2 chunks of 64 bytes of low 4 bits
//         (byte 4j + i of chunk m: value 4m + i of unit j low, value 8 + 4m + i
//         high), then 64 bytes of high 2 bits (byte 4j + i: bits 2c, 2c+1 for
//         value 4c + i, c = 0..3). Scales: 16 int8 sc_u; the fp16 d.
// Activations: chunk c of a group holds codes 4c .. 4c+3 of each unit (unit j
// at byte 4j), with each unit's code sum and, per lane, the Q8_K scale of the
// unit's super-block. A last, partial Q4_K group (an odd number of
// super-blocks) is padded with zero units, which leaves the sum unchanged.
#pragma once

#include <cstddef>
#include <cstdint>

#include "core/cpu_features.h"
#include "core/kquant.h"
#include "core/page_buffer.h"

namespace hearth::cpu {

enum class KType : std::uint8_t { kQ4_K, kQ6_K };

inline constexpr int kKLanes = 16;
inline constexpr int k_unit_values(KType t) noexcept { return t == KType::kQ4_K ? 32 : 16; }
inline constexpr int k_group_values(KType t) noexcept { return kKLanes * k_unit_values(t); }  // 512 or 256
inline constexpr int k_group_supers(KType t) noexcept { return k_group_values(t) / quant::kSuperBlock; }
inline constexpr std::int64_t k_group_quant_bytes(KType t) noexcept { return t == KType::kQ4_K ? 256 : 192; }
inline constexpr std::int64_t k_groups_for(KType t, std::int64_t k) noexcept {
  return (k + k_group_values(t) - 1) / k_group_values(t);
}

// What the kernels read. `q` is 64-byte aligned.
struct KPackedView {
  KType type = KType::kQ4_K;
  std::int64_t rows = 0, k = 0, groups = 0;
  const std::uint8_t* q = nullptr;      // [rows][groups][k_group_quant_bytes]
  const std::uint8_t* sc = nullptr;     // [rows][groups][16]: Q4_K 6-bit scales, Q6_K int8 scales
  const std::uint8_t* mn = nullptr;     // Q4_K: [rows][groups][16] 6-bit mins; Q6_K: nullptr
  const std::uint16_t* d = nullptr;     // [rows][groups][2] fp16: d of the group's super-blocks (Q6_K: [1] = 0)
  const std::uint16_t* dmin = nullptr;  // Q4_K: [rows][groups][2] fp16; Q6_K: nullptr
};

// Bytes of a rows x k matrix in the packed layout. k is a multiple of 256.
std::size_t k_packed_bytes(KType type, std::int64_t rows, std::int64_t k);

// Repacks rows * k / 256 GGUF blocks (BlockQ4_K or BlockQ6_K) into `dst`,
// k_packed_bytes(type, rows, k) bytes that must be zero on entry and 64-byte
// aligned, and returns the view of it.
KPackedView pack_k_matrix(KType type, const void* blocks, std::int64_t rows, std::int64_t k, std::byte* dst);

// A weight matrix repacked from GGUF's blocks (one allocation, page aligned).
class KMatrix {
 public:
  KMatrix() = default;
  KMatrix(KType type, const void* blocks, std::int64_t rows, std::int64_t k);
  KPackedView view() const noexcept { return view_; }
  std::size_t bytes() const noexcept { return buf_.size(); }

 private:
  PageBuffer buf_;
  KPackedView view_;
};

// Q8_K activations of `tokens` rows of `k` values in the layout of a weight
// type. The storage grows as needed and is reused across calls.
class KActs {
 public:
  void resize(KType type, std::int64_t tokens, std::int64_t k);
  // Quantizes rows t0 .. t1 - 1 of x (row t at x + t * k) exactly as ggml's
  // quantize_row_q8_K_ref does (quant::quantize_row_q8_k), into this layout.
  // kScalar and kAvx512 give the same bytes.
  void quantize(Isa isa, const float* x, std::int64_t t0, std::int64_t t1);

  KType type() const noexcept { return type_; }
  std::int64_t tokens() const noexcept { return tokens_; }
  std::int64_t k() const noexcept { return k_; }
  std::int64_t groups() const noexcept { return groups_; }
  // Token t, group g: the interleaved codes, 16 lane scales, 16 unit code sums.
  const std::int8_t* q(std::int64_t t, std::int64_t g) const noexcept {
    return reinterpret_cast<const std::int8_t*>(buf_.data()) + (t * groups_ + g) * k_group_values(type_);
  }
  const float* d(std::int64_t t, std::int64_t g) const noexcept { return d_ + (t * groups_ + g) * kKLanes; }
  const std::int32_t* s(std::int64_t t, std::int64_t g) const noexcept { return s_ + (t * groups_ + g) * kKLanes; }

 private:
  PageBuffer buf_;
  float* d_ = nullptr;
  std::int32_t* s_ = nullptr;
  KType type_ = KType::kQ4_K;
  std::int64_t tokens_ = 0, k_ = 0, groups_ = 0;
};

// y[t * ldy + r] for rows r in [r0, r1) and every token t of x. kScalar works
// on any x86-64; kAvx512 needs AVX-512 VNNI (callers check isa_supported()).
// Both give the canonical bits.
void kgemm(Isa isa, const KPackedView& w, const KActs& x, float* y, std::int64_t ldy, std::int64_t r0, std::int64_t r1);

// The canonical order computed directly from GGUF blocks and ggml's Q8_K
// blocks (k / 256 of each): the executable form of ADR 0021, written for
// clarity, that every kernel is tested against bit for bit.
float canonical_dot(const quant::BlockQ4_K* row, const quant::BlockQ8_K* x, std::int64_t k);
float canonical_dot(const quant::BlockQ6_K* row, const quant::BlockQ8_K* x, std::int64_t k);

namespace detail {
// The 256 Q8_K codes and the scale of one super-block, as quantize_row_q8_k.
void quantize_super_avx512(const float* x, std::int8_t* codes, float& d);
// `scales`: scratch of k_tile_scale_floats(w) floats for prefill (x.tokens() > 1),
// allocated by the caller: the AVX-512 file must not instantiate library code
//, so it holds no std::vector of its own.
void kgemm_avx512(const KPackedView& w, const KActs& x, float* y, std::int64_t ldy, std::int64_t r0, std::int64_t r1,
                  float* scales);
inline std::int64_t k_tile_scale_floats(const KPackedView& w) noexcept { return 4 * w.groups * 32; }
}  // namespace detail

}  // namespace hearth::cpu
