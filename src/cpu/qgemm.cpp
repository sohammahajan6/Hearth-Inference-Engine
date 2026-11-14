// Repacking, activation quantization, the canonical order written out
// (canonical_dot), the scalar kernel and dispatch. Compiled for baseline
// x86-64: no FMA instruction exists here, and /fp:precise (MSVC) and
// -ffp-contract=off (clang) forbid contracting a * b + c anyway.
#include "cpu/qgemm.h"

#include <algorithm>
#include <cstring>

#include "core/error.h"
#include "core/fp16.h"
#include "core/thread_pool.h"

namespace hearth::cpu {
namespace {

using quant::kBlock;

// Q8_0 quants are stored offset by 128, so VPDPBUSD can treat them as unsigned.
constexpr std::uint8_t kQ8Zero = 128;

std::size_t checked_size(std::int64_t a, std::int64_t b) {
  HEARTH_CHECK(a >= 0 && b >= 0 && (a == 0 || b <= INT64_MAX / a));
  return static_cast<std::size_t>(a * b);
}

// The 16 lanes and the halving tree of ADR 0009, shared by canonical_dot and
// the scalar kernel.
struct Lanes {
  float v[kGroupBlocks] = {};  // +0.0f
  void add(int j, float t) noexcept { v[j] = v[j] + t; }
  float reduce() noexcept {
    for (int w = kGroupBlocks / 2; w >= 1; w /= 2) {
      for (int j = 0; j < w; ++j) v[j] = v[j] + v[j + w];
    }
    return v[0];
  }
};

float block_term(std::int32_t isum, float dw, float dx) noexcept {
  const float s = dw * dx;
  return static_cast<float>(isum) * s;
}

template <class Block, class Dot>
float canonical(const Block* row, const quant::ActQ8& x, Dot dot) {
  HEARTH_CHECK(x.k % kBlock == 0);
  Lanes lanes;
  for (std::int64_t b = 0; b < x.k / kBlock; ++b) {
    lanes.add(static_cast<int>(b % kGroupBlocks),
              block_term(dot(row[b], x.q + b * kBlock), fp16_to_fp32(row[b].d), x.d[b]));
  }
  return lanes.reduce();
}

// One row and one token in the scalar kernel: the same arithmetic as
// canonical_dot, reading the packed bytes.
float row_token_scalar(const PackedView& w, std::int64_t r, const PackedActs& x, std::int64_t t) {
  const std::int64_t gq = group_quant_bytes(w.type);
  Lanes lanes;
  for (std::int64_t g = 0; g < w.groups; ++g) {
    const std::uint8_t* wq = w.q + (r * w.groups + g) * gq;
    const std::uint16_t* wd = w.d + (r * w.groups + g) * kGroupBlocks;
    const std::int8_t* xq = x.q(t, g);
    const float* xd = x.d(t, g);
    for (int j = 0; j < kGroupBlocks; ++j) {
      std::int32_t isum = 0;
      if (w.type == QType::kQ8_0) {
        for (int c = 0; c < 8; ++c) {
          for (int i = 0; i < 4; ++i) {
            const int at = c * kChunkBytes + j * 4 + i;
            isum += (static_cast<std::int32_t>(wq[at]) - kQ8Zero) * xq[at];
          }
        }
      } else {
        for (int m = 0; m < 4; ++m) {
          for (int i = 0; i < 4; ++i) {
            const std::uint8_t byte = wq[m * kChunkBytes + j * 4 + i];
            isum += (static_cast<std::int32_t>(byte & 0x0F) - 8) * xq[m * kChunkBytes + j * 4 + i];
            isum += (static_cast<std::int32_t>(byte >> 4) - 8) * xq[(m + 4) * kChunkBytes + j * 4 + i];
          }
        }
      }
      lanes.add(j, block_term(isum, fp16_to_fp32(wd[j]), xd[j]));
    }
  }
  return lanes.reduce();
}

}  // namespace

std::size_t packed_quant_bytes(QType type, std::int64_t rows, std::int64_t k) {
  if (rows <= 0 || k <= 0 || k % kBlock != 0) {
    fail(ErrorKind::kInvalidArgument, "packed matrix: {} x {} is not a quantized shape", rows, k);
  }
  return checked_size(static_cast<std::int64_t>(checked_size(rows, groups_for(k))), group_quant_bytes(type));
}

std::size_t packed_bytes(QType type, std::int64_t rows, std::int64_t k) {
  const std::size_t d_bytes =
      checked_size(static_cast<std::int64_t>(checked_size(rows, groups_for(k))), kGroupBlocks * 2);
  return packed_quant_bytes(type, rows, k) + d_bytes;
}

PackedMatrix::PackedMatrix(QType type, const void* blocks, std::int64_t rows, std::int64_t k) {
  buf_ = PageBuffer(packed_bytes(type, rows, k));  // zero-filled by the OS: padding blocks have scale 0
  view_ = pack_matrix(type, blocks, rows, k, buf_.data());
}

PackedView pack_matrix(QType type, const void* blocks, std::int64_t rows, std::int64_t k, std::byte* dst_bytes) {
  const std::size_t q_bytes = packed_quant_bytes(type, rows, k);
  HEARTH_CHECK(reinterpret_cast<std::uintptr_t>(dst_bytes) % 64 == 0);
  const std::int64_t nb = k / kBlock;
  const std::int64_t groups = groups_for(k);
  const std::int64_t gq = group_quant_bytes(type);
  auto* q = reinterpret_cast<std::uint8_t*>(dst_bytes);
  auto* d = reinterpret_cast<std::uint16_t*>(dst_bytes + q_bytes);
  if (type == QType::kQ8_0) {
    std::memset(q, kQ8Zero, q_bytes);  // padding quants: value 0
    const auto* src = static_cast<const quant::BlockQ8_0*>(blocks);
    for (std::int64_t r = 0; r < rows; ++r) {
      for (std::int64_t b = 0; b < nb; ++b) {
        const quant::BlockQ8_0& blk = src[r * nb + b];
        const std::int64_t g = b / kGroupBlocks, j = b % kGroupBlocks;
        std::uint8_t* dst = q + (r * groups + g) * gq;
        for (int v = 0; v < kBlock; ++v) {
          dst[(v / 4) * kChunkBytes + j * 4 + v % 4] = static_cast<std::uint8_t>(blk.q[v] + kQ8Zero);
        }
        d[(r * groups + g) * kGroupBlocks + j] = blk.d;
      }
    }
  } else {
    std::memset(q, 0x88, q_bytes);  // padding nibbles: 8, value 0
    const auto* src = static_cast<const quant::BlockQ4_0*>(blocks);
    for (std::int64_t r = 0; r < rows; ++r) {
      for (std::int64_t b = 0; b < nb; ++b) {
        const quant::BlockQ4_0& blk = src[r * nb + b];
        const std::int64_t g = b / kGroupBlocks, j = b % kGroupBlocks;
        std::uint8_t* dst = q + (r * groups + g) * gq;
        for (int v = 0; v < kBlock / 2; ++v) dst[(v / 4) * kChunkBytes + j * 4 + v % 4] = blk.q[v];
        d[(r * groups + g) * kGroupBlocks + j] = blk.d;
      }
    }
  }
  return PackedView{type, rows, k, groups, q, d};
}

void PackedActs::resize(std::int64_t tokens, std::int64_t k) {
  HEARTH_CHECK(tokens > 0 && k > 0 && k % kBlock == 0);
  const std::int64_t groups = groups_for(k);
  const std::size_t n = checked_size(tokens, groups);
  const std::size_t q_bytes = checked_size(static_cast<std::int64_t>(n), kGroupValues);
  const std::size_t ds_bytes = checked_size(static_cast<std::int64_t>(n), kGroupBlocks * 4);
  const std::size_t need = q_bytes + 2 * ds_bytes;
  if (buf_.size() < need) buf_ = PageBuffer(need);
  d_ = reinterpret_cast<float*>(buf_.data() + q_bytes);
  s_ = reinterpret_cast<std::int32_t*>(buf_.data() + q_bytes + ds_bytes);
  tokens_ = tokens;
  k_ = k;
  groups_ = groups;
}

void PackedActs::quantize(Isa isa, const float* x, std::int64_t t0, std::int64_t t1) {
  HEARTH_CHECK(0 <= t0 && t0 <= t1 && t1 <= tokens_ && (isa == Isa::kScalar || isa == Isa::kAvx512));
  const std::int64_t nb = k_ / kBlock;
  for (std::int64_t t = t0; t < t1; ++t) {
    for (std::int64_t g = 0; g < groups_; ++g) {
      auto* q = reinterpret_cast<std::int8_t*>(buf_.data()) + (t * groups_ + g) * kGroupValues;
      float* d = d_ + (t * groups_ + g) * kGroupBlocks;
      std::int32_t* s = s_ + (t * groups_ + g) * kGroupBlocks;
      const float* xg = x + t * k_ + g * kGroupValues;
      const auto blocks = static_cast<int>(std::min<std::int64_t>(kGroupBlocks, nb - g * kGroupBlocks));
      if (isa == Isa::kAvx512) {
        detail::quantize_group_avx512(xg, blocks, q, d, s);
        continue;
      }
      for (int j = 0; j < kGroupBlocks; ++j) {
        std::int8_t codes[kBlock] = {};
        if (j < blocks) {
          quant::quantize_act_q8_0(xg + j * kBlock, kBlock, codes, &d[j], &s[j]);
        } else {
          d[j] = 0.0f;
          s[j] = 0;
        }
        for (int v = 0; v < kBlock; ++v) q[(v / 4) * kChunkBytes + j * 4 + v % 4] = codes[v];
      }
    }
  }
}

void qgemm(Isa isa, const PackedView& w, const PackedActs& x, float* y, std::int64_t ldy, std::int64_t r0,
           std::int64_t r1) {
  HEARTH_CHECK(w.k == x.k() && w.groups == x.groups() && 0 <= r0 && r0 <= r1 && r1 <= w.rows && ldy >= w.rows);
  if (isa == Isa::kAvx512) return detail::qgemm_avx512(w, x, y, ldy, r0, r1);
  HEARTH_CHECK(isa == Isa::kScalar);
  for (std::int64_t t = 0; t < x.tokens(); ++t) {
    for (std::int64_t r = r0; r < r1; ++r) y[t * ldy + r] = row_token_scalar(w, r, x, t);
  }
}

void tile_range(std::int64_t rows, int parts, int index, std::int64_t& r0, std::int64_t& r1) noexcept {
  constexpr std::int64_t kTile = 4;
  std::int64_t a = 0, b = 0;
  ThreadPool::split_range((rows + kTile - 1) / kTile, parts, index, a, b);
  r0 = std::min(a * kTile, rows);
  r1 = std::min(b * kTile, rows);
}

float canonical_dot(const quant::BlockQ8_0* row, const quant::ActQ8& x) {
  return canonical(row, x,
                   [](const quant::BlockQ8_0& b, const std::int8_t* xq) { return quant::block_dot_q8_0(b, xq); });
}

float canonical_dot(const quant::BlockQ4_0* row, const quant::ActQ8& x) {
  return canonical(row, x,
                   [](const quant::BlockQ4_0& b, const std::int8_t* xq) { return quant::block_dot_q4_0(b, xq); });
}

}  // namespace hearth::cpu
