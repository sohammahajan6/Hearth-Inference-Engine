// K-quant repacking, Q8_K activations in the packed layout, the canonical
// order of ADR 0021 written out (canonical_dot), the scalar kernel and
// dispatch. Compiled for baseline x86-64: no FMA instruction exists here, and
// /fp:precise (MSVC) and -ffp-contract=off (clang) forbid contracting
// a * b + c anyway.
#include "cpu/kquant.h"

#include <cstring>
#include <vector>

#include "core/error.h"
#include "core/fp16.h"

namespace hearth::cpu {
namespace {

using quant::kSuperBlock;

std::size_t checked_size(std::int64_t a, std::int64_t b) {
  HEARTH_CHECK(a >= 0 && b >= 0 && (a == 0 || b <= INT64_MAX / a));
  return static_cast<std::size_t>(a * b);
}

// The byte offsets of one matrix's arrays in its buffer.
struct Layout {
  std::int64_t groups = 0;
  std::size_t q = 0, sc = 0, mn = 0, d = 0, dmin = 0, total = 0;
};

Layout layout(KType type, std::int64_t rows, std::int64_t k) {
  if (rows <= 0 || k <= 0 || k % kSuperBlock != 0) {
    fail(ErrorKind::kInvalidArgument, "packed K-quant matrix: {} x {} is not a K-quant shape", rows, k);
  }
  Layout l;
  l.groups = k_groups_for(type, k);
  const auto units = static_cast<std::int64_t>(checked_size(rows, l.groups));  // row-groups
  const bool q4 = type == KType::kQ4_K;
  l.q = 0;
  l.sc = checked_size(units, k_group_quant_bytes(type));
  l.mn = l.sc + checked_size(units, kKLanes);
  l.d = l.mn + (q4 ? checked_size(units, kKLanes) : 0);
  l.dmin = l.d + checked_size(units, 4);
  l.total = l.dmin + (q4 ? checked_size(units, 4) : 0);
  return l;
}

// The 16 lanes and the halving tree of ADR 0009 step 5.
struct Lanes {
  float v[kKLanes] = {};  // +0.0f
  void add(int j, float t) noexcept { v[j] = v[j] + t; }
  float reduce() noexcept {
    for (int w = kKLanes / 2; w >= 1; w /= 2) {
      for (int j = 0; j < w; ++j) v[j] = v[j] + v[j + w];
    }
    return v[0];
  }
};

// One unit's term, the operations of ADR 0021 in their order.
float term_q4_k(std::int32_t isum, std::int32_t xsum, float ws, float wm, float dx) noexcept {
  const float s1 = ws * dx;
  const float p1 = static_cast<float>(isum) * s1;
  const float s2 = wm * dx;
  const float p2 = static_cast<float>(xsum) * s2;
  return p1 - p2;
}

float term_q6_k(std::int32_t isum, float ws, float dx) noexcept {
  const float s = ws * dx;
  return static_cast<float>(isum) * s;
}

// Byte offset, in a group's activation codes or Q4_K/Q6_K low-bit chunks, of
// value v of unit j: chunk v / 4, then unit j's 4 bytes.
constexpr std::int64_t chunk_at(int j, int v) noexcept { return (v / 4) * 64 + 4 * j + v % 4; }

// The code of value v (0..31) of unit j in a packed Q4_K group.
int packed_q4_k_code(const std::uint8_t* wq, int j, int v) noexcept {
  return v < 16 ? (wq[chunk_at(j, v)] & 0x0F) : (wq[chunk_at(j, v - 16)] >> 4);
}

// The code of value v (0..15) of unit j in a packed Q6_K group.
int packed_q6_k_code(const std::uint8_t* wq, int j, int v) noexcept {
  const int lo = v < 8 ? (wq[chunk_at(j, v)] & 0x0F) : (wq[chunk_at(j, v - 8)] >> 4);
  const int hi = (wq[128 + 4 * j + v % 4] >> (2 * (v / 4))) & 3;
  return lo | (hi << 4);
}

// One weight row decoded for the scalar kernel: every unit's codes in value
// order (Q6_K already minus 32) and its ws and wm (exact products, ADR 0021),
// so that the codes are unpacked once per row, not once per token.
struct DecodedRow {
  std::vector<std::int8_t> codes;  // [groups][16][unit]
  std::vector<float> ws, wm;       // [groups][16]; wm is unused for Q6_K
};

void decode_row(const KPackedView& w, std::int64_t r, DecodedRow& out) {
  const std::int64_t gq = k_group_quant_bytes(w.type);
  const int unit = k_unit_values(w.type);
  out.codes.resize(static_cast<std::size_t>(w.groups * kKLanes * unit));
  out.ws.resize(static_cast<std::size_t>(w.groups * kKLanes));
  out.wm.resize(out.ws.size());
  for (std::int64_t g = 0; g < w.groups; ++g) {
    const std::int64_t rg = r * w.groups + g;
    const std::uint8_t* wq = w.q + rg * gq;
    for (int j = 0; j < kKLanes; ++j) {
      const auto u = static_cast<std::size_t>(g * kKLanes + j);
      std::int8_t* c = out.codes.data() + u * static_cast<std::size_t>(unit);
      if (w.type == KType::kQ4_K) {
        for (int v = 0; v < unit; ++v) c[v] = static_cast<std::int8_t>(packed_q4_k_code(wq, j, v));
        out.ws[u] = fp16_to_fp32(w.d[rg * 2 + j / 8]) * static_cast<float>(w.sc[rg * kKLanes + j]);
        out.wm[u] = fp16_to_fp32(w.dmin[rg * 2 + j / 8]) * static_cast<float>(w.mn[rg * kKLanes + j]);
      } else {
        for (int v = 0; v < unit; ++v) c[v] = static_cast<std::int8_t>(packed_q6_k_code(wq, j, v) - 32);
        out.ws[u] = fp16_to_fp32(w.d[rg * 2]) * static_cast<float>(static_cast<std::int8_t>(w.sc[rg * kKLanes + j]));
        out.wm[u] = 0.0f;
      }
    }
  }
}

// One token against a decoded row: the arithmetic of canonical_dot.
float row_token_scalar(const KPackedView& w, const DecodedRow& row, const KActs& x, std::int64_t t) {
  const int unit = k_unit_values(w.type);
  Lanes lanes;
  for (std::int64_t g = 0; g < w.groups; ++g) {
    const std::int8_t* xq = x.q(t, g);
    const float* xd = x.d(t, g);
    const std::int32_t* xs = x.s(t, g);
    for (int j = 0; j < kKLanes; ++j) {
      const auto u = static_cast<std::size_t>(g * kKLanes + j);
      const std::int8_t* c = row.codes.data() + u * static_cast<std::size_t>(unit);
      std::int32_t isum = 0;
      for (int chunk = 0; chunk < unit / 4; ++chunk) {
        const std::int8_t* xc = xq + chunk * 64 + 4 * j;  // chunk_at(j, 4 * chunk + i)
        for (int i = 0; i < 4; ++i) isum += c[4 * chunk + i] * xc[i];
      }
      lanes.add(j, w.type == KType::kQ4_K ? term_q4_k(isum, xs[j], row.ws[u], row.wm[u], xd[j])
                                          : term_q6_k(isum, row.ws[u], xd[j]));
    }
  }
  return lanes.reduce();
}

}  // namespace

std::size_t k_packed_bytes(KType type, std::int64_t rows, std::int64_t k) { return layout(type, rows, k).total; }

KPackedView pack_k_matrix(KType type, const void* blocks, std::int64_t rows, std::int64_t k, std::byte* dst) {
  const Layout l = layout(type, rows, k);
  HEARTH_CHECK(reinterpret_cast<std::uintptr_t>(dst) % 64 == 0);
  auto* q = reinterpret_cast<std::uint8_t*>(dst + l.q);
  auto* sc = reinterpret_cast<std::uint8_t*>(dst + l.sc);
  auto* mn = reinterpret_cast<std::uint8_t*>(dst + l.mn);
  auto* d = reinterpret_cast<std::uint16_t*>(dst + l.d);
  auto* dmin = reinterpret_cast<std::uint16_t*>(dst + l.dmin);
  const std::int64_t nb = k / kSuperBlock, gq = k_group_quant_bytes(type);
  // The buffer is zero on entry: the padding of a partial Q4_K group keeps
  // code 0 and scales 0.
  for (std::int64_t r = 0; r < rows; ++r) {
    for (std::int64_t b = 0; b < nb; ++b) {
      if (type == KType::kQ4_K) {
        const quant::BlockQ4_K& blk = static_cast<const quant::BlockQ4_K*>(blocks)[r * nb + b];
        const std::int64_t rg = r * l.groups + b / 2;
        const int half = static_cast<int>(b % 2);
        std::uint8_t* gq_dst = q + rg * gq;
        for (int jj = 0; jj < 8; ++jj) {
          const int j = 8 * half + jj;
          std::uint8_t s = 0, m = 0;
          quant::q4_k_scale_min(blk, jj, s, m);
          sc[rg * kKLanes + j] = s;
          mn[rg * kKLanes + j] = m;
          for (int v = 0; v < 32; ++v) {
            const auto code = static_cast<std::uint8_t>(quant::q4_k_code(blk, 32 * jj + v));
            if (v < 16) {
              gq_dst[chunk_at(j, v)] |= code;
            } else {
              gq_dst[chunk_at(j, v - 16)] |= static_cast<std::uint8_t>(code << 4);
            }
          }
        }
        d[rg * 2 + half] = blk.d;
        dmin[rg * 2 + half] = blk.dmin;
      } else {
        const quant::BlockQ6_K& blk = static_cast<const quant::BlockQ6_K*>(blocks)[r * nb + b];
        const std::int64_t rg = r * l.groups + b;
        std::uint8_t* gq_dst = q + rg * gq;
        for (int j = 0; j < kKLanes; ++j) {
          sc[rg * kKLanes + j] = static_cast<std::uint8_t>(blk.scales[j]);
          for (int v = 0; v < 16; ++v) {
            const int code = quant::q6_k_code(blk, 16 * j + v);
            const auto lo = static_cast<std::uint8_t>(code & 0x0F);
            if (v < 8) {
              gq_dst[chunk_at(j, v)] |= lo;
            } else {
              gq_dst[chunk_at(j, v - 8)] |= static_cast<std::uint8_t>(lo << 4);
            }
            gq_dst[128 + 4 * j + v % 4] |= static_cast<std::uint8_t>((code >> 4) << (2 * (v / 4)));
          }
        }
        d[rg * 2] = blk.d;
      }
    }
  }
  return KPackedView{
      type, rows, k, l.groups, q, sc, type == KType::kQ4_K ? mn : nullptr, d, type == KType::kQ4_K ? dmin : nullptr};
}

KMatrix::KMatrix(KType type, const void* blocks, std::int64_t rows, std::int64_t k) {
  buf_ = PageBuffer(k_packed_bytes(type, rows, k));  // zero-filled by the OS
  view_ = pack_k_matrix(type, blocks, rows, k, buf_.data());
}

void KActs::resize(KType type, std::int64_t tokens, std::int64_t k) {
  HEARTH_CHECK(tokens > 0 && k > 0 && k % kSuperBlock == 0);
  const std::int64_t groups = k_groups_for(type, k);
  const std::size_t n = checked_size(tokens, groups);
  const std::size_t q_bytes = checked_size(static_cast<std::int64_t>(n), k_group_values(type));
  const std::size_t ds_bytes = checked_size(static_cast<std::int64_t>(n), kKLanes * 4);
  const std::size_t need = q_bytes + 2 * ds_bytes;
  if (buf_.size() < need) buf_ = PageBuffer(need);
  d_ = reinterpret_cast<float*>(buf_.data() + q_bytes);
  s_ = reinterpret_cast<std::int32_t*>(buf_.data() + q_bytes + ds_bytes);
  type_ = type;
  tokens_ = tokens;
  k_ = k;
  groups_ = groups;
}

void KActs::quantize(Isa isa, const float* x, std::int64_t t0, std::int64_t t1) {
  HEARTH_CHECK(0 <= t0 && t0 <= t1 && t1 <= tokens_ && (isa == Isa::kScalar || isa == Isa::kAvx512));
  const std::int64_t nb = k_ / kSuperBlock;
  const int unit = k_unit_values(type_), units_per_super = kSuperBlock / unit;
  auto* base = reinterpret_cast<std::int8_t*>(buf_.data());
  for (std::int64_t t = t0; t < t1; ++t) {
    for (std::int64_t b = 0; b < nb; ++b) {
      std::int8_t codes[kSuperBlock];
      float dx = 0.0f;
      if (isa == Isa::kAvx512) {
        detail::quantize_super_avx512(x + t * k_ + b * kSuperBlock, codes, dx);
      } else {
        quant::BlockQ8_K blk;
        quant::quantize_row_q8_k(x + t * k_ + b * kSuperBlock, kSuperBlock, &blk);
        std::memcpy(codes, blk.qs, sizeof(codes));
        dx = blk.d;
      }
      const std::int64_t g = b / k_group_supers(type_);
      const int first = static_cast<int>(b % k_group_supers(type_)) * units_per_super;  // first lane of b
      std::int8_t* q = base + (t * groups_ + g) * k_group_values(type_);
      float* d = d_ + (t * groups_ + g) * kKLanes;
      std::int32_t* s = s_ + (t * groups_ + g) * kKLanes;
      for (int u = 0; u < units_per_super; ++u) {
        const int j = first + u;
        std::int32_t sum = 0;
        for (int v = 0; v < unit; ++v) {
          q[chunk_at(j, v)] = codes[u * unit + v];
          sum += codes[u * unit + v];
        }
        s[j] = sum;
        d[j] = dx;
      }
      // The padding lanes of a last, half-filled Q4_K group (reused storage
      // may hold anything there).
      if (b == nb - 1 && first + units_per_super < kKLanes) {
        for (int j = first + units_per_super; j < kKLanes; ++j) {
          for (int v = 0; v < unit; ++v) q[chunk_at(j, v)] = 0;
          s[j] = 0;
          d[j] = 0.0f;
        }
      }
    }
  }
}

void kgemm(Isa isa, const KPackedView& w, const KActs& x, float* y, std::int64_t ldy, std::int64_t r0,
           std::int64_t r1) {
  HEARTH_CHECK(w.type == x.type() && w.k == x.k() && w.groups == x.groups() && 0 <= r0 && r0 <= r1 && r1 <= w.rows &&
               ldy >= w.rows);
  if (isa == Isa::kAvx512) {
    // Prefill decodes the scales of each 4-row tile once per block of tokens
    // into this scratch (Step K2); decode needs none.
    std::vector<float> scales(x.tokens() > 1 ? static_cast<std::size_t>(detail::k_tile_scale_floats(w)) : 0);
    return detail::kgemm_avx512(w, x, y, ldy, r0, r1, scales.data());
  }
  HEARTH_CHECK(isa == Isa::kScalar);
  DecodedRow row;
  for (std::int64_t r = r0; r < r1; ++r) {
    decode_row(w, r, row);
    for (std::int64_t t = 0; t < x.tokens(); ++t) y[t * ldy + r] = row_token_scalar(w, row, x, t);
  }
}

float canonical_dot(const quant::BlockQ4_K* row, const quant::BlockQ8_K* x, std::int64_t k) {
  HEARTH_CHECK(k % kSuperBlock == 0);
  Lanes lanes;
  for (std::int64_t b = 0; b < k / kSuperBlock; ++b) {
    const float d = fp16_to_fp32(row[b].d), dmin = fp16_to_fp32(row[b].dmin);
    for (int jj = 0; jj < 8; ++jj) {
      std::uint8_t sc = 0, m = 0;
      quant::q4_k_scale_min(row[b], jj, sc, m);
      std::int32_t isum = 0, xsum = 0;
      for (int i = 0; i < 32; ++i) {
        isum += quant::q4_k_code(row[b], 32 * jj + i) * x[b].qs[32 * jj + i];
        xsum += x[b].qs[32 * jj + i];
      }
      const std::int64_t u = b * 8 + jj;
      lanes.add(static_cast<int>(u % kKLanes),
                term_q4_k(isum, xsum, d * static_cast<float>(sc), dmin * static_cast<float>(m), x[b].d));
    }
  }
  return lanes.reduce();
}

float canonical_dot(const quant::BlockQ6_K* row, const quant::BlockQ8_K* x, std::int64_t k) {
  HEARTH_CHECK(k % kSuperBlock == 0);
  Lanes lanes;
  for (std::int64_t b = 0; b < k / kSuperBlock; ++b) {
    const float d = fp16_to_fp32(row[b].d);
    for (int jj = 0; jj < 16; ++jj) {
      std::int32_t isum = 0;
      for (int i = 0; i < 16; ++i) isum += (quant::q6_k_code(row[b], 16 * jj + i) - 32) * x[b].qs[16 * jj + i];
      lanes.add(jj, term_q6_k(isum, d * static_cast<float>(row[b].scales[jj]), x[b].d));
    }
  }
  return lanes.reduce();
}

}  // namespace hearth::cpu
