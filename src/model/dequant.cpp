#include "model/dequant.h"

#include <cstring>

#include "core/error.h"
#include "core/fp16.h"
#include "core/kquant.h"
#include "core/quant.h"

namespace hearth::model {
namespace {

// The formats below are decoded exactly as ggml-quants.c (llama.cpp b11960)
// decodes them, operation for operation, so a file reads to ggml's floats
// (tests/kernels/dequant_more_test.cpp checks them against ggml's own code).
// They run on the CPU's FP32 path; only Q8_0/Q4_0 and Q4_K/Q6_K have
// dedicated kernels.

float half(const std::byte* p) {
  std::uint16_t h;
  std::memcpy(&h, p, 2);
  return fp16_to_fp32(h);
}
std::uint8_t u8(const std::byte* p) { return std::to_integer<std::uint8_t>(*p); }

// get_scale_min_k4 of ggml-quants.c: the 6-bit scale and min `j` of 8.
void scale_min_k4(int j, const std::byte* q, std::uint8_t& d, std::uint8_t& m) {
  if (j < 4) {
    d = u8(q + j) & 63;
    m = u8(q + j + 4) & 63;
  } else {
    d = static_cast<std::uint8_t>((u8(q + j + 4) & 0xF) | ((u8(q + j - 4) >> 6) << 4));
    m = static_cast<std::uint8_t>((u8(q + j + 4) >> 4) | ((u8(q + j) >> 6) << 4));
  }
}

void q4_1(const std::byte* x, float* y) {  // d, m, qs[16]
  const float d = half(x), m = half(x + 2);
  for (int j = 0; j < 16; ++j) {
    const int x0 = u8(x + 4 + j) & 0x0F, x1 = u8(x + 4 + j) >> 4;
    y[j] = x0 * d + m;
    y[j + 16] = x1 * d + m;
  }
}

void q5_0(const std::byte* x, float* y) {  // d, qh[4], qs[16]
  const float d = half(x);
  std::uint32_t qh;
  std::memcpy(&qh, x + 2, 4);
  for (int j = 0; j < 16; ++j) {
    const auto xh_0 = static_cast<std::uint8_t>(((qh >> (j + 0)) << 4) & 0x10);
    const auto xh_1 = static_cast<std::uint8_t>((qh >> (j + 12)) & 0x10);
    const std::int32_t x0 = ((u8(x + 6 + j) & 0x0F) | xh_0) - 16;
    const std::int32_t x1 = ((u8(x + 6 + j) >> 4) | xh_1) - 16;
    y[j] = x0 * d;
    y[j + 16] = x1 * d;
  }
}

void q5_1(const std::byte* x, float* y) {  // d, m, qh[4], qs[16]
  const float d = half(x), m = half(x + 2);
  std::uint32_t qh;
  std::memcpy(&qh, x + 4, 4);
  for (int j = 0; j < 16; ++j) {
    const auto xh_0 = static_cast<std::uint8_t>(((qh >> (j + 0)) << 4) & 0x10);
    const auto xh_1 = static_cast<std::uint8_t>((qh >> (j + 12)) & 0x10);
    const int x0 = (u8(x + 8 + j) & 0x0F) | xh_0;
    const int x1 = (u8(x + 8 + j) >> 4) | xh_1;
    y[j] = x0 * d + m;
    y[j + 16] = x1 * d + m;
  }
}

void q2_k(const std::byte* x, float* y) {  // scales[16], qs[64], d, dmin
  const std::byte* scales = x;
  const std::byte* q = x + 16;
  const float d = half(x + 80), min = half(x + 82);
  int is = 0;
  for (int n = 0; n < 256; n += 128) {
    int shift = 0;
    for (int j = 0; j < 4; ++j) {
      std::uint8_t sc = u8(scales + is++);
      float dl = d * (sc & 0xF), ml = min * (sc >> 4);
      for (int l = 0; l < 16; ++l) *y++ = dl * static_cast<std::int8_t>((u8(q + l) >> shift) & 3) - ml;
      sc = u8(scales + is++);
      dl = d * (sc & 0xF);
      ml = min * (sc >> 4);
      for (int l = 0; l < 16; ++l) *y++ = dl * static_cast<std::int8_t>((u8(q + l + 16) >> shift) & 3) - ml;
      shift += 2;
    }
    q += 32;
  }
}

void q3_k(const std::byte* x, float* y) {  // hmask[32], qs[64], scales[12], d
  const std::byte* hm = x;
  const std::byte* q = x + 32;
  const float d_all = half(x + 108);
  constexpr std::uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
  std::uint32_t aux[4] = {};  // aux[3] is written below, from aux[1] and tmp
  std::memcpy(aux, x + 96, 12);
  const std::uint32_t tmp = aux[2];
  aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
  aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
  aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
  aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
  std::int8_t scales[16];
  std::memcpy(scales, aux, 16);
  int is = 0;
  std::uint8_t m = 1;
  for (int n = 0; n < 256; n += 128) {
    int shift = 0;
    for (int j = 0; j < 4; ++j) {
      float dl = d_all * (scales[is++] - 32);
      for (int l = 0; l < 16; ++l) {
        *y++ = dl * (static_cast<std::int8_t>((u8(q + l) >> shift) & 3) - ((u8(hm + l) & m) ? 0 : 4));
      }
      dl = d_all * (scales[is++] - 32);
      for (int l = 0; l < 16; ++l) {
        *y++ = dl * (static_cast<std::int8_t>((u8(q + l + 16) >> shift) & 3) - ((u8(hm + l + 16) & m) ? 0 : 4));
      }
      shift += 2;
      m = static_cast<std::uint8_t>(m << 1);
    }
    q += 32;
  }
}

void q5_k(const std::byte* x, float* y) {  // d, dmin, scales[12], qh[32], qs[128]
  const float d = half(x), min = half(x + 2);
  const std::byte* sc_bytes = x + 4;
  const std::byte* qh = x + 16;
  const std::byte* ql = x + 48;
  int is = 0;
  std::uint8_t sc, m, u1 = 1, u2 = 2;
  for (int j = 0; j < 256; j += 64) {
    scale_min_k4(is + 0, sc_bytes, sc, m);
    const float d1 = d * sc, m1 = min * m;
    scale_min_k4(is + 1, sc_bytes, sc, m);
    const float d2 = d * sc, m2 = min * m;
    for (int l = 0; l < 32; ++l) *y++ = d1 * ((u8(ql + l) & 0xF) + ((u8(qh + l) & u1) ? 16 : 0)) - m1;
    for (int l = 0; l < 32; ++l) *y++ = d2 * ((u8(ql + l) >> 4) + ((u8(qh + l) & u2) ? 16 : 0)) - m2;
    ql += 32;
    is += 2;
    u1 = static_cast<std::uint8_t>(u1 << 2);
    u2 = static_cast<std::uint8_t>(u2 << 2);
  }
}

}  // namespace

void dequant_row(const gguf::TensorInfo& t, std::span<const std::byte> data, std::int64_t row, float* out) {
  const std::int64_t n = static_cast<std::int64_t>(t.ne[0]);
  const std::uint64_t row_bytes = static_cast<std::uint64_t>(n) / t.traits->block_size * t.traits->type_size;
  HEARTH_CHECK(row >= 0 && static_cast<std::uint64_t>(row + 1) * row_bytes <= data.size());
  const std::byte* p = data.data() + static_cast<std::uint64_t>(row) * row_bytes;
  switch (static_cast<gguf::GgmlType>(t.type)) {
    case gguf::GgmlType::kF32:
      std::memcpy(out, p, static_cast<std::size_t>(n) * 4);
      return;
    case gguf::GgmlType::kF16:
      for (std::int64_t i = 0; i < n; ++i) {
        std::uint16_t h;
        std::memcpy(&h, p + 2 * i, 2);
        out[i] = fp16_to_fp32(h);
      }
      return;
    case gguf::GgmlType::kBF16:
      for (std::int64_t i = 0; i < n; ++i) {
        std::uint16_t h;
        std::memcpy(&h, p + 2 * i, 2);
        out[i] = bf16_to_fp32(h);
      }
      return;
    case gguf::GgmlType::kQ8_0:
      for (std::int64_t b = 0; b < n / quant::kBlock; ++b) {
        quant::BlockQ8_0 blk;
        std::memcpy(&blk, p + b * static_cast<std::int64_t>(sizeof(blk)), sizeof(blk));
        for (int j = 0; j < quant::kBlock; ++j) out[b * quant::kBlock + j] = quant::dequant(blk, j);
      }
      return;
    case gguf::GgmlType::kQ4_0:
      for (std::int64_t b = 0; b < n / quant::kBlock; ++b) {
        quant::BlockQ4_0 blk;
        std::memcpy(&blk, p + b * static_cast<std::int64_t>(sizeof(blk)), sizeof(blk));
        for (int j = 0; j < quant::kBlock; ++j) out[b * quant::kBlock + j] = quant::dequant(blk, j);
      }
      return;
    case gguf::GgmlType::kQ4_K:
      for (std::int64_t b = 0; b < n / quant::kSuperBlock; ++b) {
        quant::BlockQ4_K blk;
        std::memcpy(&blk, p + b * static_cast<std::int64_t>(sizeof(blk)), sizeof(blk));
        quant::dequant_q4_k(blk, out + b * quant::kSuperBlock);
      }
      return;
    case gguf::GgmlType::kQ6_K:
      for (std::int64_t b = 0; b < n / quant::kSuperBlock; ++b) {
        quant::BlockQ6_K blk;
        std::memcpy(&blk, p + b * static_cast<std::int64_t>(sizeof(blk)), sizeof(blk));
        quant::dequant_q6_k(blk, out + b * quant::kSuperBlock);
      }
      return;
    case gguf::GgmlType::kQ4_1:
    case gguf::GgmlType::kQ5_0:
    case gguf::GgmlType::kQ5_1:
    case gguf::GgmlType::kQ2_K:
    case gguf::GgmlType::kQ3_K:
    case gguf::GgmlType::kQ5_K: {
      const auto type = static_cast<gguf::GgmlType>(t.type);
      const std::int64_t bs = t.traits->block_size, bytes = t.traits->type_size;
      void (*decode)(const std::byte*, float*) = type == gguf::GgmlType::kQ4_1   ? q4_1
                                                 : type == gguf::GgmlType::kQ5_0 ? q5_0
                                                 : type == gguf::GgmlType::kQ5_1 ? q5_1
                                                 : type == gguf::GgmlType::kQ2_K ? q2_k
                                                 : type == gguf::GgmlType::kQ3_K ? q3_k
                                                                                 : q5_k;
      for (std::int64_t b = 0; b < n / bs; ++b) decode(p + b * bytes, out + b * bs);
      return;
    }
  }
  fail(ErrorKind::kUnsupported, "cannot dequantize tensor '{}' of type {}", t.name, t.traits->name);
}

}  // namespace hearth::model
