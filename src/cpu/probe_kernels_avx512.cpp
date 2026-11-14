// AVX-512 probe loops. Compiled with AVX-512 (+VNNI) flags; only called after
// dispatch confirmed the CPU and OS support them.
#include <immintrin.h>

#include "cpu/probe_kernels.h"

namespace hearth::cpu::detail {

std::uint64_t probe_read_xor_avx512(const std::byte* data, std::size_t bytes) {
  __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;
  for (std::size_t off = 0; off < bytes; off += 256) {
    const auto* p = reinterpret_cast<const __m512i*>(data + off);
    a0 = _mm512_xor_si512(a0, _mm512_load_si512(p + 0));
    a1 = _mm512_xor_si512(a1, _mm512_load_si512(p + 1));
    a2 = _mm512_xor_si512(a2, _mm512_load_si512(p + 2));
    a3 = _mm512_xor_si512(a3, _mm512_load_si512(p + 3));
  }
  const __m512i x = _mm512_xor_si512(_mm512_xor_si512(a0, a1), _mm512_xor_si512(a2, a3));
  alignas(64) std::uint64_t w[8];
  _mm512_store_si512(w, x);
  return w[0] ^ w[1] ^ w[2] ^ w[3] ^ w[4] ^ w[5] ^ w[6] ^ w[7];
}

std::uint64_t probe_read_xor_streaming_avx512(const std::byte* data, std::size_t bytes) {
  __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;
  for (std::size_t off = 0; off < bytes; off += 256) {
    // The intrinsic takes a non-const pointer but only reads through it.
    auto* p = reinterpret_cast<__m512i*>(const_cast<std::byte*>(data + off));
    a0 = _mm512_xor_si512(a0, _mm512_stream_load_si512(p + 0));
    a1 = _mm512_xor_si512(a1, _mm512_stream_load_si512(p + 1));
    a2 = _mm512_xor_si512(a2, _mm512_stream_load_si512(p + 2));
    a3 = _mm512_xor_si512(a3, _mm512_stream_load_si512(p + 3));
  }
  const __m512i x = _mm512_xor_si512(_mm512_xor_si512(a0, a1), _mm512_xor_si512(a2, a3));
  alignas(64) std::uint64_t w[8];
  _mm512_store_si512(w, x);
  return w[0] ^ w[1] ^ w[2] ^ w[3] ^ w[4] ^ w[5] ^ w[6] ^ w[7];
}

// 12 chains: VPDPBUSD has 5-cycle latency and 1/cycle throughput on this
// core, so at least 5 independent chains are needed to keep it busy; 12 gives
// margin. The chains are named variables, not an array: MSVC keeps a
// __m512i array in memory and turns every step into load + op + store, which
// measured the store-forwarding path (half speed) instead of the instruction.
std::uint32_t probe_vnni_loop_avx512(std::uint64_t iters, std::uint32_t seed) {
  const int s = static_cast<int>(seed);
  __m512i c0 = _mm512_set1_epi32(s + 0), c1 = _mm512_set1_epi32(s + 1), c2 = _mm512_set1_epi32(s + 2);
  __m512i c3 = _mm512_set1_epi32(s + 3), c4 = _mm512_set1_epi32(s + 4), c5 = _mm512_set1_epi32(s + 5);
  __m512i c6 = _mm512_set1_epi32(s + 6), c7 = _mm512_set1_epi32(s + 7), c8 = _mm512_set1_epi32(s + 8);
  __m512i c9 = _mm512_set1_epi32(s + 9), c10 = _mm512_set1_epi32(s + 10), c11 = _mm512_set1_epi32(s + 11);
  const __m512i a = _mm512_set1_epi32(0x01020304 ^ s);
  const __m512i b = _mm512_set1_epi32(0x05060708);
  for (std::uint64_t i = 0; i < iters; ++i) {
    c0 = _mm512_dpbusd_epi32(c0, a, b);
    c1 = _mm512_dpbusd_epi32(c1, a, b);
    c2 = _mm512_dpbusd_epi32(c2, a, b);
    c3 = _mm512_dpbusd_epi32(c3, a, b);
    c4 = _mm512_dpbusd_epi32(c4, a, b);
    c5 = _mm512_dpbusd_epi32(c5, a, b);
    c6 = _mm512_dpbusd_epi32(c6, a, b);
    c7 = _mm512_dpbusd_epi32(c7, a, b);
    c8 = _mm512_dpbusd_epi32(c8, a, b);
    c9 = _mm512_dpbusd_epi32(c9, a, b);
    c10 = _mm512_dpbusd_epi32(c10, a, b);
    c11 = _mm512_dpbusd_epi32(c11, a, b);
  }
  const __m512i x =
      _mm512_xor_si512(_mm512_xor_si512(_mm512_xor_si512(_mm512_xor_si512(c0, c1), _mm512_xor_si512(c2, c3)),
                                        _mm512_xor_si512(_mm512_xor_si512(c4, c5), _mm512_xor_si512(c6, c7))),
                       _mm512_xor_si512(_mm512_xor_si512(c8, c9), _mm512_xor_si512(c10, c11)));
  return static_cast<std::uint32_t>(_mm512_reduce_add_epi32(x));
}

float probe_fma_loop_avx512(std::uint64_t iters, float seed) {
  __m512 c0 = _mm512_set1_ps(seed + 0.000f), c1 = _mm512_set1_ps(seed + 0.125f), c2 = _mm512_set1_ps(seed + 0.250f);
  __m512 c3 = _mm512_set1_ps(seed + 0.375f), c4 = _mm512_set1_ps(seed + 0.500f), c5 = _mm512_set1_ps(seed + 0.625f);
  __m512 c6 = _mm512_set1_ps(seed + 0.750f), c7 = _mm512_set1_ps(seed + 0.875f), c8 = _mm512_set1_ps(seed + 1.000f);
  __m512 c9 = _mm512_set1_ps(seed + 1.125f), c10 = _mm512_set1_ps(seed + 1.250f), c11 = _mm512_set1_ps(seed + 1.375f);
  const __m512 m = _mm512_set1_ps(0.9999999f);
  const __m512 c = _mm512_set1_ps(1e-7f);
  for (std::uint64_t i = 0; i < iters; ++i) {
    c0 = _mm512_fmadd_ps(c0, m, c);
    c1 = _mm512_fmadd_ps(c1, m, c);
    c2 = _mm512_fmadd_ps(c2, m, c);
    c3 = _mm512_fmadd_ps(c3, m, c);
    c4 = _mm512_fmadd_ps(c4, m, c);
    c5 = _mm512_fmadd_ps(c5, m, c);
    c6 = _mm512_fmadd_ps(c6, m, c);
    c7 = _mm512_fmadd_ps(c7, m, c);
    c8 = _mm512_fmadd_ps(c8, m, c);
    c9 = _mm512_fmadd_ps(c9, m, c);
    c10 = _mm512_fmadd_ps(c10, m, c);
    c11 = _mm512_fmadd_ps(c11, m, c);
  }
  const __m512 x = _mm512_add_ps(_mm512_add_ps(_mm512_add_ps(_mm512_add_ps(c0, c1), _mm512_add_ps(c2, c3)),
                                               _mm512_add_ps(_mm512_add_ps(c4, c5), _mm512_add_ps(c6, c7))),
                                 _mm512_add_ps(_mm512_add_ps(c8, c9), _mm512_add_ps(c10, c11)));
  return _mm512_reduce_add_ps(x);
}

// 256-bit versions. Compiled in this AVX-512 translation unit, so
// _mm256_dpbusd_epi32 is the EVEX AVX512_VNNI+VL instruction (the VEX
// AVX-VNNI form is a different intrinsic and is illegal on Tiger Lake).
std::uint32_t probe_vnni_loop_ymm(std::uint64_t iters, std::uint32_t seed) {
  const int s = static_cast<int>(seed);
  __m256i c0 = _mm256_set1_epi32(s + 0), c1 = _mm256_set1_epi32(s + 1), c2 = _mm256_set1_epi32(s + 2);
  __m256i c3 = _mm256_set1_epi32(s + 3), c4 = _mm256_set1_epi32(s + 4), c5 = _mm256_set1_epi32(s + 5);
  __m256i c6 = _mm256_set1_epi32(s + 6), c7 = _mm256_set1_epi32(s + 7), c8 = _mm256_set1_epi32(s + 8);
  __m256i c9 = _mm256_set1_epi32(s + 9), c10 = _mm256_set1_epi32(s + 10), c11 = _mm256_set1_epi32(s + 11);
  const __m256i a = _mm256_set1_epi32(0x01020304 ^ s);
  const __m256i b = _mm256_set1_epi32(0x05060708);
  for (std::uint64_t i = 0; i < iters; ++i) {
    c0 = _mm256_dpbusd_epi32(c0, a, b);
    c1 = _mm256_dpbusd_epi32(c1, a, b);
    c2 = _mm256_dpbusd_epi32(c2, a, b);
    c3 = _mm256_dpbusd_epi32(c3, a, b);
    c4 = _mm256_dpbusd_epi32(c4, a, b);
    c5 = _mm256_dpbusd_epi32(c5, a, b);
    c6 = _mm256_dpbusd_epi32(c6, a, b);
    c7 = _mm256_dpbusd_epi32(c7, a, b);
    c8 = _mm256_dpbusd_epi32(c8, a, b);
    c9 = _mm256_dpbusd_epi32(c9, a, b);
    c10 = _mm256_dpbusd_epi32(c10, a, b);
    c11 = _mm256_dpbusd_epi32(c11, a, b);
  }
  const __m256i x =
      _mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(c0, c1), _mm256_xor_si256(c2, c3)),
                                        _mm256_xor_si256(_mm256_xor_si256(c4, c5), _mm256_xor_si256(c6, c7))),
                       _mm256_xor_si256(_mm256_xor_si256(c8, c9), _mm256_xor_si256(c10, c11)));
  alignas(32) std::uint32_t w[8];
  _mm256_store_si256(reinterpret_cast<__m256i*>(w), x);
  return w[0] + w[1] + w[2] + w[3] + w[4] + w[5] + w[6] + w[7];
}

float probe_fma_loop_ymm(std::uint64_t iters, float seed) {
  __m256 c0 = _mm256_set1_ps(seed + 0.000f), c1 = _mm256_set1_ps(seed + 0.125f), c2 = _mm256_set1_ps(seed + 0.250f);
  __m256 c3 = _mm256_set1_ps(seed + 0.375f), c4 = _mm256_set1_ps(seed + 0.500f), c5 = _mm256_set1_ps(seed + 0.625f);
  __m256 c6 = _mm256_set1_ps(seed + 0.750f), c7 = _mm256_set1_ps(seed + 0.875f), c8 = _mm256_set1_ps(seed + 1.000f);
  __m256 c9 = _mm256_set1_ps(seed + 1.125f), c10 = _mm256_set1_ps(seed + 1.250f), c11 = _mm256_set1_ps(seed + 1.375f);
  const __m256 m = _mm256_set1_ps(0.9999999f);
  const __m256 c = _mm256_set1_ps(1e-7f);
  for (std::uint64_t i = 0; i < iters; ++i) {
    c0 = _mm256_fmadd_ps(c0, m, c);
    c1 = _mm256_fmadd_ps(c1, m, c);
    c2 = _mm256_fmadd_ps(c2, m, c);
    c3 = _mm256_fmadd_ps(c3, m, c);
    c4 = _mm256_fmadd_ps(c4, m, c);
    c5 = _mm256_fmadd_ps(c5, m, c);
    c6 = _mm256_fmadd_ps(c6, m, c);
    c7 = _mm256_fmadd_ps(c7, m, c);
    c8 = _mm256_fmadd_ps(c8, m, c);
    c9 = _mm256_fmadd_ps(c9, m, c);
    c10 = _mm256_fmadd_ps(c10, m, c);
    c11 = _mm256_fmadd_ps(c11, m, c);
  }
  const __m256 x = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(c0, c1), _mm256_add_ps(c2, c3)),
                                               _mm256_add_ps(_mm256_add_ps(c4, c5), _mm256_add_ps(c6, c7))),
                                 _mm256_add_ps(_mm256_add_ps(c8, c9), _mm256_add_ps(c10, c11)));
  alignas(32) float w[8];
  _mm256_store_ps(w, x);
  return ((w[0] + w[1]) + (w[2] + w[3])) + ((w[4] + w[5]) + (w[6] + w[7]));
}

}  // namespace hearth::cpu::detail
