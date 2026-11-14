// AVX2 probe loops. Compiled with AVX2 flags; only called after dispatch.
#include <immintrin.h>

#include "cpu/probe_kernels.h"

namespace hearth::cpu::detail {

std::uint64_t probe_read_xor_avx2(const std::byte* data, std::size_t bytes) {
  __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
  for (std::size_t off = 0; off < bytes; off += 128) {
    const auto* p = reinterpret_cast<const __m256i*>(data + off);
    a0 = _mm256_xor_si256(a0, _mm256_load_si256(p + 0));
    a1 = _mm256_xor_si256(a1, _mm256_load_si256(p + 1));
    a2 = _mm256_xor_si256(a2, _mm256_load_si256(p + 2));
    a3 = _mm256_xor_si256(a3, _mm256_load_si256(p + 3));
  }
  const __m256i x = _mm256_xor_si256(_mm256_xor_si256(a0, a1), _mm256_xor_si256(a2, a3));
  alignas(32) std::uint64_t w[4];
  _mm256_store_si256(reinterpret_cast<__m256i*>(w), x);
  return w[0] ^ w[1] ^ w[2] ^ w[3];
}

}  // namespace hearth::cpu::detail
