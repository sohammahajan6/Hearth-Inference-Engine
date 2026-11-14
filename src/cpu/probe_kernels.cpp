// Dispatch and the portable (scalar) variants. Compiled for baseline x86-64.
#include "cpu/probe_kernels.h"

#include <cstring>

#include "core/error.h"

namespace hearth::cpu {

namespace detail {
std::uint64_t probe_read_xor_scalar(const std::byte* data, std::size_t bytes) {
  std::uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  for (std::size_t off = 0; off < bytes; off += 32) {
    std::uint64_t w[4];
    std::memcpy(w, data + off, sizeof(w));
    a0 ^= w[0];
    a1 ^= w[1];
    a2 ^= w[2];
    a3 ^= w[3];
  }
  return a0 ^ a1 ^ a2 ^ a3;
}
}  // namespace detail

std::uint64_t probe_read_xor(Isa isa, const std::byte* data, std::size_t bytes) {
  HEARTH_CHECK(reinterpret_cast<std::uintptr_t>(data) % 64 == 0 && bytes % 256 == 0);
  HEARTH_CHECK(isa_supported(isa));
  switch (isa) {
    case Isa::kAvx512:
      return detail::probe_read_xor_avx512(data, bytes);
    case Isa::kAvx2:
      return detail::probe_read_xor_avx2(data, bytes);
    case Isa::kScalar:
      return detail::probe_read_xor_scalar(data, bytes);
  }
  return 0;
}

std::uint64_t probe_read_xor_streaming(const std::byte* data, std::size_t bytes) {
  HEARTH_CHECK(reinterpret_cast<std::uintptr_t>(data) % 64 == 0 && bytes % 256 == 0);
  if (!isa_supported(Isa::kAvx512)) fail(ErrorKind::kUnsupported, "the streaming-read probe needs AVX-512");
  return detail::probe_read_xor_streaming_avx512(data, bytes);
}

int vec_lanes32(VecWidth w) noexcept { return w == VecWidth::kZmm ? 16 : 8; }

std::uint32_t probe_vnni_loop(VecWidth w, std::uint64_t iters, std::uint32_t seed) {
  if (!isa_supported(Isa::kAvx512)) fail(ErrorKind::kUnsupported, "the VNNI probe needs AVX-512 VNNI");
  return w == VecWidth::kZmm ? detail::probe_vnni_loop_avx512(iters, seed) : detail::probe_vnni_loop_ymm(iters, seed);
}

float probe_fma_loop(VecWidth w, std::uint64_t iters, float seed) {
  if (!isa_supported(Isa::kAvx512)) fail(ErrorKind::kUnsupported, "the FMA probe needs AVX-512");
  return w == VecWidth::kZmm ? detail::probe_fma_loop_avx512(iters, seed) : detail::probe_fma_loop_ymm(iters, seed);
}

}  // namespace hearth::cpu
