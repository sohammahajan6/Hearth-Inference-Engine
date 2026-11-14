// Inner loops for `hearth probe`, one implementation per instruction set.
// Each ISA's functions live in their own .cpp compiled with that ISA's flags;
// callers pick a variant with cpu::probe_read_xor() etc., which dispatch on
// the detected (or forced) ISA.
#pragma once

#include <cstddef>
#include <cstdint>

#include "core/cpu_features.h"

namespace hearth::cpu {

// XOR of every 64-bit word in [data, data + bytes). `data` must be 64-byte
// aligned and `bytes` a multiple of 256.
std::uint64_t probe_read_xor(Isa isa, const std::byte* data, std::size_t bytes);

// Like probe_read_xor, but with streaming loads (VMOVNTDQA). That is the
// documented fast way to read write-combined memory; on normal cached memory
// it behaves like an ordinary load. Needs AVX-512.
std::uint64_t probe_read_xor_streaming(const std::byte* data, std::size_t bytes);

// Vector width of the compute loops. On this core 512-bit FMA/VNNI issue on
// one port and 256-bit on two, so both widths can reach the same ops/clock;
// which is faster depends on the clocks each sustains.
enum class VecWidth { kYmm, kZmm };
int vec_lanes32(VecWidth w) noexcept;  // 32-bit lanes per register: 8 or 16

// `iters` iterations of 12 independent VPDPBUSD (int8 dot) chains.
// Ops = iters * 12 * lanes * 4 * 2. Both widths use the EVEX AVX512_VNNI(+VL)
// encoding, never AVX-VNNI. Returns a value that depends on every result.
std::uint32_t probe_vnni_loop(VecWidth w, std::uint64_t iters, std::uint32_t seed);
// `iters` iterations of 12 independent FP32 FMA chains. FLOPs = iters * 12 * lanes * 2.
float probe_fma_loop(VecWidth w, std::uint64_t iters, float seed);

namespace detail {
std::uint64_t probe_read_xor_scalar(const std::byte* data, std::size_t bytes);
std::uint64_t probe_read_xor_avx2(const std::byte* data, std::size_t bytes);
std::uint64_t probe_read_xor_avx512(const std::byte* data, std::size_t bytes);
std::uint64_t probe_read_xor_streaming_avx512(const std::byte* data, std::size_t bytes);
std::uint32_t probe_vnni_loop_avx512(std::uint64_t iters, std::uint32_t seed);
float probe_fma_loop_avx512(std::uint64_t iters, float seed);
std::uint32_t probe_vnni_loop_ymm(std::uint64_t iters, std::uint32_t seed);
float probe_fma_loop_ymm(std::uint64_t iters, float seed);
}  // namespace detail

}  // namespace hearth::cpu
