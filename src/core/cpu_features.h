// What the CPU (and the OS) can execute, detected once at startup.
//
// A feature is only usable if the CPU reports it (CPUID) AND the OS saves the
// matching registers on context switches (XGETBV/XCR0). Checking CPUID alone
// is a classic bug: the program then crashes with 0xC000001D (illegal
// instruction) on systems where the OS disabled AVX or AVX-512.
#pragma once

#include <string>

namespace hearth {

struct CpuFeatures {
  std::string brand;
  int physical_cores = 0;
  int logical_cores = 0;
  // Each flag below already includes the OS-support check.
  bool sse42 = false;
  bool avx = false;
  bool avx2 = false;
  bool fma = false;
  bool f16c = false;
  bool bmi2 = false;
  bool avx512f = false;
  bool avx512bw = false;
  bool avx512dq = false;
  bool avx512vl = false;
  bool avx512vnni = false;  // EVEX-encoded VPDPBUSD (Ice Lake / Tiger Lake and later)
  bool avx512bf16 = false;
  bool avx_vnni = false;  // VEX-encoded VPDPBUSD (Alder Lake and later); NOT on Tiger Lake
};

const CpuFeatures& cpu_features();

// Kernel families, in increasing order of capability. Each one is compiled in
// its own translation unit with matching compiler flags.
enum class Isa {
  kScalar = 0,  // plain C++, any x86-64
  kAvx2 = 1,    // AVX2 + FMA + F16C
  kAvx512 = 2,  // AVX-512 F/BW/DQ/VL + VNNI
};

const char* to_string(Isa isa) noexcept;
bool parse_isa(const std::string& text, Isa& out) noexcept;
bool isa_supported(Isa isa) noexcept;
Isa best_isa() noexcept;

}  // namespace hearth
