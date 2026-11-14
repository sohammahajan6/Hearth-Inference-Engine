#include "core/cpu_features.h"

#include <windows.h>

#include <cstdint>
#include <cstring>
#include <vector>

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace hearth {
namespace {

struct Regs {
  std::uint32_t eax, ebx, ecx, edx;
};

Regs cpuid(std::uint32_t leaf, std::uint32_t subleaf) {
  Regs r{};
#if defined(_MSC_VER) && !defined(__clang__)
  int v[4];
  __cpuidex(v, static_cast<int>(leaf), static_cast<int>(subleaf));
  r = {static_cast<std::uint32_t>(v[0]), static_cast<std::uint32_t>(v[1]), static_cast<std::uint32_t>(v[2]),
       static_cast<std::uint32_t>(v[3])};
#else
  __cpuid_count(leaf, subleaf, r.eax, r.ebx, r.ecx, r.edx);
#endif
  return r;
}

std::uint64_t xgetbv0() {
#if defined(_MSC_VER) && !defined(__clang__)
  return _xgetbv(0);
#else
  std::uint32_t lo = 0, hi = 0;
  __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
  return (static_cast<std::uint64_t>(hi) << 32) | lo;
#endif
}

bool bit(std::uint32_t v, int b) { return ((v >> b) & 1u) != 0; }

void count_cores(CpuFeatures& f) {
  DWORD len = 0;
  GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
  std::vector<std::byte> buf(len);
  auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data());
  if (len == 0 || !GetLogicalProcessorInformationEx(RelationProcessorCore, info, &len)) return;
  for (DWORD off = 0; off < len;) {
    auto* p = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + off);
    if (p->Relationship == RelationProcessorCore) {
      ++f.physical_cores;
      for (WORD g = 0; g < p->Processor.GroupCount; ++g) {
        KAFFINITY mask = p->Processor.GroupMask[g].Mask;
        while (mask != 0) {
          f.logical_cores += static_cast<int>(mask & 1u);
          mask >>= 1;
        }
      }
    }
    off += p->Size;
  }
}

CpuFeatures detect() {
  CpuFeatures f;
  const Regs r0 = cpuid(0, 0);
  const std::uint32_t max_leaf = r0.eax;
  const Regs r1 = cpuid(1, 0);
  const Regs r7 = max_leaf >= 7 ? cpuid(7, 0) : Regs{};
  const Regs r71 = (max_leaf >= 7 && r7.eax >= 1) ? cpuid(7, 1) : Regs{};

  const bool osxsave = bit(r1.ecx, 27);
  const std::uint64_t xcr0 = osxsave ? xgetbv0() : 0;
  const bool os_avx = (xcr0 & 0x6) == 0x6;       // XMM + YMM state
  const bool os_avx512 = (xcr0 & 0xE6) == 0xE6;  // + opmask, ZMM_Hi256, Hi16_ZMM

  f.sse42 = bit(r1.ecx, 20);
  f.avx = os_avx && bit(r1.ecx, 28);
  f.fma = f.avx && bit(r1.ecx, 12);
  f.f16c = f.avx && bit(r1.ecx, 29);
  f.avx2 = f.avx && bit(r7.ebx, 5);
  f.bmi2 = bit(r7.ebx, 8);
  f.avx512f = os_avx512 && bit(r7.ebx, 16);
  f.avx512dq = f.avx512f && bit(r7.ebx, 17);
  f.avx512bw = f.avx512f && bit(r7.ebx, 30);
  f.avx512vl = f.avx512f && bit(r7.ebx, 31);
  f.avx512vnni = f.avx512f && bit(r7.ecx, 11);
  f.avx512bf16 = f.avx512f && bit(r71.eax, 5);
  f.avx_vnni = f.avx && bit(r71.eax, 4);

  const Regs ext = cpuid(0x80000000u, 0);
  if (ext.eax >= 0x80000004u) {
    char brand[49] = {};
    for (std::uint32_t i = 0; i < 3; ++i) {
      const Regs b = cpuid(0x80000002u + i, 0);
      std::memcpy(brand + 16 * i + 0, &b.eax, 4);
      std::memcpy(brand + 16 * i + 4, &b.ebx, 4);
      std::memcpy(brand + 16 * i + 8, &b.ecx, 4);
      std::memcpy(brand + 16 * i + 12, &b.edx, 4);
    }
    f.brand = brand;
    const auto first = f.brand.find_first_not_of(' ');
    f.brand = first == std::string::npos ? std::string() : f.brand.substr(first);
  }
  count_cores(f);
  return f;
}

}  // namespace

const CpuFeatures& cpu_features() {
  static const CpuFeatures features = detect();
  return features;
}

const char* to_string(Isa isa) noexcept {
  switch (isa) {
    case Isa::kScalar:
      return "scalar";
    case Isa::kAvx2:
      return "avx2";
    case Isa::kAvx512:
      return "avx512";
  }
  return "unknown";
}

bool parse_isa(const std::string& text, Isa& out) noexcept {
  for (Isa i : {Isa::kScalar, Isa::kAvx2, Isa::kAvx512}) {
    if (text == to_string(i)) {
      out = i;
      return true;
    }
  }
  return false;
}

bool isa_supported(Isa isa) noexcept {
  const CpuFeatures& f = cpu_features();
  switch (isa) {
    case Isa::kScalar:
      return true;
    case Isa::kAvx2:
      return f.avx2 && f.fma && f.f16c;
    case Isa::kAvx512:
      return f.avx512f && f.avx512bw && f.avx512dq && f.avx512vl && f.avx512vnni && f.fma && f.f16c;
  }
  return false;
}

Isa best_isa() noexcept {
  if (isa_supported(Isa::kAvx512)) return Isa::kAvx512;
  if (isa_supported(Isa::kAvx2)) return Isa::kAvx2;
  return Isa::kScalar;
}

}  // namespace hearth
