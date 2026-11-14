#include <doctest/doctest.h>

#include "core/cpu_features.h"

TEST_SUITE("quick") {
  TEST_CASE("cpu features: detection is self-consistent") {
    const hearth::CpuFeatures& f = hearth::cpu_features();
    CHECK(f.physical_cores >= 1);
    CHECK(f.logical_cores >= f.physical_cores);
    CHECK_FALSE(f.brand.empty());
    // Implications that must hold on any CPU (each flag includes the OS check).
    if (f.avx2) CHECK(f.avx);
    if (f.avx512bw || f.avx512dq || f.avx512vl || f.avx512vnni) CHECK(f.avx512f);
    if (f.avx512f) CHECK(f.avx2);
    CHECK(hearth::isa_supported(hearth::Isa::kScalar));
    CHECK(hearth::isa_supported(hearth::best_isa()));
  }

  TEST_CASE("cpu features: ISA names round-trip") {
    for (hearth::Isa isa : {hearth::Isa::kScalar, hearth::Isa::kAvx2, hearth::Isa::kAvx512}) {
      hearth::Isa parsed{};
      REQUIRE(hearth::parse_isa(hearth::to_string(isa), parsed));
      CHECK(parsed == isa);
    }
    hearth::Isa unused{};
    CHECK_FALSE(hearth::parse_isa("avx-512", unused));
  }
}
