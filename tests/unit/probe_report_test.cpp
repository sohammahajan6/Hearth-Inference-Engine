#include <doctest/doctest.h>

#include <string>

#include "probe/report.h"

TEST_SUITE("quick") {
  TEST_CASE("probe report: references from add() stay valid while more measurements are added") {
    hearth::probe::Report r;
    hearth::probe::Measurement& first = r.add("s", "first", "GB/s");
    for (int i = 0; i < 1000; ++i) r.add("s", "filler " + std::to_string(i), "GB/s");
    first.samples.push_back(1.0);  // would write to freed memory if add() had reallocated
    CHECK(r.measurements.front().samples.size() == 1);
  }

  TEST_CASE("probe report: statistics") {
    hearth::probe::Measurement m;
    m.samples = {5, 1, 4, 2, 3};
    CHECK(m.median() == 3);
    CHECK(m.min() == 1);
    CHECK(m.max() == 5);
    CHECK(m.percentile(25) == 2);
    CHECK(m.percentile(75) == 4);
    CHECK(m.percentile(100) == 5);
    m.samples = {1, 2, 3, 4};
    CHECK(m.median() == 2.5);
  }
}
