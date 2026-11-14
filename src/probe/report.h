// Results of `hearth probe`: a flat list of measurements, each a series of
// samples with a unit and named parameters, plus the environment they were
// taken in. One shape for every section keeps printing, JSON output and the
// sanity checks uniform.
#pragma once

#include <cstdio>
#include <deque>
#include <string>
#include <utility>
#include <vector>

#include "core/environment.h"

namespace hearth::probe {

struct Measurement {
  std::string section;  // e.g. "cpu-bw"
  std::string name;     // e.g. "read, 4 threads, virtualalloc"
  std::string unit;     // e.g. "GB/s"
  std::vector<double> samples;
  std::vector<std::pair<std::string, std::string>> params;
  std::string note;  // optional caveat shown in the table

  double median() const;
  double min() const;
  double max() const;
  double percentile(double p) const;  // p in [0, 100], nearest-rank
};

struct Report {
  Environment env;
  // A deque, not a vector: add() hands out references that callers keep while
  // adding more measurements, and a deque never moves existing elements on
  // push_back. (With a vector, the first probe wrote samples through a
  // dangling reference.)
  std::deque<Measurement> measurements;
  std::vector<std::string> warnings;  // sanity-check failures and noise notes

  Measurement& add(std::string section, std::string name, std::string unit);
};

void print_table(const Report& report, std::FILE* out);
std::string to_json(const Report& report);

}  // namespace hearth::probe
