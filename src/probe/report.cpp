#include "probe/report.h"

#include <algorithm>
#include <cmath>

#include "core/error.h"
#include "core/json_writer.h"

namespace hearth::probe {

double Measurement::percentile(double p) const {
  HEARTH_CHECK(!samples.empty());
  std::vector<double> s = samples;
  std::sort(s.begin(), s.end());
  const auto rank = static_cast<std::size_t>(std::ceil(p / 100.0 * static_cast<double>(s.size())));
  return s[std::min(s.size() - 1, rank == 0 ? 0 : rank - 1)];
}

double Measurement::median() const {
  HEARTH_CHECK(!samples.empty());
  std::vector<double> s = samples;
  std::sort(s.begin(), s.end());
  const std::size_t n = s.size();
  return n % 2 == 1 ? s[n / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
}

double Measurement::min() const {
  HEARTH_CHECK(!samples.empty());
  return *std::min_element(samples.begin(), samples.end());
}

double Measurement::max() const {
  HEARTH_CHECK(!samples.empty());
  return *std::max_element(samples.begin(), samples.end());
}

Measurement& Report::add(std::string section, std::string name, std::string unit) {
  measurements.push_back({std::move(section), std::move(name), std::move(unit), {}, {}, {}});
  return measurements.back();
}

void print_table(const Report& r, std::FILE* out) {
  // Median and interquartile range (p25..p75);
  // the JSON output keeps every sample plus min and max.
  std::fprintf(out, "%-14s %-64s %10s %10s %10s  %s\n", "section", "measurement", "median", "p25", "p75", "unit");
  std::string last;
  for (const Measurement& m : r.measurements) {
    if (m.samples.empty()) continue;
    if (m.section != last && !last.empty()) std::fputc('\n', out);
    last = m.section;
    std::fprintf(out, "%-14s %-64s %10.4g %10.4g %10.4g  %s%s%s\n", m.section.c_str(), m.name.c_str(), m.median(),
                 m.percentile(25), m.percentile(75), m.unit.c_str(), m.note.empty() ? "" : "  ", m.note.c_str());
  }
  for (const std::string& w : r.warnings) std::fprintf(out, "warning: %s\n", w.c_str());
}

std::string to_json(const Report& r) {
  JsonWriter w;
  w.begin_object();
  w.field("tool", "hearth probe");
  w.key("environment");
  write_json(w, r.env);
  w.key("measurements");
  w.begin_array();
  for (const Measurement& m : r.measurements) {
    if (m.samples.empty()) continue;
    w.begin_object();
    w.field("section", m.section);
    w.field("name", m.name);
    w.field("unit", m.unit);
    w.field("median", m.median());
    w.field("p25", m.percentile(25));
    w.field("p75", m.percentile(75));
    w.field("min", m.min());
    w.field("max", m.max());
    w.field("n", m.samples.size());
    if (!m.note.empty()) w.field("note", m.note);
    w.key("params");
    w.begin_object();
    for (const auto& [k, v] : m.params) w.field(k, v);
    w.end_object();
    w.key("samples");
    w.begin_array();
    for (double s : m.samples) w.value(s);
    w.end_array();
    w.end_object();
  }
  w.end_array();
  w.key("warnings");
  w.begin_array();
  for (const std::string& s : r.warnings) w.value(s);
  w.end_array();
  w.end_object();
  return w.str() + "\n";
}

}  // namespace hearth::probe
