// `hearth probe`: measured hardware ceilings. Design: docs/architecture/02-probe.md.
#pragma once

#include <string>
#include <vector>

#include "probe/report.h"

namespace hearth::probe {

struct Options {
  bool quick = false;                 // smaller buffers, fewer repetitions (~1 min instead of ~5)
  std::vector<std::string> sections;  // empty = all default sections
  int reps = 5;                       // measured repetitions per data point (after warm-up)
};

// The default sections, in run order. "throttle" (~5-10 min), "layout"
// (~2 min, ~2.5 GB of RAM) and "qgemm" (~1 min, up to ~0.5 GB) are opt-in.
const std::vector<std::string>& default_sections();
const std::vector<std::string>& all_sections();

Report run(const Options& options);

}  // namespace hearth::probe
