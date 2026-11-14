// `hearth probe`: measure this machine's bandwidth, latency and compute limits.
#include <cstdio>
#include <format>
#include <fstream>
#include <sstream>

#include "commands.h"
#include "core/error.h"
#include "core/log.h"
#include "core/platform.h"
#include "hearth/version.h"
#include "probe/probe.h"

namespace hearth::cli {

int cmd_probe(Args& args) {
  probe::Options o;
  o.quick = args.flag("quick");
  o.reps = static_cast<int>(args.int_or("reps", o.quick ? 3 : 5, 1, 50));
  const bool allow_battery = args.flag("allow-battery");
  // For correctness runs (e.g. the whole probe under ASan):
  // the numbers are then marked as not comparable.
  const bool allow_non_release = args.flag("allow-non-release");
  const auto json_path = args.value("json");
  if (auto list = args.value("sections")) {
    std::stringstream ss(*list);
    for (std::string s; std::getline(ss, s, ',');) {
      if (!s.empty()) o.sections.push_back(s);
    }
  }
  args.finish();

  const bool release = build_info().build_type == std::string("Release");
  if (!release && !allow_non_release) {
    fail(ErrorKind::kInvalidArgument,
         "probe measures performance; use a Release build (this is {}; --allow-non-release runs it anyway)",
         build_info().build_type);
  }
  if (power_source() != PowerSource::kAc && !allow_battery) {
    fail(ErrorKind::kInvalidArgument,
         "the machine is not on AC power; battery limits clocks and makes numbers incomparable "
         "(pass --allow-battery to measure anyway)");
  }
  probe::Report report = probe::run(o);
  if (!release) {
    report.warnings.push_back(std::format("{} build{}: these numbers are not comparable to Release measurements",
                                          build_info().build_type, build_info().sanitized ? " with ASan/UBSan" : ""));
  }
  probe::print_table(report, stdout);
  if (json_path) {
    std::ofstream f(path_from_utf8(*json_path), std::ios::binary);
    if (!f) fail(ErrorKind::kIo, "cannot write '{}'", *json_path);
    f << probe::to_json(report);
    if (!f) fail(ErrorKind::kIo, "error while writing '{}'", *json_path);
    log_info("probe: wrote {}", *json_path);
  }
  return report.warnings.empty() ? 0 : 1;
}

}  // namespace hearth::cli
