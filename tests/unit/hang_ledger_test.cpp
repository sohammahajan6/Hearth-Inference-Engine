// gpu/hang_ledger: one tab-separated line per entry, device losses noted by
// note_device_loss and nothing else, and a default path in the build folder.
#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>
#include <regex>
#include <string>
#include <vector>

#include "core/error.h"
#include "core/log.h"
#include "gpu/hang_ledger.h"
#include "test_util.h"

using namespace hearth;

namespace {

std::vector<std::string> lines_of(const std::filesystem::path& p) {
  std::vector<std::string> out;
  std::ifstream in(p, std::ios::binary);
  for (std::string line; std::getline(in, line);) out.push_back(line);
  return out;
}

// Points the ledger at a test file for the scope, then restores the default.
struct LedgerAt {
  explicit LedgerAt(const std::filesystem::path& p) { gpu::set_hang_ledger_path(p); }
  ~LedgerAt() { gpu::set_hang_ledger_path({}); }
  LedgerAt(const LedgerAt&) = delete;
  LedgerAt& operator=(const LedgerAt&) = delete;
};

}  // namespace

TEST_SUITE("quick") {
  TEST_CASE("hang ledger: entries are single tab-separated lines; only kDeviceLost is noted") {
    const auto dir = test::repo_dir() / "build" / "test-tmp";
    std::filesystem::create_directories(dir);
    const auto path = dir / "hang_ledger_test.txt";
    std::filesystem::remove(path);
    {
      const LedgerAt at(path);
      // Each entry is also reported as a warning; catch and check them here.
      const LogCapture log;
      CHECK(gpu::hang_ledger_path() == path);
      gpu::record_hang("near-miss", "layers 0-3\ttook 300 ms\nsecond line");
      // A device loss is noted and rethrown; other errors pass through unnoted.
      CHECK_THROWS_AS(gpu::note_device_loss("test", [] { fail(ErrorKind::kDeviceLost, "the GPU was lost"); }), Error);
      CHECK_THROWS_AS(gpu::note_device_loss("test", [] { fail(ErrorKind::kInvalidArgument, "not a hang"); }), Error);
      CHECK(gpu::note_device_loss("test", [] { return 7; }) == 7);
      CHECK(log.count(LogLevel::kWarn) == 2);
      CHECK(log.count("warning: GPU hang ledger: near-miss recorded in ") == 1);
      CHECK(log.count("warning: GPU hang ledger: device-lost recorded in ") == 1);
      CHECK(log.lines() == 2);
    }
    const std::vector<std::string> lines = lines_of(path);
    std::filesystem::remove(path);
    REQUIRE(lines.size() == 2);
    const std::regex format(R"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ\t\d{10}\t[a-z-]+\t[^\t]*)");
    for (const std::string& l : lines) CHECK_MESSAGE(std::regex_match(l, format), l);
    CHECK(lines[0].ends_with("\tnear-miss\tlayers 0-3 took 300 ms second line"));
    CHECK(lines[1].find("\tdevice-lost\ttest: ") != std::string::npos);
    CHECK(lines[1].find("the GPU was lost") != std::string::npos);
    // The default is the build folder of this executable (build\<preset>).
    const std::filesystem::path def = gpu::hang_ledger_path();
    CHECK(def.filename() == "gpu-hang-ledger.txt");
    CHECK(std::filesystem::exists(def.parent_path() / "CMakeCache.txt"));
  }
}
