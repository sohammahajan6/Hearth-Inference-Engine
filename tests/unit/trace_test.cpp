// core/trace: events in Chrome trace format, nothing recorded while off; named lanes.
#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <regex>
#include <string>
#include <vector>

#include "core/trace.h"
#include "test_util.h"

using namespace hearth;

namespace {

std::string read_all(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::size_t count(const std::string& s, const std::string& what) {
  std::size_t n = 0;
  for (std::size_t at = s.find(what); at != std::string::npos; at = s.find(what, at + 1)) ++n;
  return n;
}

}  // namespace

TEST_SUITE("quick") {
  TEST_CASE("trace: scopes become complete events with their layer and shape; nothing while off") {
    const auto dir = test::repo_dir() / "build" / "test-tmp";
    std::filesystem::create_directories(dir);
    const auto path = dir / "trace_test.json";
    {
      const trace::Scope ignored("before_start", 1, 2, 3);
    }
    trace::start();
    CHECK(trace::on());
    {
      const trace::Scope outer("outer");
      {
        const trace::Scope inner("matmul", 7, 2048, 16);
      }
    }
    trace::write(path);
    CHECK(!trace::on());
    {
      const trace::Scope ignored("after_write");
    }
    const std::string json = read_all(path);
    std::filesystem::remove(path);
    CHECK(json.find("\"traceEvents\"") != std::string::npos);
    CHECK(count(json, "\"ph\":\"X\"") == 2);
    CHECK(json.find("\"name\":\"matmul\"") != std::string::npos);
    CHECK(json.find("\"layer\":7") != std::string::npos);
    CHECK(json.find("\"rows\":2048") != std::string::npos);
    CHECK(json.find("\"tokens\":16") != std::string::npos);
    CHECK(json.find("before_start") == std::string::npos);
    CHECK(json.find("after_write") == std::string::npos);
    CHECK(json.find("\"dropped_events\":0") != std::string::npos);
    // The earliest start is time 0: the outer scope began first, though it ended last.
    CHECK(std::regex_search(json, std::regex(R"re("name":"outer","ph":"X","pid":1,"tid":[0-9]+,"ts":0,)re")));
  }

  TEST_CASE("trace: events on a named lane (the GPU's) carry the lane's ID, and the lane gets a title") {
    const auto dir = test::repo_dir() / "build" / "test-tmp";
    std::filesystem::create_directories(dir);
    const auto path = dir / "trace_lane_test.json";
    trace::name_lane(trace::kGpuLane, "GPU: test device");
    trace::start();
    trace::record_on(trace::kGpuLane, "gpu qgemm_q8_0", 5'000, 7'500, -1, 192, 16);
    {
      const trace::Scope cpu("cpu_rows", -1, 64, 16);
    }
    const std::vector<trace::Event> ev = trace::events();
    REQUIRE(ev.size() == 2);
    CHECK(ev[0].lane == trace::kGpuLane);
    CHECK(ev[0].t1_ns - ev[0].t0_ns == 2'500);
    CHECK(ev[1].lane != trace::kGpuLane);
    CHECK(ev[1].lane % 4 == 0);  // a Windows thread ID, which can never equal the odd lane ID
    trace::write(path);
    const std::string json = read_all(path);
    std::filesystem::remove(path);
    const std::string lane = std::to_string(trace::kGpuLane);
    CHECK(count(json, "\"ph\":\"X\"") == 2);
    CHECK(json.find(R"("name":"thread_name","ph":"M","pid":1,"tid":)" + lane +
                    R"(,"args":{"name":"GPU: test device"})") != std::string::npos);
    CHECK(json.find(R"("name":"gpu qgemm_q8_0","ph":"X","pid":1,"tid":)" + lane + R"(,"ts":0,"dur":2.5,)") !=
          std::string::npos);
  }
}
