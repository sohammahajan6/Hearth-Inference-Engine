// core/log: LogCapture catches every line (from any thread) instead of
// stderr, counts by text and by level, does not nest, and keeps caught lines
// out of log_problem_count(), which the test runner requires to stay 0.
#include <doctest/doctest.h>

#include <stdexcept>
#include <thread>

#include "core/log.h"

using namespace hearth;

TEST_SUITE("quick") {
  TEST_CASE("log: LogCapture catches lines from every thread, counts them, and keeps them out of the problem count") {
    const unsigned long long problems = log_problem_count();
    {
      const LogCapture log;
      log_warn("first {}", 1);
      log_error("second {}", "two");
      std::thread([] { log_warn("from another thread"); }).join();
      log_info("plain");
      CHECK(log.lines() == 4);
      CHECK(log.count(LogLevel::kWarn) == 2);
      CHECK(log.count(LogLevel::kError) == 1);
      CHECK(log.count(LogLevel::kInfo) == 1);
      CHECK(log.count("warning: first 1") == 1);
      CHECK(log.count("error: second two") == 1);
      CHECK(log.count("from another thread") == 1);
      CHECK(log.count("no such text") == 0);
      CHECK(log.text() == "warning: first 1\nerror: second two\nwarning: from another thread\nplain\n");
      CHECK_THROWS_AS(LogCapture(), std::logic_error);  // captures do not nest
    }
    CHECK(log_problem_count() == problems);
    {
      const LogCapture again;  // the first one ended: a new capture starts empty
      CHECK(again.lines() == 0);
    }
  }
}
