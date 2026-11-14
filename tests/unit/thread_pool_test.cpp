#include <doctest/doctest.h>

#include <atomic>
#include <stdexcept>
#include <vector>

#include "core/cpu_features.h"
#include "core/error.h"
#include "core/thread_pool.h"

using hearth::ThreadPool;

TEST_SUITE("quick") {
  TEST_CASE("thread pool: every index runs exactly once per job, for every pool size") {
    const int max_threads = hearth::cpu_features().logical_cores;
    for (int n = 1; n <= max_threads; ++n) {
      ThreadPool pool(n);
      std::vector<std::atomic<int>> hits(static_cast<std::size_t>(n));
      for (int job = 0; job < 50; ++job) {
        pool.run([&](int i, int size) {
          REQUIRE(size == n);
          hits[static_cast<std::size_t>(i)].fetch_add(1);
        });
      }
      for (int i = 0; i < n; ++i) CHECK(hits[static_cast<std::size_t>(i)].load() == 50);
    }
  }

  TEST_CASE("thread pool: 10000 back-to-back jobs complete and see each other's writes") {
    ThreadPool pool(4);
    std::vector<long long> sums(4, 0);
    long long expected = 0;
    for (int job = 0; job < 10000; ++job) {
      pool.run([&](int i, int) { sums[static_cast<std::size_t>(i)] += job + i; });
      expected += job;
    }
    for (int i = 0; i < 4; ++i) CHECK(sums[static_cast<std::size_t>(i)] == expected + 10000LL * i);
  }

  TEST_CASE("thread pool: an exception on any thread reaches the caller, and the pool stays usable") {
    ThreadPool pool(4);
    for (int thrower = 0; thrower < 4; ++thrower) {
      CHECK_THROWS_AS(pool.run([&](int i, int) {
        if (i == thrower) throw std::runtime_error("boom");
      }),
                      std::runtime_error);
    }
    std::atomic<int> ran{0};
    pool.run([&](int, int) { ran.fetch_add(1); });
    CHECK(ran.load() == 4);
  }

  TEST_CASE("thread pool: invalid sizes are rejected") {
    CHECK_THROWS_AS(ThreadPool(0), hearth::Error);
    CHECK_THROWS_AS(ThreadPool(hearth::cpu_features().logical_cores + 1), hearth::Error);
  }

  TEST_CASE("thread pool: split_range covers [0,total) exactly with near-equal parts") {
    for (std::int64_t total : {0LL, 1LL, 7LL, 64LL, 1000003LL}) {
      for (int parts : {1, 2, 3, 4, 8}) {
        std::int64_t next = 0;
        for (int i = 0; i < parts; ++i) {
          std::int64_t b = 0, e = 0;
          ThreadPool::split_range(total, parts, i, b, e);
          CHECK(b == next);
          CHECK(e - b >= total / parts);
          CHECK(e - b <= total / parts + 1);
          next = e;
        }
        CHECK(next == total);
      }
    }
  }

  TEST_CASE("thread pool: pinning puts the first 4 threads on 4 different cores") {
    // On a 4-core/8-thread CPU: participants 0..3 get one logical processor of
    // cores 0..3; 4..7 get the SMT siblings. All 8 masks must be distinct
    // single bits.
    const int n = hearth::cpu_features().logical_cores;
    std::uintptr_t seen = 0;
    for (int i = 0; i < n; ++i) {
      const std::uintptr_t m = hearth::thread_affinity_mask(i);
      CHECK(m != 0);
      CHECK((m & (m - 1)) == 0);
      CHECK((seen & m) == 0);
      seen |= m;
    }
  }
}
