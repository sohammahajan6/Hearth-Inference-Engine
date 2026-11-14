// A fixed set of pinned worker threads that all run the same job.
//
// Inference calls into the pool thousands of times per second (once per
// matrix operation), so starting threads per call would cost more than the
// work. Workers stay alive, spin briefly waiting for the next job (so a job
// that follows quickly starts within a microsecond or so), then sleep in
// WaitOnAddress so an idle pool costs no CPU.
//
// Every job runs on all threads at once; thread i receives (i, size). Work is
// split by index, never by "whoever is free first", so the same input always
// gives the same per-thread work and bit-identical results.
#pragma once

#include <atomic>
#include <cstdint>
#include <exception>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

namespace hearth {

class ThreadPool {
 public:
  // `threads` participants in total, including the calling thread (index 0).
  // With `pin`, participant i is bound to physical core i mod cores, and SMT
  // siblings are used only after every core has one thread. The caller's
  // original affinity is restored by the destructor, so create and destroy the
  // pool on the same thread.
  explicit ThreadPool(int threads, bool pin = true);
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  int size() const noexcept { return size_; }

  // Runs fn(index, size) on every participant and returns when all are done.
  // If any participant throws, the first exception is rethrown here (after
  // all participants have finished, because they may use the caller's stack).
  template <class F>
  void run(F&& fn) {
    using Fn = std::remove_reference_t<F>;
    run_job(Job{[](void* ctx, int i, int n) { (*static_cast<Fn*>(ctx))(i, n); }, &fn});
  }

  // Splits [0, total) into `parts` contiguous ranges that differ in length by
  // at most one, and returns range `index` as [begin, end).
  static void split_range(std::int64_t total, int parts, int index, std::int64_t& begin, std::int64_t& end) noexcept;

 private:
  struct Job {
    void (*call)(void*, int, int) = nullptr;
    void* ctx = nullptr;
  };

  void run_job(const Job& job);
  void worker_main(int index);
  void record_error(std::exception_ptr e);

  int size_ = 1;
  bool pin_ = false;
  std::uintptr_t caller_old_affinity_ = 0;
  std::vector<std::thread> workers_;
  Job job_;
  alignas(64) std::atomic<std::uint32_t> generation_{0};
  alignas(64) std::atomic<std::int32_t> remaining_{0};
  std::atomic<bool> stop_{false};
  std::mutex error_mutex_;
  std::exception_ptr error_;
};

// Affinity mask for participant `index` under the pinning rule above.
std::uintptr_t thread_affinity_mask(int index);

}  // namespace hearth
