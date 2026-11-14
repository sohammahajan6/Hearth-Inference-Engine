#include "core/thread_pool.h"

#include <immintrin.h>
#include <windows.h>

#include <vector>

#include "core/cpu_features.h"
#include "core/error.h"

namespace hearth {
namespace {

// About 100 us of spinning (PAUSE is ~140 cycles on this CPU) before sleeping.
constexpr int kSpinIterations = 2048;

static_assert(sizeof(std::atomic<std::uint32_t>) == sizeof(std::uint32_t));
static_assert(sizeof(std::atomic<std::int32_t>) == sizeof(std::int32_t));

template <class T>
void wait_while_equal(std::atomic<T>& a, T value) {
  for (int i = 0; i < kSpinIterations; ++i) {
    if (a.load(std::memory_order_acquire) != value) return;
    _mm_pause();
  }
  while (a.load(std::memory_order_acquire) == value) {
    WaitOnAddress(reinterpret_cast<volatile void*>(&a), &value, sizeof(T), INFINITE);
  }
}

// Logical-processor masks of the physical cores, in OS order (processor group 0).
const std::vector<KAFFINITY>& core_masks() {
  static const std::vector<KAFFINITY> masks = [] {
    std::vector<KAFFINITY> out;
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    std::vector<std::byte> buf(len);
    if (len == 0 ||
        !GetLogicalProcessorInformationEx(
            RelationProcessorCore, reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data()), &len)) {
      return out;
    }
    for (DWORD off = 0; off < len;) {
      auto* p = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + off);
      if (p->Relationship == RelationProcessorCore && p->Processor.GroupMask[0].Group == 0) {
        out.push_back(p->Processor.GroupMask[0].Mask);
      }
      off += p->Size;
    }
    return out;
  }();
  return masks;
}

}  // namespace

std::uintptr_t thread_affinity_mask(int index) {
  const auto& cores = core_masks();
  if (cores.empty()) return 0;
  const auto ncores = static_cast<int>(cores.size());
  const KAFFINITY core = cores[static_cast<std::size_t>(index % ncores)];
  int sibling = index / ncores;
  // Pick the sibling-th set bit of this core's mask (wrapping if the core has fewer).
  int bits = 0;
  for (KAFFINITY m = core; m != 0; m &= m - 1) ++bits;
  if (bits == 0) return 0;  // a core with no logical processor: leave the thread unpinned
  sibling %= bits;
  KAFFINITY m = core;
  for (int s = 0; s < sibling; ++s) m &= m - 1;
  return static_cast<std::uintptr_t>(m & (~m + 1));  // lowest remaining set bit
}

void ThreadPool::split_range(std::int64_t total, int parts, int index, std::int64_t& begin,
                             std::int64_t& end) noexcept {
  const std::int64_t base = total / parts;
  const std::int64_t extra = total % parts;
  begin = index * base + (index < extra ? index : extra);
  end = begin + base + (index < extra ? 1 : 0);
}

ThreadPool::ThreadPool(int threads, bool pin) : size_(threads), pin_(pin) {
  const int logical = cpu_features().logical_cores;
  if (threads < 1 || threads > logical) {
    fail(ErrorKind::kInvalidArgument, "thread count must be between 1 and {} (logical processors), got {}", logical,
         threads);
  }
  if (pin_) {
    caller_old_affinity_ = SetThreadAffinityMask(GetCurrentThread(), thread_affinity_mask(0));
  }
  workers_.reserve(static_cast<std::size_t>(threads - 1));
  try {
    for (int i = 1; i < threads; ++i) workers_.emplace_back([this, i] { worker_main(i); });
  } catch (...) {
    stop_.store(true, std::memory_order_release);
    generation_.fetch_add(1, std::memory_order_release);
    WakeByAddressAll(&generation_);
    for (auto& t : workers_) t.join();
    throw;
  }
}

ThreadPool::~ThreadPool() {
  stop_.store(true, std::memory_order_release);
  generation_.fetch_add(1, std::memory_order_release);
  WakeByAddressAll(&generation_);
  for (auto& t : workers_) t.join();
  if (pin_ && caller_old_affinity_ != 0) SetThreadAffinityMask(GetCurrentThread(), caller_old_affinity_);
}

void ThreadPool::record_error(std::exception_ptr e) {
  std::lock_guard<std::mutex> lock(error_mutex_);
  if (!error_) error_ = std::move(e);
}

void ThreadPool::worker_main(int index) {
  if (pin_) SetThreadAffinityMask(GetCurrentThread(), thread_affinity_mask(index));
  std::uint32_t seen = 0;
  for (;;) {
    wait_while_equal(generation_, seen);
    seen = generation_.load(std::memory_order_acquire);
    if (stop_.load(std::memory_order_acquire)) return;
    try {
      job_.call(job_.ctx, index, size_);
    } catch (...) {
      record_error(std::current_exception());
    }
    if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) WakeByAddressSingle(&remaining_);
  }
}

void ThreadPool::run_job(const Job& job) {
  if (size_ == 1) {
    job.call(job.ctx, 0, 1);
    return;
  }
  job_ = job;
  error_ = nullptr;
  remaining_.store(size_ - 1, std::memory_order_relaxed);
  generation_.fetch_add(1, std::memory_order_release);  // publishes job_ and remaining_
  WakeByAddressAll(&generation_);
  try {
    job.call(job.ctx, 0, size_);
  } catch (...) {
    record_error(std::current_exception());
  }
  for (std::int32_t r = remaining_.load(std::memory_order_acquire); r != 0;
       r = remaining_.load(std::memory_order_acquire)) {
    wait_while_equal(remaining_, r);
  }
  if (error_) std::rethrow_exception(error_);
}

}  // namespace hearth
