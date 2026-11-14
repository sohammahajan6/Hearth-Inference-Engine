// GPU timestamps: how long work took *on the GPU*, excluding the CPU-side
// cost of submitting it and waking up afterwards.
//
// On Iris Xe a tick is 52.08 ns and only the low 36 bits are valid, so the
// counter wraps about once an hour; differences are taken modulo 2^36.
#pragma once

#include <cstdint>
#include <vector>

#include "gpu/vk.h"

namespace hearth::gpu {

class Context;

class TimestampPool {
 public:
  TimestampPool(Context& ctx, std::uint32_t count);
  ~TimestampPool();
  TimestampPool(const TimestampPool&) = delete;
  TimestampPool& operator=(const TimestampPool&) = delete;

  // Host-side reset of all queries (needs the hostQueryReset feature, which
  // gpu::Context requires). Call before recording writes into a pool again.
  void reset();

  // Records "write the GPU clock once all previously recorded compute work has
  // finished" into the command buffer.
  void write(VkCommandBuffer cmd, std::uint32_t index) const;

  // Raw tick values for queries [first, first + n). Only call after the work
  // that writes them has completed (e.g. after its fence was waited on).
  std::vector<std::uint64_t> read(std::uint32_t first, std::uint32_t n) const;

  // Nanoseconds between two tick values, handling counter wrap.
  double elapsed_ns(std::uint64_t begin_ticks, std::uint64_t end_ticks) const noexcept;

  std::uint32_t count() const noexcept { return count_; }

 private:
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueryPool pool_ = VK_NULL_HANDLE;
  std::uint32_t count_ = 0;
  double period_ns_ = 0.0;
  std::uint64_t mask_ = 0;
};

// Maps GPU timestamp ticks onto now_ns()'s clock (QueryPerformanceCounter),
// so GPU work can be drawn on the same time axis as CPU threads (--trace).
// Built from one simultaneous reading of both clocks
// (VK_EXT_calibrated_timestamps); calibrate() keeps the reading with the
// smallest uncertainty of several, since a single one was off by up to
// 160 us in Phase 1. The two clocks drift apart slowly, so a long run takes
// a new reading now and then (LlamaGpu: every second).
class ClockMap {
 public:
  // `gpu_ticks` (masked to `valid_bits`) were read at `cpu_ns`, give or take
  // `deviation_ns`.
  ClockMap(std::uint64_t gpu_ticks, std::int64_t cpu_ns, std::uint64_t deviation_ns, double period_ns,
           std::uint32_t valid_bits) noexcept;
  // The best of `samples` readings. Needs ctx.calibrated_timestamps_enabled().
  static ClockMap calibrate(const Context& ctx, int samples);

  // The now_ns() time of a tick value. Ticks up to half the counter's range
  // before or after the reading map correctly across a wrap (36 bits on
  // Iris Xe: about 30 minutes either way).
  std::int64_t to_cpu_ns(std::uint64_t ticks) const noexcept;
  std::int64_t cpu_ns() const noexcept { return cpu_ns_; }
  std::uint64_t deviation_ns() const noexcept { return deviation_ns_; }

 private:
  std::uint64_t gpu_ticks_;
  std::int64_t cpu_ns_;
  std::uint64_t deviation_ns_;
  double period_ns_;
  std::uint64_t mask_;
};

}  // namespace hearth::gpu
