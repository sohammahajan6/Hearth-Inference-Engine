#include "gpu/timestamps.h"

#include "core/error.h"
#include "gpu/context.h"

namespace hearth::gpu {

TimestampPool::TimestampPool(Context& ctx, std::uint32_t count)
    : device_(ctx.device()), count_(count), period_ns_(ctx.info().timestamp_period_ns) {
  const std::uint32_t bits = ctx.info().timestamp_valid_bits;
  if (bits == 0)
    fail(ErrorKind::kUnsupported, "GPU '{}' does not support timestamps on its compute queue", ctx.info().name);
  mask_ = bits >= 64 ? ~0ull : ((1ull << bits) - 1);
  VkQueryPoolCreateInfo ci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
  ci.queryType = VK_QUERY_TYPE_TIMESTAMP;
  ci.queryCount = count;
  VK_CHECK(vkCreateQueryPool(device_, &ci, nullptr, &pool_));
  reset();
}

TimestampPool::~TimestampPool() {
  if (pool_ != VK_NULL_HANDLE) vkDestroyQueryPool(device_, pool_, nullptr);
}

void TimestampPool::reset() { vkResetQueryPool(device_, pool_, 0, count_); }

void TimestampPool::write(VkCommandBuffer cmd, std::uint32_t index) const {
  HEARTH_CHECK(index < count_);
  vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, pool_, index);
}

std::vector<std::uint64_t> TimestampPool::read(std::uint32_t first, std::uint32_t n) const {
  HEARTH_CHECK(first + n <= count_);
  std::vector<std::uint64_t> out(n);
  // WAIT_BIT is safe here because callers only read after the writing
  // submission's fence has signalled; it cannot block on unwritten queries.
  VK_CHECK(vkGetQueryPoolResults(device_, pool_, first, n, n * sizeof(std::uint64_t), out.data(), sizeof(std::uint64_t),
                                 VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
  for (auto& t : out) t &= mask_;
  return out;
}

double TimestampPool::elapsed_ns(std::uint64_t begin_ticks, std::uint64_t end_ticks) const noexcept {
  const std::uint64_t delta = (end_ticks - begin_ticks) & mask_;
  return static_cast<double>(delta) * period_ns_;
}

ClockMap::ClockMap(std::uint64_t gpu_ticks, std::int64_t cpu_ns, std::uint64_t deviation_ns, double period_ns,
                   std::uint32_t valid_bits) noexcept
    : cpu_ns_(cpu_ns),
      deviation_ns_(deviation_ns),
      period_ns_(period_ns),
      mask_(valid_bits >= 64 ? ~0ull : ((1ull << valid_bits) - 1)) {
  gpu_ticks_ = gpu_ticks & mask_;
}

ClockMap ClockMap::calibrate(const Context& ctx, int samples) {
  HEARTH_CHECK(samples >= 1);
  Context::ClockPair best = ctx.sample_clocks();
  for (int i = 1; i < samples; ++i) {
    const Context::ClockPair p = ctx.sample_clocks();
    if (p.max_deviation_ns < best.max_deviation_ns) best = p;
  }
  return {best.gpu_ticks, best.cpu_ns, best.max_deviation_ns, ctx.info().timestamp_period_ns,
          ctx.info().timestamp_valid_bits};
}

std::int64_t ClockMap::to_cpu_ns(std::uint64_t ticks) const noexcept {
  // The distance from the reading modulo the counter's range, taken as
  // negative when it is more than half of it (the tick came first).
  const std::uint64_t forward = (ticks - gpu_ticks_) & mask_;
  const std::uint64_t half = mask_ / 2 + 1;
  const double delta_ticks =
      forward < half ? static_cast<double>(forward) : -static_cast<double>(((gpu_ticks_ - ticks) & mask_));
  return cpu_ns_ + static_cast<std::int64_t>(delta_ticks * period_ns_);
}

}  // namespace hearth::gpu
