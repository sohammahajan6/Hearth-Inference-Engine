// Reusable command buffers and timeline semaphores.
//
// A timeline semaphore is a 64-bit counter shared by the CPU and GPU. GPU work
// can wait until it reaches a value and raise it when done; the CPU can do the
// same. This is the primitive the hearth scheduler (Phase 7) uses to chain
// CPU and GPU halves of one operation. Every host-side wait has a finite
// timeout: a GPU that never signals is treated as lost, never waited on forever.
#pragma once

#include <cstdint>
#include <span>

#include "gpu/context.h"
#include "gpu/vk.h"

namespace hearth::gpu {

class TimelineSemaphore {
 public:
  explicit TimelineSemaphore(Context& ctx, std::uint64_t initial = 0);
  ~TimelineSemaphore();
  TimelineSemaphore(const TimelineSemaphore&) = delete;
  TimelineSemaphore& operator=(const TimelineSemaphore&) = delete;

  VkSemaphore handle() const noexcept { return sem_; }
  std::uint64_t value() const;       // current counter (non-blocking)
  void signal(std::uint64_t value);  // raise the counter from the CPU
  // Block until the counter reaches `value`. Throws kDeviceLost on timeout.
  void wait(std::uint64_t value, std::uint64_t timeout_ns = Context::kDefaultTimeoutNs) const;

 private:
  VkDevice device_ = VK_NULL_HANDLE;
  VkSemaphore sem_ = VK_NULL_HANDLE;
};

struct SemaphoreValue {
  VkSemaphore semaphore;
  std::uint64_t value;
};

// A primary command buffer with its own pool and fence, for work that is
// recorded once and submitted many times (or recorded fresh each time).
class CommandBuffer {
 public:
  explicit CommandBuffer(Context& ctx);
  ~CommandBuffer();
  CommandBuffer(const CommandBuffer&) = delete;
  CommandBuffer& operator=(const CommandBuffer&) = delete;

  // Resets and begins recording. Not allowed while a submission is in flight.
  VkCommandBuffer begin();
  void end();
  VkCommandBuffer handle() const noexcept { return cmd_; }

  // Submits the recorded commands. Waits happen before any of the commands
  // start; signals happen after all complete. The internal fence signals too.
  void submit(std::span<const SemaphoreValue> waits = {}, std::span<const SemaphoreValue> signals = {});
  void wait(std::uint64_t timeout_ns = Context::kDefaultTimeoutNs);  // blocking fence wait
  bool done();  // non-blocking fence check; true also clears the in-flight state

 private:
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  VkCommandPool pool_ = VK_NULL_HANDLE;
  VkCommandBuffer cmd_ = VK_NULL_HANDLE;
  VkFence fence_ = VK_NULL_HANDLE;
  bool in_flight_ = false;
};

}  // namespace hearth::gpu
