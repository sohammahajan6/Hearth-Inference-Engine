#include "gpu/sync.h"

#include <vector>

#include "core/error.h"

namespace hearth::gpu {

TimelineSemaphore::TimelineSemaphore(Context& ctx, std::uint64_t initial) : device_(ctx.device()) {
  VkSemaphoreTypeCreateInfo tci{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
  tci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  tci.initialValue = initial;
  VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  sci.pNext = &tci;
  VK_CHECK(vkCreateSemaphore(device_, &sci, nullptr, &sem_));
}

TimelineSemaphore::~TimelineSemaphore() {
  if (sem_ != VK_NULL_HANDLE) vkDestroySemaphore(device_, sem_, nullptr);
}

std::uint64_t TimelineSemaphore::value() const {
  std::uint64_t v = 0;
  VK_CHECK(vkGetSemaphoreCounterValue(device_, sem_, &v));
  return v;
}

void TimelineSemaphore::signal(std::uint64_t value) {
  VkSemaphoreSignalInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
  si.semaphore = sem_;
  si.value = value;
  VK_CHECK(vkSignalSemaphore(device_, &si));
}

void TimelineSemaphore::wait(std::uint64_t value, std::uint64_t timeout_ns) const {
  VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
  wi.semaphoreCount = 1;
  wi.pSemaphores = &sem_;
  wi.pValues = &value;
  const VkResult r = vkWaitSemaphores(device_, &wi, timeout_ns);
  if (r == VK_TIMEOUT) {
    fail(ErrorKind::kDeviceLost, "timeline semaphore did not reach {} within {:.1f} s; treating the GPU as hung", value,
         static_cast<double>(timeout_ns) / 1e9);
  }
  VK_CHECK(r);
}

CommandBuffer::CommandBuffer(Context& ctx) : device_(ctx.device()), queue_(ctx.queue()) {
  try {
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = ctx.queue_family();
    VK_CHECK(vkCreateCommandPool(device_, &pci, nullptr, &pool_));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool_;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(device_, &cai, &cmd_));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK(vkCreateFence(device_, &fci, nullptr, &fence_));
  } catch (...) {
    if (fence_ != VK_NULL_HANDLE) vkDestroyFence(device_, fence_, nullptr);
    if (pool_ != VK_NULL_HANDLE) vkDestroyCommandPool(device_, pool_, nullptr);
    throw;
  }
}

CommandBuffer::~CommandBuffer() {
  // A submission still in flight must finish before its command buffer dies.
  if (in_flight_) vkWaitForFences(device_, 1, &fence_, VK_TRUE, Context::kDefaultTimeoutNs);
  vkDestroyFence(device_, fence_, nullptr);
  vkDestroyCommandPool(device_, pool_, nullptr);
}

VkCommandBuffer CommandBuffer::begin() {
  HEARTH_CHECK(!in_flight_);
  VK_CHECK(vkResetCommandBuffer(cmd_, 0));
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  VK_CHECK(vkBeginCommandBuffer(cmd_, &bi));
  return cmd_;
}

void CommandBuffer::end() { VK_CHECK(vkEndCommandBuffer(cmd_)); }

void CommandBuffer::submit(std::span<const SemaphoreValue> waits, std::span<const SemaphoreValue> signals) {
  HEARTH_CHECK(!in_flight_);
  std::vector<VkSemaphoreSubmitInfo> w(waits.size());
  std::vector<VkSemaphoreSubmitInfo> s(signals.size());
  for (std::size_t i = 0; i < waits.size(); ++i) {
    w[i] = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, nullptr, waits[i].semaphore, waits[i].value,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,    0};
  }
  for (std::size_t i = 0; i < signals.size(); ++i) {
    s[i] = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, nullptr, signals[i].semaphore, signals[i].value,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,    0};
  }
  VkCommandBufferSubmitInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
  cbi.commandBuffer = cmd_;
  VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
  si.waitSemaphoreInfoCount = static_cast<std::uint32_t>(w.size());
  si.pWaitSemaphoreInfos = w.data();
  si.commandBufferInfoCount = 1;
  si.pCommandBufferInfos = &cbi;
  si.signalSemaphoreInfoCount = static_cast<std::uint32_t>(s.size());
  si.pSignalSemaphoreInfos = s.data();
  VK_CHECK(vkResetFences(device_, 1, &fence_));
  VK_CHECK(vkQueueSubmit2(queue_, 1, &si, fence_));
  in_flight_ = true;
}

void CommandBuffer::wait(std::uint64_t timeout_ns) {
  if (!in_flight_) return;
  wait_fence(device_, fence_, timeout_ns);
  in_flight_ = false;
}

bool CommandBuffer::done() {
  if (!in_flight_) return true;
  const VkResult r = vkGetFenceStatus(device_, fence_);
  if (r == VK_NOT_READY) return false;
  VK_CHECK(r);
  in_flight_ = false;
  return true;
}

}  // namespace hearth::gpu
