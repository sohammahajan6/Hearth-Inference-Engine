// Vulkan instance + device + queue for one GPU, with the checks this project
// relies on: device selection by UUID/LUID, required-vs-optional features,
// validation-layer message counting, and finite timeouts on every wait.
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "gpu/vk.h"

namespace hearth::gpu {

struct MemoryTypeInfo {
  VkMemoryPropertyFlags flags = 0;
  std::uint32_t heap_index = 0;
};

struct DeviceInfo {
  std::string name;
  std::uint32_t vendor_id = 0;
  std::uint32_t device_id = 0;
  std::uint32_t api_version = 0;
  std::uint32_t driver_version = 0;
  std::string driver_version_text;  // vendor-specific decoding, e.g. "101.1692" for Intel
  VkPhysicalDeviceType type = VK_PHYSICAL_DEVICE_TYPE_OTHER;
  std::array<std::uint8_t, VK_UUID_SIZE> uuid{};
  std::array<std::uint8_t, VK_LUID_SIZE> luid{};
  bool luid_valid = false;
  bool luid_matches_dxgi = false;  // LUID equals a hardware (non-software) DXGI adapter
  int duplicates = 0;              // how many other physical devices shared this UUID

  VkPhysicalDeviceLimits limits{};
  VkDeviceSize max_buffer_size = 0;             // largest single VkBuffer
  VkDeviceSize max_memory_allocation_size = 0;  // largest single vkAllocateMemory
  std::uint32_t subgroup_size = 0;
  std::uint32_t min_subgroup_size = 0;
  std::uint32_t max_subgroup_size = 0;
  bool int8_dot_mixed_accelerated = false;   // dot(uint8x4, int8x4) in hardware
  bool int8_dot_signed_accelerated = false;  // dot(int8x4, int8x4) in hardware

  // Features (true = supported by the device).
  bool shader_float16 = false;
  bool shader_int8 = false;
  bool shader_int16 = false;
  bool shader_int64 = false;
  bool shader_float64 = false;
  bool storage_buffer_8bit = false;
  bool storage_buffer_16bit = false;
  bool timeline_semaphore = false;
  bool synchronization2 = false;
  bool subgroup_size_control = false;
  bool compute_full_subgroups = false;
  bool integer_dot_product = false;
  bool host_query_reset = false;
  bool maintenance4 = false;
  bool ext_external_memory_host = false;
  bool ext_memory_budget = false;
  bool ext_calibrated_timestamps = false;  // GPU ticks <-> QPC correlation (VK_EXT_calibrated_timestamps)
  VkDeviceSize min_imported_host_pointer_alignment = 0;

  std::vector<MemoryTypeInfo> memory_types;
  std::vector<VkMemoryHeap> memory_heaps;
  float timestamp_period_ns = 0.0f;
  std::uint32_t timestamp_valid_bits = 0;
  std::uint32_t compute_queue_family = UINT32_MAX;
  std::uint32_t compute_queue_count = 0;
};

// True if the Vulkan loader (vulkan-1.dll) is present and initialises.
bool vulkan_available() noexcept;

// Every Vulkan GPU, deduplicated by UUID (the duplicate with a hardware DXGI
// LUID is kept). Empty if Vulkan is unavailable.
std::vector<DeviceInfo> list_devices();

struct ContextOptions {
  // Require VK_LAYER_KHRONOS_validation (with synchronization validation).
  // Fails if the layer cannot be found; scripts set VK_ADD_LAYER_PATH.
  bool validation = false;
  // Index into list_devices(); -1 picks the integrated GPU whose LUID is a
  // hardware adapter, else the first device.
  int device_index = -1;
};

class Context {
 public:
  explicit Context(const ContextOptions& options = {});
  ~Context();
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  VkInstance instance() const noexcept { return instance_; }
  VkPhysicalDevice physical_device() const noexcept { return physical_; }
  VkDevice device() const noexcept { return device_; }
  VkQueue queue() const noexcept { return queue_; }
  std::uint32_t queue_family() const noexcept { return info_.compute_queue_family; }
  const DeviceInfo& info() const noexcept { return info_; }
  bool validation_enabled() const noexcept { return validation_; }
  bool external_memory_host_enabled() const noexcept { return ext_memory_host_enabled_; }

  bool calibrated_timestamps_enabled() const noexcept { return calibrated_timestamps_enabled_; }

  // One simultaneous reading of the GPU timestamp counter and the CPU's
  // QueryPerformanceCounter (as now_ns() nanoseconds). Requires
  // calibrated_timestamps_enabled(). max_deviation_ns bounds how far apart
  // the two readings may really be.
  struct ClockPair {
    std::uint64_t gpu_ticks;
    std::int64_t cpu_ns;
    std::uint64_t max_deviation_ns;
  };
  ClockPair sample_clocks() const;

  // Validation warnings + errors reported since creation. The acceptance bar is 0.
  std::uint32_t validation_message_count() const noexcept { return validation_messages_.load(); }

  // A memory type allowed by `type_bits` with every `required` flag; among
  // those, the first that also has all `preferred` flags. Throws if none.
  std::uint32_t find_memory_type(std::uint32_t type_bits, VkMemoryPropertyFlags required,
                                 VkMemoryPropertyFlags preferred = 0) const;

  // Record with `record`, submit, wait. Throws kDeviceLost on a lost device or
  // if the GPU has not finished after `timeout_ns`.
  void run_once(const std::function<void(VkCommandBuffer)>& record, std::uint64_t timeout_ns = kDefaultTimeoutNs);

  // Waits longer than Windows' 2 s GPU-hang detection (TDR), so a real hang
  // is reported by the driver as VK_ERROR_DEVICE_LOST rather than by us.
  static constexpr std::uint64_t kDefaultTimeoutNs = 10'000'000'000ull;

 private:
  static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                       VkDebugUtilsMessageTypeFlagsEXT types,
                                                       const VkDebugUtilsMessengerCallbackDataEXT* data, void* user);
  void destroy() noexcept;

  VkInstance instance_ = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  VkCommandPool once_pool_ = VK_NULL_HANDLE;
  VkCommandBuffer once_cmd_ = VK_NULL_HANDLE;
  VkFence once_fence_ = VK_NULL_HANDLE;
  DeviceInfo info_;
  bool validation_ = false;
  bool ext_memory_host_enabled_ = false;
  bool calibrated_timestamps_enabled_ = false;
  std::atomic<std::uint32_t> validation_messages_{0};
};

// Wait for a fence with a finite timeout; throws kDeviceLost on timeout.
void wait_fence(VkDevice device, VkFence fence, std::uint64_t timeout_ns);

}  // namespace hearth::gpu
