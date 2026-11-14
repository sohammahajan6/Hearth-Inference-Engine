// Vulkan entry point for the library. Functions are loaded at run time by
// volk, so the executable starts (and runs on the CPU) even on a machine with
// no Vulkan driver at all.
#pragma once

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <volk.h>

namespace hearth::gpu {

const char* vk_result_name(VkResult r) noexcept;

// Throws Error: kDeviceLost for VK_ERROR_DEVICE_LOST, kOutOfMemory for the
// out-of-memory results, kGpu otherwise.
[[noreturn]] void vk_fail(VkResult r, const char* expr, const char* file, int line);

}  // namespace hearth::gpu

#define VK_CHECK(expr)                                                                                       \
  do {                                                                                                       \
    const VkResult vk_check_result_ = (expr);                                                                \
    if (vk_check_result_ != VK_SUCCESS) ::hearth::gpu::vk_fail(vk_check_result_, #expr, __FILE__, __LINE__); \
  } while (0)
