#include "gpu/vk.h"

#include <cstring>

#include "core/error.h"

namespace hearth::gpu {

const char* vk_result_name(VkResult r) noexcept {
  switch (r) {
#define HEARTH_VK_CASE(x) \
  case x:                 \
    return #x;
    HEARTH_VK_CASE(VK_SUCCESS)
    HEARTH_VK_CASE(VK_NOT_READY)
    HEARTH_VK_CASE(VK_TIMEOUT)
    HEARTH_VK_CASE(VK_EVENT_SET)
    HEARTH_VK_CASE(VK_EVENT_RESET)
    HEARTH_VK_CASE(VK_INCOMPLETE)
    HEARTH_VK_CASE(VK_ERROR_OUT_OF_HOST_MEMORY)
    HEARTH_VK_CASE(VK_ERROR_OUT_OF_DEVICE_MEMORY)
    HEARTH_VK_CASE(VK_ERROR_INITIALIZATION_FAILED)
    HEARTH_VK_CASE(VK_ERROR_DEVICE_LOST)
    HEARTH_VK_CASE(VK_ERROR_MEMORY_MAP_FAILED)
    HEARTH_VK_CASE(VK_ERROR_LAYER_NOT_PRESENT)
    HEARTH_VK_CASE(VK_ERROR_EXTENSION_NOT_PRESENT)
    HEARTH_VK_CASE(VK_ERROR_FEATURE_NOT_PRESENT)
    HEARTH_VK_CASE(VK_ERROR_INCOMPATIBLE_DRIVER)
    HEARTH_VK_CASE(VK_ERROR_TOO_MANY_OBJECTS)
    HEARTH_VK_CASE(VK_ERROR_FORMAT_NOT_SUPPORTED)
    HEARTH_VK_CASE(VK_ERROR_FRAGMENTED_POOL)
    HEARTH_VK_CASE(VK_ERROR_UNKNOWN)
    HEARTH_VK_CASE(VK_ERROR_OUT_OF_POOL_MEMORY)
    HEARTH_VK_CASE(VK_ERROR_INVALID_EXTERNAL_HANDLE)
    HEARTH_VK_CASE(VK_ERROR_FRAGMENTATION)
    HEARTH_VK_CASE(VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS)
#undef HEARTH_VK_CASE
    default:
      return "VkResult(unknown)";
  }
}

void vk_fail(VkResult r, const char* expr, const char* file, int line) {
  const char* base = std::strrchr(file, '\\');
  const char* slash = std::strrchr(file, '/');
  if (slash != nullptr && (base == nullptr || slash > base)) base = slash;
  const char* short_file = base != nullptr ? base + 1 : file;
  ErrorKind kind = ErrorKind::kGpu;
  if (r == VK_ERROR_DEVICE_LOST) kind = ErrorKind::kDeviceLost;
  if (r == VK_ERROR_OUT_OF_DEVICE_MEMORY || r == VK_ERROR_OUT_OF_HOST_MEMORY) kind = ErrorKind::kOutOfMemory;
  throw Error(kind,
              std::format("Vulkan call failed: {} returned {} ({}:{})", expr, vk_result_name(r), short_file, line));
}

}  // namespace hearth::gpu
