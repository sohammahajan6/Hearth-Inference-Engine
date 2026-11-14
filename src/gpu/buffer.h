// A Vulkan storage buffer with its own memory allocation.
//
// On this integrated GPU all memory is the same system RAM; the "kinds" below
// differ only in how the CPU may access it:
//   kDeviceLocal  - GPU only (memory type 0 on Iris Xe).
//   kHostCoherent - CPU-visible, write-combined: fast CPU writes, very slow CPU reads.
//   kHostCached   - CPU-visible and CPU-cached: fast CPU reads and writes (type 2).
#pragma once

#include <cstdint>

#include "gpu/vk.h"

namespace hearth::gpu {

class Context;

enum class MemoryKind { kDeviceLocal, kHostCoherent, kHostCached };
const char* to_string(MemoryKind kind) noexcept;

class Buffer {
 public:
  Buffer() = default;
  Buffer(Context& ctx, VkDeviceSize size, MemoryKind kind);

  // Wraps existing CPU memory as a GPU buffer without copying it
  // (VK_EXT_external_memory_host). `host` and `size` must be multiples of the
  // GPU's import alignment (4 KiB here), and the memory must outlive the
  // buffer: the caller keeps ownership, and mapped() returns `host`.
  static Buffer import_host(Context& ctx, void* host, VkDeviceSize size);
  ~Buffer();
  Buffer(Buffer&& other) noexcept;
  Buffer& operator=(Buffer&& other) noexcept;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  VkBuffer handle() const noexcept { return buffer_; }
  VkDeviceSize size() const noexcept { return size_; }
  // CPU pointer to the contents, or nullptr for kDeviceLocal.
  void* mapped() const noexcept { return mapped_; }
  std::uint32_t memory_type() const noexcept { return memory_type_; }

 private:
  void release() noexcept;

  VkDevice device_ = VK_NULL_HANDLE;
  VkBuffer buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory memory_ = VK_NULL_HANDLE;
  void* mapped_ = nullptr;
  VkDeviceSize size_ = 0;
  std::uint32_t memory_type_ = 0;
};

}  // namespace hearth::gpu
