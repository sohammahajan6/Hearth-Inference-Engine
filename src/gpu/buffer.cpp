#include "gpu/buffer.h"

#include <utility>

#include "core/error.h"
#include "gpu/context.h"

namespace hearth::gpu {

const char* to_string(MemoryKind kind) noexcept {
  switch (kind) {
    case MemoryKind::kDeviceLocal:
      return "device-local";
    case MemoryKind::kHostCoherent:
      return "host-coherent";
    case MemoryKind::kHostCached:
      return "host-cached";
  }
  return "unknown";
}

Buffer::Buffer(Context& ctx, VkDeviceSize size, MemoryKind kind) : device_(ctx.device()), size_(size) {
  if (size == 0) fail(ErrorKind::kInvalidArgument, "cannot create an empty GPU buffer");
  try {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(device_, &bci, nullptr, &buffer_));
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device_, buffer_, &req);
    VkMemoryPropertyFlags required = 0;
    VkMemoryPropertyFlags preferred = 0;
    switch (kind) {
      case MemoryKind::kDeviceLocal:
        required = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        break;
      case MemoryKind::kHostCoherent:
        required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        preferred = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        break;
      case MemoryKind::kHostCached:
        required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        preferred = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        break;
    }
    memory_type_ = ctx.find_memory_type(req.memoryTypeBits, required, preferred);
    if (kind == MemoryKind::kHostCoherent &&
        (ctx.info().memory_types[memory_type_].flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0) {
      // find_memory_type returns the first match, which on every driver we
      // know is the uncached type. Say so loudly if that ever changes.
      fail(ErrorKind::kInternal, "host-coherent request resolved to a cached memory type {}", memory_type_);
    }
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = memory_type_;
    VK_CHECK(vkAllocateMemory(device_, &mai, nullptr, &memory_));
    VK_CHECK(vkBindBufferMemory(device_, buffer_, memory_, 0));
    if (kind != MemoryKind::kDeviceLocal) VK_CHECK(vkMapMemory(device_, memory_, 0, VK_WHOLE_SIZE, 0, &mapped_));
  } catch (...) {
    release();
    throw;
  }
}

Buffer Buffer::import_host(Context& ctx, void* host, VkDeviceSize size) {
  if (!ctx.external_memory_host_enabled()) {
    fail(ErrorKind::kUnsupported, "GPU '{}' cannot import host memory (no VK_EXT_external_memory_host)",
         ctx.info().name);
  }
  const VkDeviceSize align = ctx.info().min_imported_host_pointer_alignment;
  if (size == 0 || reinterpret_cast<std::uintptr_t>(host) % align != 0 || size % align != 0) {
    fail(ErrorKind::kInvalidArgument, "host import needs pointer and size aligned to {} bytes", align);
  }
  constexpr auto kHandle = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
  Buffer b;
  b.device_ = ctx.device();
  b.size_ = size;
  try {
    VkMemoryHostPointerPropertiesEXT hp{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    VK_CHECK(vkGetMemoryHostPointerPropertiesEXT(b.device_, kHandle, host, &hp));
    VkExternalMemoryBufferCreateInfo ext{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    ext.handleTypes = kHandle;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.pNext = &ext;
    bci.size = size;
    bci.usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(b.device_, &bci, nullptr, &b.buffer_));
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(b.device_, b.buffer_, &req);
    const std::uint32_t allowed = req.memoryTypeBits & hp.memoryTypeBits;
    if (allowed == 0) fail(ErrorKind::kUnsupported, "no memory type can both hold this buffer and import host memory");
    b.memory_type_ = ctx.find_memory_type(allowed, 0, VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    VkImportMemoryHostPointerInfoEXT import{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
    import.handleType = kHandle;
    import.pHostPointer = host;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.pNext = &import;
    mai.allocationSize = size;
    mai.memoryTypeIndex = b.memory_type_;
    VK_CHECK(vkAllocateMemory(b.device_, &mai, nullptr, &b.memory_));
    VK_CHECK(vkBindBufferMemory(b.device_, b.buffer_, b.memory_, 0));
    b.mapped_ = host;
  } catch (...) {
    b.release();
    throw;
  }
  return b;
}

Buffer::~Buffer() { release(); }

Buffer::Buffer(Buffer&& other) noexcept
    : device_(std::exchange(other.device_, VK_NULL_HANDLE)),
      buffer_(std::exchange(other.buffer_, VK_NULL_HANDLE)),
      memory_(std::exchange(other.memory_, VK_NULL_HANDLE)),
      mapped_(std::exchange(other.mapped_, nullptr)),
      size_(std::exchange(other.size_, 0)),
      memory_type_(std::exchange(other.memory_type_, 0)) {}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
  if (this != &other) {
    release();
    device_ = std::exchange(other.device_, VK_NULL_HANDLE);
    buffer_ = std::exchange(other.buffer_, VK_NULL_HANDLE);
    memory_ = std::exchange(other.memory_, VK_NULL_HANDLE);
    mapped_ = std::exchange(other.mapped_, nullptr);
    size_ = std::exchange(other.size_, 0);
    memory_type_ = std::exchange(other.memory_type_, 0);
  }
  return *this;
}

void Buffer::release() noexcept {
  if (device_ == VK_NULL_HANDLE) return;
  if (buffer_ != VK_NULL_HANDLE) vkDestroyBuffer(device_, buffer_, nullptr);
  if (memory_ != VK_NULL_HANDLE) vkFreeMemory(device_, memory_, nullptr);  // also unmaps
  buffer_ = VK_NULL_HANDLE;
  memory_ = VK_NULL_HANDLE;
  mapped_ = nullptr;
}

}  // namespace hearth::gpu
