#include "core/page_buffer.h"

#include <windows.h>

#include <utility>

#include "core/error.h"
#include "core/mapped_file.h"

namespace hearth {

std::size_t PageBuffer::page_size() noexcept {
  static const std::size_t size = [] {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return static_cast<std::size_t>(si.dwPageSize);
  }();
  return size;
}

PageBuffer::PageBuffer(std::size_t bytes) {
  if (bytes == 0) return;
  void* p = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
  if (p == nullptr) {
    fail(ErrorKind::kOutOfMemory, "VirtualAlloc of {} bytes failed: {}", bytes, win32_error_text(GetLastError()));
  }
  data_ = static_cast<std::byte*>(p);
  size_ = bytes;
  // Touch every page so it is backed by RAM before anyone times anything.
  const std::size_t page = page_size();
  for (std::size_t off = 0; off < bytes; off += page) data_[off] = std::byte{0};
}

PageBuffer::~PageBuffer() {
  if (data_ != nullptr) VirtualFree(data_, 0, MEM_RELEASE);
}

PageBuffer::PageBuffer(PageBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}

PageBuffer& PageBuffer::operator=(PageBuffer&& other) noexcept {
  if (this != &other) {
    if (data_ != nullptr) VirtualFree(data_, 0, MEM_RELEASE);
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
  }
  return *this;
}

}  // namespace hearth
