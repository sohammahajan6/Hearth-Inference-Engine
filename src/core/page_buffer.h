// Page-aligned memory straight from the OS (VirtualAlloc), committed and
// touched up front.
//
// Two reasons not to use `new`: (1) the GPU can only import host memory that
// is aligned to its import granularity (4 KiB here), and (2) touching every
// page before timing anything moves the page-fault cost out of the
// measurement.
#pragma once

#include <cstddef>

namespace hearth {

class PageBuffer {
 public:
  PageBuffer() = default;
  // Throws Error(kOutOfMemory) if the OS refuses the allocation.
  explicit PageBuffer(std::size_t bytes);
  ~PageBuffer();
  PageBuffer(PageBuffer&& other) noexcept;
  PageBuffer& operator=(PageBuffer&& other) noexcept;
  PageBuffer(const PageBuffer&) = delete;
  PageBuffer& operator=(const PageBuffer&) = delete;

  std::byte* data() const noexcept { return data_; }
  std::size_t size() const noexcept { return size_; }

  static std::size_t page_size() noexcept;

 private:
  std::byte* data_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace hearth
