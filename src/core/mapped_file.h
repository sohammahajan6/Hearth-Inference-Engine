// A read-only memory-mapped file.
//
// Mapping (instead of reading) a multi-gigabyte model means the OS loads pages
// on first touch and can drop them under memory pressure, because they are
// backed by the file itself rather than by the pagefile.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>

namespace hearth {

class MappedFile {
 public:
  MappedFile() = default;
  // Throws Error(kIo) with the OS error text if the file cannot be opened or mapped.
  explicit MappedFile(const std::filesystem::path& path);
  ~MappedFile();
  MappedFile(MappedFile&& other) noexcept;
  MappedFile& operator=(MappedFile&& other) noexcept;
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  std::span<const std::byte> bytes() const noexcept { return {data_, static_cast<std::size_t>(size_)}; }
  std::uint64_t size() const noexcept { return size_; }
  const std::filesystem::path& path() const noexcept { return path_; }

 private:
  void close() noexcept;

  std::filesystem::path path_;
  void* file_ = nullptr;     // HANDLE
  void* mapping_ = nullptr;  // HANDLE
  const std::byte* data_ = nullptr;
  std::uint64_t size_ = 0;
};

// Text of a Win32 error code, e.g. "The system cannot find the file specified."
std::string win32_error_text(unsigned long code);

}  // namespace hearth
