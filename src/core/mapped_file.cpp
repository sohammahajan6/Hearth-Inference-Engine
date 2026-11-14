#include "core/mapped_file.h"

#include <windows.h>

#include <string>
#include <utility>

#include "core/error.h"
#include "core/platform.h"

namespace hearth {

std::string win32_error_text(unsigned long code) {
  wchar_t* buf = nullptr;
  const DWORD n = FormatMessageW(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, code,
      MAKELANGID(LANG_ENGLISH, SUBLANG_ENGLISH_US), reinterpret_cast<wchar_t*>(&buf), 0, nullptr);
  std::string text;
  if (n != 0 && buf != nullptr) {
    const int len = WideCharToMultiByte(CP_UTF8, 0, buf, static_cast<int>(n), nullptr, 0, nullptr, nullptr);
    text.resize(static_cast<std::size_t>(len));
    WideCharToMultiByte(CP_UTF8, 0, buf, static_cast<int>(n), text.data(), len, nullptr, nullptr);
    LocalFree(buf);
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' ')) text.pop_back();
  }
  return std::format("{} (Win32 error {})", text.empty() ? "unknown error" : text, code);
}

MappedFile::MappedFile(const std::filesystem::path& path) : path_(path) {
  HANDLE file =
      CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE) {
    fail(ErrorKind::kIo, "cannot open '{}': {}", path_to_utf8(path), win32_error_text(GetLastError()));
  }
  file_ = file;
  LARGE_INTEGER size{};
  if (!GetFileSizeEx(file, &size)) {
    const DWORD err = GetLastError();
    close();
    fail(ErrorKind::kIo, "cannot get size of '{}': {}", path_to_utf8(path), win32_error_text(err));
  }
  size_ = static_cast<std::uint64_t>(size.QuadPart);
  if (size_ == 0) return;  // empty file: valid object with no bytes (a zero-length mapping is not allowed)
  HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (mapping == nullptr) {
    const DWORD err = GetLastError();
    close();
    fail(ErrorKind::kIo, "cannot map '{}': {}", path_to_utf8(path), win32_error_text(err));
  }
  mapping_ = mapping;
  const void* view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
  if (view == nullptr) {
    const DWORD err = GetLastError();
    close();
    fail(ErrorKind::kIo, "cannot map view of '{}': {}", path_to_utf8(path), win32_error_text(err));
  }
  data_ = static_cast<const std::byte*>(view);
}

MappedFile::~MappedFile() { close(); }

MappedFile::MappedFile(MappedFile&& other) noexcept
    : path_(std::move(other.path_)),
      file_(std::exchange(other.file_, nullptr)),
      mapping_(std::exchange(other.mapping_, nullptr)),
      data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0)) {}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
  if (this != &other) {
    close();
    path_ = std::move(other.path_);
    file_ = std::exchange(other.file_, nullptr);
    mapping_ = std::exchange(other.mapping_, nullptr);
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
  }
  return *this;
}

void MappedFile::close() noexcept {
  if (data_ != nullptr) UnmapViewOfFile(data_);
  if (mapping_ != nullptr) CloseHandle(mapping_);
  if (file_ != nullptr) CloseHandle(file_);
  data_ = nullptr;
  mapping_ = nullptr;
  file_ = nullptr;
  size_ = 0;
}

}  // namespace hearth
