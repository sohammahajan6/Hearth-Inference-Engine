// GGUF model files: metadata and tensor directory, parsed strictly.
// Design: docs/architecture/04-gguf-loader.md.
//
// The file is untrusted input. Every read is bounds-checked, every size
// computation is overflow-checked, and no count read from the file is used
// to allocate memory before it has been checked against the bytes that are
// actually there. Problems are reported as Error(kBadModel) naming the byte
// offset and what was expected.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/mapped_file.h"
#include "gguf/types.h"

namespace hearth::gguf {

enum class ValueType : std::uint32_t {
  kU8 = 0,
  kI8 = 1,
  kU16 = 2,
  kI16 = 3,
  kU32 = 4,
  kI32 = 5,
  kF32 = 6,
  kBool = 7,
  kString = 8,
  kArray = 9,
  kU64 = 10,
  kI64 = 11,
  kF64 = 12,
};
const char* to_string(ValueType t) noexcept;

// One metadata value. Scalars are widened when parsed (unsigned -> u,
// signed -> i, float -> f); `type` keeps the type the file used. Strings
// point into the file's bytes.
struct Value {
  ValueType type = ValueType::kU8;
  std::uint64_t u = 0;
  std::int64_t i = 0;
  double f = 0;
  bool b = false;
  std::string_view s;
  // Arrays: element type, count, the packed little-endian elements in the
  // file (numeric arrays), or the strings (string arrays).
  ValueType elem = ValueType::kU8;
  std::uint64_t count = 0;
  const std::byte* raw = nullptr;
  std::vector<std::string_view> strings;
};

struct Kv {
  std::string_view key;
  Value value;
};

struct TensorInfo {
  std::string_view name;
  std::uint32_t n_dims = 0;
  std::array<std::uint64_t, 4> ne{1, 1, 1, 1};  // ne[0] is the contiguous dimension
  std::uint32_t type = 0;
  const TypeTraits* traits = nullptr;
  std::uint64_t offset = 0;  // from the start of the data section
  std::uint64_t elements = 0;
  std::uint64_t bytes = 0;
};

class File {
 public:
  // Maps and parses a file. Throws kIo (cannot read) or kBadModel (malformed).
  static File open(const std::filesystem::path& path);
  // Parses bytes the caller keeps alive for the File's lifetime. With
  // `logical_size` > bytes.size(), `bytes` is treated as the first part of a
  // file of that size: the metadata and tensor directory are parsed and
  // checked against the logical size, and tensor_bytes() is available only
  // for tensors that lie inside `bytes` (used by the fuzzer, and for reading
  // just the header of a large file).
  static File parse(std::span<const std::byte> bytes, std::uint64_t logical_size = 0);

  File(File&&) noexcept = default;
  File& operator=(File&&) noexcept = default;
  File(const File&) = delete;
  File& operator=(const File&) = delete;

  std::uint32_t version() const noexcept { return version_; }
  std::uint64_t alignment() const noexcept { return alignment_; }
  std::uint64_t data_offset() const noexcept { return data_offset_; }
  std::uint64_t size() const noexcept { return logical_size_; }

  const std::vector<Kv>& metadata() const noexcept { return kvs_; }
  const Value* find(std::string_view key) const;

  // Typed accessors. They throw kBadModel naming the key if it is missing or
  // has an incompatible type, so model code reads like a specification.
  std::uint64_t get_uint(std::string_view key) const;  // any integer type with a non-negative value
  double get_float(std::string_view key) const;        // f32 or f64
  std::string_view get_string(std::string_view key) const;
  bool get_bool(std::string_view key) const;
  const Value& get_array(std::string_view key, ValueType elem) const;

  const std::vector<TensorInfo>& tensors() const noexcept { return tensors_; }
  const TensorInfo* find_tensor(std::string_view name) const;
  std::span<const std::byte> tensor_bytes(const TensorInfo& t) const;

 private:
  File() = default;
  void parse_all();

  MappedFile mapped_;  // empty when parsing caller-owned bytes
  std::span<const std::byte> bytes_;
  std::uint32_t version_ = 0;
  std::uint64_t alignment_ = 32;
  std::uint64_t data_offset_ = 0;
  std::uint64_t logical_size_ = 0;
  std::vector<Kv> kvs_;
  std::vector<TensorInfo> tensors_;
  std::unordered_map<std::string_view, std::size_t> kv_index_;
  std::unordered_map<std::string_view, std::size_t> tensor_index_;
};

}  // namespace hearth::gguf
