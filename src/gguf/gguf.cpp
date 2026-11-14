#include "gguf/gguf.h"

#include <algorithm>
#include <cstring>
#include <format>
#include <limits>
#include <string>

#include "core/error.h"
#include "core/utf8.h"

namespace hearth::gguf {
namespace {

constexpr std::uint64_t kMaxAlignment = 1u << 20;
constexpr std::size_t kMaxNameBytes = 63;  // ggml's GGML_MAX_NAME is 64 including the terminator
constexpr std::uint32_t kMaxDims = 4;
// Hard caps, far above any real model (the Llama-3.2 files have 32 keys, 147
// tensors, arrays of <= 280147 elements and 7.8 MB of metadata). They bound
// memory use for hostile files: limits derived only from the file size would
// still let a 5 GB file request tens of GB of heap.
constexpr std::uint64_t kMaxKvs = 4096;
constexpr std::uint64_t kMaxTensors = 65536;
constexpr std::uint64_t kMaxArrayElements = std::uint64_t{1} << 24;
constexpr std::size_t kMaxKeyBytes = 256;
constexpr std::uint64_t kMaxMetadataBytes = std::uint64_t{64} << 20;

// A bounds-checked little-endian reader over the file's bytes.
class Cursor {
 public:
  explicit Cursor(std::span<const std::byte> b) : b_(b) {}

  std::uint64_t pos() const noexcept { return pos_; }
  std::uint64_t remaining() const noexcept { return b_.size() - pos_; }

  template <class T>
  T read(const char* what) {
    need(sizeof(T), what);
    T v;
    std::memcpy(&v, b_.data() + pos_, sizeof(T));  // x86 is little-endian, like the format
    pos_ += sizeof(T);
    return v;
  }

  std::string_view read_string(const char* what) {
    const auto len = read<std::uint64_t>(what);
    need(len, what);
    std::string_view s(reinterpret_cast<const char*>(b_.data() + pos_), static_cast<std::size_t>(len));
    pos_ += len;
    return s;
  }

  const std::byte* skip(std::uint64_t n, const char* what) {
    need(n, what);
    const std::byte* p = b_.data() + pos_;
    pos_ += n;
    return p;
  }

  void need(std::uint64_t n, const char* what) const {
    if (n > remaining()) {
      fail(ErrorKind::kBadModel, "GGUF: {} at byte {} needs {} bytes but only {} remain (file truncated or corrupt)",
           what, pos_, n, remaining());
    }
  }

 private:
  std::span<const std::byte> b_;
  std::uint64_t pos_ = 0;
};

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b, const char* what) {
  if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
    fail(ErrorKind::kBadModel, "GGUF: {} overflows 64 bits ({} x {})", what, a, b);
  }
  return a * b;
}

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, const char* what) {
  if (b > std::numeric_limits<std::uint64_t>::max() - a) {
    fail(ErrorKind::kBadModel, "GGUF: {} overflows 64 bits ({} + {})", what, a, b);
  }
  return a + b;
}

// Size in bytes of one element of a fixed-size value type; 0 for string/array.
std::uint64_t scalar_size(ValueType t) {
  switch (t) {
    case ValueType::kU8:
    case ValueType::kI8:
    case ValueType::kBool:
      return 1;
    case ValueType::kU16:
    case ValueType::kI16:
      return 2;
    case ValueType::kU32:
    case ValueType::kI32:
    case ValueType::kF32:
      return 4;
    case ValueType::kU64:
    case ValueType::kI64:
    case ValueType::kF64:
      return 8;
    case ValueType::kString:
    case ValueType::kArray:
      return 0;
  }
  return 0;
}

ValueType checked_type(std::uint32_t code, std::uint64_t at) {
  if (code > static_cast<std::uint32_t>(ValueType::kF64)) {
    fail(ErrorKind::kBadModel, "GGUF: unknown value type {} at byte {}", code, at);
  }
  return static_cast<ValueType>(code);
}

void read_scalar(Cursor& c, ValueType t, Value& v, std::string_view key) {
  switch (t) {
    case ValueType::kU8:
      v.u = c.read<std::uint8_t>("u8 value");
      break;
    case ValueType::kI8: {
      // Two's complement, decoded explicitly (no signed char conversion).
      const std::uint8_t b = c.read<std::uint8_t>("i8 value");
      v.i = b < 128 ? std::int64_t{b} : std::int64_t{b} - 256;
      break;
    }
    case ValueType::kU16:
      v.u = c.read<std::uint16_t>("u16 value");
      break;
    case ValueType::kI16:
      v.i = c.read<std::int16_t>("i16 value");
      break;
    case ValueType::kU32:
      v.u = c.read<std::uint32_t>("u32 value");
      break;
    case ValueType::kI32:
      v.i = c.read<std::int32_t>("i32 value");
      break;
    case ValueType::kF32:
      v.f = c.read<float>("f32 value");
      break;
    case ValueType::kU64:
      v.u = c.read<std::uint64_t>("u64 value");
      break;
    case ValueType::kI64:
      v.i = c.read<std::int64_t>("i64 value");
      break;
    case ValueType::kF64:
      v.f = c.read<double>("f64 value");
      break;
    case ValueType::kBool: {
      const auto at = c.pos();
      const auto raw = c.read<std::uint8_t>("bool value");
      if (raw > 1) fail(ErrorKind::kBadModel, "GGUF: key '{}': bool at byte {} is {}, not 0 or 1", key, at, raw);
      v.b = raw == 1;
      break;
    }
    case ValueType::kString: {
      v.s = c.read_string("string value");
      if (!valid_utf8(v.s)) fail(ErrorKind::kBadModel, "GGUF: key '{}': string value is not valid UTF-8", key);
      break;
    }
    case ValueType::kArray:
      HEARTH_CHECK(false);
      break;  // handled by the caller
  }
}

Value read_value(Cursor& c, ValueType t, std::string_view key) {
  Value v;
  v.type = t;
  if (t != ValueType::kArray) {
    read_scalar(c, t, v, key);
    return v;
  }
  const auto at = c.pos();
  v.elem = checked_type(c.read<std::uint32_t>("array element type"), at);
  if (v.elem == ValueType::kArray)
    fail(ErrorKind::kBadModel, "GGUF: key '{}': arrays of arrays are not supported", key);
  v.count = c.read<std::uint64_t>("array length");
  if (v.count > kMaxArrayElements) {
    fail(ErrorKind::kBadModel, "GGUF: key '{}': array of {} elements exceeds the limit of {}", key, v.count,
         kMaxArrayElements);
  }
  if (v.elem == ValueType::kString) {
    // Each string needs at least its 8-byte length, which bounds the count
    // before anything is allocated.
    if (v.count > c.remaining() / 8) {
      fail(ErrorKind::kBadModel, "GGUF: key '{}': string array of {} elements cannot fit in the {} remaining bytes",
           key, v.count, c.remaining());
    }
    v.strings.reserve(static_cast<std::size_t>(v.count));
    for (std::uint64_t k = 0; k < v.count; ++k) {
      const std::string_view s = c.read_string("string array element");
      if (!valid_utf8(s)) fail(ErrorKind::kBadModel, "GGUF: key '{}': element {} is not valid UTF-8", key, k);
      v.strings.push_back(s);
    }
    return v;
  }
  const std::uint64_t bytes = checked_mul(v.count, scalar_size(v.elem), "array size");
  v.raw = c.skip(bytes, "array data");
  if (v.elem == ValueType::kBool) {
    for (std::uint64_t k = 0; k < v.count; ++k) {
      const auto raw = static_cast<std::uint8_t>(v.raw[k]);
      if (raw > 1) fail(ErrorKind::kBadModel, "GGUF: key '{}': bool array element {} is {}, not 0 or 1", key, k, raw);
    }
  }
  return v;
}

bool is_unsigned(ValueType t) {
  return t == ValueType::kU8 || t == ValueType::kU16 || t == ValueType::kU32 || t == ValueType::kU64;
}
bool is_signed(ValueType t) {
  return t == ValueType::kI8 || t == ValueType::kI16 || t == ValueType::kI32 || t == ValueType::kI64;
}

}  // namespace

const char* to_string(ValueType t) noexcept {
  switch (t) {
    case ValueType::kU8:
      return "u8";
    case ValueType::kI8:
      return "i8";
    case ValueType::kU16:
      return "u16";
    case ValueType::kI16:
      return "i16";
    case ValueType::kU32:
      return "u32";
    case ValueType::kI32:
      return "i32";
    case ValueType::kF32:
      return "f32";
    case ValueType::kBool:
      return "bool";
    case ValueType::kString:
      return "string";
    case ValueType::kArray:
      return "array";
    case ValueType::kU64:
      return "u64";
    case ValueType::kI64:
      return "i64";
    case ValueType::kF64:
      return "f64";
  }
  return "?";
}

File File::open(const std::filesystem::path& path) {
  File f;
  f.mapped_ = MappedFile(path);
  f.bytes_ = f.mapped_.bytes();
  f.logical_size_ = f.bytes_.size();
  f.parse_all();
  return f;
}

File File::parse(std::span<const std::byte> bytes, std::uint64_t logical_size) {
  File f;
  f.bytes_ = bytes;
  f.logical_size_ = std::max<std::uint64_t>(logical_size, bytes.size());
  f.parse_all();
  return f;
}

void File::parse_all() {
  Cursor c(bytes_);
  const auto magic = c.read<std::array<char, 4>>("magic");
  if (std::memcmp(magic.data(), "GGUF", 4) != 0) {
    fail(ErrorKind::kBadModel, "not a GGUF file: the first 4 bytes are not 'GGUF'");
  }
  version_ = c.read<std::uint32_t>("version");
  if ((version_ & 0xFFFFu) == 0 && version_ != 0) {
    fail(ErrorKind::kBadModel,
         "GGUF: this is a big-endian file (version bytes 0x{:08x}); only little-endian is supported", version_);
  }
  if (version_ != 2 && version_ != 3) {
    fail(ErrorKind::kBadModel, "GGUF version {} is not supported (only 2 and 3; version 1 used 32-bit counts)",
         version_);
  }
  const auto n_tensors = c.read<std::uint64_t>("tensor count");
  const auto n_kv = c.read<std::uint64_t>("metadata count");
  // A key-value pair takes at least 8 (key length) + 1 (key) + 4 (type) + 1 bytes;
  // a tensor info at least 8 + 1 + 4 + 8 + 4 + 8.
  if (n_kv > c.remaining() / 14 || n_tensors > c.remaining() / 33) {
    fail(ErrorKind::kBadModel, "GGUF: {} metadata entries and {} tensors cannot fit in a {}-byte file", n_kv, n_tensors,
         bytes_.size());
  }
  if (n_kv > kMaxKvs || n_tensors > kMaxTensors) {
    fail(ErrorKind::kBadModel, "GGUF: {} metadata entries / {} tensors exceed the limits of {} / {}", n_kv, n_tensors,
         kMaxKvs, kMaxTensors);
  }

  kvs_.reserve(static_cast<std::size_t>(n_kv));
  for (std::uint64_t k = 0; k < n_kv; ++k) {
    const auto at = c.pos();
    const std::string_view key = c.read_string("metadata key");
    if (key.empty() || key.size() > kMaxKeyBytes || key.find('\0') != std::string_view::npos || !valid_utf8(key)) {
      fail(ErrorKind::kBadModel,
           "GGUF: metadata key {} at byte {} is empty, longer than {} bytes, contains NUL, or is not valid UTF-8", k,
           at, kMaxKeyBytes);
    }
    const ValueType t = checked_type(c.read<std::uint32_t>("value type"), c.pos() - 4);
    Kv kv{key, read_value(c, t, key)};
    if (!kv_index_.emplace(key, kvs_.size()).second)
      fail(ErrorKind::kBadModel, "GGUF: duplicate metadata key '{}'", key);
    kvs_.push_back(std::move(kv));
    if (c.pos() > kMaxMetadataBytes) {
      fail(ErrorKind::kBadModel, "GGUF: metadata larger than the limit of {} bytes", kMaxMetadataBytes);
    }
  }

  if (const Value* a = find("general.alignment")) {
    if (a->type != ValueType::kU32)
      fail(ErrorKind::kBadModel, "GGUF: general.alignment must be u32, not {}", to_string(a->type));
    alignment_ = a->u;
    if (alignment_ < 8 || alignment_ > kMaxAlignment || (alignment_ & (alignment_ - 1)) != 0) {
      fail(ErrorKind::kBadModel, "GGUF: general.alignment {} must be a power of two in [8, {}]", alignment_,
           kMaxAlignment);
    }
  }

  tensors_.reserve(static_cast<std::size_t>(n_tensors));
  for (std::uint64_t k = 0; k < n_tensors; ++k) {
    TensorInfo t;
    const auto at = c.pos();
    t.name = c.read_string("tensor name");
    if (t.name.empty() || t.name.size() > kMaxNameBytes || t.name.find('\0') != std::string_view::npos ||
        !valid_utf8(t.name)) {
      fail(ErrorKind::kBadModel,
           "GGUF: tensor {} at byte {}: name is empty, longer than {} bytes, contains NUL, or is not UTF-8", k, at,
           kMaxNameBytes);
    }
    t.n_dims = c.read<std::uint32_t>("tensor dimension count");
    if (t.n_dims < 1 || t.n_dims > kMaxDims) {
      fail(ErrorKind::kBadModel, "GGUF: tensor '{}' has {} dimensions (1 to {} supported)", t.name, t.n_dims, kMaxDims);
    }
    t.elements = 1;
    for (std::uint32_t d = 0; d < t.n_dims; ++d) {
      t.ne[d] = c.read<std::uint64_t>("tensor dimension");
      if (t.ne[d] == 0 || t.ne[d] > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        fail(ErrorKind::kBadModel, "GGUF: tensor '{}' dimension {} is {}", t.name, d, t.ne[d]);
      }
      t.elements = checked_mul(t.elements, t.ne[d], "tensor element count");
    }
    t.type = c.read<std::uint32_t>("tensor type");
    t.traits = type_traits(t.type);
    if (t.traits == nullptr) fail(ErrorKind::kBadModel, "GGUF: tensor '{}' has unknown type code {}", t.name, t.type);
    if (t.ne[0] % t.traits->block_size != 0) {
      fail(ErrorKind::kBadModel, "GGUF: tensor '{}': first dimension {} is not a multiple of the {} block size {}",
           t.name, t.ne[0], t.traits->name, t.traits->block_size);
    }
    t.bytes = checked_mul(t.elements / t.traits->block_size, t.traits->type_size, "tensor byte size");
    t.offset = c.read<std::uint64_t>("tensor data offset");
    if (!tensor_index_.emplace(t.name, tensors_.size()).second) {
      fail(ErrorKind::kBadModel, "GGUF: duplicate tensor name '{}'", t.name);
    }
    tensors_.push_back(t);
  }

  // As llama.cpp's gguf.cpp: the data section starts at the next multiple of
  // the alignment (only if there are tensors), and tensors are packed in
  // directory order: each one starts exactly where the previous one's
  // alignment-padded data ends. That one rule also implies aligned offsets
  // and no overlaps.
  data_offset_ = n_tensors > 0
                     ? checked_add(c.pos(), (alignment_ - c.pos() % alignment_) % alignment_, "data section offset")
                     : c.pos();
  std::uint64_t expected = 0;
  for (const TensorInfo& t : tensors_) {
    if (t.offset != expected) {
      fail(ErrorKind::kBadModel, "GGUF: tensor '{}' has data offset {}, expected {} (tensors must be packed in order)",
           t.name, t.offset, expected);
    }
    const std::uint64_t padded =
        checked_add(t.bytes, (alignment_ - t.bytes % alignment_) % alignment_, "padded tensor size");
    expected = checked_add(expected, padded, "data section size");
  }
  if (!tensors_.empty()) {
    const TensorInfo& last = tensors_.back();
    const std::uint64_t end_of_data =
        checked_add(checked_add(data_offset_, last.offset, "tensor end"), last.bytes, "tensor end");
    if (end_of_data > logical_size_) {
      fail(ErrorKind::kBadModel, "GGUF: tensor '{}' data ends at byte {}, past the end of the {}-byte file", last.name,
           end_of_data, logical_size_);
    }
  } else if (data_offset_ > logical_size_) {
    fail(ErrorKind::kBadModel, "GGUF: the data section would start past the end of the file");
  }
}

const Value* File::find(std::string_view key) const {
  const auto it = kv_index_.find(key);
  return it == kv_index_.end() ? nullptr : &kvs_[it->second].value;
}

std::uint64_t File::get_uint(std::string_view key) const {
  const Value* v = find(key);
  if (v == nullptr) fail(ErrorKind::kBadModel, "GGUF: required key '{}' is missing", key);
  if (is_unsigned(v->type)) return v->u;
  if (is_signed(v->type) && v->i >= 0) return static_cast<std::uint64_t>(v->i);
  fail(ErrorKind::kBadModel, "GGUF: key '{}' must be a non-negative integer, not {}", key, to_string(v->type));
}

double File::get_float(std::string_view key) const {
  const Value* v = find(key);
  if (v == nullptr) fail(ErrorKind::kBadModel, "GGUF: required key '{}' is missing", key);
  if (v->type == ValueType::kF32 || v->type == ValueType::kF64) return v->f;
  fail(ErrorKind::kBadModel, "GGUF: key '{}' must be a float, not {}", key, to_string(v->type));
}

std::string_view File::get_string(std::string_view key) const {
  const Value* v = find(key);
  if (v == nullptr) fail(ErrorKind::kBadModel, "GGUF: required key '{}' is missing", key);
  if (v->type != ValueType::kString)
    fail(ErrorKind::kBadModel, "GGUF: key '{}' must be a string, not {}", key, to_string(v->type));
  return v->s;
}

bool File::get_bool(std::string_view key) const {
  const Value* v = find(key);
  if (v == nullptr) fail(ErrorKind::kBadModel, "GGUF: required key '{}' is missing", key);
  if (v->type != ValueType::kBool)
    fail(ErrorKind::kBadModel, "GGUF: key '{}' must be a bool, not {}", key, to_string(v->type));
  return v->b;
}

const Value& File::get_array(std::string_view key, ValueType elem) const {
  const Value* v = find(key);
  if (v == nullptr) fail(ErrorKind::kBadModel, "GGUF: required key '{}' is missing", key);
  if (v->type != ValueType::kArray || v->elem != elem) {
    fail(ErrorKind::kBadModel, "GGUF: key '{}' must be an array of {}", key, to_string(elem));
  }
  return *v;
}

const TensorInfo* File::find_tensor(std::string_view name) const {
  const auto it = tensor_index_.find(name);
  return it == tensor_index_.end() ? nullptr : &tensors_[it->second];
}

std::span<const std::byte> File::tensor_bytes(const TensorInfo& t) const {
  const std::uint64_t begin = data_offset_ + t.offset;  // cannot overflow: checked when parsed
  if (begin + t.bytes > bytes_.size()) {
    fail(ErrorKind::kInvalidArgument, "GGUF: tensor '{}' lies beyond the bytes that were parsed", t.name);
  }
  return bytes_.subspan(static_cast<std::size_t>(begin), static_cast<std::size_t>(t.bytes));
}

}  // namespace hearth::gguf
