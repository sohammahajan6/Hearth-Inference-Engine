// The code under test for the GGUF fuzzer. Shared by the libFuzzer build
// (fuzz_gguf.cpp, clang + ASan/UBSan), the replay build (same file, any
// compiler) and the quick-tier regression test.
//
// Input layout: an 8-byte little-endian "logical file size", then the first
// bytes of a GGUF file. The logical size lets small inputs describe real
// model shapes (gigabytes of tensor data) without containing that data, so
// fuzzed headers get past the bounds checks into the config and binder code.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

#include "core/error.h"
#include "core/json_writer.h"
#include "gguf/gguf.h"
#include "model/llama.h"

namespace hearth::fuzz {

inline void gguf_one_input(const std::uint8_t* data, std::size_t size) {
  if (size < 8) return;
  std::uint64_t logical = 0;
  std::memcpy(&logical, data, 8);
  logical &= (std::uint64_t{1} << 40) - 1;  // at most 1 TiB: a large but real file size
  const std::span<const std::byte> bytes(reinterpret_cast<const std::byte*>(data + 8), size - 8);
  try {
    const gguf::File f = gguf::File::parse(bytes, logical);
    // `hearth info --json` escapes every key and string; hostile ones included.
    JsonWriter w(false);
    w.begin_array();
    for (const gguf::Kv& kv : f.metadata()) {
      w.value(kv.key);
      if (kv.value.type == gguf::ValueType::kString) w.value(kv.value.s);
      for (std::string_view s : kv.value.strings) w.value(s);
    }
    w.end_array();
    for (const gguf::TensorInfo& t : f.tensors()) {
      try {
        (void)f.tensor_bytes(t);  // throws kInvalidArgument for tensors beyond the parsed bytes
      } catch (const Error& e) {
        if (e.kind() == ErrorKind::kInternal) throw;
      }
    }
    const model::LlamaConfig c = model::LlamaConfig::from_gguf(f);
    (void)model::LlamaTensors::bind(f, c);
  } catch (const Error& e) {
    // A clean rejection is the expected outcome for most inputs. A failed
    // internal check (kInternal) is a bug, and so is anything else escaping
    // (std::bad_alloc, std::length_error, ...): both reach libFuzzer as a crash.
    if (e.kind() == ErrorKind::kInternal) throw;
  }
}

}  // namespace hearth::fuzz
