// ggml tensor types as they appear in GGUF files.
//
// The code -> (block size, bytes per block) table covers every type ggml
// defines at the pinned llama.cpp commit (copied from gguf-py's
// GGML_QUANT_SIZES and checked on 2026-11-03), so any file can be described
// and sized, including tensors this engine cannot compute with.
#pragma once

#include <cstdint>

namespace hearth::gguf {

enum class GgmlType : std::uint32_t {
  kF32 = 0,
  kF16 = 1,
  kQ4_0 = 2,
  kQ4_1 = 3,
  kQ5_0 = 6,
  kQ5_1 = 7,
  kQ8_0 = 8,
  kQ2_K = 10,
  kQ3_K = 11,
  kQ4_K = 12,
  kQ5_K = 13,
  kQ6_K = 14,
  kBF16 = 30,
};

struct TypeTraits {
  const char* name;
  std::uint32_t block_size;  // elements per block
  std::uint32_t type_size;   // bytes per block
  bool supported;            // this engine can load tensors of this type
};

// nullptr for a code ggml does not define (e.g. the removed 4, 5, 31-33, 36-38).
const TypeTraits* type_traits(std::uint32_t code) noexcept;

}  // namespace hearth::gguf
