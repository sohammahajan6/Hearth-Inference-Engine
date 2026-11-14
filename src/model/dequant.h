// Converting one weight row to float, for every tensor type the engine loads.
#pragma once

#include <cstdint>
#include <span>

#include "gguf/gguf.h"

namespace hearth::model {

// Writes row `row` of a 2-D tensor (ne[0] contiguous values) as float.
// `data` is the tensor's bytes (gguf::File::tensor_bytes). For 1-D tensors
// row must be 0. Supported types: F32, F16, BF16, Q8_0, Q4_0, Q4_K, Q6_K
// (each exactly as ggml dequantizes it).
void dequant_row(const gguf::TensorInfo& t, std::span<const std::byte> data, std::int64_t row, float* out);

}  // namespace hearth::model
