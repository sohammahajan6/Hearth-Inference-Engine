// Hardware reference sequences for the bit-exactness tests
// (unit/fp16_hw_test.cpp, kernels/quant_hw_test.cpp). They live in their own
// AVX2-compiled file, which includes no shared header: an inline function
// (from core/ or doctest) compiled with AVX2 flags could otherwise be linked
// into the whole test binary.
// Callers check cpu_features() before calling them.
#pragma once

#include <cstdint>

namespace hearth::test {

// F16C VCVTPS2PH (round to nearest even) and VCVTPH2PS, one value each.
std::uint16_t hw_fp32_to_fp16(float f);
float hw_fp16_to_fp32(std::uint16_t h);

// ggml's AVX2 Q8_0 quantization of one 32-value block.
void ggml_quantize_block(const float* x, std::uint16_t& d_bits, std::int8_t* q);

}  // namespace hearth::test
