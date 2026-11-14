#include "gguf/types.h"

namespace hearth::gguf {
namespace {

struct Entry {
  std::uint32_t code;
  TypeTraits traits;
};

// From gguf-py GGML_QUANT_SIZES at llama.cpp b11960. Codes 4, 5, 31-33 and
// 36-38 belonged to formats removed from ggml and are not valid in files.
constexpr Entry kTypes[] = {
    {0, {"F32", 1, 4, true}},          {1, {"F16", 1, 2, true}},          {2, {"Q4_0", 32, 18, true}},
    {3, {"Q4_1", 32, 20, true}},       {6, {"Q5_0", 32, 22, true}},       {7, {"Q5_1", 32, 24, true}},
    {8, {"Q8_0", 32, 34, true}},       {9, {"Q8_1", 32, 36, false}},      {10, {"Q2_K", 256, 84, true}},
    {11, {"Q3_K", 256, 110, true}},    {12, {"Q4_K", 256, 144, true}},    {13, {"Q5_K", 256, 176, true}},
    {14, {"Q6_K", 256, 210, true}},    {15, {"Q8_K", 256, 292, false}},   {16, {"IQ2_XXS", 256, 66, false}},
    {17, {"IQ2_XS", 256, 74, false}},  {18, {"IQ3_XXS", 256, 98, false}}, {19, {"IQ1_S", 256, 50, false}},
    {20, {"IQ4_NL", 32, 18, false}},   {21, {"IQ3_S", 256, 110, false}},  {22, {"IQ2_S", 256, 82, false}},
    {23, {"IQ4_XS", 256, 136, false}}, {24, {"I8", 1, 1, false}},         {25, {"I16", 1, 2, false}},
    {26, {"I32", 1, 4, false}},        {27, {"I64", 1, 8, false}},        {28, {"F64", 1, 8, false}},
    {29, {"IQ1_M", 256, 56, false}},   {30, {"BF16", 1, 2, true}},        {34, {"TQ1_0", 256, 54, false}},
    {35, {"TQ2_0", 256, 66, false}},   {39, {"MXFP4", 32, 17, false}},    {40, {"NVFP4", 64, 36, false}},
    {41, {"Q1_0", 128, 18, false}},    {42, {"Q2_0", 64, 18, false}},
};

}  // namespace

const TypeTraits* type_traits(std::uint32_t code) noexcept {
  for (const Entry& e : kTypes) {
    if (e.code == code) return &e.traits;
  }
  return nullptr;
}

}  // namespace hearth::gguf
