// The character classes of the Llama-3 pre-tokenizer, generated from the
// Unicode Character Database 15.1.0.
#pragma once

#include <cstddef>
#include <cstdint>

namespace hearth::tokenizer::unicode {

struct Range {
  char32_t first, last;  // inclusive
};

extern const Range kLetters[];
extern const std::size_t kLettersCount;
extern const Range kNumbers[];
extern const std::size_t kNumbersCount;
extern const char32_t kWhiteSpace[];
extern const std::size_t kWhiteSpaceCount;

bool is_letter(char32_t c) noexcept;      // \p{L}
bool is_number(char32_t c) noexcept;      // \p{N}
bool is_whitespace(char32_t c) noexcept;  // \s (White_Space)

}  // namespace hearth::tokenizer::unicode
