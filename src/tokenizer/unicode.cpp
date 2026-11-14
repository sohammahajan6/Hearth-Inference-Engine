// Lookups into the generated tables (unicode_tables.cpp).
#include <algorithm>

#include "tokenizer/unicode_tables.h"

namespace hearth::tokenizer::unicode {
namespace {

bool in_ranges(const Range* r, std::size_t n, char32_t c) noexcept {
  // The last range whose start is <= c; then is c inside it?
  const Range* end = r + n;
  const Range* it = std::upper_bound(r, end, c, [](char32_t v, const Range& x) { return v < x.first; });
  return it != r && c <= (it - 1)->last;
}

}  // namespace

bool is_letter(char32_t c) noexcept {
  if (c < 0x80) return (c | 0x20) >= 'a' && (c | 0x20) <= 'z';
  return in_ranges(kLetters, kLettersCount, c);
}

bool is_number(char32_t c) noexcept {
  if (c < 0x80) return c >= '0' && c <= '9';
  return in_ranges(kNumbers, kNumbersCount, c);
}

bool is_whitespace(char32_t c) noexcept { return std::binary_search(kWhiteSpace, kWhiteSpace + kWhiteSpaceCount, c); }

}  // namespace hearth::tokenizer::unicode
