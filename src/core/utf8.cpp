#include "core/utf8.h"

#include <cstdint>

namespace hearth {

std::optional<std::size_t> first_invalid_utf8(std::string_view s) noexcept {
  const auto* begin = reinterpret_cast<const unsigned char*>(s.data());
  const auto* p = begin;
  const unsigned char* end = p + s.size();
  while (p < end) {
    const unsigned c = *p;
    if (c < 0x80) {
      ++p;
      continue;
    }
    const auto at = static_cast<std::size_t>(p - begin);
    int len = 0;
    std::uint32_t cp = 0;
    std::uint32_t min = 0;
    if ((c & 0xE0) == 0xC0) {
      len = 2;
      cp = c & 0x1F;
      min = 0x80;
    } else if ((c & 0xF0) == 0xE0) {
      len = 3;
      cp = c & 0x0F;
      min = 0x800;
    } else if ((c & 0xF8) == 0xF0) {
      len = 4;
      cp = c & 0x07;
      min = 0x10000;
    } else {
      return at;  // continuation byte without a lead, or 0xF8-0xFF
    }
    if (end - p < len) return at;
    for (int k = 1; k < len; ++k) {
      if ((p[k] & 0xC0) != 0x80) return at;
      cp = (cp << 6) | (p[k] & 0x3Fu);
    }
    if (cp < min || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return at;
    p += len;
  }
  return std::nullopt;
}

}  // namespace hearth
