// UTF-8 helpers.
#pragma once

#include <cstddef>
#include <optional>
#include <string_view>

namespace hearth {

// The byte offset of the first ill-formed sequence in `s`, or nullopt if `s`
// is well-formed UTF-8 (RFC 3629): no overlong encodings, no surrogates
// (U+D800-DFFF), nothing above U+10FFFF, no truncated sequences.
std::optional<std::size_t> first_invalid_utf8(std::string_view s) noexcept;

inline bool valid_utf8(std::string_view s) noexcept { return !first_invalid_utf8(s); }

}  // namespace hearth
