// The code under test for the tokenizer fuzzer. Shared by the libFuzzer build
// (fuzz_tokenizer.cpp, clang + ASan/UBSan), the replay build and the
// model-tier regression test. The vocabulary is the real Llama-3.2 one.
//
// Input: any bytes. The first byte chooses where the bytes are split before
// they are fed to Utf8Stream; the rest is the text.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "core/error.h"
#include "core/utf8.h"
#include "tokenizer/tokenizer.h"

namespace hearth::fuzz {

// Violations throw kInternal (HEARTH_CHECK), which callers treat as a finding.
inline void tokenizer_one_input(const tokenizer::Tokenizer& tok, const std::uint8_t* data, std::size_t size) {
  if (size == 0) return;
  const std::size_t stride = 1 + data[0] % 7;
  const std::string_view text(reinterpret_cast<const char*>(data + 1), size - 1);

  // Utf8Stream: valid UTF-8 out for any input; valid input comes out unchanged.
  tokenizer::Utf8Stream stream;
  std::string streamed;
  for (std::size_t i = 0; i < text.size(); i += stride) streamed += stream.push(text.substr(i, stride));
  streamed += stream.flush();
  HEARTH_CHECK(valid_utf8(streamed));
  const auto bad = first_invalid_utf8(text);
  HEARTH_CHECK(bad.has_value() || streamed == text);

  if (bad) {  // invalid UTF-8 is a clean error naming the first bad byte
    try {
      std::vector<std::int32_t> ids;
      tok.encode(text, ids);
    } catch (const Error& e) {
      HEARTH_CHECK(e.kind() == ErrorKind::kInvalidArgument);
      HEARTH_CHECK(std::string_view(e.what()).find("at byte " + std::to_string(*bad)) != std::string_view::npos);
      return;
    }
    fail(ErrorKind::kInternal, "encode accepted invalid UTF-8 (first bad byte {})", *bad);
  }
  // Pre-tokenizer pieces tile the text exactly.
  std::vector<std::string_view> pieces;
  tokenizer::pretokenize(text, pieces);
  std::size_t at = 0;
  for (std::string_view p : pieces) {
    HEARTH_CHECK(!p.empty() && p.data() == text.data() + at);
    at += p.size();
  }
  HEARTH_CHECK(at == text.size());
  // Encoding: normal tokens only, and decoding gives the text back.
  std::vector<std::int32_t> ids;
  tok.encode(text, ids);
  std::string decoded;
  for (std::int32_t id : ids) {
    HEARTH_CHECK(id >= 0 && id < tok.n_vocab() && !tok.is_special(id));
    tok.decode(id, decoded);
  }
  HEARTH_CHECK(decoded == text);
}

}  // namespace hearth::fuzz
