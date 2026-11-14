// Byte-level BPE tokenizer of Llama 3 and Qwen2 (docs/architecture/06-tokenizer-chat-sampling.md,
// 15-qwen.md). Produces the same token IDs as llama.cpp b11960 for valid UTF-8
// text, as llama-tokenize does without --parse-special: control-token text
// is treated as ordinary text (control tokens are inserted by ID only), while
// user-defined tokens (Qwen's <tool_call>) are always matched whole.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "gguf/gguf.h"

namespace hearth::tokenizer {

// Splits valid UTF-8 `text` into pre-tokenizer pieces (views into `text`),
// with the Llama-3 regex semantics of llama.cpp's custom splitter. Digits go
// in runs of at most `max_digits`: 3 for Llama 3 (\p{N}{1,3}), 1 for Qwen2
// (\p{N}; its regex is otherwise the same).
void pretokenize(std::string_view text, std::vector<std::string_view>& pieces, int max_digits = 3);

class Tokenizer {
 public:
  // Reads `tokenizer.ggml.*` from a GGUF file. Supports model "gpt2" with
  // pre-tokenizer "llama-bpe" or "qwen2" (kUnsupported otherwise); token types
  // normal (1), control (3), user-defined (4) and unused (5); kBadModel if the
  // vocabulary or merges are inconsistent. Copies everything it needs.
  static Tokenizer from_gguf(const gguf::File& file);

  // Move-only: the lookup table holds views into the vocabulary strings,
  // which a move keeps in place and a copy would not.
  Tokenizer(Tokenizer&&) noexcept = default;
  Tokenizer& operator=(Tokenizer&&) noexcept = default;
  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;

  std::int32_t n_vocab() const noexcept { return static_cast<std::int32_t>(text_.size()); }
  std::int32_t bos() const noexcept { return bos_; }
  std::int32_t eos() const noexcept { return eos_; }
  bool add_bos() const noexcept { return add_bos_; }
  bool is_special(std::int32_t id) const;  // control token (type 3)
  bool is_eog(std::int32_t id) const;      // ends a generated reply
  // The ID of special token `text` ("<|eot_id|>"), if the vocabulary has it.
  std::optional<std::int32_t> special(std::string_view text) const;

  // Appends the token IDs of `text` to `out`. `text` must be valid UTF-8
  // (kInvalidArgument naming the byte offset otherwise).
  void encode(std::string_view text, std::vector<std::int32_t>& out) const;
  // BOS (if `bos`) followed by the IDs of `text`.
  std::vector<std::int32_t> encode(std::string_view text, bool bos) const;

  // Appends the bytes token `id` stands for. A control token contributes its
  // text if `special`, nothing otherwise; a user-defined token always its
  // text; an unused token nothing (as llama.cpp's token_to_piece). Bytes may be partial UTF-8: stream
  // them through Utf8Stream before printing.
  void decode(std::int32_t id, std::string& out, bool special = false) const;
  // The vocabulary string of `id` (in the byte-level alphabet for normal tokens).
  std::string_view text(std::int32_t id) const;

 private:
  Tokenizer() = default;
  void bpe(std::string_view piece, std::vector<std::int32_t>& out) const;
  // Pre-tokenizes and BPE-encodes text that holds no user-defined token.
  void encode_plain(std::string_view s, std::vector<std::int32_t>& out) const;

  struct Merge {
    std::uint32_t rank;
    std::int32_t result;
  };
  std::vector<std::string> text_;                             // vocabulary strings, by ID
  std::vector<std::string> bytes_;                            // what each normal token decodes to
  std::vector<std::uint8_t> special_;                         // 1 for control tokens
  std::vector<std::int32_t> user_defined_;                    // user-defined tokens, longest text first
  std::unordered_map<std::string_view, std::int32_t> id_of_;  // views into text_
  std::unordered_map<std::uint64_t, Merge> merges_;           // (left << 32 | right)
  std::int32_t byte_token_[256] = {};                         // the single-byte token of each byte
  std::vector<std::int32_t> eog_;
  std::int32_t bos_ = -1, eos_ = -1;
  bool add_bos_ = false;
  int max_digits_ = 3;         // pretokenize's digit runs
  bool ignore_merges_ = true;  // a piece that is a token is emitted whole (llama.cpp: llama-bpe yes, qwen2 no)
};

// Turns a stream of byte chunks (decoded tokens) into valid UTF-8: an
// incomplete sequence is held back until a later chunk completes it; a
// sequence that cannot become valid is replaced by U+FFFD, one per maximal
// invalid subpart (Unicode 15.1, section 3.9 "U+FFFD Substitution").
class Utf8Stream {
 public:
  std::string push(std::string_view bytes);
  std::string flush();  // an unfinished tail becomes U+FFFD

 private:
  std::string pending_;
};

}  // namespace hearth::tokenizer
