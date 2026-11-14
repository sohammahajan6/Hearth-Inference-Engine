#include "tokenizer/tokenizer.h"

#include <algorithm>
#include <cstring>
#include <queue>

#include "core/error.h"
#include "core/utf8.h"
#include "tokenizer/unicode_tables.h"

namespace hearth::tokenizer {
namespace {

constexpr char32_t kOutOfRange = 0xFFFFFFFF;

// Decodes one code point of valid UTF-8 at s[i]; advances i.
char32_t next_cp(std::string_view s, std::size_t& i) {
  const auto b0 = static_cast<unsigned char>(s[i]);
  if (b0 < 0x80) {
    ++i;
    return b0;
  }
  const int n = b0 >= 0xF0 ? 4 : b0 >= 0xE0 ? 3 : 2;
  char32_t c = b0 & (0x7F >> n);
  for (int k = 1; k < n; ++k) c = (c << 6) | (static_cast<unsigned char>(s[i + static_cast<std::size_t>(k)]) & 0x3F);
  i += static_cast<std::size_t>(n);
  return c;
}

void put_utf8(char32_t c, std::string& out) {
  if (c < 0x80) {
    out += static_cast<char>(c);
  } else if (c < 0x800) {
    out += static_cast<char>(0xC0 | (c >> 6));
    out += static_cast<char>(0x80 | (c & 0x3F));
  } else if (c < 0x10000) {
    out += static_cast<char>(0xE0 | (c >> 12));
    out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (c & 0x3F));
  } else {
    out += static_cast<char>(0xF0 | (c >> 18));
    out += static_cast<char>(0x80 | ((c >> 12) & 0x3F));
    out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (c & 0x3F));
  }
}

// GPT-2's bytes_to_unicode: printable bytes stand for themselves, the other
// 68 map in byte order to U+0100, U+0101, ... Built at compile time.
struct ByteAlphabet {
  char32_t to_cp[256] = {};
  std::int16_t to_byte[0x144] = {};
  constexpr ByteAlphabet() {
    for (std::int16_t& v : to_byte) v = -1;
    int next = 0;
    for (int b = 0; b < 256; ++b) {
      const bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
      to_cp[b] = printable ? static_cast<char32_t>(b) : static_cast<char32_t>(0x100 + next++);
      to_byte[to_cp[b]] = static_cast<std::int16_t>(b);
    }
  }
};
constexpr ByteAlphabet kAlphabet;
static_assert(kAlphabet.to_cp[' '] == 0x120 && kAlphabet.to_cp['\n'] == 0x10A && kAlphabet.to_byte[0x143] == 0xAD);

}  // namespace

// ------------------------------------------------------------ pretokenize

void pretokenize(std::string_view text, std::vector<std::string_view>& pieces, int max_digits) {
  HEARTH_CHECK(max_digits >= 1);
  std::vector<char32_t> cps;
  std::vector<std::size_t> off;  // byte offset of each code point, plus the end
  cps.reserve(text.size());
  off.reserve(text.size() + 1);
  for (std::size_t i = 0; i < text.size();) {
    off.push_back(i);
    cps.push_back(next_cp(text, i));
  }
  off.push_back(text.size());
  const std::size_t n = cps.size();
  const auto cp = [&](std::size_t p) { return p < n ? cps[p] : kOutOfRange; };
  const auto letter = [&](std::size_t p) { return p < n && unicode::is_letter(cps[p]); };
  const auto number = [&](std::size_t p) { return p < n && unicode::is_number(cps[p]); };
  const auto space = [&](std::size_t p) { return p < n && unicode::is_whitespace(cps[p]); };
  const auto lower = [](char32_t c) { return c >= 'A' && c <= 'Z' ? c + 32 : c; };  // ASCII is exact here

  std::size_t prev = 0;
  const auto add = [&](std::size_t end) {
    if (end > prev) pieces.push_back(text.substr(off[prev], off[end] - off[prev]));
    prev = end;
    return end;
  };
  for (std::size_t pos = 0; pos < n;) {
    const char32_t c = cps[pos];
    // (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if (c == '\'' && pos + 1 < n) {
      const char32_t a = lower(cp(pos + 1));
      if (a == 's' || a == 't' || a == 'm' || a == 'd') {
        pos = add(pos + 2);
        continue;
      }
      if (pos + 2 < n) {
        const char32_t b = lower(cp(pos + 2));
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) {
          pos = add(pos + 3);
          continue;
        }
      }
    }
    // [^\r\n\p{L}\p{N}]?\p{L}+
    if (!(c == '\r' || c == '\n' || number(pos)) && (letter(pos) || letter(pos + 1))) {
      ++pos;
      while (letter(pos)) ++pos;
      add(pos);
      continue;
    }
    // \p{N}{1,max_digits}
    if (number(pos)) {
      std::size_t start = pos;
      while (number(pos)) {
        if (++pos - start >= static_cast<std::size_t>(max_digits)) {
          add(pos);
          start = pos;
        }
      }
      add(pos);
      continue;
    }
    // <space>?[^\s\p{L}\p{N}]+[\r\n]*  (a lone trailing space also lands here, as in llama.cpp)
    const std::size_t q = c == ' ' ? pos + 1 : pos;
    if (!(space(q) || letter(q) || number(q))) {
      pos = q;
      while (pos < n && !(space(pos) || letter(pos) || number(pos))) ++pos;
      while (cp(pos) == '\r' || cp(pos) == '\n') ++pos;
      add(pos);
      continue;
    }
    std::size_t run = 0, last_rn = 0;
    while (space(pos + run)) {
      if (cps[pos + run] == '\r' || cps[pos + run] == '\n') last_rn = pos + run + 1;
      ++run;
    }
    if (last_rn > 0) {  // \s*[\r\n]+
      pos = add(last_rn);
      continue;
    }
    if (run > 1 && pos + run < n) {  // \s+(?!\S)
      pos = add(pos + run - 1);
      continue;
    }
    // \s+. Every code point is a letter, a number, whitespace or "other", and
    // all but whitespace were handled above, so the run is not empty.
    HEARTH_CHECK(run > 0);
    pos = add(pos + run);
  }
}

// ------------------------------------------------------------ Tokenizer

Tokenizer Tokenizer::from_gguf(const gguf::File& f) {
  const std::string_view model = f.get_string("tokenizer.ggml.model");
  if (model != "gpt2") fail(ErrorKind::kUnsupported, "tokenizer model '{}' is not supported (only gpt2)", model);
  const std::string_view pre = f.get_string("tokenizer.ggml.pre");
  if (pre != "llama-bpe" && pre != "qwen2") {
    fail(ErrorKind::kUnsupported, "pre-tokenizer '{}' is not supported (only llama-bpe and qwen2)", pre);
  }
  Tokenizer t;
  // llama.cpp b11960: llama-bpe = LLAMA3 (\p{N}{1,3}, ignore_merges); qwen2 =
  // QWEN2 (\p{N}, merges always applied).
  t.max_digits_ = pre == "qwen2" ? 1 : 3;
  t.ignore_merges_ = pre == "llama-bpe";
  const auto& tokens = f.get_array("tokenizer.ggml.tokens", gguf::ValueType::kString).strings;
  const gguf::Value& types = f.get_array("tokenizer.ggml.token_type", gguf::ValueType::kI32);
  if (types.count != tokens.size()) {
    fail(ErrorKind::kBadModel, "tokenizer: {} token types for {} tokens", types.count, tokens.size());
  }
  if (tokens.empty() || tokens.size() > 0x7FFFFFFF) {
    fail(ErrorKind::kBadModel, "tokenizer: {} tokens", tokens.size());
  }
  const std::size_t n = tokens.size();
  t.text_.reserve(n);
  t.bytes_.resize(n);
  t.special_.assign(n, 0);
  std::vector<std::uint8_t> raw_text(n, 0);  // user-defined and unused: not in the byte-level alphabet's sense
  for (std::size_t i = 0; i < n; ++i) {
    std::int32_t type;
    std::memcpy(&type, types.raw + 4 * i, 4);
    t.text_.emplace_back(tokens[i]);
    if (type == 3) {
      t.special_[i] = 1;
    } else if (type == 4) {
      // Matched whole in any text, before pre-tokenization (llama.cpp's
      // tokenizer_st_partition), and printed as its text.
      raw_text[i] = 1;
      t.user_defined_.push_back(static_cast<std::int32_t>(i));
    } else if (type == 5) {
      raw_text[i] = 2;  // never produced by BPE (no merge leads to it); prints nothing
    } else if (type != 1) {
      fail(ErrorKind::kUnsupported,
           "tokenizer: token {} '{}' has type {}; only normal (1), control (3), user-defined (4) and unused (5) are "
           "supported",
           i, tokens[i], type);
    }
  }
  // Longest first, as llama.cpp sorts its special-token cache (stable: equal
  // lengths keep vocabulary order).
  std::stable_sort(t.user_defined_.begin(), t.user_defined_.end(), [&](std::int32_t a, std::int32_t b) {
    return t.text_[static_cast<std::size_t>(a)].size() > t.text_[static_cast<std::size_t>(b)].size();
  });
  t.id_of_.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    const auto id = static_cast<std::int32_t>(i);
    if (!t.id_of_.emplace(t.text_[i], id).second) {
      fail(ErrorKind::kBadModel, "tokenizer: token {} duplicates the text of token {}", i, t.id_of_[t.text_[i]]);
    }
    if (t.special_[i]) continue;
    if (raw_text[i] == 1) {
      t.bytes_[i] = t.text_[i];
      continue;
    }
    if (raw_text[i] == 2) continue;
    // A normal token is a string in the byte-level alphabet: decode it once.
    const std::string_view s = t.text_[i];
    for (std::size_t k = 0; k < s.size();) {
      const char32_t c = next_cp(s, k);
      if (c >= 0x144 || kAlphabet.to_byte[c] < 0) {
        fail(ErrorKind::kBadModel, "tokenizer: token {} contains U+{:04X}, outside the byte-level alphabet", i,
             static_cast<std::uint32_t>(c));
      }
      t.bytes_[i] += static_cast<char>(kAlphabet.to_byte[c]);
    }
  }
  for (int b = 0; b < 256; ++b) {
    std::string s;
    put_utf8(kAlphabet.to_cp[b], s);
    const auto it = t.id_of_.find(s);
    if (it == t.id_of_.end() || t.special_[static_cast<std::size_t>(it->second)]) {
      fail(ErrorKind::kBadModel, "tokenizer: byte 0x{:02X} has no token", b);
    }
    t.byte_token_[b] = it->second;
  }
  const auto& merges = f.get_array("tokenizer.ggml.merges", gguf::ValueType::kString).strings;
  t.merges_.reserve(merges.size());
  for (std::size_t r = 0; r < merges.size(); ++r) {
    const std::string_view m = merges[r];
    const auto sp = m.find(' ');
    if (sp == std::string_view::npos || m.find(' ', sp + 1) != std::string_view::npos) {
      fail(ErrorKind::kBadModel, "tokenizer: merge {} '{}' does not have exactly one space", r, m);
    }
    const std::string_view left = m.substr(0, sp), right = m.substr(sp + 1);
    const auto l = t.id_of_.find(left), rr = t.id_of_.find(right);
    const auto res = t.id_of_.find(std::string(left) + std::string(right));
    if (l == t.id_of_.end() || rr == t.id_of_.end() || res == t.id_of_.end()) {
      fail(ErrorKind::kBadModel, "tokenizer: merge {} '{}' names or makes a string that is not a token", r, m);
    }
    for (const auto& it : {l, rr, res}) {
      if (t.special_[static_cast<std::size_t>(it->second)]) {
        fail(ErrorKind::kBadModel, "tokenizer: merge {} '{}' involves control token {}", r, m, it->second);
      }
    }
    const std::uint64_t key = (std::uint64_t(std::uint32_t(l->second)) << 32) | std::uint32_t(rr->second);
    if (!t.merges_.emplace(key, Merge{static_cast<std::uint32_t>(r), res->second}).second) {
      fail(ErrorKind::kBadModel, "tokenizer: merge {} '{}' repeats an earlier merge", r, m);
    }
  }
  const auto id = [&](const char* key) {
    const std::uint64_t v = f.get_uint(key);
    if (v >= n) fail(ErrorKind::kBadModel, "tokenizer: {} = {} is not below the vocabulary size {}", key, v, n);
    return static_cast<std::int32_t>(v);
  };
  t.bos_ = id("tokenizer.ggml.bos_token_id");
  t.eos_ = id("tokenizer.ggml.eos_token_id");
  t.add_bos_ = f.find("tokenizer.ggml.add_bos_token") != nullptr && f.get_bool("tokenizer.ggml.add_bos_token");
  // End of generation: EOS plus the end markers llama.cpp recognizes by name:
  // Llama 3's, and Qwen's (<|im_end|>, <|endoftext|>, and its FIM pad, repo
  // and file-separator tokens, which llama.cpp b11960 also adds).
  t.eog_.push_back(t.eos_);
  for (const char* s : {"<|eot_id|>", "<|eom_id|>", "<|end_of_text|>", "<|im_end|>", "<|endoftext|>", "<|fim_pad|>",
                        "<|repo_name|>", "<|file_sep|>"}) {
    if (const auto s_id = t.special(s); s_id && *s_id != t.eos_) t.eog_.push_back(*s_id);
  }
  return t;
}

bool Tokenizer::is_special(std::int32_t id) const {
  HEARTH_CHECK(id >= 0 && id < n_vocab());
  return special_[static_cast<std::size_t>(id)] != 0;
}

bool Tokenizer::is_eog(std::int32_t id) const {
  for (std::int32_t e : eog_) {
    if (e == id) return true;
  }
  return false;
}

std::optional<std::int32_t> Tokenizer::special(std::string_view s) const {
  const auto it = id_of_.find(s);
  if (it == id_of_.end() || !special_[static_cast<std::size_t>(it->second)]) return std::nullopt;
  return it->second;
}

std::string_view Tokenizer::text(std::int32_t id) const {
  HEARTH_CHECK(id >= 0 && id < n_vocab());
  return text_[static_cast<std::size_t>(id)];
}

void Tokenizer::encode(std::string_view s, std::vector<std::int32_t>& out) const {
  if (const auto bad = first_invalid_utf8(s)) {
    fail(ErrorKind::kInvalidArgument, "text is not valid UTF-8 at byte {}", *bad);
  }
  if (user_defined_.empty()) return encode_plain(s, out);
  // llama.cpp's tokenizer_st_partition: for each user-defined token, longest
  // first, split every text fragment at its occurrences, left to right; the
  // remaining text fragments are encoded on their own.
  struct Fragment {
    std::string_view text;
    std::int32_t token;  // -1: text
  };
  std::vector<Fragment> frags{{s, -1}};
  for (const std::int32_t id : user_defined_) {
    const std::string_view word = text_[static_cast<std::size_t>(id)];
    if (word.empty()) continue;
    std::vector<Fragment> next;
    for (const Fragment& f : frags) {
      if (f.token >= 0) {
        next.push_back(f);
        continue;
      }
      std::string_view rest = f.text;
      for (std::size_t at = rest.find(word); at != std::string_view::npos; at = rest.find(word)) {
        if (at > 0) next.push_back({rest.substr(0, at), -1});
        next.push_back({{}, id});
        rest.remove_prefix(at + word.size());
      }
      if (!rest.empty()) next.push_back({rest, -1});
    }
    frags = std::move(next);
  }
  for (const Fragment& f : frags) {
    if (f.token >= 0) {
      out.push_back(f.token);
    } else {
      encode_plain(f.text, out);
    }
  }
}

void Tokenizer::encode_plain(std::string_view s, std::vector<std::int32_t>& out) const {
  std::vector<std::string_view> pieces;
  pretokenize(s, pieces, max_digits_);
  for (std::string_view p : pieces) bpe(p, out);
}

std::vector<std::int32_t> Tokenizer::encode(std::string_view s, bool bos) const {
  std::vector<std::int32_t> out;
  if (bos) out.push_back(bos_);
  encode(s, out);
  return out;
}

void Tokenizer::bpe(std::string_view piece, std::vector<std::int32_t>& out) const {
  // ignore_merges: a piece that is itself a (normal) token is emitted whole.
  std::string mapped;
  for (char ch : piece) put_utf8(kAlphabet.to_cp[static_cast<unsigned char>(ch)], mapped);
  if (const auto it = id_of_.find(mapped);
      ignore_merges_ && it != id_of_.end() && !special_[static_cast<std::size_t>(it->second)]) {
    out.push_back(it->second);
    return;
  }
  // One symbol per byte; repeatedly merge the adjacent pair with the lowest
  // rank, the leftmost on a tie (a lazily invalidated priority queue).
  struct Sym {
    std::int32_t id;
    int prev, next;
  };
  std::vector<Sym> sym(piece.size());
  for (std::size_t i = 0; i < piece.size(); ++i) {
    sym[i] = {byte_token_[static_cast<unsigned char>(piece[i])], static_cast<int>(i) - 1,
              i + 1 < piece.size() ? static_cast<int>(i) + 1 : -1};
  }
  struct Cand {
    std::uint32_t rank;
    int left, right;
    std::int32_t lid, rid, result;
  };
  const auto worse = [](const Cand& a, const Cand& b) {
    return a.rank > b.rank || (a.rank == b.rank && a.left > b.left);
  };
  std::priority_queue<Cand, std::vector<Cand>, decltype(worse)> queue(worse);
  const auto consider = [&](int l, int r) {
    if (l < 0 || r < 0) return;
    const std::uint64_t key = (std::uint64_t(std::uint32_t(sym[l].id)) << 32) | std::uint32_t(sym[r].id);
    if (const auto it = merges_.find(key); it != merges_.end()) {
      queue.push({it->second.rank, l, r, sym[l].id, sym[r].id, it->second.result});
    }
  };
  for (int i = 0; i + 1 < static_cast<int>(sym.size()); ++i) consider(i, i + 1);
  while (!queue.empty()) {
    const Cand c = queue.top();
    queue.pop();
    Sym& l = sym[c.left];
    if (l.next != c.right || l.id != c.lid || sym[c.right].id != c.rid) continue;  // stale
    const int after = sym[c.right].next;
    l.id = c.result;
    l.next = after;
    sym[c.right].id = -1;
    if (after >= 0) sym[after].prev = c.left;
    consider(l.prev, c.left);
    consider(c.left, after);
  }
  for (int i = 0; i >= 0; i = sym[i].next) out.push_back(sym[i].id);
}

void Tokenizer::decode(std::int32_t id, std::string& out, bool special) const {
  HEARTH_CHECK(id >= 0 && id < n_vocab());
  const auto i = static_cast<std::size_t>(id);
  if (special_[i]) {
    if (special) out += text_[i];
    return;
  }
  out += bytes_[i];
}

// ------------------------------------------------------------ Utf8Stream

std::string Utf8Stream::push(std::string_view bytes) {
  pending_ += bytes;
  std::string out;
  std::size_t i = 0;
  const std::size_t n = pending_.size();
  while (i < n) {
    const auto b0 = static_cast<unsigned char>(pending_[i]);
    if (b0 < 0x80) {
      out += static_cast<char>(b0);
      ++i;
      continue;
    }
    // Expected length and the allowed range of the second byte (Unicode 15.1, table 3-7).
    int len = 0;
    unsigned char lo = 0x80, hi = 0xBF;
    if (b0 >= 0xC2 && b0 <= 0xDF) {
      len = 2;
    } else if (b0 >= 0xE0 && b0 <= 0xEF) {
      len = 3;
      if (b0 == 0xE0) lo = 0xA0;
      if (b0 == 0xED) hi = 0x9F;
    } else if (b0 >= 0xF0 && b0 <= 0xF4) {
      len = 4;
      if (b0 == 0xF0) lo = 0x90;
      if (b0 == 0xF4) hi = 0x8F;
    }
    if (len == 0) {  // not a lead byte
      out += "\xEF\xBF\xBD";
      ++i;
      continue;
    }
    std::size_t k = 1;
    bool bad = false;
    while (k < static_cast<std::size_t>(len) && i + k < n) {
      const auto b = static_cast<unsigned char>(pending_[i + k]);
      if (k == 1 ? (b < lo || b > hi) : (b < 0x80 || b > 0xBF)) {
        bad = true;
        break;
      }
      ++k;
    }
    if (bad) {  // the maximal subpart [i, i + k) becomes one U+FFFD
      out += "\xEF\xBF\xBD";
      i += k;
      continue;
    }
    if (k < static_cast<std::size_t>(len)) break;  // incomplete so far: hold back
    out.append(pending_, i, static_cast<std::size_t>(len));
    i += static_cast<std::size_t>(len);
  }
  pending_.erase(0, i);
  return out;
}

std::string Utf8Stream::flush() {
  std::string out = pending_.empty() ? std::string() : std::string("\xEF\xBF\xBD");
  pending_.clear();
  return out;
}

}  // namespace hearth::tokenizer
