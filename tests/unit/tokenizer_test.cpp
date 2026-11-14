// The Llama-3 tokenizer: every pre-tokenizer rule, the BPE mechanics on a
// small synthetic vocabulary, loader strictness, Utf8Stream, and (model tier)
// the real vocabulary against IDs recorded from llama-tokenize b11960.
#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "core/error.h"
#include "core/utf8.h"
#include "gguf/gguf.h"
#include "gguf_builder.h"
#include "test_util.h"
#include "tokenizer/tokenizer.h"

using namespace hearth;
using tokenizer::Tokenizer;
using tokenizer::Utf8Stream;

namespace {

std::vector<std::string> split(std::string_view text) {
  std::vector<std::string_view> pieces;
  tokenizer::pretokenize(text, pieces);
  return {pieces.begin(), pieces.end()};
}

// GPT-2's byte alphabet, written independently of the tokenizer's.
std::string byte_char(int b) {
  int cp = b;
  if (!((b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF))) {
    int n = 0;
    for (int x = 0; x < b; ++x) {
      if (!((x >= '!' && x <= '~') || (x >= 0xA1 && x <= 0xAC) || (x >= 0xAE && x <= 0xFF))) ++n;
    }
    cp = 0x100 + n;
  }
  std::string s;
  if (cp < 0x80) {
    s += static_cast<char>(cp);
  } else {
    s += static_cast<char>(0xC0 | (cp >> 6));
    s += static_cast<char>(0x80 | (cp & 0x3F));
  }
  return s;
}

// A tiny byte-level vocabulary: the 256 byte tokens, a few merges, two
// control tokens. ID 256 + i is the i-th extra token.
struct Tiny {
  std::vector<std::string> tokens;
  std::vector<std::int32_t> types;
  std::vector<std::string> merges;
};

Tiny tiny_vocab_and_merges() {
  Tiny t;
  for (int b = 0; b < 256; ++b) t.tokens.push_back(byte_char(b));
  const std::string sp = byte_char(' ');  // "Ġ"
  for (const std::string& s : {std::string("he"), std::string("ll"), std::string("hell"), std::string("hello"),
                               sp + "w", std::string("or"), sp + "wor", std::string("ld"), sp + "world"}) {
    t.tokens.push_back(s);
  }
  t.types.assign(t.tokens.size(), 1);
  t.tokens.push_back("<|end|>");
  t.tokens.push_back("<|bos|>");
  t.types.push_back(3);
  t.types.push_back(3);
  t.merges = {"h e", "l l", "he ll", "hell o", sp + " w", "o r", sp + "w or", "l d", sp + "wor ld"};
  return t;
}

std::vector<std::byte> tiny_gguf(const Tiny& t, const std::string& pre = "llama-bpe") {
  return test::GgufBuilder()
      .kv_string("tokenizer.ggml.model", "gpt2")
      .kv_string("tokenizer.ggml.pre", pre)
      .kv_strings("tokenizer.ggml.tokens", t.tokens)
      .kv_i32s("tokenizer.ggml.token_type", t.types)
      .kv_strings("tokenizer.ggml.merges", t.merges)
      .kv_u32("tokenizer.ggml.bos_token_id", 266)
      .kv_u32("tokenizer.ggml.eos_token_id", 265)
      .kv_bool("tokenizer.ggml.add_bos_token", true)
      .build();
}

void expect_load_error(const Tiny& t, const char* needle, ErrorKind kind = ErrorKind::kBadModel,
                       const std::string& pre = "llama-bpe") {
  const auto bytes = tiny_gguf(t, pre);
  const gguf::File f = gguf::File::parse(bytes);
  try {
    (void)Tokenizer::from_gguf(f);
    FAIL("loaded, but expected an error mentioning '" << needle << "'");
  } catch (const Error& e) {
    CHECK(e.kind() == kind);
    const std::string what = e.what();
    CHECK_MESSAGE(what.find(needle) != std::string::npos, what);
  }
}

std::string decode_all(const Tokenizer& tok, const std::vector<std::int32_t>& ids) {
  std::string out;
  for (std::int32_t id : ids) tok.decode(id, out);
  return out;
}

}  // namespace

TEST_SUITE("quick") {
  TEST_CASE("pretokenize: every rule of the Llama-3 split") {
    using V = std::vector<std::string>;
    CHECK(split("") == V{});
    CHECK(split("Hello world") == V{"Hello", " world"});
    // Contractions, case-insensitive; other apostrophes start a word or a punctuation run.
    CHECK(split("it's") == V{"it", "'s"});
    CHECK(split("IT'S") == V{"IT", "'S"});
    CHECK(split("we're they've I'M he'd") == V{"we", "'re", " they", "'ve", " I", "'M", " he", "'d"});
    CHECK(split("y'all") == V{"y", "'all"});
    CHECK(split("'s") == V{"'s"});
    CHECK(split("'") == V{"'"});
    CHECK(split("' s") == V{"'", " s"});
    // Words take one optional prefix that is not \r, \n, a letter or a number.
    CHECK(split("\tword") == V{"\tword"});
    CHECK(split("\nword") == V{"\n", "word"});
    CHECK(split("a b") == V{"a", " b"});  // NBSP is a valid prefix
    CHECK(split("éx") == V{"e", "́x"});    // a combining mark is not a letter
    CHECK(split("café 中文") == V{"café", " 中文"});
    // Numbers in groups of at most three, in any script; never joined to words.
    CHECK(split("12345") == V{"123", "45"});
    CHECK(split("abc123def") == V{"abc", "123", "def"});
    CHECK(split("1a a1") == V{"1", "a", " a", "1"});
    CHECK(split("١٢٣٤") == V{"١٢٣", "٤"});
    CHECK(split(" 1") == V{" ", "1"});
    // Punctuation runs, with one optional leading space and trailing newlines.
    CHECK(split("Hi!!! there") == V{"Hi", "!!!", " there"});
    CHECK(split("a !!\n\nb") == V{"a", " !!\n\n", "b"});
    CHECK(split("$5") == V{"$", "5"});
    CHECK(split("\U0001F44D\U0001F44D") == V{"\U0001F44D\U0001F44D"});
    // Whitespace runs.
    CHECK(split("a  b") == V{"a", " ", " b"});  // \s+(?!\S) leaves one space for the word
    CHECK(split("  hello") == V{" ", " hello"});
    CHECK(split("a\n\nb") == V{"a", "\n\n", "b"});  // up to the last newline
    CHECK(split("a \nb") == V{"a", " \n", "b"});
    CHECK(split("\r\n\r\nX") == V{"\r\n\r\n", "X"});
    CHECK(split("x  ") == V{"x", "  "});  // a trailing run is kept whole
    CHECK(split("x ") == V{"x", " "});
  }

  TEST_CASE("tokenizer: BPE merges by rank, leftmost on a tie, and ignore_merges") {
    const auto bytes = tiny_gguf(tiny_vocab_and_merges());
    const gguf::File f = gguf::File::parse(bytes);
    const Tokenizer tok = Tokenizer::from_gguf(f);
    CHECK(tok.n_vocab() == 267);
    CHECK(tok.bos() == 266);
    CHECK(tok.add_bos());
    // Whole pieces that are tokens are emitted as they are.
    CHECK(tok.encode("hello world", false) == std::vector<std::int32_t>{259, 264});
    // "hellos": h e -> he, l l -> ll, he ll -> hell, hell o -> hello; s stays a byte.
    CHECK(tok.encode("hellos", false) == std::vector<std::int32_t>{259, 's'});
    // Equal ranks: the leftmost pair merges first ("ll" + "l", not "l" + "ll").
    CHECK(tok.encode("lll", false) == std::vector<std::int32_t>{257, 'l'});
    // A lower rank wins over a position further left.
    CHECK(tok.encode("llhe", false) == std::vector<std::int32_t>{257, 256});
    // Special-token text is ordinary text.
    const auto ids = tok.encode("<|end|>", true);
    CHECK(ids.front() == 266);
    for (std::size_t i = 1; i < ids.size(); ++i) CHECK_FALSE(tok.is_special(ids[i]));
    // Decoding: bytes back, special tokens only on request.
    std::string s;
    tok.decode(264, s);
    CHECK(s == " world");
    tok.decode(265, s);
    CHECK(s == " world");
    tok.decode(265, s, true);
    CHECK(s == " world<|end|>");
    CHECK(tok.is_eog(265));
    CHECK(tok.special("<|end|>") == 265);
    CHECK_FALSE(tok.special("hello").has_value());
    // Any valid UTF-8 round-trips through the byte tokens.
    const std::string text = "héllo \U0001F642\t\r\n  x";
    CHECK(decode_all(tok, tok.encode(text, false)) == text);
    CHECK_THROWS_WITH_AS(tok.encode("ok\xC3", false), doctest::Contains("not valid UTF-8 at byte 2"), Error);
  }

  TEST_CASE("tokenizer: Qwen2's rules: single digits, merges always, user-defined and unused tokens") {
    using V = std::vector<std::string>;
    // \p{N} (Qwen2) against \p{N}{1,3} (Llama 3); the rest of the regex is the same.
    std::vector<std::string_view> pieces;
    tokenizer::pretokenize("a 12345", pieces, 1);
    CHECK(V(pieces.begin(), pieces.end()) == V{"a", " ", "1", "2", "3", "4", "5"});
    CHECK(split("a 12345") == V{"a", " ", "123", "45"});
    Tiny t = tiny_vocab_and_merges();
    // 267 "xyz" (normal, no merge leads to it), 268 "<tc>" (user-defined), 269 "[PAD]" (unused),
    // 270 "<|endoftext|>" and 271 "<|im_end|>" (control).
    for (const auto& [text, type] :
         {std::pair<std::string, int>{"xyz", 1}, {"<tc>", 4}, {"[PAD]", 5}, {"<|endoftext|>", 3}, {"<|im_end|>", 3}}) {
      t.tokens.push_back(text);
      t.types.push_back(type);
    }
    const auto llama_bytes = tiny_gguf(t, "llama-bpe"), qwen_bytes = tiny_gguf(t, "qwen2");
    const gguf::File lf = gguf::File::parse(llama_bytes), qf = gguf::File::parse(qwen_bytes);
    const Tokenizer llama = Tokenizer::from_gguf(lf), qwen = Tokenizer::from_gguf(qf);
    // ignore_merges: Llama 3 emits a piece that is a token whole; Qwen2 always merges.
    CHECK(llama.encode("xyz", false) == std::vector<std::int32_t>{267});
    CHECK(qwen.encode("xyz", false) == std::vector<std::int32_t>{'x', 'y', 'z'});
    // User-defined tokens are matched whole anywhere in the text, before
    // pre-tokenization (llama.cpp's tokenizer_st_partition), for both.
    for (const Tokenizer* tok : {&llama, &qwen}) {
      CHECK(tok->encode("a<tc>b<tc>", false) == std::vector<std::int32_t>{'a', 268, 'b', 268});
      CHECK(tok->encode("<tc", false) == std::vector<std::int32_t>{'<', 't', 'c'});
      CHECK_FALSE(tok->is_special(268));
    }
    // Decoding: a user-defined token is its text; an unused one is nothing.
    std::string s;
    qwen.decode(268, s);
    CHECK(s == "<tc>");
    qwen.decode(269, s);
    qwen.decode(269, s, true);
    CHECK(s == "<tc>");
    // Qwen's end markers end a reply, as in llama.cpp.
    CHECK(qwen.is_eog(265));
    CHECK(qwen.is_eog(270));
    CHECK(qwen.is_eog(271));
    CHECK_FALSE(qwen.is_eog(266));
  }

  TEST_CASE("tokenizer: the loader rejects inconsistent or foreign vocabularies") {
    const Tiny good = tiny_vocab_and_merges();
    expect_load_error(good, "pre-tokenizer 'gpt-4o' is not supported", ErrorKind::kUnsupported, "gpt-4o");
    Tiny t = good;
    t.tokens[260] = t.tokens[1];  // two tokens with the same text
    expect_load_error(t, "token 260 duplicates the text of token 1");
    t = good;
    t.merges.push_back("he ll o");
    expect_load_error(t, "does not have exactly one space");
    t = good;
    t.merges.push_back("hello s");  // "hellos" is not a token
    expect_load_error(t, "is not a token");
    t = good;
    t.merges.push_back("h e");
    expect_load_error(t, "repeats an earlier merge");
    t = good;
    t.types[260] = 6;  // byte tokens (SPM vocabularies); user-defined (4) and unused (5) are supported (Qwen2)
    expect_load_error(t, "has type 6", ErrorKind::kUnsupported);
    t = good;
    t.tokens[65] = "AA";  // byte 'A' has no single-byte token any more
    expect_load_error(t, "byte 0x41 has no token");
    t = good;
    t.tokens[261] = "Ѐ";  // outside the byte-level alphabet
    expect_load_error(t, "token 261 contains U+0400, outside the byte-level alphabet");
    t = good;
    t.types.pop_back();
    expect_load_error(t, "token types for");
  }

  TEST_CASE("Utf8Stream: valid text survives any split; invalid bytes become U+FFFD") {
    const std::string text = "aé中\U0001F642z";
    for (std::size_t cut = 0; cut <= text.size(); ++cut) {
      for (std::size_t cut2 = cut; cut2 <= text.size(); ++cut2) {
        Utf8Stream s;
        std::string out = s.push(text.substr(0, cut));
        out += s.push(text.substr(cut, cut2 - cut));
        out += s.push(text.substr(cut2));
        out += s.flush();
        CHECK(out == text);
      }
    }
    const std::string r = "\xEF\xBF\xBD";
    const auto once = [](std::string_view bytes) {
      Utf8Stream s;
      std::string out = s.push(bytes);
      return out + s.flush();
    };
    CHECK(once("\x80") == r);                          // a lone continuation byte
    CHECK(once("\xC0\xAF") == r + r);                  // C0 is never a lead byte
    CHECK(once("\xE0\x80\x80") == r + r + r);          // E0 needs A0..BF next
    CHECK(once("\xED\xA0\x80") == r + r + r);          // a surrogate
    CHECK(once("\xF0\x9F\x99") == r);                  // truncated at the end: one maximal subpart
    CHECK(once("\xF0\x9F\x99x") == r + "x");           // truncated before an ASCII byte
    CHECK(once("\xF4\x90\x80\x80") == r + r + r + r);  // above U+10FFFF
    CHECK(once("ok\xFF!") == "ok" + r + "!");
    Utf8Stream s;
    CHECK(s.push("\xE4\xB8") == "");  // incomplete: held back
    CHECK(s.push("\xAD") == "中");
    CHECK(s.flush() == "");
  }
}

TEST_SUITE("model") {
  TEST_CASE("tokenizer: the real Llama-3.2 vocabulary gives llama-tokenize's IDs") {
    const auto path = test::repo_dir() / "models" / "Llama-3.2-1B-Instruct-Q8_0-pure.gguf";
    if (!std::filesystem::exists(path)) {
      MESSAGE("missing (run scripts\\fetch-models.ps1): Llama-3.2-1B-Instruct-Q8_0-pure.gguf");
      return;
    }
    const gguf::File f = gguf::File::open(path);
    const Tokenizer tok = Tokenizer::from_gguf(f);
    CHECK(tok.n_vocab() == 128256);
    CHECK(tok.bos() == 128000);
    CHECK(tok.eos() == 128009);
    CHECK(tok.is_eog(128009));  // <|eot_id|>
    CHECK(tok.is_eog(128008));  // <|eom_id|>
    CHECK(tok.is_eog(128001));  // <|end_of_text|>
    CHECK_FALSE(tok.is_eog(12366));
    // Recorded from llama-tokenize (b11960 CPU zip) --ids --no-escape --no-parse-special.
    CHECK(tok.encode("The capital of France is", true) == std::vector<std::int32_t>{128000, 791, 6864, 315, 9822, 374});
    CHECK(tok.encode("Hello world! It's 2024-11-04, I've 12345 apples.", true) ==
          std::vector<std::int32_t>{128000, 9906, 1917, 0,   1102, 596, 220,  2366, 19,    12, 806,
                                    12,     2371, 11,   358, 3077, 220, 4513, 1774, 41776, 13});
    CHECK(tok.encode("  leading   spaces\ttab\r\nCRLF end ", true) ==
          std::vector<std::int32_t>{128000, 220, 6522, 256, 12908, 59249, 319, 34, 81758, 842, 220});
    // The adversarial corpus round-trips byte for byte.
    std::ifstream in(test::repo_dir() / "tests" / "fixtures" / "tokenizer" / "adversarial.txt", std::ios::binary);
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    REQUIRE(text.size() > 1000);
    const auto ids = tok.encode(text, false);
    CHECK(ids.size() == 2466);  // llama-tokenize: 2467 with BOS
    CHECK(decode_all(tok, ids) == text);
  }
}
