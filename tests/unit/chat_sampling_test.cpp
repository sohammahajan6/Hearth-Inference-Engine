// The chat template's structure on a tiny vocabulary, the template's date and
// trim rules, and the sampler; on the real model, a multi-turn conversation
// against IDs recorded from llama.cpp. (Single-turn prompts on the real model
// are checked against llama.cpp by scripts/chat-check.ps1.)
#include <doctest/doctest.h>

#include <cmath>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include "core/error.h"
#include "gguf/gguf.h"
#include "gguf_builder.h"
#include "sampling/sampler.h"
#include "test_util.h"
#include "tokenizer/chat.h"
#include "tokenizer/tokenizer.h"

using namespace hearth;

namespace {

std::string byte_char(int b) {
  int cp = b;
  const auto printable = [](int x) {
    return (x >= '!' && x <= '~') || (x >= 0xA1 && x <= 0xAC) || (x >= 0xAE && x <= 0xFF);
  };
  if (!printable(b)) {
    int n = 0;
    for (int x = 0; x < b; ++x) n += printable(x) ? 0 : 1;
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

// The 256 byte tokens, then BOS and the three Llama-3 chat control tokens.
std::vector<std::byte> chat_vocab(const std::string& chat_template) {
  std::vector<std::string> tokens;
  for (int b = 0; b < 256; ++b) tokens.push_back(byte_char(b));
  std::vector<std::int32_t> types(256, 1);
  for (const char* s : {"<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"}) {
    tokens.emplace_back(s);
    types.push_back(3);
  }
  return test::GgufBuilder()
      .kv_string("tokenizer.ggml.model", "gpt2")
      .kv_string("tokenizer.ggml.pre", "llama-bpe")
      .kv_strings("tokenizer.ggml.tokens", tokens)
      .kv_i32s("tokenizer.ggml.token_type", types)
      .kv_strings("tokenizer.ggml.merges", {})
      .kv_u32("tokenizer.ggml.bos_token_id", 256)
      .kv_u32("tokenizer.ggml.eos_token_id", 259)
      .kv_bool("tokenizer.ggml.add_bos_token", true)
      .kv_string("tokenizer.chat_template", chat_template)
      .build();
}

// A Qwen2.5-style vocabulary: byte tokens, then <|endoftext|> (256),
// <|im_start|> (257), <|im_end|> (258); pre-tokenizer qwen2, no BOS added.
std::vector<std::byte> chatml_vocab(const std::string& chat_template) {
  std::vector<std::string> tokens;
  for (int b = 0; b < 256; ++b) tokens.push_back(byte_char(b));
  std::vector<std::int32_t> types(256, 1);
  for (const char* s : {"<|endoftext|>", "<|im_start|>", "<|im_end|>"}) {
    tokens.emplace_back(s);
    types.push_back(3);
  }
  return test::GgufBuilder()
      .kv_string("tokenizer.ggml.model", "gpt2")
      .kv_string("tokenizer.ggml.pre", "qwen2")
      .kv_strings("tokenizer.ggml.tokens", tokens)
      .kv_i32s("tokenizer.ggml.token_type", types)
      .kv_strings("tokenizer.ggml.merges", {})
      .kv_u32("tokenizer.ggml.bos_token_id", 256)
      .kv_u32("tokenizer.ggml.eos_token_id", 258)
      .kv_bool("tokenizer.ggml.add_bos_token", false)
      .kv_string("tokenizer.chat_template", chat_template)
      .build();
}

// The parts of Qwen2.5's template that ChatTemplate::detect looks for.
const char* kQwenMarkers =
    "{{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}"
    "{{- '<|im_start|>assistant\\n' }}";

const char* kLlamaMarkers =
    "{{- bos_token }}<|start_header_id|>system<|end_header_id|> Cutting Knowledge Date: December 2023\n"
    "Today Date: {{ date_string }} <|eot_id|>";

}  // namespace

TEST_SUITE("quick") {
  TEST_CASE("chat template: Llama-3.1/3.2 blocks by ID, trimmed content, date line") {
    const auto bytes = chat_vocab(kLlamaMarkers);
    const gguf::File f = gguf::File::parse(bytes);
    const auto tok = tokenizer::Tokenizer::from_gguf(f);
    const auto tmpl = tokenizer::ChatTemplate::detect(f, tok);
    const std::int32_t bos = 256, start = 257, end = 258, eot = 259;
    const auto enc = [&](const std::string& s) { return tok.encode(s, false); };
    const auto ids = tmpl.prompt(tok, " Be brief.\n", {{"user", " Hi! "}, {"assistant", "Hello."}}, "04 Nov 2026");
    std::vector<std::int32_t> want{bos, start};
    const auto add = [&](const std::vector<std::int32_t>& v) { want.insert(want.end(), v.begin(), v.end()); };
    add(enc("system"));
    want.push_back(end);
    add(enc("\n\nCutting Knowledge Date: December 2023\nToday Date: 04 Nov 2026\n\nBe brief."));
    want.insert(want.end(), {eot, start});
    add(enc("user"));
    want.push_back(end);
    add(enc("\n\nHi!"));
    want.insert(want.end(), {eot, start});
    add(enc("assistant"));
    want.push_back(end);
    add(enc("\n\nHello."));
    want.insert(want.end(), {eot, start});
    add(enc("assistant"));
    want.push_back(end);
    add(enc("\n\n"));
    CHECK(ids == want);
    // A further turn closes the reply, adds the user block and a new assistant header.
    std::vector<std::int32_t> next{eot, start};
    const auto cat = [&](const std::vector<std::int32_t>& v) { next.insert(next.end(), v.begin(), v.end()); };
    cat(enc("user"));
    next.push_back(end);
    cat(enc("\n\nAnd you?"));
    next.insert(next.end(), {eot, start});
    cat(enc("assistant"));
    next.push_back(end);
    cat(enc("\n\n"));
    CHECK(tmpl.next_turn(tok, "And you?\t") == next);
    CHECK_THROWS_WITH_AS(tmpl.prompt(tok, "", {{"tool", "x"}}, "04 Nov 2026"), doctest::Contains("role 'tool'"), Error);
    // Special-token text in a message stays ordinary text.
    for (std::int32_t id : enc("<|eot_id|>")) CHECK_FALSE(tok.is_special(id));
  }

  TEST_CASE("chat template: Qwen2.5's ChatML by ID, default system prompt, no BOS, content not trimmed") {
    const auto bytes = chatml_vocab(kQwenMarkers);
    const gguf::File f = gguf::File::parse(bytes);
    const auto tok = tokenizer::Tokenizer::from_gguf(f);
    const auto tmpl = tokenizer::ChatTemplate::detect(f, tok);
    const std::int32_t start = 257, end = 258;
    const auto enc = [&](const std::string& s) { return tok.encode(s, false); };
    std::vector<std::int32_t> want;
    const auto add = [&](const std::vector<std::int32_t>& v) { want.insert(want.end(), v.begin(), v.end()); };
    const auto block = [&](const std::string& text) {
      want.push_back(start);
      add(enc(text));
      want.push_back(end);
      add(enc("\n"));
    };
    block("system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.");
    block("user\n Hi! ");
    block("assistant\nHello.");
    want.push_back(start);
    add(enc("assistant\n"));
    CHECK(tmpl.prompt(tok, "", {{"user", " Hi! "}, {"assistant", "Hello."}}, "ignored") == want);
    // A system prompt replaces the default.
    want.clear();
    block("system\nBe brief.");
    block("user\nx");
    want.push_back(start);
    add(enc("assistant\n"));
    CHECK(tmpl.prompt(tok, "Be brief.", {{"user", "x"}}, "ignored") == want);
    // The next turn closes the reply with <|im_end|> and a newline first.
    want = {end};
    add(enc("\n"));
    block("user\nAnd you?");
    want.push_back(start);
    add(enc("assistant\n"));
    CHECK(tmpl.next_turn(tok, "And you?") == want);
  }

  // Step W3: ChatML templates other than Qwen2.5's (SmolLM2, many
  // fine-tunes) write a system turn only when one is given.
  TEST_CASE("chat template: other ChatML templates have no default system prompt") {
    const auto bytes = chatml_vocab(
        "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>' + '\\n' }}"
        "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}");
    const gguf::File f = gguf::File::parse(bytes);
    const auto tok = tokenizer::Tokenizer::from_gguf(f);
    const auto tmpl = tokenizer::ChatTemplate::detect(f, tok);
    const std::int32_t start = 257, end = 258;
    const auto enc = [&](const std::string& s) { return tok.encode(s, false); };
    std::vector<std::int32_t> want;
    const auto add = [&](const std::vector<std::int32_t>& v) { want.insert(want.end(), v.begin(), v.end()); };
    const auto block = [&](const std::string& text) {
      want.push_back(start);
      add(enc(text));
      want.push_back(end);
      add(enc("\n"));
    };
    block("user\nHi");
    want.push_back(start);
    add(enc("assistant\n"));
    CHECK(tmpl.prompt(tok, "", {{"user", "Hi"}}, "ignored") == want);
    want.clear();
    block("system\nBe brief.");
    block("user\nHi");
    want.push_back(start);
    add(enc("assistant\n"));
    CHECK(tmpl.prompt(tok, "Be brief.", {{"user", "Hi"}}, "ignored") == want);
  }

  // Step W3: Llama 3.0's template has no date lines, and a system block only
  // when there is a system message.
  TEST_CASE("chat template: Llama 3.0 without the date lines") {
    const auto bytes = chat_vocab(
        "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + "
        "message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == "
        "0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}");
    const gguf::File f = gguf::File::parse(bytes);
    const auto tok = tokenizer::Tokenizer::from_gguf(f);
    const auto tmpl = tokenizer::ChatTemplate::detect(f, tok);
    const std::int32_t bos = 256, start = 257, end = 258, eot = 259;
    const auto enc = [&](const std::string& s) { return tok.encode(s, false); };
    std::vector<std::int32_t> want{bos};
    const auto add = [&](const std::vector<std::int32_t>& v) { want.insert(want.end(), v.begin(), v.end()); };
    const auto block = [&](const std::string& role, const std::string& text) {
      want.push_back(start);
      add(enc(role));
      want.push_back(end);
      add(enc("\n\n" + text));
      want.push_back(eot);
    };
    block("user", "Hi!");
    want.push_back(start);
    add(enc("assistant"));
    want.push_back(end);
    add(enc("\n\n"));
    CHECK(tmpl.prompt(tok, "", {{"user", " Hi! "}}, "04 Nov 2026") == want);
    want = {bos};
    block("system", "Be brief.");
    block("user", "Hi!");
    want.push_back(start);
    add(enc("assistant"));
    want.push_back(end);
    add(enc("\n\n"));
    CHECK(tmpl.prompt(tok, " Be brief. ", {{"user", "Hi!"}}, "04 Nov 2026") == want);
  }

  TEST_CASE("chat template: other templates are not guessed at") {
    {
      // ChatML without <|im_end|>.
      const auto bytes = chat_vocab("{% for m in messages %}<|im_start|>{{ m.role }}{% endfor %}");
      const gguf::File f = gguf::File::parse(bytes);
      const auto tok = tokenizer::Tokenizer::from_gguf(f);
      CHECK_THROWS_WITH_AS((void)tokenizer::ChatTemplate::detect(f, tok), doctest::Contains("not a standard one"),
                           Error);
    }
    {
      const auto bytes = chat_vocab("{% for m in messages %}[INST] {{ m.content }} [/INST]{% endfor %}");
      const gguf::File f = gguf::File::parse(bytes);
      const auto tok = tokenizer::Tokenizer::from_gguf(f);
      CHECK_THROWS_WITH_AS((void)tokenizer::ChatTemplate::detect(f, tok),
                           doctest::Contains("neither a Llama 3 one nor ChatML"), Error);
    }
  }

  TEST_CASE("chat template: date format and validation, trim rules") {
    CHECK(tokenizer::template_date(2026, 11, 4) == "04 Nov 2026");
    CHECK(tokenizer::template_date("2024-02-29") == "29 Feb 2024");
    CHECK(tokenizer::template_date("2000-02-29") == "29 Feb 2000");
    CHECK_THROWS_AS((void)tokenizer::template_date("1900-02-29"), Error);
    CHECK_THROWS_AS((void)tokenizer::template_date("2023-02-29"), Error);
    CHECK_THROWS_AS((void)tokenizer::template_date("2026-13-01"), Error);
    CHECK_THROWS_AS((void)tokenizer::template_date("26-11-04"), Error);
    CHECK_THROWS_AS((void)tokenizer::template_date("2026-11-4x"), Error);
    CHECK(tokenizer::template_date_today().size() == 11);
    CHECK(tokenizer::trim(" \t\n\v\f\rx y\r\n ") == "x y");
    CHECK(tokenizer::trim(" x ") == " x ");  // C isspace only: NBSP stays
    CHECK(tokenizer::trim("") == "");
  }

  TEST_CASE("sampler: greedy, seeds, top-k, top-p and the distribution") {
    const std::vector<float> ties{1.0f, 3.0f, 3.0f, 2.0f};
    CHECK(sampling::greedy(ties) == 1);  // the lowest ID among equal maxima
    sampling::Sampler g({0.0f, 40, 0.95f, 1});
    CHECK(g.sample(ties) == 1);

    const std::vector<float> l{0.0f, 0.5f, 1.0f, 1.5f, 2.0f, -1.0f};
    sampling::Sampler a({1.0f, 0, 1.0f, 42}), b({1.0f, 0, 1.0f, 42}), c({1.0f, 0, 1.0f, 43});
    std::vector<std::int32_t> sa, sb, sc;
    for (int i = 0; i < 200; ++i) {
      sa.push_back(a.sample(l));
      sb.push_back(b.sample(l));
      sc.push_back(c.sample(l));
    }
    CHECK(sa == sb);  // the same seed gives the same draws
    CHECK(sa != sc);

    sampling::Sampler k1({5.0f, 1, 1.0f, 7});  // top-k 1 is greedy at any temperature
    for (int i = 0; i < 100; ++i) CHECK(k1.sample(l) == 4);

    // Probabilities 0.5, 0.3, 0.2 at temperature 1.
    const std::vector<float> p{std::log(0.5f), std::log(0.3f), std::log(0.2f)};
    sampling::Sampler half({1.0f, 0, 0.45f, 9});  // the first alone (0.5) reaches top-p 0.45
    sampling::Sampler six({1.0f, 0, 0.6f, 9});    // needs the first two (0.8)
    int third = 0;
    for (int i = 0; i < 2000; ++i) {
      CHECK(half.sample(p) == 0);
      third += six.sample(p) == 2;
    }
    CHECK(third == 0);

    // softmax({0, log 3}) = {0.25, 0.75}: 20000 draws stay within 5 sigma.
    sampling::Sampler d({1.0f, 0, 1.0f, 2026});
    const std::vector<float> two{0.0f, std::log(3.0f)};
    int ones = 0;
    for (int i = 0; i < 20000; ++i) ones += d.sample(two);
    const double frac = ones / 20000.0, sigma = std::sqrt(0.75 * 0.25 / 20000.0);
    CHECK(std::fabs(frac - 0.75) <= 5 * sigma);
    CHECK_THROWS_AS((void)sampling::Sampler({1.0f, 0, 0.0f, 1}), Error);
  }
}

TEST_SUITE("model") {
  TEST_CASE("sampler: bad temperatures and NaN logits are clear errors") {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    for (const float t : {nan, -1.0f, inf}) {
      CHECK_THROWS_AS(sampling::Sampler({t, 0, 1.0f, 1}), Error);
    }
    sampling::Sampler s({1.0f, 0, 1.0f, 1});
    const std::vector<float> bad{0.5f, nan, 0.1f};
    CHECK_THROWS_WITH_AS(s.sample(bad), doctest::Contains("NaN"), Error);
  }

  TEST_CASE("chat template: a multi-turn conversation gives llama.cpp's IDs, also turn by turn") {
    const auto path = test::repo_dir() / "models" / "Llama-3.2-1B-Instruct-Q8_0-pure.gguf";
    if (!std::filesystem::exists(path)) {
      MESSAGE("missing (run scripts\\fetch-models.ps1): Llama-3.2-1B-Instruct-Q8_0-pure.gguf");
      return;
    }
    const gguf::File f = gguf::File::open(path);
    const auto tok = tokenizer::Tokenizer::from_gguf(f);
    const auto tmpl = tokenizer::ChatTemplate::detect(f, tok);
    // Recorded from hetero-oracle-chat (llama.cpp b11960's Jinja engine and server tokenization) for
    // [system "Be brief.", user "Hi!", assistant "Hello! How can I help you today?", user "What is 2+2?"]
    // on 2026-11-04.
    const std::vector<std::int32_t> want{
        128000, 128006, 9125, 128007, 271, 38766,  1303,   33025,  2696,   25,    6790,   220,  2366,  18,
        198,    15724,  2696, 25,     220, 2371,   4723,   220,    2366,   21,    271,    3513, 10015, 13,
        128009, 128006, 882,  128007, 271, 13347,  0,      128009, 128006, 78191, 128007, 271,  9906,  0,
        2650,   649,    358,  1520,   499, 3432,   30,     128009, 128006, 882,   128007, 271,  3923,  374,
        220,    17,     10,   17,     30,  128009, 128006, 78191,  128007, 271};
    const std::string reply = "Hello! How can I help you today?";
    const std::string date = tokenizer::template_date(2026, 11, 4);
    CHECK(tmpl.prompt(tok, "Be brief.", {{"user", "Hi!"}, {"assistant", reply}, {"user", "What is 2+2?"}}, date) ==
          want);
    // What `hearth chat` feeds: the first turn's prompt, the reply's tokens (here
    // the canonical ones), then next_turn for the second user message.
    auto turns = tmpl.prompt(tok, "Be brief.", {{"user", "Hi!"}}, date);
    tok.encode(reply, turns);
    const auto next = tmpl.next_turn(tok, "What is 2+2?");
    turns.insert(turns.end(), next.begin(), next.end());
    CHECK(turns == want);
  }
}
