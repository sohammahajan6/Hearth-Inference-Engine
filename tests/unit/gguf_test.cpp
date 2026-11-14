// The GGUF parser and Llama binder: every strictness rule of
// docs/architecture/04-gguf-loader.md has a test that violates exactly that rule.
#include <doctest/doctest.h>

#include <filesystem>
#include <limits>
#include <string>

#include "core/error.h"
#include "gguf/gguf.h"
#include "gguf_builder.h"
#include "model/llama.h"
#include "test_util.h"

using hearth::Error;
using hearth::ErrorKind;
using hearth::gguf::File;
using hearth::gguf::ValueType;
using hearth::test::GgufBuilder;
using hearth::test::tiny_llama;

namespace {

// Parses and expects an Error of `kind` whose message contains `needle`.
void expect_reject(const std::vector<std::byte>& bytes, const char* needle, ErrorKind kind = ErrorKind::kBadModel) {
  try {
    (void)File::parse(bytes);
    FAIL("parsed, but expected an error mentioning '" << needle << "'");
  } catch (const Error& e) {
    CHECK(e.kind() == kind);
    const std::string what = e.what();  // (a const char* message would print as a bool)
    CHECK_MESSAGE(what.find(needle) != std::string::npos, what);
  }
}

std::vector<std::byte> raw(std::initializer_list<int> v) {
  std::vector<std::byte> b;
  for (int x : v) b.push_back(static_cast<std::byte>(x));
  return b;
}

GgufBuilder one_tensor(std::vector<std::uint64_t> ne, std::uint32_t type, std::uint64_t bytes,
                       std::int64_t offset = -1) {
  GgufBuilder b;
  b.tensor({"t", std::move(ne), type, bytes, offset});
  return b;
}

}  // namespace

TEST_SUITE("quick") {
  TEST_CASE("gguf: a valid file round-trips every value type") {
    GgufBuilder b;
    b.kv_raw("u8", 0, raw({200}))
        .kv_raw("i8", 1, raw({0xFE}))
        .kv_raw("u16", 2, GgufBuilder::bytes_of(std::uint16_t{60000}))
        .kv_raw("i16", 3, GgufBuilder::bytes_of(std::int16_t{-30000}))
        .kv_u32("u32", 4000000000u)
        .kv_i32("i32", -5)
        .kv_f32("f32", 0.25f)
        .kv_bool("bool", true)
        .kv_string("str", "h\xC3\xA9llo")
        .kv_raw("u64", 10, GgufBuilder::bytes_of(std::uint64_t{1} << 40))
        .kv_raw("i64", 11, GgufBuilder::bytes_of(std::int64_t{-1} << 40))
        .kv_raw("f64", 12, GgufBuilder::bytes_of(0.1))
        .kv_u32s("arr", {1, 2, 3})
        .kv_strings("strs", {"x", "", "yz"});
    const auto bytes = b.build();
    const File f = File::parse(bytes);
    CHECK(f.version() == 3);
    CHECK(f.metadata().size() == 14);
    CHECK(f.get_uint("u8") == 200);
    CHECK(f.find("i8")->i == -2);
    CHECK(f.get_uint("u16") == 60000);
    CHECK(f.find("i16")->i == -30000);
    CHECK(f.get_uint("u32") == 4000000000u);
    CHECK(f.find("i32")->i == -5);
    CHECK(f.get_float("f32") == 0.25);
    CHECK(f.get_bool("bool"));
    CHECK(f.get_string("str") == "h\xC3\xA9llo");
    CHECK(f.get_uint("u64") == (std::uint64_t{1} << 40));
    CHECK(f.find("i64")->i == (std::int64_t{-1} << 40));
    CHECK(f.get_float("f64") == 0.1);
    CHECK(f.get_array("arr", ValueType::kU32).count == 3);
    const auto& strs = f.get_array("strs", ValueType::kString).strings;
    REQUIRE(strs.size() == 3);
    CHECK(strs[2] == "yz");
    CHECK_THROWS_AS(f.get_uint("i32"), Error);    // negative
    CHECK_THROWS_AS(f.get_string("u32"), Error);  // wrong type
    CHECK_THROWS_AS(f.get_uint("missing"), Error);
  }

  TEST_CASE("gguf: header rules") {
    expect_reject(GgufBuilder().magic("GGUG").build(), "not a GGUF file");
    expect_reject(GgufBuilder().version(1).build(), "version 1");
    expect_reject(GgufBuilder().version(4).build(), "version 4");
    expect_reject(GgufBuilder().kv_count(std::uint64_t{1} << 62).build(), "cannot fit");
    expect_reject(GgufBuilder().tensor_count(std::uint64_t{1} << 62).build(), "cannot fit");
    expect_reject(raw({'G', 'G', 'U', 'F', 3, 0}), "truncated");
  }

  TEST_CASE("gguf: every truncation of a valid file is rejected cleanly") {
    const auto bytes = tiny_llama().build();
    for (std::size_t n = 0; n < bytes.size(); ++n) {
      const std::vector<std::byte> prefix(bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(n));
      try {
        (void)File::parse(prefix);
        FAIL("prefix of " << n << " bytes parsed");
      } catch (const Error& e) {
        REQUIRE(e.kind() == ErrorKind::kBadModel);
      }
    }
    CHECK_NOTHROW((void)File::parse(bytes));
  }

  TEST_CASE("gguf: key and value rules") {
    expect_reject(GgufBuilder().kv_u32("a", 1).kv_u32("a", 2).build(), "duplicate metadata key 'a'");
    expect_reject(GgufBuilder().kv_u32("", 1).build(), "is empty, longer than");
    expect_reject(GgufBuilder().kv_u32("bad\xC3", 1).build(), "or is not valid UTF-8");
    expect_reject(GgufBuilder().kv_u32("\xC0\xAF", 1).build(), "or is not valid UTF-8");  // overlong '/'
    expect_reject(GgufBuilder().kv_raw("t", 13, raw({0})).build(), "unknown value type 13");
    expect_reject(GgufBuilder().kv_raw("b", 7, raw({2})).build(), "not 0 or 1");
    expect_reject(GgufBuilder().kv_string("s", "\xED\xA0\x80").build(), "not valid UTF-8");  // surrogate
    // Array of arrays.
    std::vector<std::byte> nested = GgufBuilder::bytes_of(std::uint32_t{9});
    const auto count = GgufBuilder::bytes_of(std::uint64_t{0});
    nested.insert(nested.end(), count.begin(), count.end());
    expect_reject(GgufBuilder().kv_raw("n", 9, nested).build(), "arrays of arrays");
    // A string array claiming more elements than the file could hold (below
    // the hard cap, so the bytes-remaining bound is what rejects it).
    std::vector<std::byte> huge = GgufBuilder::bytes_of(std::uint32_t{8});
    const auto n = GgufBuilder::bytes_of(std::uint64_t{1} << 20);
    huge.insert(huge.end(), n.begin(), n.end());
    expect_reject(GgufBuilder().kv_raw("h", 9, huge).build(), "cannot fit");
    // A numeric array whose data is missing.
    std::vector<std::byte> wide = GgufBuilder::bytes_of(std::uint32_t{10});
    const auto m = GgufBuilder::bytes_of(std::uint64_t{1} << 20);
    wide.insert(wide.end(), m.begin(), m.end());
    expect_reject(GgufBuilder().kv_raw("w", 9, wide).build(), "array data at byte");
  }

  TEST_CASE("gguf: hard limits and NUL bytes (review findings)") {
    expect_reject(GgufBuilder().kv_count(5000).build(),
                  "cannot fit");  // also > kMaxKvs, but fails the size bound first
    std::vector<std::byte> many = GgufBuilder::bytes_of(std::uint32_t{0});  // u8 array
    const auto n = GgufBuilder::bytes_of((std::uint64_t{1} << 24) + 1);
    many.insert(many.end(), n.begin(), n.end());
    expect_reject(GgufBuilder().kv_raw("a", 9, many).build(), "exceeds the limit");
    expect_reject(GgufBuilder().kv_u32(std::string(257, 'k'), 1).build(), "longer than 256 bytes");
    expect_reject(GgufBuilder().kv_u32(std::string("a\0b", 3), 1).build(), "contains NUL");
    expect_reject(GgufBuilder().tensor({std::string("t\0", 2), {8}, 0, 32}).build(), "contains NUL");
    expect_reject(GgufBuilder().version(0x03000000u).build(), "big-endian");
    // A file with no tensors has no padding before its (empty) data section,
    // as llama.cpp's vocabulary-only files show: header (24 bytes) + one key
    // "x" (8 + 1) + type (4) + string "abcde" (8 + 5) = 50 bytes, not a
    // multiple of the 32-byte alignment, and still valid.
    const auto built = GgufBuilder().kv_string("x", "abcde").build();  // the builder pads to 64
    const File f = File::parse(std::span<const std::byte>(built.data(), 50));
    CHECK(f.tensors().empty());
    CHECK(f.data_offset() == 50);
  }

  TEST_CASE("gguf: alignment rules") {
    expect_reject(GgufBuilder().kv_f32("general.alignment", 64.0f).build(), "must be u32");
    expect_reject(GgufBuilder().kv_u32("general.alignment", 48).alignment(48).build(), "power of two");
    expect_reject(GgufBuilder().kv_u32("general.alignment", 4).alignment(4).build(), "power of two");
    const auto ok = GgufBuilder().kv_u32("general.alignment", 64).alignment(64).tensor({"t", {8}, 0, 32}).build();
    const File f = File::parse(ok);
    CHECK(f.alignment() == 64);
    CHECK(f.data_offset() % 64 == 0);
  }

  TEST_CASE("gguf: tensor rules") {
    expect_reject(one_tensor({}, 0, 0).build(), "dimensions");
    expect_reject(one_tensor({1, 1, 1, 1, 1}, 0, 4).build(), "dimensions");
    expect_reject(one_tensor({8, 0}, 0, 0).build(), "dimension 1 is 0");
    expect_reject(one_tensor({std::uint64_t{1} << 40, std::uint64_t{1} << 40}, 0, 4).build(), "overflows");
    expect_reject(one_tensor({32}, 4, 18).build(), "unknown type code 4");
    expect_reject(one_tensor({33}, 8, 34).build(), "not a multiple of the Q8_0 block size 32");
    expect_reject(one_tensor({8}, 0, 32, 16).build(), "has data offset 16, expected 0");
    // Claims 64 bytes of data but the file holds only 32.
    {
      GgufBuilder b;
      b.tensor({"t", {16}, 0, 32});
      auto bytes = b.build();
      std::vector<std::byte> lying = GgufBuilder().tensor({"t", {16}, 0, 64}).build();
      lying.resize(bytes.size());  // drop the second half of its data
      expect_reject(lying, "past the end of the");
    }
    {
      GgufBuilder b;
      b.tensor({"a", {16}, 0, 64, 0}).tensor({"b", {16}, 0, 64, 32});
      expect_reject(b.build(), "has data offset 32, expected 64");
    }
    {
      GgufBuilder b;
      b.tensor({"t", {8}, 0, 32}).tensor({"t", {8}, 0, 32});
      expect_reject(b.build(), "duplicate tensor name 't'");
    }
    expect_reject(GgufBuilder().tensor({std::string(64, 'x'), {8}, 0, 32}).build(), "longer than 63 bytes");
    const File f = File::parse(GgufBuilder().tensor({"q", {64, 3}, 8, 2 * 34 * 3}).build());
    REQUIRE(f.tensors().size() == 1);
    CHECK(f.tensors()[0].bytes == 204);
    CHECK(f.tensors()[0].elements == 192);
    CHECK(f.tensor_bytes(f.tensors()[0]).size() == 204);
  }

  TEST_CASE("llama: config and tensor binding of a tiny model") {
    const auto bytes = tiny_llama().build();
    const File f = File::parse(bytes);
    const auto c = hearth::model::LlamaConfig::from_gguf(f);
    CHECK(c.n_layer == 1);
    CHECK(c.n_embd == 64);
    CHECK(c.head_dim == 32);
    CHECK(c.n_vocab == 32);
    CHECK(c.tied_embeddings);
    CHECK(c.has_rope_freqs);
    const auto t = hearth::model::LlamaTensors::bind(f, c);
    CHECK(t.layers.size() == 1);
    CHECK(t.output == nullptr);
  }

  TEST_CASE("llama: binder rejects missing, misshapen and unclaimed tensors") {
    auto bind = [](const GgufBuilder& b) {
      const auto bytes = b.build();
      const File f = File::parse(bytes);
      (void)hearth::model::LlamaTensors::bind(f, hearth::model::LlamaConfig::from_gguf(f));
    };
    auto with_extra = tiny_llama();
    with_extra.tensor({"blk.0.mystery.weight", {64}, 0, 256});
    CHECK_THROWS_WITH_AS(bind(with_extra), doctest::Contains("does not use: blk.0.mystery.weight"), Error);
    auto wrong_arch = tiny_llama();
    wrong_arch.kv_string("general.architecture", "gpt2");  // duplicate key: rejected by the parser first
    CHECK_THROWS_AS(bind(wrong_arch), Error);
    GgufBuilder misshapen;
    misshapen.kv_string("general.architecture", "llama");
    CHECK_THROWS_WITH_AS(bind(misshapen), doctest::Contains("llama.block_count"), Error);
  }
}

TEST_SUITE("model") {
  TEST_CASE("gguf: the local Llama-3.2-1B files parse and bind") {
    const auto dir = hearth::test::repo_dir() / "models";
    for (const char* name : {"Llama-3.2-1B-Instruct-BF16.gguf", "Llama-3.2-1B-Instruct-F32-pure.gguf",
                             "Llama-3.2-1B-Instruct-Q8_0-pure.gguf", "Llama-3.2-1B-Instruct-Q4_0-pure.gguf"}) {
      CAPTURE(name);
      const auto path = dir / name;
      if (!std::filesystem::exists(path)) {
        MESSAGE("missing (run scripts\\fetch-models.ps1): " << name);
        continue;
      }
      const File f = File::open(path);
      CHECK(f.tensors().size() == 147);
      const auto c = hearth::model::LlamaConfig::from_gguf(f);
      CHECK(c.n_layer == 16);
      CHECK(c.n_embd == 2048);
      CHECK(c.n_ff == 8192);
      CHECK(c.n_head == 32);
      CHECK(c.n_head_kv == 8);
      CHECK(c.head_dim == 64);
      CHECK(c.n_vocab == 128256);
      CHECK(c.rope_theta == 500000.0f);
      CHECK(c.rms_eps == doctest::Approx(1e-5f).epsilon(1e-6));
      CHECK(c.tied_embeddings);
      CHECK(c.has_rope_freqs);
      const auto t = hearth::model::LlamaTensors::bind(f, c);
      CHECK(t.layers.size() == 16);
      CHECK(f.get_array("tokenizer.ggml.tokens", ValueType::kString).count == 128256);
      CHECK(f.get_array("tokenizer.ggml.merges", ValueType::kString).count == 280147);
    }
  }
}
