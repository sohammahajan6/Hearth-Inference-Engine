// Speculative decoding (model/speculative.h): on tiny random Llamas, greedy
// generation with any draft gives model::generate's tokens, stop reason and
// context, bit for bit, across draft lengths, draft probabilities, token
// limits, a full context, an end-of-generation token and a second turn; the
// rejection step and end-to-end sampling follow the target's distribution;
// on the real 1B files, the public API with a draft gives the same text.
#include <doctest/doctest.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <string>
#include <vector>

#include "core/error.h"
#include "core/quant.h"
#include "core/thread_pool.h"
#include "gguf/gguf.h"
#include "gguf_builder.h"
#include "hearth/engine.h"
#include "model/generate.h"
#include "model/llama_ref.h"
#include "model/speculative.h"
#include "sampling/sampler.h"
#include "test_util.h"
#include "tokenizer/tokenizer.h"

using namespace hearth;

namespace {

// The GPT-2 byte-level spelling of byte b (the tokenizer's alphabet).
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

// A tokenizer whose end-of-generation token is `eos` (the tiny models have 32
// tokens, so an `eos` of 32 or more never ends a reply).
std::vector<std::byte> tokenizer_file(std::uint32_t eos) {
  std::vector<std::string> tokens;
  for (int b = 0; b < 256; ++b) tokens.push_back(byte_char(b));
  std::vector<std::int32_t> types(256, 1);
  for (const char* s : {"<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|reserved|>"}) {
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
      .kv_u32("tokenizer.ggml.eos_token_id", eos)
      .build();
}

// The tiny Llama of seed `seed` in F32, with the layers' matrices scaled by
// `gain`. The builder's weights (+-0.08) leave the tied embeddings dominant,
// and such a model repeats one token; a larger gain gives varied text.
std::vector<std::byte> tiny_f32(std::uint64_t seed, float gain) {
  std::vector<std::byte> b = test::tiny_llama(2, 4, 2).random_f32_data(seed).build();
  const gguf::File f = gguf::File::parse(b);
  for (const gguf::TensorInfo& t : f.tensors()) {
    if (t.n_dims != 2 || t.name == "token_embd.weight") continue;
    std::byte* p = b.data() + (f.tensor_bytes(t).data() - b.data());
    for (std::size_t i = 0; i < f.tensor_bytes(t).size() / 4; ++i) {
      float v;
      std::memcpy(&v, p + 4 * i, 4);
      v *= gain;
      std::memcpy(p + 4 * i, &v, 4);
    }
  }
  return b;
}

// The same model with Q8_0 matrices (the cpu backend's quantized path).
std::vector<std::byte> tiny_q8(std::uint64_t seed, float gain = 1.0f) {
  const std::vector<std::byte> f32 = tiny_f32(seed, gain);
  std::vector<std::byte> out = test::tiny_llama(2, 4, 2, 8).build();
  const gguf::File src = gguf::File::parse(f32);
  const gguf::File dst = gguf::File::parse(out);
  for (const gguf::TensorInfo& t : src.tensors()) {
    const gguf::TensorInfo* d = dst.find_tensor(t.name);
    REQUIRE(d != nullptr);
    const std::span<const std::byte> from = src.tensor_bytes(t);
    std::byte* to = out.data() + (dst.tensor_bytes(*d).data() - out.data());
    if (t.n_dims == 1) {
      std::memcpy(to, from.data(), from.size());
      continue;
    }
    const auto k = static_cast<std::int64_t>(t.ne[0]);
    std::vector<float> row(static_cast<std::size_t>(k));
    for (std::int64_t r = 0; r < static_cast<std::int64_t>(t.ne[1]); ++r) {
      std::memcpy(row.data(), from.data() + r * k * 4, static_cast<std::size_t>(k) * 4);
      quant::quantize_row_q8_0(row.data(), k, reinterpret_cast<quant::BlockQ8_0*>(to) + r * (k / quant::kBlock));
    }
  }
  return out;
}

struct Outcome {
  std::vector<std::int32_t> ids;
  model::GenerateResult r;
  std::vector<float> after;  // the target's logits for one more token: shows its whole context
};

// Two turns (prompt, then `second`), then one more token; plain when `draft`
// is nullptr. With `state` the draft may lag between the turns.
Outcome two_turns(model::LlamaRef& target, model::LlamaRef* draft, const tokenizer::Tokenizer& tok,
                  sampling::Sampler& sampler, int max_new, const model::SpeculativeOptions& so,
                  model::SpeculativeState* state = nullptr) {
  const std::vector<std::int32_t> prompt = {1, 17, 3, 29, 5}, second = {11, 7};
  Outcome o;
  const auto emit = [&](std::int32_t id) { o.ids.push_back(id); };
  for (const auto* p : {&prompt, &second}) {
    if (target.n_past() + p->size() > target.n_ctx()) break;
    o.r = draft ? model::generate_speculative(target, *draft, tok, sampler, *p, max_new, so, emit, state)
                : model::generate(target, tok, sampler, *p, max_new, emit);
    o.ids.push_back(-1);  // turn boundary
    if (draft) {
      REQUIRE(draft->n_past() + (state != nullptr ? state->skipped() + state->lag().size() : 0) == target.n_past());
    }
  }
  if (target.n_past() < target.n_ctx()) {
    const std::int32_t t = 9;
    o.after.resize(target.config().n_vocab);
    target.forward(std::span(&t, 1), o.after, false);
  }
  return o;
}

}  // namespace

TEST_SUITE("quick") {
  TEST_CASE("speculative greedy: tokens, stop reason and context equal model::generate's, with any draft") {
    ThreadPool pool(2);
    const std::vector<std::byte> no_eos = tokenizer_file(259);
    const gguf::File no_eos_file = gguf::File::parse(no_eos);
    const tokenizer::Tokenizer tok_none = tokenizer::Tokenizer::from_gguf(no_eos_file);
    // Many tiny random models repeat one token. The target is the first seed
    // whose greedy text has at least 8 different tokens, so drafts are
    // rejected at varied places; its end token for the end-of-generation
    // cases is one that first appears 3 or more tokens in.
    constexpr float kGain = 12.0f;
    std::uint64_t seed = 0;
    std::uint32_t eos = 259;
    std::vector<std::byte> target_bytes;
    for (std::uint64_t candidate = 1; candidate <= 200 && eos == 259; ++candidate) {
      target_bytes = tiny_q8(candidate, kGain);
      const gguf::File f = gguf::File::parse(target_bytes);
      model::LlamaRef t(f, 128, pool, model::Matmul::kQ8Activations);
      sampling::Sampler s({0.0f, 40, 0.95f, 0});
      const Outcome plain = two_turns(t, nullptr, tok_none, s, 40, {});
      std::vector<bool> seen(32, false);
      int distinct = 0;
      std::uint32_t first_late = 259;
      for (std::size_t i = 0; i < plain.ids.size(); ++i) {
        const std::int32_t id = plain.ids[i];
        if (id < 0 || seen[static_cast<std::size_t>(id)]) continue;
        seen[static_cast<std::size_t>(id)] = true;
        ++distinct;
        if (i >= 3 && first_late == 259) first_late = static_cast<std::uint32_t>(id);
      }
      if (distinct >= 8 && first_late < 32) {
        seed = candidate;
        eos = first_late;
      }
    }
    REQUIRE(eos < 32);
    MESSAGE(std::format("target seed {}, end token {}", seed, eos));
    const std::vector<std::byte> other_bytes = tiny_q8(seed + 1000, kGain);
    const std::vector<std::byte> f32_bytes = tiny_f32(seed, kGain);
    const gguf::File target_file = gguf::File::parse(target_bytes), other_file = gguf::File::parse(other_bytes),
                     f32_file = gguf::File::parse(f32_bytes);
    const std::vector<std::byte> with_eos = tokenizer_file(eos);
    const gguf::File with_eos_file = gguf::File::parse(with_eos);
    const tokenizer::Tokenizer tok_eos = tokenizer::Tokenizer::from_gguf(with_eos_file);

    // Drafts: the target itself (every draft kept), another random model
    // (most rejected), and the target's F32 original on cpu-ref (most kept).
    struct DraftKind {
      const char* name;
      const gguf::File* file;
      model::Matmul matmul;
    };
    const DraftKind drafts[] = {{"same", &target_file, model::Matmul::kQ8Activations},
                                {"other", &other_file, model::Matmul::kQ8Activations},
                                {"f32", &f32_file, model::Matmul::kFp32}};
    int cases = 0, kept_all = 0, rejected_some = 0, looked_some = 0;
    for (const tokenizer::Tokenizer* tok : {&tok_none, &tok_eos}) {
      for (const std::uint32_t ctx : {128u, 20u}) {
        for (const int max_new : {0, 1, 2, 5, 40}) {
          model::LlamaRef plain_llm(target_file, ctx, pool, model::Matmul::kQ8Activations);
          sampling::Sampler plain_s({0.0f, 40, 0.95f, 0});
          const Outcome want = two_turns(plain_llm, nullptr, *tok, plain_s, max_new, {});
          for (const DraftKind& dk : drafts) {
            for (const int dmax : {1, 2, 3, 4, 7}) {
              for (const float p_min : {0.0f, 0.3f, -1.0f}) {
                // No state; a state kept across the turns (the draft may lag); and a
                // state with a reading limit of 3 tokens (head 1), so the draft skips
                // most of every catch-up.
                for (const int mode : {0, 1, 2}) {
                  CAPTURE(dk.name);
                  CAPTURE(ctx);
                  CAPTURE(max_new);
                  CAPTURE(dmax);
                  CAPTURE(p_min);
                  CAPTURE(tok == &tok_eos);
                  CAPTURE(mode);
                  model::LlamaRef target(target_file, ctx, pool, model::Matmul::kQ8Activations);
                  model::LlamaRef draft(*dk.file, ctx, pool, dk.matmul);
                  sampling::Sampler s({0.0f, 40, 0.95f, 0});
                  model::SpeculativeOptions so{dmax, p_min};
                  so.lookup_ngram = mode == 0 ? 0 : 3;  // the draft model alone; modes 1 and 2 also look up
                  if (mode == 2) {
                    so.draft_read_max = 3;
                    so.draft_read_head = 1;
                  }
                  model::SpeculativeState kept;
                  const Outcome got = two_turns(target, &draft, *tok, s, max_new, so, mode >= 1 ? &kept : nullptr);
                  CHECK(got.ids == want.ids);
                  CHECK(got.r.stop == want.r.stop);
                  CHECK(got.r.eog_token == want.r.eog_token);
                  CHECK(got.r.generated == want.r.generated);
                  CHECK(target.n_past() == plain_llm.n_past());
                  REQUIRE(got.after.size() == want.after.size());
                  CHECK(std::memcmp(got.after.data(), want.after.data(), got.after.size() * sizeof(float)) == 0);
                  CHECK(got.r.accepted <= got.r.drafted);
                  looked_some += got.r.looked_up > 0;
                  if (mode == 0) CHECK(got.r.looked_up == 0);
                  if (got.r.drafted > 0) {
                    kept_all += got.r.accepted == got.r.drafted;
                    rejected_some += got.r.accepted < got.r.drafted;
                  }
                  ++cases;
                }
              }
            }
          }
        }
      }
    }
    CHECK(cases == 2 * 2 * 5 * 3 * 5 * 3 * 3);
    // Both paths of the acceptance loop ran.
    CHECK(kept_all > 0);
    CHECK(rejected_some > 0);
    CHECK(looked_some > 0);  // prompt lookup proposed drafts in some runs
  }

  TEST_CASE("speculative greedy: the target as its own draft keeps every draft; a full context stops as plain") {
    const std::vector<std::byte> bytes = tiny_q8(5);
    const gguf::File file = gguf::File::parse(bytes);
    const std::vector<std::byte> tb = tokenizer_file(259);
    const gguf::File tf = gguf::File::parse(tb);
    const tokenizer::Tokenizer tok = tokenizer::Tokenizer::from_gguf(tf);
    ThreadPool pool(3);
    model::LlamaRef target(file, 64, pool, model::Matmul::kQ8Activations);
    model::LlamaRef draft(file, 64, pool, model::Matmul::kQ8Activations);
    sampling::Sampler s({0.0f, 40, 0.95f, 0});
    const std::vector<std::int32_t> prompt = {1, 2, 3};
    std::vector<std::int32_t> ids;
    const model::GenerateResult r = model::generate_speculative(target, draft, tok, s, prompt, 100, {4, 0.0f},
                                                                [&](std::int32_t id) { ids.push_back(id); });
    // 3 prompt positions + 61 fed tokens fill the context; the 62nd emitted token is not fed.
    CHECK(r.stop == model::Stop::kContextFull);
    CHECK(r.generated == 62);
    CHECK(target.n_past() == 64);
    CHECK(draft.n_past() == 64);
    CHECK(r.accepted == r.drafted);
    CHECK(r.drafted >= 40);
    // A full context is a stop reason at the next call, not an error.
    const model::GenerateResult again =
        model::generate_speculative(target, draft, tok, s, prompt, 5, {4, 0.0f}, [](std::int32_t) {});
    CHECK(again.stop == model::Stop::kContextFull);
    CHECK(again.generated == 0);
  }

  TEST_CASE("speculative: with a kept state the draft lags after a reply, catches up when it drafts, and is reset") {
    const std::vector<std::byte> bytes = tiny_q8(5);
    const gguf::File file = gguf::File::parse(bytes);
    const std::vector<std::byte> tb = tokenizer_file(259);
    const gguf::File tf = gguf::File::parse(tb);
    const tokenizer::Tokenizer tok = tokenizer::Tokenizer::from_gguf(tf);
    ThreadPool pool(2);
    model::LlamaRef target(file, 128, pool, model::Matmul::kQ8Activations);
    model::LlamaRef draft(file, 128, pool, model::Matmul::kQ8Activations);
    model::LlamaRef plain(file, 128, pool, model::Matmul::kQ8Activations);
    model::SpeculativeState st;
    sampling::Sampler s1({0.0f, 40, 0.95f, 0}), s2({0.0f, 40, 0.95f, 0});
    std::vector<std::int32_t> a, b;
    const auto turn = [&](const std::vector<std::int32_t>& p, int n) {
      model::generate_speculative(target, draft, tok, s1, p, n, {}, [&](std::int32_t id) { a.push_back(id); }, &st);
      model::generate(plain, tok, s2, p, n, [&](std::int32_t id) { b.push_back(id); });
      CHECK(a == b);
      CHECK(draft.n_past() + st.lag().size() == target.n_past());
    };
    // A reply of one token: the draft never drafted, so it has read nothing.
    turn({1, 2, 3}, 1);
    CHECK(draft.n_past() == 0);
    CHECK(st.lag().size() == 4);
    // The next reply drafts: the draft reads the 4 pending tokens first.
    turn({4, 5}, 20);
    CHECK(draft.n_past() > 6);
    CHECK(st.lag().size() <= 2);  // at most the last kept draft token and the token after it
    std::vector<float> la(32), lb(32);
    const std::int32_t t = 9;
    target.forward(std::span(&t, 1), la, false);
    plain.forward(std::span(&t, 1), lb, false);
    CHECK(std::memcmp(la.data(), lb.data(), la.size() * sizeof(float)) == 0);
    st.forget_context();
    CHECK(st.lag().empty());
    CHECK(st.skipped() == 0);

    // A reading limit: a 12-token prompt with limit 5 (head 2): the draft
    // reads 2 + 3 tokens of the catch-up and skips 8; output still equal.
    target.reset();
    draft.reset();
    plain.reset();
    a.clear();
    b.clear();
    model::SpeculativeOptions lim;
    lim.draft_read_max = 5;
    lim.draft_read_head = 2;
    const std::vector<std::int32_t> long_prompt = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    model::generate_speculative(
        target, draft, tok, s1, long_prompt, 10, lim, [&](std::int32_t id) { a.push_back(id); }, &st);
    model::generate(plain, tok, s2, long_prompt, 10, [&](std::int32_t id) { b.push_back(id); });
    CHECK(a == b);
    CHECK(st.skipped() == 8);  // the first catch-up: 12 prompt tokens + the first token = 13, 5 read
    CHECK(draft.n_past() + st.skipped() + st.lag().size() == target.n_past());
    model::SpeculativeOptions bad;
    bad.draft_read_head = bad.draft_read_max;
    CHECK_THROWS_AS(model::generate_speculative(
                        target, draft, tok, s1, std::vector<std::int32_t>{4}, 2, bad, [](std::int32_t) {}, &st),
                    Error);
  }

  TEST_CASE("speculative: bad options and mismatched models are clear errors that leave the contexts unchanged") {
    const std::vector<std::byte> bytes = tiny_q8(5);
    const gguf::File file = gguf::File::parse(bytes);
    const std::vector<std::byte> tb = tokenizer_file(259);
    const gguf::File tf = gguf::File::parse(tb);
    const tokenizer::Tokenizer tok = tokenizer::Tokenizer::from_gguf(tf);
    ThreadPool pool(1);
    model::LlamaRef target(file, 32, pool, model::Matmul::kQ8Activations);
    model::LlamaRef draft(file, 32, pool, model::Matmul::kQ8Activations);
    model::LlamaRef small(file, 16, pool, model::Matmul::kQ8Activations);
    sampling::Sampler s({0.0f, 40, 0.95f, 0});
    const std::vector<std::int32_t> prompt = {1, 2, 3};
    const auto kind = [&](model::LlamaRef& d, const model::SpeculativeOptions& o, std::span<const std::int32_t> p) {
      try {
        model::generate_speculative(target, d, tok, s, p, 4, o, [](std::int32_t) {});
      } catch (const Error& e) {
        return e.kind();
      }
      return ErrorKind::kInternal;  // no error: fails the checks below
    };
    CHECK(kind(draft, {0, 0.0f}, prompt) == ErrorKind::kInvalidArgument);
    CHECK(kind(draft, {17, 0.0f}, prompt) == ErrorKind::kInvalidArgument);
    CHECK(kind(draft, {4, 1.5f}, prompt) == ErrorKind::kInvalidArgument);
    CHECK(kind(small, {4, 0.0f}, prompt) == ErrorKind::kInvalidArgument);  // the draft's context is smaller
    CHECK(kind(draft, {4, 0.0f}, {}) == ErrorKind::kInvalidArgument);      // empty prompt
    const std::vector<std::int32_t> bad = {1, 40};                         // a token outside the vocabulary
    CHECK(kind(draft, {4, 0.0f}, bad) == ErrorKind::kInvalidArgument);
    CHECK(target.n_past() == 0);
    CHECK(draft.n_past() == 0);
    // The draft must hold what the target holds.
    const std::int32_t one = 1;
    std::vector<float> l(32);
    draft.forward(std::span(&one, 1), l, false);
    CHECK(kind(draft, {4, 0.0f}, prompt) == ErrorKind::kInternal);
    // An error thrown during decode (here by the callback) leaves both contexts as they were.
    draft.reset();
    target.forward(std::span(&one, 1), l, false);
    draft.forward(std::span(&one, 1), l, false);
    int calls = 0;
    CHECK_THROWS_AS(model::generate_speculative(target, draft, tok, s, prompt, 20, {3, 0.0f},
                                                [&](std::int32_t) {
                                                  if (++calls == 4) fail(ErrorKind::kIo, "test: callback failed");
                                                }),
                    Error);
    CHECK(target.n_past() == 1);
    CHECK(draft.n_past() == 1);
  }

  TEST_CASE("speculative sampling: the rejection step returns tokens with the target's probabilities") {
    // p and q over 6 tokens with different supports: q has token 5, which p
    // lacks; p has token 4, which q lacks.
    const std::vector<sampling::Candidate> p = {{0, 0.40}, {1, 0.25}, {2, 0.15}, {3, 0.12}, {4, 0.08}};
    const std::vector<sampling::Candidate> q = {{2, 0.30}, {0, 0.30}, {1, 0.10}, {3, 0.05}, {5, 0.25}};
    sampling::Sampler s({1.0f, 0, 1.0f, 2026});
    sampling::Rng rng(7);
    std::vector<double> scratch(6, 0.0);
    constexpr int kN = 400000;
    std::vector<int> count(6, 0);
    int accepted_n = 0;
    for (int n = 0; n < kN; ++n) {
      // d ~ q
      const double u = rng.uniform();
      double acc = 0;
      std::int32_t d = q.back().id;
      for (const sampling::Candidate& c : q) {
        acc += c.p;
        if (u < acc) {
          d = c.id;
          break;
        }
      }
      bool accepted = false;
      const std::int32_t t = model::detail::accept_or_resample(p, q, d, s, scratch, accepted);
      REQUIRE(t >= 0);
      REQUIRE(t < 6);
      ++count[static_cast<std::size_t>(t)];
      accepted_n += accepted;
      if (accepted) CHECK(t == d);
    }
    for (const double v : scratch) CHECK(v == 0.0);  // left as zeros
    CHECK(count[5] == 0);                            // p(5) = 0
    for (const sampling::Candidate& c : p) {
      const double frac = static_cast<double>(count[static_cast<std::size_t>(c.id)]) / kN;
      const double sigma = std::sqrt(c.p * (1 - c.p) / kN);
      CAPTURE(c.id);
      CHECK(std::fabs(frac - c.p) <= 5 * sigma);
    }
    // A proposal of probability 1 (prompt lookup, sampled): token 2 always
    // proposed; the output must still follow p (kept with p(2) = 0.15).
    {
      const std::vector<sampling::Candidate> one = {{2, 1.0}};
      std::vector<int> c1(6, 0);
      for (int n = 0; n < kN; ++n) {
        bool acc = false;
        ++c1[static_cast<std::size_t>(model::detail::accept_or_resample(p, one, 2, s, scratch, acc))];
      }
      for (const sampling::Candidate& c : p) {
        const double frac = static_cast<double>(c1[static_cast<std::size_t>(c.id)]) / kN;
        CAPTURE(c.id);
        CHECK(std::fabs(frac - c.p) <= 5 * std::sqrt(c.p * (1 - c.p) / kN));
      }
    }
    // The acceptance rate is sum over x of min(p(x), q(x)) = 0.3 + 0.1 + 0.15 + 0.05 = 0.6.
    const double rate = static_cast<double>(accepted_n) / kN, rs = std::sqrt(0.6 * 0.4 / kN);
    CHECK(std::fabs(rate - 0.6) <= 5 * rs);
  }

  TEST_CASE("speculative sampling: the second and third tokens follow the target's distribution end to end") {
    // Exact marginals of the 2nd and 3rd generated tokens under the target at
    // temperature 1 (no top-k, no top-p), against 6000 speculative runs with a
    // different draft model, which forces rejections.
    const std::vector<std::byte> target_bytes = tiny_q8(23), draft_bytes = tiny_q8(41);
    const gguf::File tf = gguf::File::parse(target_bytes), df = gguf::File::parse(draft_bytes);
    const std::vector<std::byte> tb = tokenizer_file(259);
    const gguf::File tokf = gguf::File::parse(tb);
    const tokenizer::Tokenizer tok = tokenizer::Tokenizer::from_gguf(tokf);
    ThreadPool pool(1);
    const std::vector<std::int32_t> prompt = {1, 9, 4};
    constexpr std::size_t V = 32;
    const auto softmax = [](const std::vector<float>& l) {
      std::vector<double> p(l.size());
      double m = l[0], sum = 0;
      for (const float x : l) m = std::max(m, static_cast<double>(x));
      for (std::size_t i = 0; i < l.size(); ++i) sum += p[i] = std::exp(l[i] - m);
      for (double& x : p) x /= sum;
      return p;
    };
    model::LlamaRef ref(tf, 16, pool, model::Matmul::kQ8Activations);
    std::vector<float> l(V);
    ref.forward(prompt, l, false);
    const std::vector<double> p1 = softmax(l);
    std::vector<double> m2(V, 0.0), m3(V, 0.0);
    for (std::int32_t a = 0; a < static_cast<std::int32_t>(V); ++a) {
      ref.truncate(3);
      ref.forward(std::span(&a, 1), l, false);
      const std::vector<double> p2 = softmax(l);
      for (std::int32_t b = 0; b < static_cast<std::int32_t>(V); ++b) {
        m2[static_cast<std::size_t>(b)] += p1[static_cast<std::size_t>(a)] * p2[static_cast<std::size_t>(b)];
        ref.truncate(4);
        ref.forward(std::span(&b, 1), l, false);
        const std::vector<double> p3 = softmax(l);
        for (std::size_t c = 0; c < V; ++c) {
          m3[c] += p1[static_cast<std::size_t>(a)] * p2[static_cast<std::size_t>(b)] * p3[c];
        }
      }
    }
    model::LlamaRef target(tf, 16, pool, model::Matmul::kQ8Activations);
    model::LlamaRef draft(df, 16, pool, model::Matmul::kQ8Activations);
    constexpr int kN = 6000;
    std::vector<int> c2(V, 0), c3(V, 0);
    int drafted = 0, accepted = 0;
    for (int n = 0; n < kN; ++n) {
      target.reset();
      draft.reset();
      sampling::Sampler s({1.0f, 0, 1.0f, static_cast<std::uint64_t>(n) + 1});
      std::vector<std::int32_t> ids;
      const model::GenerateResult r = model::generate_speculative(target, draft, tok, s, prompt, 3, {2, 0.0f},
                                                                  [&](std::int32_t id) { ids.push_back(id); });
      REQUIRE(ids.size() == 3);
      ++c2[static_cast<std::size_t>(ids[1])];
      ++c3[static_cast<std::size_t>(ids[2])];
      drafted += r.drafted;
      accepted += r.accepted;
    }
    CHECK(accepted < drafted);  // rejections happened
    CHECK(accepted > 0);
    for (std::size_t t = 0; t < V; ++t) {
      CAPTURE(t);
      for (const auto& [count, marginal] : {std::pair(&c2, &m2), std::pair(&c3, &m3)}) {
        const double want = (*marginal)[t], got = static_cast<double>((*count)[t]) / kN;
        const double sigma = std::sqrt(want * (1 - want) / kN);
        CHECK(std::fabs(got - want) <= 5 * sigma + 1e-9);
      }
    }
  }
}

TEST_SUITE("quick") {
  TEST_CASE("speculative: top_probability is the softmax maximum, within 3e-4 of the exact value") {
    CHECK(model::detail::top_probability(std::vector<float>{0.0f, std::log(3.0f)}) == doctest::Approx(0.75));
    CHECK(model::detail::top_probability(std::vector<float>{5.0f}) == 1.0);
    // 128,256 logits: a peaked and a flat case, against an exact double sum.
    for (const float spread : {1.0f, 25.0f}) {
      std::vector<float> l(128256);
      std::uint32_t st = 12345;
      for (float& x : l) {
        st = st * 1664525u + 1013904223u;
        x = spread * static_cast<float>(st >> 8) / 16777216.0f;
      }
      l[77] = spread + 3.0f;
      double sum = 0;
      for (const float x : l) sum += std::exp(static_cast<double>(x) - (spread + 3.0));
      const double exact = 1.0 / sum, got = model::detail::top_probability(l);
      CAPTURE(spread);
      CHECK(std::fabs(got - exact) <= 3e-4 * exact);
    }
  }

  TEST_CASE("speculative costs: the fit recovers a draft step, a target step and a per-token cost") {
    model::SpeculativeState c;
    CHECK_FALSE(c.measured());
    CHECK(c.threshold() == doctest::Approx(0.3));  // the prior
    // Draft 20 ms; verification of 1 + k tokens 100 + 6 k ms, for k = 0..8.
    for (int i = 0; i < 200; ++i) {
      c.add_draft_step(0.020);
      c.add_verification(i % 9, 0.100 + 0.006 * (i % 9));
    }
    REQUIRE(c.measured());
    CHECK(c.draft_step() == doctest::Approx(0.020));
    CHECK(c.target_step() == doctest::Approx(0.100).epsilon(1e-6));
    CHECK(c.extra_token() == doctest::Approx(0.006).epsilon(1e-6));
    CHECK(c.threshold() == doctest::Approx(0.26).epsilon(1e-6));
    // A pause of the whole process (2 s) is clipped, so it moves the
    // estimates by a bounded amount and they recover.
    c.add_verification(4, 2.0);
    c.add_draft_step(2.0);
    CHECK(c.target_step() < 0.2);
    CHECK(c.draft_step() < 0.03);
    for (int i = 0; i < 300; ++i) {
      c.add_draft_step(0.020);
      c.add_verification(i % 9, 0.100 + 0.006 * (i % 9));
    }
    CHECK(c.target_step() == doctest::Approx(0.100).epsilon(1e-3));
    // One draft length only: the prior slope (6 % of the target step).
    model::SpeculativeState one;
    for (int i = 0; i < 50; ++i) {
      one.add_draft_step(0.030);
      one.add_verification(4, 0.124);
    }
    CHECK(one.extra_token() == doctest::Approx(0.006).epsilon(1e-6));
    CHECK(one.target_step() == doctest::Approx(0.100).epsilon(1e-6));
    CHECK(one.threshold() == doctest::Approx(0.36).epsilon(1e-6));
  }
}

TEST_SUITE("model") {
  TEST_CASE("speculative: the real 1B Q8_0 with the 1B Q4_0 as draft gives the same greedy text through the API") {
    const auto dir = test::repo_dir() / "models";
    const auto main_path = dir / "Llama-3.2-1B-Instruct-Q8_0-pure.gguf";
    const auto draft_path = dir / "Llama-3.2-1B-Instruct-Q4_0-pure.gguf";
    if (!std::filesystem::exists(main_path) || !std::filesystem::exists(draft_path)) {
      MESSAGE("missing (run scripts\\fetch-models.ps1): the 1B Q8_0 and Q4_0 files");
      return;
    }
    Engine::Options o;
    o.model = main_path;
    o.context = 512;
    Engine plain(o);
    o.draft_model = draft_path;
    o.draft_max = 5;
    Engine spec(o);
    CHECK(plain.draft_description().empty());
    CHECK(spec.draft_description().starts_with("draft Llama-3.2-1B-Instruct-Q4_0-pure.gguf"));
    GenerateOptions g;
    g.max_new_tokens = 64;
    g.temperature = 0;
    const ChatMessage m{"user", "Name three rivers in Europe and one fact about each."};
    std::vector<std::int32_t> a, b;
    for (const char* next : {"", "Now one more river, in Asia."}) {
      const std::vector<std::int32_t> prompt =
          a.empty() ? plain.chat_prompt("", std::span(&m, 1), "2026-11-12") : plain.chat_next_turn(next);
      const GenerateResult ra = plain.generate(prompt, g, [&](std::int32_t id, std::string_view) { a.push_back(id); });
      const GenerateResult rb = spec.generate(prompt, g, [&](std::int32_t id, std::string_view) { b.push_back(id); });
      CHECK(a == b);
      CHECK(ra.stop == rb.stop);
      CHECK(plain.context_used() == spec.context_used());
      CHECK(ra.draft_tokens == 0);
      CHECK(rb.draft_accepted <= rb.draft_tokens);
      CHECK(rb.draft_accepted > 0);
    }
    CHECK(a.size() > 64);
    spec.reset();
    CHECK(spec.context_used() == 0);
  }
}
