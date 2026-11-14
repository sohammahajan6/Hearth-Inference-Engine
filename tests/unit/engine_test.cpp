// The public API (include/hearth/engine.h; docs/architecture/11-public-api.md).
#include <doctest/doctest.h>

#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "core/thread_pool.h"
#include "gguf/gguf.h"
#include "hearth/engine.h"
#include "model/generate.h"
#include "model/llama_ref.h"
#include "sampling/sampler.h"
#include "test_util.h"
#include "tokenizer/tokenizer.h"

using namespace hearth;

TEST_SUITE("quick") {
  TEST_CASE("engine: bad options are kInvalidArgument, before the model is opened") {
    auto bad = [](Engine::Options o) {
      try {
        Engine e(o);
      } catch (const Error& err) {
        return err.kind() == ErrorKind::kInvalidArgument;
      }
      return false;
    };
    Engine::Options o;
    o.model = "no-such-file.gguf";                   // never opened: validation comes first
    CHECK(bad([] { return Engine::Options{}; }()));  // no model path
    auto with = [&](auto f) {
      Engine::Options x = o;
      f(x);
      return x;
    };
    CHECK(bad(with([](Engine::Options& x) { x.threads = 0; })));
    CHECK(bad(with([](Engine::Options& x) { x.context = 0; })));
    CHECK(bad(with([](Engine::Options& x) { x.threads = 65; })));
    CHECK(bad(with([](Engine::Options& x) { x.backend = static_cast<Backend>(7); })));  // no such backend
    // Speculative decoding: the draft length, its probability and the backend.
    CHECK(bad(with([](Engine::Options& x) {
      x.draft_model = "no-such-draft.gguf";
      x.draft_max = 0;
    })));
    CHECK(bad(with([](Engine::Options& x) {
      x.draft_model = "no-such-draft.gguf";
      x.draft_max = 17;
    })));
    CHECK(bad(with([](Engine::Options& x) {
      x.draft_model = "no-such-draft.gguf";
      x.draft_p_min = 1.5f;
    })));
    CHECK(bad(with([](Engine::Options& x) {
      x.draft_model = "no-such-draft.gguf";
      x.backend = Backend::kGpu;
    })));
    // Valid options with a missing file: an I/O error, not an argument error.
    bool io = false;
    try {
      Engine e(o);
    } catch (const Error& err) {
      io = err.kind() == ErrorKind::kIo;
    }
    CHECK(io);
  }
}

TEST_SUITE("model") {
  TEST_CASE("engine: the real 1B Q8_0 through the public API equals the internal path, and continues its context") {
    const auto path = test::repo_dir() / "models" / "Llama-3.2-1B-Instruct-Q8_0-pure.gguf";
    if (!std::filesystem::exists(path)) {
      MESSAGE("missing (run scripts\\fetch-models.ps1): Llama-3.2-1B-Instruct-Q8_0-pure.gguf");
      return;
    }
    Engine::Options o;
    o.model = path;
    o.context = 256;
    Engine engine(o);
    CHECK(engine.backend_description().starts_with("cpu ("));
    CHECK(engine.gpu_device().empty());
    const std::vector<std::int32_t> prompt = engine.tokenize("The capital of France is");
    GenerateOptions g;
    g.max_new_tokens = 8;
    g.temperature = 0;
    std::vector<std::int32_t> ids;
    std::string text;
    const GenerateResult r = engine.generate(prompt, g, [&](std::int32_t id, std::string_view piece) {
      ids.push_back(id);
      text += piece;
    });
    text += r.trailing_text;
    CHECK(text.starts_with(" Paris"));
    CHECK(r.prompt_tokens == static_cast<int>(prompt.size()));
    CHECK(engine.context_used() == prompt.size() + ids.size());

    // The internal path (model::generate on LlamaRef) gives the same tokens.
    const gguf::File f = gguf::File::open(path);
    const tokenizer::Tokenizer tok = tokenizer::Tokenizer::from_gguf(f);
    CHECK(tok.encode("The capital of France is", true) == prompt);
    ThreadPool pool(4);
    model::LlamaRef llm(f, 256, pool, model::Matmul::kQ8Activations);
    sampling::Sampler sampler({0.0f, 40, 0.95f, 0});
    std::vector<std::int32_t> want;
    model::generate(llm, tok, sampler, prompt, 8, [&](std::int32_t id) { want.push_back(id); });
    CHECK(ids == want);

    // A chat prompt, then reset empties the context.
    const ChatMessage m{"user", "Hello"};
    const std::vector<std::int32_t> chat = engine.chat_prompt("", std::span(&m, 1), "2026-11-08");
    CHECK(chat.size() > prompt.size());
    CHECK_FALSE(engine.chat_next_turn("Next").empty());
    engine.reset();
    CHECK(engine.context_used() == 0);
  }
}
