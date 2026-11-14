// cli/ui_app: the routes of `hearth ui` through a fake connection. Quick
// tier: everything that needs no model (the page, status, refusals, stop and
// quit). Model tier: load the real 1B Q8_0 on the CPU and chat with it. The
// benchmark and the GPU mode run end to end in scripts/ui-check.ps1.
#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

#include "core/platform.h"
#include "gguf_builder.h"
#include "test_util.h"
#include "ui_app.h"

using namespace hearth;

namespace {

struct FakeConnection final : ui::Connection {
  int status = 0;
  std::string type, body;
  bool streamed = false;
  bool respond(int s, std::string_view t, std::string_view b) override {
    status = s;
    type = t;
    body = b;
    return true;
  }
  bool begin_stream(std::string_view t) override {
    status = 200;
    type = t;
    streamed = true;
    return true;
  }
  bool send(std::string_view b) override {
    body += b;
    return true;
  }
  bool alive() const override { return true; }

  // The events of a stream: (name, data).
  std::vector<std::pair<std::string, std::string>> events() const {
    std::vector<std::pair<std::string, std::string>> out;
    std::size_t pos = 0;
    for (;;) {
      const std::size_t end = body.find("\n\n", pos);
      if (end == std::string::npos) break;
      const std::string block = body.substr(pos, end - pos);
      pos = end + 2;
      const std::size_t nl = block.find('\n');
      if (!block.starts_with("event: ") || nl == std::string::npos || block.compare(nl + 1, 6, "data: ") != 0) continue;
      out.emplace_back(block.substr(7, nl - 7), block.substr(nl + 7));
    }
    return out;
  }
};

FakeConnection call(cli::UiApp& app, const std::string& method, const std::string& path, const std::string& body = {}) {
  ui::Request r;
  r.method = method;
  r.path = path;
  r.body = body;
  FakeConnection c;
  app.handle(r, c);
  return c;
}

cli::UiOptions test_options() {
  cli::UiOptions o;
  o.model_dirs = {test::repo_dir() / "models"};
  const auto dir = test::repo_dir() / "build" / "test-tmp";
  std::filesystem::create_directories(dir);
  o.results_file = dir / "ui-results-test.jsonl";
  o.models_file = dir / "ui-models-test.txt";
  std::filesystem::remove(o.models_file);
  o.idle_minutes = 0;
  return o;
}

}  // namespace

TEST_SUITE("quick") {
  TEST_CASE("ui app: find_draft picks the smallest Q4_0 file with the model's vocabulary, at most 30 % of its size") {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "hearth-find-draft-test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const auto write = [&](const char* name, const std::vector<std::byte>& bytes) {
      std::ofstream out(dir / name, std::ios::binary);
      out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
      return dir / name;
    };
    // The "target": 6 F32 layers. Candidates of 1 layer: Q4_0 (the draft), Q8_0 (not Q4_0), Q4_0 with
    // one token spelled differently (another vocabulary); a broken file; and a 6-layer Q4_0 (larger than
    // the draft, so not chosen while the draft exists).
    const auto target = write("target.gguf", test::tiny_llama(6, 2, 1, 0).build());
    CHECK(cli::find_draft(target).empty());  // no candidate yet
    std::vector<std::string> other_vocab = test::tiny_vocab();
    other_vocab[5] = "zz";
    write("other-vocab-q4.gguf", test::tiny_llama(1, 2, 1, 2, 64, 128, other_vocab).build());
    write("small-q8.gguf", test::tiny_llama(1, 2, 1, 8).build());
    write("broken.gguf", std::vector<std::byte>(64, std::byte{7}));
    CHECK(cli::find_draft(target).empty());  // none qualifies
    const auto big = write("big-q4.gguf", test::tiny_llama(6, 2, 1, 2).build());
    CHECK(cli::find_draft(target) == big);
    const auto draft = write("draft-q4.gguf", test::tiny_llama(1, 2, 1, 2).build());
    CHECK(cli::find_draft(target) == draft);
    // A Q4_0 model has no draft among these: none is at most 30 % of its size.
    CHECK(cli::find_draft(draft).empty());
    std::filesystem::remove_all(dir);
  }

  TEST_CASE("ui app: the page, the status, and every refusal that needs no model") {
    cli::UiApp app(test_options());
    FakeConnection page = call(app, "GET", "/");
    CHECK(page.status == 200);
    CHECK(page.type == "text/html; charset=utf-8");
    CHECK(page.body.find("<title>Hearth</title>") != std::string::npos);
    CHECK(page.body.find("X-Hearth-Token") != std::string::npos);  // the page sends the token on every call
    CHECK(page.body.find("innerHTML") == std::string::npos);       // model text is never inserted as HTML

    FakeConnection st = call(app, "GET", "/api/status");
    CHECK(st.status == 200);
    CHECK(st.body.find("\"loaded\":null") != std::string::npos);
    CHECK(st.body.find("\"job\":\"\"") != std::string::npos);
    CHECK(st.body.find("\"machine\":{\"cpu\":") != std::string::npos);
    // The page compares the power source case-sensitively with "ac" and
    // "battery"; "AC" made it warn "on battery" while plugged in.
    const PowerSource ps = power_source();
    const char* power = ps == PowerSource::kAc        ? "\"power\":\"ac\""
                        : ps == PowerSource::kBattery ? "\"power\":\"battery\""
                                                      : "\"power\":\"unknown\"";
    CHECK_MESSAGE(st.body.find(power) != std::string::npos, "expected " << power);
    CHECK(page.body.find("power === \"battery\"") != std::string::npos);
    CHECK(page.body.find("power !== \"ac\"") != std::string::npos);
    CHECK(call(app, "GET", "/api/history").status == 200);

    CHECK(call(app, "GET", "/api/chat").status == 405);
    CHECK(call(app, "POST", "/api/status").status == 405);
    CHECK(call(app, "POST", "/").status == 405);
    CHECK(call(app, "GET", "/nothing").status == 404);
    CHECK(call(app, "POST", "/api/nothing").status == 404);
    CHECK(call(app, "POST", "/api/chat", "message=%zz").status == 400);  // malformed form
    FakeConnection chat = call(app, "POST", "/api/chat", "message=hello");
    CHECK(chat.status == 409);
    CHECK(chat.body.find("load a model first") != std::string::npos);
    CHECK(call(app, "POST", "/api/chat", "message=").status == 400);  // empty message
    CHECK(call(app, "POST", "/api/chat", "message=hi&temperature=abc").status == 400);
    CHECK(call(app, "POST", "/api/chat", "message=hi&max_tokens=999999").status == 400);
    CHECK(call(app, "POST", "/api/load", "model=x.gguf&mode=fast").status == 400);  // unknown mode
    CHECK(call(app, "POST", "/api/load", "model=x.gguf&mode=auto").status == 400);  // CPU + GPU modes are gone
    CHECK(call(app, "POST", "/api/load", "model=x.gguf&mode=both").status == 400);
    CHECK(call(app, "POST", "/api/load", "model=no-such-file.gguf&mode=cpu").status == 400);
    CHECK(call(app, "POST", "/api/bench", "reps=10").status == 400);          // at most 9 runs
    CHECK(call(app, "POST", "/api/bench", "prompt_tokens=8").status == 400);  // at least 16 tokens
    CHECK(call(app, "POST", "/api/bench", "backends=hetero").status == 400);  // current or both
    CHECK(call(app, "POST", "/api/compare", "text=hi").status == 404);        // the CPU + GPU speed test is gone
    CHECK(call(app, "POST", "/api/calibrate").status == 404);                 // and so is the calibration

    // A benchmark without a model is an error event, not a crash.
    FakeConnection bench = call(app, "POST", "/api/bench", "reps=1");
    CHECK(bench.streamed);
    const auto ev = bench.events();
    REQUIRE(ev.size() == 1);
    CHECK(ev[0].first == "error");

    // Stop with no job is harmless: it leaves no pending stop behind.
    CHECK(call(app, "POST", "/api/stop").status == 200);
    CHECK_FALSE(cancel_requested());
    CHECK(app.keep_running());
    CHECK(call(app, "POST", "/api/quit").status == 200);
    CHECK_FALSE(app.keep_running());
    CHECK_FALSE(cancel_requested());
  }
}

TEST_SUITE("quick") {
  // Step W3: "Add model" (the file dialog itself, /api/pick, is driven by
  // hand: it needs a desktop) and the per-file check.
  TEST_CASE("ui app: add a model by path, see why it cannot run, forget it; the list survives a restart") {
    const cli::UiOptions o = test_options();
    const auto dir = test::repo_dir() / "build" / "test-tmp";
    // A valid GGUF file of a model family this version does not run.
    const auto mistral = dir / "ui-add-test-family.gguf";
    {
      std::vector<std::byte> bytes = test::tiny_llama(1, 4, 2).build();
      // The same file, with another architecture name of the same length.
      const std::string from = "llama", to = "mamba";
      auto* p = reinterpret_cast<char*>(bytes.data());
      std::string_view view(p, bytes.size());
      const std::size_t at = view.find("general.architecture");
      REQUIRE(at != std::string_view::npos);
      const std::size_t value = view.find(from, at);
      REQUIRE(value != std::string_view::npos);
      std::copy(to.begin(), to.end(), p + value);
      std::ofstream(mistral, std::ios::binary).write(p, static_cast<std::streamsize>(bytes.size()));
    }
    const auto not_gguf = dir / "ui-add-test.txt";
    std::ofstream(not_gguf) << "not a model";
    {
      cli::UiApp app(o);
      CHECK(call(app, "POST", "/api/add", "model=").status == 400);
      FakeConnection missing = call(app, "POST", "/api/add", "model=no-such-file.gguf");
      CHECK(missing.status == 400);
      CHECK(missing.body.find("is not a file") != std::string::npos);
      FakeConnection txt = call(app, "POST", "/api/add", "model=" + path_to_utf8(not_gguf));
      CHECK(txt.status == 400);
      CHECK(txt.body.find("Only .gguf") != std::string::npos);
      // Quotes from Explorer's "Copy as path" are accepted.
      FakeConnection add = call(app, "POST", "/api/add", "model=%22" + path_to_utf8(mistral) + "%22");
      REQUIRE(add.status == 200);
      CHECK(add.body.find("\"status\":\"unsupported\"") != std::string::npos);
      CHECK(add.body.find("Model family \\\"mamba\\\" is not supported yet") != std::string::npos);
      FakeConnection st = call(app, "GET", "/api/status");
      CHECK(st.body.find("\"name\":\"ui-add-test-family.gguf\"") != std::string::npos);
      CHECK(st.body.find("\"added\":true") != std::string::npos);
      CHECK(st.body.find("\"arch\":\"mamba\"") != std::string::npos);
      // Loading it fails with the reason, not a crash.
      CHECK(call(app, "POST", "/api/load", "model=" + path_to_utf8(mistral) + "&mode=cpu").status == 400);
    }
    {
      cli::UiApp app(o);  // a restart: the added file is still listed
      CHECK(call(app, "GET", "/api/status").body.find("ui-add-test-family.gguf") != std::string::npos);
      const std::string canon = path_to_utf8(std::filesystem::weakly_canonical(mistral));
      CHECK(call(app, "POST", "/api/forget", "model=" + canon).status == 200);
      CHECK(call(app, "GET", "/api/status").body.find("ui-add-test-family.gguf") == std::string::npos);
    }
    {
      cli::UiApp app(o);  // and forgetting it lasts too
      CHECK(call(app, "GET", "/api/status").body.find("ui-add-test-family.gguf") == std::string::npos);
    }
    std::filesystem::remove(mistral);
    std::filesystem::remove(not_gguf);
    std::filesystem::remove(o.models_file);
  }
}

TEST_SUITE("model") {
  TEST_CASE("ui app: load the 1B Q8_0 on the CPU, chat, continue the conversation, start over") {
    const auto path = test::repo_dir() / "models" / "Llama-3.2-1B-Instruct-Q8_0-pure.gguf";
    if (!std::filesystem::exists(path)) {
      MESSAGE("missing (run scripts\\fetch-models.ps1): Llama-3.2-1B-Instruct-Q8_0-pure.gguf");
      return;
    }
    cli::UiApp app(test_options());
    const std::string model = "model=" + path_to_utf8(path);
    FakeConnection st = call(app, "POST", "/api/load", model + "&mode=cpu");
    REQUIRE(st.status == 200);
    CHECK(st.body.find("\"mode\":\"cpu\"") != std::string::npos);
    CHECK(st.body.find("\"label\":\"Processor (CPU), ") != std::string::npos);
    CHECK(st.body.find("\"working_set_gib\":") != std::string::npos);  // the app's own memory use
    CHECK(st.body.find("\"name\":\"Llama-3.2-1B-Instruct-Q8_0-pure.gguf\"") != std::string::npos);

    auto chat = [&](const std::string& msg) {
      FakeConnection c = call(app, "POST", "/api/chat", "message=" + msg + "&temperature=0&max_tokens=24");
      std::string text, done;
      for (const auto& [name, data] : c.events()) {
        if (name == "token") {  // {"text":"..."}: the model's text here is plain words, no escapes
          REQUIRE(data.starts_with("{\"text\":\""));
          REQUIRE(data.ends_with("\"}"));
          text += data.substr(9, data.size() - 11);
        }
        if (name == "done") done = data;
        CHECK(name != "error");
      }
      CHECK_FALSE(done.empty());
      CHECK(done.find("\"prompt_tokens\":") != std::string::npos);
      CHECK(done.find("\"total_s\":") != std::string::npos);
      CHECK(done.find("\"mode\":\"cpu\"") != std::string::npos);
      return text;
    };
    const std::string first = chat("What+is+the+capital+of+France%3F+Answer+in+one+word.");
    CAPTURE(first);
    CHECK(first.find("Paris") != std::string::npos);
    st = call(app, "GET", "/api/status");
    CHECK(st.body.find("\"context_used\":0") == std::string::npos);  // the conversation is kept
    CHECK(st.body.find("\"replies\":[{") != std::string::npos);
    const std::string second = chat("And+of+Italy%3F");
    CAPTURE(second);
    CHECK(second.find("Rome") != std::string::npos);  // it remembered the question's form
    CHECK(call(app, "POST", "/api/reset").status == 200);
    st = call(app, "GET", "/api/status");
    CHECK(st.body.find("\"context_used\":0") != std::string::npos);
  }
}
