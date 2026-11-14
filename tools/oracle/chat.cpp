// hetero-oracle-chat: llama.cpp's own rendering of the model's chat template
// (its Jinja engine, common/chat.cpp) and its tokenization of the result, for
// comparison with hetero's template.
//
//   hetero-oracle-chat --model F --messages FILE.json --date YYYY-MM-DD
//
// FILE.json is [{"role": "system"|"user"|"assistant", "content": "..."}, ...].
//
// The date. The Llama-3.1/3.2 template takes the date from a variable
// `date_string` if one is defined, and otherwise from strftime_now("%d %b %Y"),
// which llama.cpp b11960 always evaluates at the real current time (its Jinja
// context reads std::time(nullptr)). So a fixed date is passed the way
// llama.cpp users pass template variables, as chat_template_kwargs
// (`--chat-template-kwargs` on its command line). Its text is produced by the
// call strftime_now makes, strftime("%d %b %Y", localtime(t)), with t = noon
// on --date. That this is equivalent to llama.cpp's own clock path is checked
// on every run: today's prompt is rendered both ways and must be identical.
//
// The prompt is tokenized as llama.cpp's server tokenizes a chat prompt: with
// BOS and with special-token parsing on.
// Output (stdout): {"prompt": "...", "tokens": [...], "date_string": "..."}.
//
// Exit status: 0 on success, 1 on a run-time error, 2 on a usage error.
#include <charconv>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "chat.h"
#include "common.h"
#include "llama.h"
#include "nlohmann/json.hpp"

namespace {

struct UsageError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

constexpr const char* kUsage = "usage: hetero-oracle-chat --model F --messages FILE.json --date YYYY-MM-DD\n";

// What the template's strftime_now("%d %b %Y") returns when the clock reads `t`.
std::string strftime_now_at(std::time_t t) {
  std::tm local{};
  if (localtime_s(&local, &t) != 0) throw std::runtime_error("localtime_s failed");
  char buf[100];
  if (std::strftime(buf, sizeof buf, "%d %b %Y", &local) == 0) throw std::runtime_error("strftime failed");
  return buf;
}

// Noon, local time, on YYYY-MM-DD; a UsageError unless that is a real date.
std::time_t noon_on(const std::string& date) {
  std::tm tm{};
  const auto field = [&](std::size_t at, std::size_t len, int& out) {
    const char* first = date.data() + at;
    const auto [end, ec] = std::from_chars(first, first + len, out);
    return ec == std::errc() && end == first + len && first[0] >= '0' && first[0] <= '9';
  };
  if (date.size() != 10 || date[4] != '-' || date[7] != '-' || !field(0, 4, tm.tm_year) || !field(5, 2, tm.tm_mon) ||
      !field(8, 2, tm.tm_mday)) {
    throw UsageError("--date must be YYYY-MM-DD");
  }
  tm.tm_year -= 1900;
  tm.tm_mon -= 1;
  tm.tm_hour = 12;
  tm.tm_isdst = -1;
  const std::tm asked = tm;
  const std::time_t t = std::mktime(&tm);
  // mktime normalises 2026-02-30 to 2 March; such a date is refused instead.
  if (t == -1 || tm.tm_year != asked.tm_year || tm.tm_mon != asked.tm_mon || tm.tm_mday != asked.tm_mday) {
    throw UsageError("--date " + date + " is not a date");
  }
  return t;
}

// The rendered prompt; with `date_string`, the template variable of that name is set.
std::string render(const common_chat_templates* tmpls, const std::vector<common_chat_msg>& msgs,
                   const std::string* date_string) {
  common_chat_templates_inputs inputs;
  inputs.messages = msgs;
  inputs.add_generation_prompt = true;
  inputs.use_jinja = true;
  if (date_string != nullptr) inputs.chat_template_kwargs["date_string"] = nlohmann::json(*date_string).dump();
  return common_chat_templates_apply(tmpls, inputs).prompt.str();
}

int run(int argc, char** argv) {
  std::string model, messages, date;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (i + 1 >= argc) throw UsageError(a + " needs a value");
    if (a == "--model") {
      model = argv[++i];
    } else if (a == "--messages") {
      messages = argv[++i];
    } else if (a == "--date") {
      date = argv[++i];
    } else {
      throw UsageError("unknown argument '" + a + "'");
    }
  }
  if (model.empty() || messages.empty() || date.empty()) {
    throw UsageError("--model, --messages and --date are required");
  }
  const std::string date_string = strftime_now_at(noon_on(date));

  std::ifstream in(messages, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + messages);
  const nlohmann::json j = nlohmann::json::parse(std::string(std::istreambuf_iterator<char>(in), {}));
  std::vector<common_chat_msg> msgs;
  for (const auto& m : j) {
    common_chat_msg msg;
    msg.role = m.at("role").get<std::string>();
    msg.content = m.at("content").get<std::string>();
    msgs.push_back(std::move(msg));
  }

  llama_backend_init();
  llama_model_params mp = llama_model_default_params();
  mp.vocab_only = true;
  llama_model* lm = llama_model_load_from_file(model.c_str(), mp);
  if (lm == nullptr) throw std::runtime_error("cannot load " + model);
  {
    const auto tmpls = common_chat_templates_init(lm, "");
    // The self-check: llama.cpp's own clock path and the date_string path agree
    // today. Tried twice, in case the day changes between the two renderings.
    bool same = false;
    for (int attempt = 0; attempt < 2 && !same; ++attempt) {
      const std::string own = render(tmpls.get(), msgs, nullptr);
      const std::string today = strftime_now_at(std::time(nullptr));
      same = own == render(tmpls.get(), msgs, &today);
    }
    if (!same) {
      throw std::runtime_error("setting date_string does not reproduce llama.cpp's own rendering of today's prompt");
    }
    const std::string prompt = render(tmpls.get(), msgs, &date_string);
    const std::vector<llama_token> tokens =
        common_tokenize(llama_model_get_vocab(lm), prompt, /*add_special=*/true, /*parse_special=*/true);
    const nlohmann::json out = {{"prompt", prompt}, {"tokens", tokens}, {"date_string", date_string}};
    const std::string s = out.dump() + "\n";
    std::fwrite(s.data(), 1, s.size(), stdout);
  }
  llama_model_free(lm);
  llama_backend_free();
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const UsageError& e) {
    std::fprintf(stderr, "hetero-oracle-chat: error: %s\n%s", e.what(), kUsage);
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "hetero-oracle-chat: error: %s\n", e.what());
    return 1;
  }
}
