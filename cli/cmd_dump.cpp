// `hearth dump`: run a forward pass teacher-forced over given token ids and
// write named intermediate tensors in the oracle's format
// (tools/oracle/README.md), for tools/layer_diff.py.
#include <algorithm>
#include <charconv>
#include <format>
#include <fstream>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

#include "commands.h"
#include "core/cpu_features.h"
#include "core/error.h"
#include "core/log.h"
#include "core/platform.h"
#include "core/thread_pool.h"
#include "gguf/gguf.h"
#include "model/llama_ref.h"
#include "session.h"

namespace hearth::cli {
namespace {

// Splits on `sep`; an empty item is an error (a stray comma is a typo).
std::vector<std::string> split(std::string_view s, char sep, const char* what) {
  std::vector<std::string> out;
  for (;;) {
    const auto at = s.find(sep);
    const std::string_view item = s.substr(0, at);
    if (item.empty()) fail(ErrorKind::kInvalidArgument, "--{} has an empty item", what);
    out.emplace_back(item);
    if (at == std::string_view::npos) return out;
    s.remove_prefix(at + 1);
  }
}

std::int64_t to_int(std::string_view s, const char* what) {
  std::int64_t v = 0;
  const auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
  if (s.empty() || ec != std::errc() || p != s.data() + s.size()) {
    fail(ErrorKind::kInvalidArgument, "bad {} '{}'", what, s);
  }
  return v;
}

// "all" -> every position; "0-63,100,2040-2047" -> those positions, each < n.
std::vector<std::int64_t> parse_positions(const std::string& s, std::int64_t n) {
  std::vector<std::int64_t> out;
  if (s == "all") {
    out.resize(static_cast<std::size_t>(n));
    std::iota(out.begin(), out.end(), std::int64_t{0});
    return out;
  }
  for (const std::string& part : split(s, ',', "positions")) {
    const auto dash = part.find('-');
    const std::int64_t a = to_int(std::string_view(part).substr(0, dash), "position");
    const std::int64_t b = dash == std::string::npos ? a : to_int(std::string_view(part).substr(dash + 1), "position");
    if (a < 0 || b < a || b >= n) {
      fail(ErrorKind::kInvalidArgument, "position range '{}' is not inside 0..{} ({} tokens)", part, n - 1, n);
    }
    for (std::int64_t p = a; p <= b; ++p) out.push_back(p);
  }
  return out;
}

}  // namespace

int cmd_dump(Args& args) {
  const ModelOptions mo = model_options(args, 4096, "cpu-ref");
  const auto tokens_arg = args.value("tokens");
  const auto tokens_file = args.value("tokens-file");
  const auto out = args.value("out");
  const std::string names = args.value_or("names", "l_out,result_output");
  const std::string positions = args.value_or("positions", "all");
  args.finish();
  if (!out || tokens_arg.has_value() == tokens_file.has_value()) {
    fail(ErrorKind::kInvalidArgument, "dump needs --model F, --out DIR, and one of --tokens LIST and --tokens-file F");
  }

  std::vector<std::int32_t> tokens;
  if (tokens_arg) {
    for (const std::string& t : split(*tokens_arg, ',', "tokens")) {
      tokens.push_back(static_cast<std::int32_t>(to_int(t, "token")));
    }
  } else {
    std::ifstream f(path_from_utf8(*tokens_file));
    if (!f) fail(ErrorKind::kIo, "cannot read '{}'", *tokens_file);
    for (std::string line; std::getline(f, line);) {
      while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
      if (!line.empty()) tokens.push_back(static_cast<std::int32_t>(to_int(line, "token")));
    }
  }
  if (tokens.empty()) fail(ErrorKind::kInvalidArgument, "no tokens given");
  const std::vector<std::string> wanted = split(names, ',', "names");
  for (const std::string& n : wanted) {
    if (!model::TensorDump::known(n)) {
      fail(ErrorKind::kInvalidArgument,
           "--names: '{}' is not a tensor this forward pass records (attn_norm, Qcur, Kcur, Vcur, attn_out, ffn_inp, "
           "ffn_norm, ffn_out, l_out, each also as NAME-<layer>, result_norm, result_output)",
           n);
    }
  }

  Session s(mo);
  model::TensorDump dump(wanted, parse_positions(positions, static_cast<std::int64_t>(tokens.size())));
  const std::string producer = "hearth " + mo.backend;
  log_info("dump: {} tokens through {} ({} threads)", tokens.size(), producer, mo.threads);
  std::vector<float> last(s.llm.config().n_vocab);
  s.llm.forward(tokens, last, false, &dump);
  if (const auto missing = dump.unmatched(); !missing.empty()) {
    std::string list;
    for (const std::string& m : missing) list += (list.empty() ? "" : ", ") + m;
    fail(ErrorKind::kInvalidArgument, "--names: no tensor of this model matches {}", list);
  }
  dump.write(path_from_utf8(*out), producer, mo.model, tokens, mo.ctx);

  // The most likely next tokens after the last position, as a quick sanity check.
  std::vector<std::int32_t> ids(last.size());
  std::iota(ids.begin(), ids.end(), 0);
  const std::size_t top = std::min<std::size_t>(5, ids.size());
  std::partial_sort(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(top), ids.end(),
                    [&](std::int32_t a, std::int32_t b) {
                      return last[static_cast<std::size_t>(a)] != last[static_cast<std::size_t>(b)]
                                 ? last[static_cast<std::size_t>(a)] > last[static_cast<std::size_t>(b)]
                                 : a < b;
                    });
  std::string line;
  for (std::size_t i = 0; i < top; ++i) {
    line += std::format("{}{} {}", i > 0 ? ", " : "", ids[i], last[static_cast<std::size_t>(ids[i])]);
  }
  log_info("dump: top-{} after position {}: {}", top, tokens.size() - 1, line);
  log_info("dump: wrote {}", *out);
  return 0;
}

}  // namespace hearth::cli
