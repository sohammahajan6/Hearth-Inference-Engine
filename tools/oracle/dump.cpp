// hetero-oracle-dump: llama.cpp's own forward pass over a list of token ids,
// teacher-forced in one batch, with chosen intermediate tensors written to
// disk exactly as llama.cpp's graph callback (cb_eval) sees them. hetero's own
// dumps (`hetero dump`) are compared with these by tools/layer_diff.py; see
// docs/architecture/05-forward-pass.md, and README.md here for the format.
//
// Exit status: 0 on success, 1 on a run-time error, 2 on a usage error.
// Standard input is never read.
#include <algorithm>
#include <atomic>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"
#include "llama-cpp.h"
#include "llama.h"

namespace {

namespace fs = std::filesystem;

static_assert(std::endian::native == std::endian::little, "the dump files are little-endian float32");

constexpr const char* kUsage =
    "usage: hetero-oracle-dump --model F (--tokens ID,ID,... | --tokens-file F) --out DIR\n"
    "                          --names NAME,... --positions all|LIST --ctx N [--f32-kv] [--threads N]\n"
    "       hetero-oracle-dump --model F (--tokens ... | --tokens-file F) --out DIR --ctx N --generate N [--f32-kv]\n";

constexpr const char* kHelp =
    "\n"
    "Evaluates the tokens teacher-forced in one llama.cpp batch (CPU backend, flash attention off,\n"
    "logits for every token) and writes the requested graph tensors to DIR: index.json and one\n"
    "NAME.bin of little-endian float32 per tensor (format: tools/oracle/README.md).\n"
    "\n"
    "  --model F         GGUF model file\n"
    "  --tokens LIST     token ids, comma separated (llama-tokenize --ids prints them)\n"
    "  --tokens-file F   token ids, one per line\n"
    "  --out DIR         output directory, created if missing\n"
    "  --names LIST      graph tensor names, comma separated. A name matches the tensor with exactly\n"
    "                    that name and every per-layer tensor NAME-<layer>: l_out matches l_out-0,\n"
    "                    l_out-1, ...; l_out-3 matches only layer 3. For Llama: attn_norm Qcur Kcur\n"
    "                    Vcur attn_out ffn_inp ffn_norm ffn_out l_out result_norm result_output\n"
    "  --positions P     all, or a comma list of positions and inclusive ranges, e.g. 0-63,100\n"
    "  --ctx N           context size; at least the number of tokens\n"
    "  --f32-kv          F32 KV cache (default: F16, as in llama.cpp)\n"
    "  --threads N       CPU threads (default 4)\n"
    "  --generate N      instead of a dump: greedy decoding of up to N tokens after the prompt, written to\n"
    "                    DIR/generate.json with each step's top-2 probability gap (no --names/--positions)\n";

struct UsageError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// ---------------------------------------------------------------- text helpers

std::string_view trim(std::string_view s) {
  const auto space = [](char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
  while (!s.empty() && space(s.front())) s.remove_prefix(1);
  while (!s.empty() && space(s.back())) s.remove_suffix(1);
  return s;
}

// Splits on `sep` and trims each item; empty items are kept so callers can reject them.
std::vector<std::string_view> split(std::string_view s, char sep) {
  std::vector<std::string_view> out;
  for (;;) {
    const size_t at = s.find(sep);
    out.push_back(trim(s.substr(0, at)));
    if (at == std::string_view::npos) return out;
    s.remove_prefix(at + 1);
  }
}

int64_t parse_int(std::string_view s, std::string_view what, int64_t lo, int64_t hi) {
  int64_t v = 0;
  const auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
  if (s.empty() || ec != std::errc() || end != s.data() + s.size()) {
    throw UsageError(std::format("{} '{}' is not an integer", what, s));
  }
  if (v < lo || v > hi) throw UsageError(std::format("{} {} is outside [{}, {}]", what, v, lo, hi));
  return v;
}

// Paths arrive as UTF-8 (the manifest makes UTF-8 the process code page).
fs::path utf8_path(std::string_view s) {
  std::u8string u;
  u.reserve(s.size());
  for (const char c : s) u.push_back(static_cast<char8_t>(c));
  return fs::path(u);
}

// The base of a graph tensor name: "l_out-3" -> "l_out". A name without a
// numeric layer suffix ("result_output", "Qcur-0 (reshaped)") is its own base.
std::string_view base_of(std::string_view name) {
  const size_t dash = name.rfind('-');
  if (dash == std::string_view::npos || dash + 1 == name.size()) return name;
  for (const char c : name.substr(dash + 1)) {
    if (c < '0' || c > '9') return name;
  }
  return name.substr(0, dash);
}

std::string shape_of(const ggml_tensor* t) {
  std::string s = "[";
  for (int i = 0; i < ggml_n_dims(t); ++i) s += std::format("{}{}", i > 0 ? ", " : "", t->ne[i]);
  return s + "]";
}

// ---------------------------------------------------------------- command line

struct Options {
  std::string model, out, names, positions, tokens, tokens_file;
  bool has_tokens = false, has_tokens_file = false, f32_kv = false, help = false;
  int generate = 0;  // --generate N: greedy decoding instead of a tensor dump
  int64_t ctx = 0;
  int threads = 4;
};

Options parse_args(int argc, char** argv) {
  Options o;
  std::set<std::string, std::less<>> given;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      o.help = true;
      continue;
    }
    if (!given.insert(arg).second) throw UsageError(std::format("{} is given twice", arg));
    if (arg == "--f32-kv") {
      o.f32_kv = true;
      continue;
    }
    const auto value = [&]() -> std::string {
      if (i + 1 >= argc || argv[i + 1][0] == '\0') throw UsageError(std::format("{} needs a value", arg));
      return argv[++i];
    };
    if (arg == "--model") {
      o.model = value();
    } else if (arg == "--tokens") {
      o.tokens = value();
      o.has_tokens = true;
    } else if (arg == "--tokens-file") {
      o.tokens_file = value();
      o.has_tokens_file = true;
    } else if (arg == "--out") {
      o.out = value();
    } else if (arg == "--names") {
      o.names = value();
    } else if (arg == "--positions") {
      o.positions = value();
    } else if (arg == "--ctx") {
      o.ctx = parse_int(value(), "--ctx", 1, int64_t{1} << 24);
    } else if (arg == "--generate") {
      o.generate = static_cast<int>(parse_int(value(), "--generate", 1, 1 << 20));
    } else if (arg == "--threads") {
      o.threads = static_cast<int>(parse_int(value(), "--threads", 1, 512));
    } else {
      throw UsageError(std::format("unknown argument '{}'", arg));
    }
  }
  if (o.help) return o;
  if (o.generate > 0) {
    for (const char* required : {"--model", "--out", "--ctx"}) {
      if (!given.contains(required)) throw UsageError(std::format("{} is required", required));
    }
    if (given.contains("--names") || given.contains("--positions")) {
      throw UsageError("--generate writes generate.json; it does not take --names or --positions");
    }
  } else {
    for (const char* required : {"--model", "--out", "--names", "--positions", "--ctx"}) {
      if (!given.contains(required)) throw UsageError(std::format("{} is required", required));
    }
  }
  if (o.has_tokens == o.has_tokens_file) throw UsageError("give exactly one of --tokens and --tokens-file");
  return o;
}

std::vector<llama_token> parse_token_list(std::string_view list) {
  std::vector<llama_token> out;
  for (const std::string_view item : split(list, ',')) {
    if (item.empty()) throw UsageError("--tokens has an empty item");
    out.push_back(static_cast<llama_token>(parse_int(item, "token id", 0, INT32_MAX)));
  }
  return out;
}

std::vector<llama_token> read_token_file(const std::string& path) {
  std::ifstream in(utf8_path(path), std::ios::binary);
  if (!in) throw std::runtime_error(std::format("cannot open --tokens-file '{}'", path));
  std::vector<llama_token> out;
  std::string line;
  for (int64_t n = 1; std::getline(in, line); ++n) {
    std::string_view item = line;
    if (n == 1 && item.starts_with("\xEF\xBB\xBF")) item.remove_prefix(3);  // UTF-8 byte order mark
    item = trim(item);
    if (item.empty()) continue;
    out.push_back(static_cast<llama_token>(parse_int(item, std::format("token id on line {}", n), 0, INT32_MAX)));
  }
  if (in.bad()) throw std::runtime_error(std::format("cannot read --tokens-file '{}'", path));
  return out;
}

// "all", or "0-63,100,2040-2047": sorted, without duplicates.
std::vector<int64_t> parse_positions(std::string_view spec, int64_t n_tokens) {
  std::set<int64_t> chosen;
  if (trim(spec) == "all") {
    for (int64_t p = 0; p < n_tokens; ++p) chosen.insert(p);
  } else {
    for (const std::string_view item : split(spec, ',')) {
      if (item.empty()) throw UsageError("--positions has an empty item");
      const size_t dash = item.find('-');
      if (dash == std::string_view::npos) {
        chosen.insert(parse_int(item, "position", 0, n_tokens - 1));
        continue;
      }
      const int64_t a = parse_int(trim(item.substr(0, dash)), "position", 0, n_tokens - 1);
      const int64_t b = parse_int(trim(item.substr(dash + 1)), "position", 0, n_tokens - 1);
      if (b < a) throw UsageError(std::format("position range '{}' is empty", item));
      for (int64_t p = a; p <= b; ++p) chosen.insert(p);
    }
  }
  return {chosen.begin(), chosen.end()};
}

std::vector<std::string> parse_names(std::string_view list) {
  std::vector<std::string> out;
  for (const std::string_view item : split(list, ',')) {
    if (item.empty()) throw UsageError("--names has an empty item");
    for (const char c : item) {
      // Each name becomes a file name.
      if (static_cast<unsigned char>(c) < 0x20 || std::strchr("<>:\"/\\|?*", c) != nullptr) {
        throw UsageError(std::format("--names: '{}' is not usable as a file name", item));
      }
    }
    if (std::find(out.begin(), out.end(), item) == out.end()) out.emplace_back(item);
  }
  return out;
}

// ---------------------------------------------------------------- capture

// One dumped tensor: the last occurrence of its name in the graph.
struct Dumped {
  std::string name;
  int64_t width = 0;             // floats per position: the product of the axes below the token axis
  std::vector<int64_t> ggml_ne;  // ggml shape of the last occurrence, up to and including the token axis
  std::string ggml_type;
  int count = 0;       // how many graph nodes carry this name
  std::string source;  // "cb_eval", or "llama_get_logits_ith" for logits the graph did not show
};

struct Capture {
  std::vector<std::string> bases;
  std::vector<int64_t> positions;
  int64_t n_tokens = 0;
  fs::path dir;

  std::vector<Dumped> dumped;  // in graph order of first appearance
  std::map<std::string, size_t, std::less<>> index;
  std::map<std::string, int, std::less<>> base_hits;  // requested name -> graph nodes captured for it
  std::set<std::string, std::less<>> seen;            // bases of all graph node names, for error messages
  std::optional<std::string> error;                   // the first failure inside the callback
  std::vector<uint8_t> raw;
  std::vector<float> row;

  bool matches(std::string_view name, std::string_view base) const { return name == base || base_of(name) == base; }

  bool wants(std::string_view name) const {
    return std::any_of(bases.begin(), bases.end(), [&](const std::string& b) { return matches(name, b); });
  }

  Dumped& entry(const std::string& name) {
    const auto it = index.find(name);
    if (it != index.end()) return dumped[it->second];
    index.emplace(name, dumped.size());
    dumped.push_back(Dumped{.name = name});
    return dumped.back();
  }

  // Writes the selected positions of graph node `t` to DIR/<name>.bin,
  // replacing what an earlier node of the same name wrote.
  void record(const ggml_tensor* t) {
    const std::string name = ggml_get_name(t);
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16) {
      throw std::runtime_error(std::format("graph tensor '{}' has type {}; only f32, f16 and bf16 can be dumped", name,
                                           ggml_type_name(t->type)));
    }
    // The token axis is the outermost axis and has n_tokens entries. With a
    // single token, the whole tensor belongs to it.
    const int n_dims = ggml_n_dims(t);
    const int k = n_tokens > 1 ? n_dims - 1 : n_dims;
    if (n_tokens > 1 && t->ne[k] != n_tokens) {
      throw std::runtime_error(
          std::format("graph tensor '{}' has shape {}: its outermost axis is not the {} tokens, so it cannot be dumped "
                      "per position",
                      name, shape_of(t), n_tokens));
    }
    int64_t width = 1;
    for (int j = 0; j < k; ++j) width *= t->ne[j];
    const size_t esize = ggml_type_size(t->type);
    const size_t token_stride = k < GGML_MAX_DIMS ? t->nb[k] : 0;  // only p = 0 exists when k == GGML_MAX_DIMS
    const bool contiguous = ggml_is_contiguous(t);
    if (!contiguous) {
      raw.resize(ggml_nbytes(t));
      ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    }
    row.resize(static_cast<size_t>(width));

    const fs::path file = dir / utf8_path(name + ".bin");
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error(std::format("cannot create {}", (dir / name).string() + ".bin"));
    for (const int64_t p : positions) {
      const size_t offset = static_cast<size_t>(p) * token_stride;
      if (contiguous) {
        raw.resize(static_cast<size_t>(width) * esize);
        ggml_backend_tensor_get(t, raw.data(), offset, raw.size());
        convert(raw.data(), t->type, row.data(), width);
      } else {
        gather(t, k, raw.data() + offset);
      }
      out.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(row.size() * sizeof(float)));
    }
    out.close();
    if (!out) throw std::runtime_error(std::format("cannot write {}.bin", (dir / name).string()));

    Dumped& d = entry(name);
    d.width = width;
    d.ggml_ne.assign(t->ne, t->ne + std::min(k + 1, GGML_MAX_DIMS));
    d.ggml_type = ggml_type_name(t->type);
    d.count += 1;
    d.source = "cb_eval";
    for (const std::string& b : bases) {
      if (matches(name, b)) base_hits[b] += 1;
    }
  }

  static void convert(const uint8_t* src, ggml_type type, float* dst, int64_t n) {
    if (type == GGML_TYPE_F32) {
      std::memcpy(dst, src, static_cast<size_t>(n) * sizeof(float));
    } else if (type == GGML_TYPE_F16) {
      ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t*>(src), dst, n);
    } else {
      ggml_bf16_to_fp32_row(reinterpret_cast<const ggml_bf16_t*>(src), dst, n);
    }
  }

  // One position of a non-contiguous tensor: the axes below the token axis
  // `k`, read through the strides nb[] from `base`, ne[0] fastest.
  void gather(const ggml_tensor* t, int k, const uint8_t* base) {
    int64_t ne[GGML_MAX_DIMS];
    for (int j = 0; j < GGML_MAX_DIMS; ++j) ne[j] = j < k ? t->ne[j] : 1;
    size_t n = 0;
    for (int64_t i3 = 0; i3 < ne[3]; ++i3) {
      for (int64_t i2 = 0; i2 < ne[2]; ++i2) {
        for (int64_t i1 = 0; i1 < ne[1]; ++i1) {
          for (int64_t i0 = 0; i0 < ne[0]; ++i0) {
            const uint8_t* e = base + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
            convert(e, t->type, &row[n++], 1);
          }
        }
      }
    }
  }
};

// llama.cpp's graph callback. With ask == true the scheduler asks whether the
// node is wanted; it then computes the graph up to that node and calls back
// with ask == false, when the node's data is final. Returning false stops it.
bool on_graph_node(ggml_tensor* t, bool ask, void* user_data) {
  auto& cap = *static_cast<Capture*>(user_data);
  if (cap.error) return false;
  const std::string_view name = ggml_get_name(t);
  if (ask) {
    cap.seen.emplace(base_of(name));
    return cap.wants(name);
  }
  try {
    cap.record(t);
    return true;
  } catch (const std::exception& e) {
    cap.error = e.what();
    return false;
  }
}

// llama.cpp and ggml logging: everything but debug output, to stderr.
void on_log(ggml_log_level level, const char* text, void* /*user_data*/) {
  static std::atomic<int> last{GGML_LOG_LEVEL_INFO};
  if (level != GGML_LOG_LEVEL_CONT) last = level;
  if (last == GGML_LOG_LEVEL_DEBUG) return;
  std::fputs(text, stderr);
}

// ---------------------------------------------------------------- outputs

// Logits as llama.cpp returns them to applications, for the selected positions.
void write_logits(llama_context* ctx, const fs::path& file, const std::vector<int64_t>& positions, int32_t n_vocab) {
  std::ofstream out(file, std::ios::binary | std::ios::trunc);
  for (const int64_t p : positions) {
    const float* logits = llama_get_logits_ith(ctx, static_cast<int32_t>(p));
    if (logits == nullptr) throw std::runtime_error(std::format("llama_get_logits_ith({}) returned no logits", p));
    out.write(reinterpret_cast<const char*>(logits), static_cast<std::streamsize>(n_vocab * sizeof(float)));
  }
  out.close();
  if (!out) throw std::runtime_error("cannot write result_output.bin");
}

// The graph's result_output must be bit-identical to the logits llama.cpp
// returns: this checks the position -> column mapping of the whole dump.
void verify_logits(llama_context* ctx, const fs::path& file, const std::vector<int64_t>& positions, int32_t n_vocab) {
  std::ifstream in(file, std::ios::binary);
  std::vector<float> row(static_cast<size_t>(n_vocab));
  for (const int64_t p : positions) {
    in.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(row.size() * sizeof(float)));
    const float* logits = llama_get_logits_ith(ctx, static_cast<int32_t>(p));
    if (!in || logits == nullptr) throw std::runtime_error(std::format("cannot compare the logits at position {}", p));
    if (std::memcmp(row.data(), logits, row.size() * sizeof(float)) != 0) {
      throw std::runtime_error(std::format(
          "result_output seen by cb_eval differs from llama_get_logits_ith at position {}: the position mapping "
          "is broken",
          p));
    }
  }
}

std::string json_string(std::string_view s) {
  std::string out = "\"";
  for (const char ch : s) {
    const auto c = static_cast<unsigned char>(ch);
    if (c == '"' || c == '\\') {
      out += '\\';
      out += ch;
    } else if (c < 0x20) {
      out += std::format("\\u{:04x}", static_cast<unsigned>(c));
    } else {
      out += ch;
    }
  }
  return out + "\"";
}

template <typename T>
std::string json_array(const std::vector<T>& v) {
  std::string out = "[";
  for (size_t i = 0; i < v.size(); ++i) out += std::format("{}{}", i > 0 ? ", " : "", v[i]);
  return out + "]";
}

void write_index(const fs::path& dir, const Options& o, const std::vector<llama_token>& tokens, uint32_t n_ctx_llama,
                 const Capture& cap) {
  std::string j = "{\n";
  j += std::format("  \"producer\": {},\n", json_string(std::format("hetero-oracle-dump (llama.cpp {}, ggml {} {})",
                                                                    llama_version(), ggml_version(), ggml_commit())));
  j += std::format("  \"model\": {},\n", json_string(o.model));
  j += std::format("  \"tokens\": {},\n", json_array(tokens));
  j += std::format("  \"ctx\": {},\n", o.ctx);
  j += std::format("  \"n_ctx_llama\": {},\n", n_ctx_llama);
  j += std::format("  \"f32_kv\": {},\n", o.f32_kv);
  j += "  \"flash_attn\": false,\n";
  j += std::format("  \"threads\": {},\n", o.threads);
  j += "  \"tensors\": [";
  for (size_t i = 0; i < cap.dumped.size(); ++i) {
    const Dumped& d = cap.dumped[i];
    j += std::format(
        "{}\n    {{\"name\": {}, \"ne\": [{}, {}], \"positions\": {}, \"file\": {}, \"count\": {}, \"ggml_ne\": {}, "
        "\"ggml_type\": {}, \"source\": {}}}",
        i > 0 ? "," : "", json_string(d.name), d.width, cap.n_tokens, json_array(cap.positions),
        json_string(d.name + ".bin"), d.count, json_array(d.ggml_ne), json_string(d.ggml_type), json_string(d.source));
  }
  j += "\n  ]\n}\n";
  // Written last, through a temporary name: a directory with an index.json holds a complete dump.
  const fs::path tmp = dir / "index.json.tmp";
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    out.write(j.data(), static_cast<std::streamsize>(j.size()));
    out.close();
    if (!out) throw std::runtime_error("cannot write index.json");
  }
  fs::rename(tmp, dir / "index.json");
}

std::string printable_piece(const llama_vocab* vocab, llama_token id) {
  char buf[256];
  const int32_t n = llama_token_to_piece(vocab, id, buf, static_cast<int32_t>(sizeof buf), 0, true);
  std::string out;
  for (int32_t i = 0; i < n; ++i) {
    const auto c = static_cast<unsigned char>(buf[i]);
    if (c >= 0x20 && c < 0x7f && c != '\'' && c != '\\') {
      out += static_cast<char>(c);
    } else {
      out += std::format("\\x{:02x}", static_cast<unsigned>(c));
    }
  }
  return out;
}

// ---------------------------------------------------------------- main flow

// --generate N: greedy decoding with llama.cpp's own forward pass, for the
// ACCEPTANCE greedy gate. The prompt is one batch, then one token per step.
// Writes DIR/generate.json: the prompt, the chosen IDs (an end-of-generation
// token, if chosen, is the last one), and at every step the top-2
// probability gap of the distribution the token was chosen from.
int generate(const Options& o, const std::vector<llama_token>& prompt, llama_model* model, const llama_vocab* vocab,
             int32_t n_vocab) {
  const auto n_prompt = static_cast<int64_t>(prompt.size());
  if (n_prompt + o.generate > o.ctx) {
    throw UsageError(
        std::format("{} prompt tokens + --generate {} do not fit in --ctx {}", n_prompt, o.generate, o.ctx));
  }
  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = static_cast<uint32_t>(o.ctx);
  cparams.n_batch = static_cast<uint32_t>(n_prompt);
  cparams.n_ubatch = static_cast<uint32_t>(n_prompt);
  cparams.n_seq_max = 1;
  cparams.n_threads = o.threads;
  cparams.n_threads_batch = o.threads;
  cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
  cparams.type_k = o.f32_kv ? GGML_TYPE_F32 : GGML_TYPE_F16;
  cparams.type_v = o.f32_kv ? GGML_TYPE_F32 : GGML_TYPE_F16;
  cparams.offload_kqv = false;
  cparams.op_offload = false;
  cparams.embeddings = false;
  const llama_context_ptr ctx(llama_init_from_model(model, cparams));
  if (!ctx) throw std::runtime_error("cannot create the llama.cpp context");
  const auto decode = [&](const llama_token* toks, int32_t n, llama_pos pos0) {
    const llama_batch_ext_ptr batch(llama_batch_ext_init(ctx.get()));
    for (int32_t i = 0; i < n; ++i) {
      const llama_pos pos = pos0 + i;
      if (llama_batch_ext_add_token(batch.get(), 0, toks[i]) != i || !llama_batch_ext_set_pos(batch.get(), i, &pos) ||
          !llama_batch_ext_set_output_logits(batch.get(), i, i == n - 1)) {
        throw std::runtime_error(std::format("cannot add token {} to the batch", i));
      }
    }
    const int32_t rc = llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get());
    if (rc != 0) throw std::runtime_error(std::format("llama_process(DECODE) failed with code {}", rc));
  };
  decode(prompt.data(), static_cast<int32_t>(n_prompt), 0);
  std::vector<llama_token> chosen;
  std::vector<double> gaps;
  std::string stop = "limit";
  for (int g = 0; g < o.generate; ++g) {
    const float* logits = llama_get_logits_ith(ctx.get(), -1);
    if (logits == nullptr) throw std::runtime_error("no logits after a decode");
    // Arg-max (the lowest ID on a tie) and the runner-up.
    int32_t best = 0, second = -1;
    for (int32_t i = 1; i < n_vocab; ++i) {
      if (logits[i] > logits[best]) {
        second = best;
        best = i;
      } else if (second < 0 || logits[i] > logits[second]) {
        second = i;
      }
    }
    double sum = 0.0;
    for (int32_t i = 0; i < n_vocab; ++i) sum += std::exp(static_cast<double>(logits[i]) - logits[best]);
    chosen.push_back(best);
    gaps.push_back((1.0 - std::exp(static_cast<double>(logits[second]) - logits[best])) / sum);
    if (llama_vocab_is_eog(vocab, best)) {
      stop = "eog";
      break;
    }
    decode(&best, 1, static_cast<llama_pos>(n_prompt + g));
  }
  const fs::path dir = utf8_path(o.out);
  fs::create_directories(dir);
  std::string j = std::format("{{\n  \"producer\": {},\n  \"f32_kv\": {},\n  \"stop\": {},\n",
                              json_string(std::format("hetero-oracle-dump --generate (llama.cpp {}, ggml {} {})",
                                                      llama_version(), ggml_version(), ggml_commit())),
                              o.f32_kv, json_string(stop));
  j +=
      std::format("  \"prompt\": {},\n  \"generated\": {},\n  \"top2_gap\": [", json_array(prompt), json_array(chosen));
  for (size_t i = 0; i < gaps.size(); ++i) j += std::format("{}{:.9g}", i > 0 ? ", " : "", gaps[i]);
  j += "]\n}\n";
  std::ofstream out(dir / "generate.json", std::ios::binary | std::ios::trunc);
  out.write(j.data(), static_cast<std::streamsize>(j.size()));
  out.close();
  if (!out) throw std::runtime_error("cannot write generate.json");
  std::fprintf(stderr, "hetero-oracle-dump: generated %zu tokens greedily (stop: %s)\n", chosen.size(), stop.c_str());
  return 0;
}

int run(const Options& o) {
  const std::vector<llama_token> tokens = o.has_tokens ? parse_token_list(o.tokens) : read_token_file(o.tokens_file);
  if (tokens.empty()) throw UsageError("no token ids given");
  const auto n_tokens = static_cast<int64_t>(tokens.size());
  if (n_tokens > o.ctx) throw UsageError(std::format("{} tokens do not fit in --ctx {}", n_tokens, o.ctx));

  Capture cap;
  if (o.generate == 0) {
    cap.bases = parse_names(o.names);
    cap.positions = parse_positions(o.positions, n_tokens);
    cap.n_tokens = n_tokens;
    cap.dir = utf8_path(o.out);
    fs::create_directories(cap.dir);
    // A failed run must not leave an older index.json next to partly rewritten files.
    fs::remove(cap.dir / "index.json");
  }

  llama_log_set(on_log, nullptr);
  llama_backend_init();
  struct BackendFree {
    ~BackendFree() { llama_backend_free(); }
  } backend_free;

  // CPU only: an empty device list means no offload device is ever used.
  llama_model_params mparams = llama_model_default_params();
  ggml_backend_dev_t no_devices[] = {nullptr};
  mparams.devices = no_devices;
  mparams.n_gpu_layers = 0;
  const llama_model_ptr model(llama_model_load_from_file(o.model.c_str(), mparams));
  if (!model) throw std::runtime_error(std::format("cannot load the model '{}'", o.model));
  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  const int32_t n_vocab = llama_vocab_n_tokens(vocab);
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (tokens[i] >= n_vocab) {
      throw UsageError(std::format("token id {} (item {}) is not below the vocabulary size {}", tokens[i], i, n_vocab));
    }
  }
  if (o.ctx > llama_model_n_ctx_train(model.get())) {
    throw UsageError(
        std::format("--ctx {} exceeds the model's trained context {}", o.ctx, llama_model_n_ctx_train(model.get())));
  }
  if (o.generate > 0) return generate(o, tokens, model.get(), vocab, n_vocab);

  // One batch, one ubatch: every token is evaluated in a single graph, with
  // the logits of every token as outputs.
  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = static_cast<uint32_t>(o.ctx);
  cparams.n_batch = static_cast<uint32_t>(n_tokens);
  cparams.n_ubatch = static_cast<uint32_t>(n_tokens);
  cparams.n_seq_max = 1;
  cparams.n_threads = o.threads;
  cparams.n_threads_batch = o.threads;
  cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
  cparams.type_k = o.f32_kv ? GGML_TYPE_F32 : GGML_TYPE_F16;
  cparams.type_v = o.f32_kv ? GGML_TYPE_F32 : GGML_TYPE_F16;
  cparams.offload_kqv = false;
  cparams.op_offload = false;
  cparams.embeddings = false;
  cparams.cb_eval = on_graph_node;
  cparams.cb_eval_user_data = &cap;
  const llama_context_ptr ctx(llama_init_from_model(model.get(), cparams));
  if (!ctx) throw std::runtime_error("cannot create the llama.cpp context");
  if (llama_n_batch(ctx.get()) < n_tokens || llama_n_ubatch(ctx.get()) < n_tokens) {
    throw std::runtime_error(std::format("llama.cpp set n_batch {} and n_ubatch {}, below the {} tokens",
                                         llama_n_batch(ctx.get()), llama_n_ubatch(ctx.get()), n_tokens));
  }
  std::fprintf(stderr,
               "hetero-oracle-dump: %lld tokens in one batch, --ctx %lld (llama.cpp allocates %u), KV cache %s, "
               "flash attention off, %d threads\n",
               static_cast<long long>(n_tokens), static_cast<long long>(o.ctx), llama_n_ctx(ctx.get()),
               o.f32_kv ? "f32" : "f16", o.threads);

  const llama_batch_ext_ptr batch(llama_batch_ext_init(ctx.get()));
  for (int32_t i = 0; i < static_cast<int32_t>(n_tokens); ++i) {
    const llama_pos pos = i;
    if (llama_batch_ext_add_token(batch.get(), 0, tokens[static_cast<size_t>(i)]) != i ||
        !llama_batch_ext_set_pos(batch.get(), i, &pos) || !llama_batch_ext_set_output_logits(batch.get(), i, true)) {
      throw std::runtime_error(std::format("cannot add token {} to the batch", i));
    }
  }
  const int32_t rc = llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get());
  if (cap.error) throw std::runtime_error(*cap.error);
  if (rc != 0) throw std::runtime_error(std::format("llama_process(DECODE) failed with code {}", rc));
  llama_synchronize(ctx.get());

  // The logits: from the graph's result_output if the callback saw it (then
  // checked against llama_get_logits_ith), otherwise from llama_get_logits_ith.
  if (std::ranges::find(cap.bases, std::string("result_output")) != cap.bases.end()) {
    const fs::path file = cap.dir / "result_output.bin";
    const auto it = cap.index.find("result_output");
    if (it != cap.index.end()) {
      if (cap.dumped[it->second].width != n_vocab) {
        throw std::runtime_error(std::format("result_output has {} values per position, not the vocabulary size {}",
                                             cap.dumped[it->second].width, n_vocab));
      }
      verify_logits(ctx.get(), file, cap.positions, n_vocab);
    } else {
      write_logits(ctx.get(), file, cap.positions, n_vocab);
      Dumped& d = cap.entry("result_output");
      d.width = n_vocab;
      d.ggml_ne = {n_vocab, n_tokens};
      d.ggml_type = "f32";
      d.source = "llama_get_logits_ith";
      cap.base_hits["result_output"] += 1;
    }
  }

  std::vector<std::string> missing;
  for (const std::string& b : cap.bases) {
    if (!cap.base_hits.contains(b)) missing.push_back(b);
  }
  if (!missing.empty()) {
    std::string have;
    for (const std::string& s : cap.seen) {
      if (!s.empty() && s.find(" (") == std::string::npos) have += (have.empty() ? "" : ", ") + s;
    }
    std::string want;
    for (const std::string& s : missing) want += (want.empty() ? "" : ", ") + s;
    throw std::runtime_error(
        std::format("no graph tensor matches {}. The graph's tensor names (without -<layer>): {}", want, have));
  }

  write_index(cap.dir, o, tokens, llama_n_ctx(ctx.get()), cap);
  for (const Dumped& d : cap.dumped) {
    std::fprintf(stderr, "  %-24s ne [%lld, %lld]  %s  count %d  (%s)\n", d.name.c_str(),
                 static_cast<long long>(d.width), static_cast<long long>(n_tokens), d.ggml_type.c_str(), d.count,
                 d.source.c_str());
  }

  // The most likely next tokens after the last position, as a quick sanity check.
  const float* last = llama_get_logits_ith(ctx.get(), static_cast<int32_t>(n_tokens - 1));
  if (last == nullptr) throw std::runtime_error("no logits for the last position");
  std::vector<llama_token> ids(static_cast<size_t>(n_vocab));
  std::iota(ids.begin(), ids.end(), 0);
  const size_t top = std::min<size_t>(5, ids.size());
  std::partial_sort(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(top), ids.end(),
                    [&](llama_token a, llama_token b) { return last[a] != last[b] ? last[a] > last[b] : a < b; });
  std::string line;
  for (size_t i = 0; i < top; ++i) {
    line += std::format("{}{} '{}' {}", i > 0 ? ", " : "", ids[i], printable_piece(vocab, ids[i]), last[ids[i]]);
  }
  std::fprintf(stderr, "hetero-oracle-dump: top-%zu after position %lld: %s\n", top,
               static_cast<long long>(n_tokens - 1), line.c_str());
  std::fprintf(stderr, "hetero-oracle-dump: wrote %zu tensors x %zu positions to %s\n", cap.dumped.size(),
               cap.positions.size(), o.out.c_str());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options o = parse_args(argc, argv);
    if (o.help) {
      std::fputs(kUsage, stdout);
      std::fputs(kHelp, stdout);
      return 0;
    }
    return run(o);
  } catch (const UsageError& e) {
    std::fprintf(stderr, "hetero-oracle-dump: error: %s\n%s(--help for details)\n", e.what(), kUsage);
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "hetero-oracle-dump: error: %s\n", e.what());
    return 1;
  }
}
