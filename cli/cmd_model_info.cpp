// `hearth info --model FILE [--tensors] [--json]`: what a GGUF file contains.
#include <cstdio>
#include <format>
#include <map>
#include <string>

#include "commands.h"
#include "core/json_writer.h"
#include "core/platform.h"
#include "gguf/gguf.h"
#include "model/llama.h"

namespace hearth::cli {
namespace {

// FNV-1a over each string's length and bytes: a fingerprint of a whole string
// array that tools/gguf_crosscheck.py recomputes with an independent reader.
std::uint64_t fnv1a(const std::vector<std::string_view>& strings) {
  std::uint64_t h = 1469598103934665603ull;
  auto mix = [&](const void* p, std::size_t n) {
    const auto* b = static_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
  };
  for (std::string_view s : strings) {
    const std::uint64_t len = s.size();
    mix(&len, sizeof(len));
    mix(s.data(), s.size());
  }
  return h;
}

std::string scalar_text(const gguf::Value& v) {
  switch (v.type) {
    case gguf::ValueType::kString:
      return v.s.size() > 80 ? std::string(v.s.substr(0, 77)) + "..." : std::string(v.s);
    case gguf::ValueType::kBool:
      return v.b ? "true" : "false";
    case gguf::ValueType::kF32:
    case gguf::ValueType::kF64:
      return std::format("{}", v.f);
    case gguf::ValueType::kI8:
    case gguf::ValueType::kI16:
    case gguf::ValueType::kI32:
    case gguf::ValueType::kI64:
      return std::format("{}", v.i);
    default:
      return std::format("{}", v.u);
  }
}

void write_value_json(JsonWriter& w, const gguf::Value& v) {
  w.field("type", gguf::to_string(v.type));
  if (v.type == gguf::ValueType::kArray) {
    w.field("elem", gguf::to_string(v.elem));
    w.field("count", v.count);
    if (v.elem == gguf::ValueType::kString) w.field("fnv1a", std::format("{:016x}", fnv1a(v.strings)));
    return;
  }
  switch (v.type) {
    case gguf::ValueType::kString:
      w.field("value", v.s);
      break;
    case gguf::ValueType::kBool:
      w.field("value", v.b);
      break;
    case gguf::ValueType::kF32:
    case gguf::ValueType::kF64:
      w.field("value", v.f);
      break;
    case gguf::ValueType::kI8:
    case gguf::ValueType::kI16:
    case gguf::ValueType::kI32:
    case gguf::ValueType::kI64:
      w.field("value", v.i);
      break;
    default:
      w.field("value", v.u);
      break;
  }
}

std::string shape(const gguf::TensorInfo& t) {
  std::string s;
  for (std::uint32_t d = 0; d < t.n_dims; ++d) s += std::format("{}{}", d ? " x " : "", t.ne[d]);
  return s;
}

}  // namespace

int model_info(const std::string& path, bool list_tensors, bool json) {
  const gguf::File f = gguf::File::open(path_from_utf8(path));
  if (json) {
    JsonWriter w;
    w.begin_object();
    w.field("version", f.version());
    w.field("alignment", f.alignment());
    w.field("data_offset", f.data_offset());
    w.key("metadata");
    w.begin_array();
    for (const gguf::Kv& kv : f.metadata()) {
      w.begin_object();
      w.field("key", kv.key);
      write_value_json(w, kv.value);
      w.end_object();
    }
    w.end_array();
    w.key("tensors");
    w.begin_array();
    for (const gguf::TensorInfo& t : f.tensors()) {
      w.begin_object();
      w.field("name", t.name);
      w.field("type", t.traits->name);
      w.key("ne");
      w.begin_array();
      for (std::uint32_t d = 0; d < t.n_dims; ++d) w.value(t.ne[d]);
      w.end_array();
      w.field("offset", t.offset);
      w.field("bytes", t.bytes);
      w.end_object();
    }
    w.end_array();
    w.end_object();
    std::printf("%s\n", w.str().c_str());
    return 0;
  }

  std::printf("file:     %s (%.3f GB, GGUF v%u, alignment %llu, data at byte %llu)\n", path.c_str(),
              static_cast<double>(f.size()) / 1e9, f.version(), static_cast<unsigned long long>(f.alignment()),
              static_cast<unsigned long long>(f.data_offset()));
  const model::LlamaConfig c = model::LlamaConfig::from_gguf(f);
  const model::LlamaTensors tensors = model::LlamaTensors::bind(f, c);
  std::printf("model:    %s: %u layers, width %u, feed-forward %u, %u heads (%u key/value), head width %u\n",
              c.name.c_str(), c.n_layer, c.n_embd, c.n_ff, c.n_head, c.n_head_kv, c.head_dim);
  std::printf("          vocabulary %u, trained context %u, rope theta %g on %u dims, rms epsilon %g\n", c.n_vocab,
              c.n_ctx_train, static_cast<double>(c.rope_theta), c.rope_dim, static_cast<double>(c.rms_eps));
  std::printf("          tied embeddings: %s, rope frequency factors: %s, %zu layers bound\n",
              c.tied_embeddings ? "yes" : "no", c.has_rope_freqs ? "yes" : "no", tensors.layers.size());
  std::map<std::string, std::pair<int, std::uint64_t>> hist;
  for (const gguf::TensorInfo& t : f.tensors()) {
    auto& e = hist[t.traits->name];
    ++e.first;
    e.second += t.bytes;
  }
  std::printf("tensors:  %zu:", f.tensors().size());
  for (const auto& [name, e] : hist) {
    std::printf(" %s %d (%.3f GB)", name.c_str(), e.first, static_cast<double>(e.second) / 1e9);
  }
  std::printf("\nmetadata: %zu keys\n", f.metadata().size());
  for (const gguf::Kv& kv : f.metadata()) {
    if (kv.value.type == gguf::ValueType::kArray) {
      std::printf("  %-42s [%llu x %s]\n", std::string(kv.key).c_str(), static_cast<unsigned long long>(kv.value.count),
                  gguf::to_string(kv.value.elem));
    } else {
      std::string text = scalar_text(kv.value);
      for (char& ch : text) {
        if (ch == '\n' || ch == '\r') ch = ' ';
      }
      std::printf("  %-42s %s\n", std::string(kv.key).c_str(), text.c_str());
    }
  }
  if (list_tensors) {
    std::printf("tensor list (name, type, shape with the contiguous dimension first, offset, bytes):\n");
    for (const gguf::TensorInfo& t : f.tensors()) {
      std::printf("  %-28s %-5s %-16s %12llu %12llu\n", std::string(t.name).c_str(), t.traits->name, shape(t).c_str(),
                  static_cast<unsigned long long>(t.offset), static_cast<unsigned long long>(t.bytes));
    }
  }
  return 0;
}

}  // namespace hearth::cli
