#include "model/llama.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <unordered_set>

#include "core/error.h"

namespace hearth::model {
namespace {

std::uint32_t u32(const gguf::File& f, std::string_view key) {
  const std::uint64_t v = f.get_uint(key);
  if (v == 0 || v > std::numeric_limits<std::uint32_t>::max()) {
    fail(ErrorKind::kBadModel, "model: '{}' = {} is out of range", key, v);
  }
  return static_cast<std::uint32_t>(v);
}

std::string shape_string(const gguf::TensorInfo& t) {
  std::string s = "[";
  for (std::uint32_t d = 0; d < t.n_dims; ++d) s += std::format("{}{}", d ? ", " : "", t.ne[d]);
  return s + "]";
}

}  // namespace

LlamaConfig LlamaConfig::from_gguf(const gguf::File& f) {
  LlamaConfig c;
  const std::string_view arch = f.get_string("general.architecture");
  if (arch != "llama" && arch != "qwen2") {
    fail(ErrorKind::kUnsupported, "model architecture '{}' is not supported (only 'llama' and 'qwen2')", arch);
  }
  c.arch = std::string(arch);
  const std::string p = c.arch + ".";  // the architecture's key prefix
  const auto key = [&](std::string_view suffix) { return p + std::string(suffix); };
  // Every <arch>.* key must be one this engine implements: a key it does not
  // know (rope scaling, experts, per-layer head counts, sliding windows, ...)
  // changes the model's meaning, so silently ignoring it would compute the
  // wrong thing.
  static constexpr std::string_view kKnown[] = {
      "block_count",          "context_length",         "embedding_length",
      "feed_forward_length",  "attention.head_count",   "attention.head_count_kv",
      "attention.key_length", "attention.value_length", "attention.layer_norm_rms_epsilon",
      "rope.freq_base",       "rope.dimension_count",   "vocab_size",
  };
  for (const gguf::Kv& kv : f.metadata()) {
    if (kv.key.rfind(p, 0) != 0) continue;
    const std::string_view suffix = std::string_view(kv.key).substr(p.size());
    if (std::find(std::begin(kKnown), std::end(kKnown), suffix) == std::end(kKnown)) {
      fail(ErrorKind::kUnsupported, "model uses '{}', which this engine does not implement", kv.key);
    }
  }
  if (const gguf::Value* n = f.find("general.name"); n != nullptr && n->type == gguf::ValueType::kString) {
    c.name = std::string(n->s);
  }
  c.n_layer = u32(f, key("block_count"));
  if (c.n_layer > 512) fail(ErrorKind::kBadModel, "model: {} layers exceeds the limit of 512", c.n_layer);
  c.n_embd = u32(f, key("embedding_length"));
  c.n_ff = u32(f, key("feed_forward_length"));
  c.n_head = u32(f, key("attention.head_count"));
  c.n_head_kv = u32(f, key("attention.head_count_kv"));
  c.n_ctx_train = u32(f, key("context_length"));
  c.rope_theta = static_cast<float>(f.get_float(key("rope.freq_base")));
  c.rms_eps = static_cast<float>(f.get_float(key("attention.layer_norm_rms_epsilon")));
  if (!std::isfinite(c.rms_eps) || !(c.rms_eps > 0.0f && c.rms_eps < 1.0f) || !std::isfinite(c.rope_theta) ||
      !(c.rope_theta > 0.0f)) {
    fail(ErrorKind::kBadModel, "model: rms epsilon {} or rope theta {} out of range", c.rms_eps, c.rope_theta);
  }
  if (c.n_embd % c.n_head != 0) {
    fail(ErrorKind::kBadModel, "model: embedding length {} is not divisible by head count {}", c.n_embd, c.n_head);
  }
  if (c.n_head % c.n_head_kv != 0) {
    fail(ErrorKind::kBadModel, "model: head count {} is not divisible by key/value head count {}", c.n_head,
         c.n_head_kv);
  }
  // As llama.cpp: head width from key_length if given, else width / heads;
  // value width and RoPE width must equal it (partial RoPE is not implemented).
  c.head_dim =
      f.find(key("attention.key_length")) != nullptr ? u32(f, key("attention.key_length")) : c.n_embd / c.n_head;
  if (f.find(key("attention.value_length")) != nullptr && u32(f, key("attention.value_length")) != c.head_dim) {
    fail(ErrorKind::kUnsupported, "model: different key and value head widths are not supported");
  }
  c.rope_dim = f.find(key("rope.dimension_count")) != nullptr ? u32(f, key("rope.dimension_count")) : c.head_dim;
  if (c.rope_dim != c.head_dim) {
    fail(ErrorKind::kUnsupported, "model: RoPE on {} of {} head dimensions (partial rotary) is not supported",
         c.rope_dim, c.head_dim);
  }
  // Vocabulary size is the tokenizer's list; llama.vocab_size, if present, must agree.
  const std::uint64_t n_tokens = f.get_array("tokenizer.ggml.tokens", gguf::ValueType::kString).count;
  if (n_tokens == 0 || n_tokens > std::numeric_limits<std::uint32_t>::max()) {
    fail(ErrorKind::kBadModel, "model: bad vocabulary size {}", n_tokens);
  }
  c.n_vocab = static_cast<std::uint32_t>(n_tokens);
  if (f.find(key("vocab_size")) != nullptr && u32(f, key("vocab_size")) != c.n_vocab) {
    fail(ErrorKind::kBadModel, "model: {} {} differs from the {} tokens in the tokenizer", key("vocab_size"),
         u32(f, key("vocab_size")), c.n_vocab);
  }
  for (const char* id_key :
       {"tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id", "tokenizer.ggml.padding_token_id"}) {
    if (f.find(id_key) != nullptr && f.get_uint(id_key) >= c.n_vocab) {
      fail(ErrorKind::kBadModel, "model: {} = {} is not a token id (vocabulary {})", id_key, f.get_uint(id_key),
           c.n_vocab);
    }
  }
  c.tied_embeddings = f.find_tensor("output.weight") == nullptr;
  c.has_rope_freqs = f.find_tensor("rope_freqs.weight") != nullptr;
  c.rope_neox = c.arch == "qwen2";  // llama.cpp: LLM_ARCH_QWEN2 -> LLAMA_ROPE_TYPE_NEOX
  return c;
}

LlamaTensors LlamaTensors::bind(const gguf::File& f, const LlamaConfig& c) {
  std::unordered_set<std::string_view> claimed;
  auto get = [&](const std::string& name, std::initializer_list<std::uint64_t> shape, bool matrix) {
    const gguf::TensorInfo* t = f.find_tensor(name);
    if (t == nullptr) fail(ErrorKind::kBadModel, "model: tensor '{}' is missing", name);
    bool ok = t->n_dims == shape.size();
    std::uint32_t d = 0;
    for (std::uint64_t n : shape) ok = ok && t->ne[d++] == n;
    if (!ok) {
      std::string want = "[";
      d = 0;
      for (std::uint64_t n : shape) want += std::format("{}{}", d++ ? ", " : "", n);
      fail(ErrorKind::kBadModel, "model: tensor '{}' has shape {}, expected {}]", name, shape_string(*t), want);
    }
    if (!matrix && t->type != static_cast<std::uint32_t>(gguf::GgmlType::kF32)) {
      fail(ErrorKind::kUnsupported, "model: tensor '{}' is {}; norm and RoPE tensors must be F32", name,
           t->traits->name);
    }
    if (matrix && !t->traits->supported) {
      fail(ErrorKind::kUnsupported,
           "model: tensor '{}' has type {}, which this engine does not support "
           "(supported: F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K)",
           name, t->traits->name);
    }
    claimed.insert(t->name);
    return t;
  };
  // An optional 1-D F32 tensor (a bias): claimed if present.
  auto get_opt = [&](const std::string& name, std::uint64_t n) -> const gguf::TensorInfo* {
    if (f.find_tensor(name) == nullptr) return nullptr;
    return get(name, {n}, false);
  };
  const std::uint64_t q_width = std::uint64_t{c.n_head} * c.head_dim;
  const std::uint64_t kv_width = std::uint64_t{c.n_head_kv} * c.head_dim;
  LlamaTensors w;
  w.token_embd = get("token_embd.weight", {c.n_embd, c.n_vocab}, true);
  w.output_norm = get("output_norm.weight", {c.n_embd}, false);
  if (!c.tied_embeddings) w.output = get("output.weight", {c.n_embd, c.n_vocab}, true);
  if (c.has_rope_freqs) w.rope_freqs = get("rope_freqs.weight", {c.rope_dim / 2}, false);
  w.layers.reserve(c.n_layer);
  for (std::uint32_t l = 0; l < c.n_layer; ++l) {
    const std::string p = std::format("blk.{}.", l);
    w.layers.push_back({
        get(p + "attn_norm.weight", {c.n_embd}, false),
        get(p + "attn_q.weight", {c.n_embd, q_width}, true),
        get(p + "attn_k.weight", {c.n_embd, kv_width}, true),
        get(p + "attn_v.weight", {c.n_embd, kv_width}, true),
        get(p + "attn_output.weight", {q_width, c.n_embd}, true),
        get(p + "ffn_norm.weight", {c.n_embd}, false),
        get(p + "ffn_gate.weight", {c.n_embd, c.n_ff}, true),
        get(p + "ffn_up.weight", {c.n_embd, c.n_ff}, true),
        get(p + "ffn_down.weight", {c.n_ff, c.n_embd}, true),
        get_opt(p + "attn_q.bias", q_width),
        get_opt(p + "attn_k.bias", kv_width),
        get_opt(p + "attn_v.bias", kv_width),
    });
  }
  if (claimed.size() != f.tensors().size()) {
    std::string extra;
    for (const gguf::TensorInfo& t : f.tensors()) {
      if (!claimed.contains(t.name)) extra += std::string(extra.empty() ? "" : ", ") + std::string(t.name);
    }
    fail(ErrorKind::kBadModel, "model: the file has tensors this Llama model does not use: {}", extra);
  }
  return w;
}

std::uint64_t decode_weight_bytes(const gguf::File& f) {
  const LlamaTensors w = LlamaTensors::bind(f, LlamaConfig::from_gguf(f));
  std::uint64_t n = f.tensor_bytes(*(w.output != nullptr ? w.output : w.token_embd)).size();
  for (const auto& l : w.layers) {
    for (const gguf::TensorInfo* t : {l.attn_q, l.attn_k, l.attn_v, l.attn_output, l.ffn_gate, l.ffn_up, l.ffn_down}) {
      n += f.tensor_bytes(*t).size();
    }
  }
  return n;
}

}  // namespace hearth::model
