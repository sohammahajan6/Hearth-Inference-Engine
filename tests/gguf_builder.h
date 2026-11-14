// Builds small GGUF files in memory for tests and fuzzer seeds. Produces
// valid files by default; every field can be overridden to produce exactly
// one kind of malformation.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "gguf/types.h"

namespace hearth::test {

class GgufBuilder {
 public:
  struct Tensor {
    std::string name;
    std::vector<std::uint64_t> ne;
    std::uint32_t type;
    std::uint64_t bytes;       // size of its data
    std::int64_t offset = -1;  // -1: the next aligned position
  };

  // Fill F32 tensor data with deterministic pseudo-random values instead of
  // zeros: weights uniform in [-0.08, 0.08], 1-D tensors (norm weights) near 1.
  GgufBuilder& random_f32_data(std::uint64_t seed) {
    random_seed_ = seed;
    return *this;
  }
  GgufBuilder& magic(const char* m) {
    magic_ = m;
    return *this;
  }
  GgufBuilder& version(std::uint32_t v) {
    version_ = v;
    return *this;
  }
  GgufBuilder& alignment(std::uint64_t a) {
    alignment_ = a;
    return *this;
  }
  // Overrides the counts written in the header (to lie about them).
  GgufBuilder& kv_count(std::uint64_t n) {
    kv_count_ = n;
    return *this;
  }
  GgufBuilder& tensor_count(std::uint64_t n) {
    tensor_count_ = n;
    return *this;
  }

  // A key with an already-encoded value: type code followed by `value` bytes.
  GgufBuilder& kv_raw(const std::string& key, std::uint32_t type, const std::vector<std::byte>& value) {
    put_string(kvs_, key);
    put(kvs_, type);
    kvs_.insert(kvs_.end(), value.begin(), value.end());
    ++n_kv_;
    return *this;
  }
  GgufBuilder& kv_u32(const std::string& key, std::uint32_t v) { return kv_raw(key, 4, bytes_of(v)); }
  GgufBuilder& kv_i32(const std::string& key, std::int32_t v) { return kv_raw(key, 5, bytes_of(v)); }
  GgufBuilder& kv_f32(const std::string& key, float v) { return kv_raw(key, 6, bytes_of(v)); }
  GgufBuilder& kv_bool(const std::string& key, bool v) {
    return kv_raw(key, 7, {std::byte{v ? std::uint8_t{1} : std::uint8_t{0}}});
  }
  GgufBuilder& kv_string(const std::string& key, const std::string& v) {
    std::vector<std::byte> b;
    put_string(b, v);
    return kv_raw(key, 8, b);
  }
  GgufBuilder& kv_strings(const std::string& key, const std::vector<std::string>& v) {
    std::vector<std::byte> b;
    put(b, std::uint32_t{8});
    put(b, static_cast<std::uint64_t>(v.size()));
    for (const std::string& s : v) put_string(b, s);
    return kv_raw(key, 9, b);
  }
  GgufBuilder& kv_i32s(const std::string& key, const std::vector<std::int32_t>& v) {
    std::vector<std::byte> b;
    put(b, std::uint32_t{5});
    put(b, static_cast<std::uint64_t>(v.size()));
    for (std::int32_t x : v) put(b, x);
    return kv_raw(key, 9, b);
  }
  GgufBuilder& kv_u32s(const std::string& key, const std::vector<std::uint32_t>& v) {
    std::vector<std::byte> b;
    put(b, std::uint32_t{4});
    put(b, static_cast<std::uint64_t>(v.size()));
    for (std::uint32_t x : v) put(b, x);
    return kv_raw(key, 9, b);
  }

  GgufBuilder& tensor(Tensor t) {
    tensors_.push_back(std::move(t));
    return *this;
  }

  std::vector<std::byte> build() const {
    std::vector<std::byte> out;
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<std::byte>(magic_[i]));
    put(out, version_);
    put(out, tensor_count_ != kUnset ? tensor_count_ : static_cast<std::uint64_t>(tensors_.size()));
    put(out, kv_count_ != kUnset ? kv_count_ : n_kv_);
    out.insert(out.end(), kvs_.begin(), kvs_.end());
    std::vector<std::uint64_t> offsets;
    std::uint64_t next = 0;
    for (const Tensor& t : tensors_) {
      const std::uint64_t off = t.offset >= 0 ? static_cast<std::uint64_t>(t.offset) : next;
      offsets.push_back(off);
      next = align(off + t.bytes);
      put_string(out, t.name);
      put(out, static_cast<std::uint32_t>(t.ne.size()));
      for (std::uint64_t n : t.ne) put(out, n);
      put(out, t.type);
      put(out, off);
    }
    out.resize(static_cast<std::size_t>(align(out.size())));
    const std::size_t data_start = out.size();
    std::uint64_t end = 0;
    for (std::size_t k = 0; k < tensors_.size(); ++k) end = std::max(end, offsets[k] + tensors_[k].bytes);
    out.resize(data_start + static_cast<std::size_t>(end));
    if (random_seed_ != 0) {
      std::uint64_t st = random_seed_;
      auto rand01 = [&st] {
        st = st * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<float>((st >> 40) & 0xFFFFFF) / static_cast<float>(0xFFFFFF);  // [0, 1]
      };
      for (std::size_t k = 0; k < tensors_.size(); ++k) {
        if (tensors_[k].type != 0) continue;
        const bool vec = tensors_[k].ne.size() == 1;
        for (std::uint64_t i = 0; i < tensors_[k].bytes / 4; ++i) {
          const float v = vec ? 0.75f + 0.5f * rand01() : 0.16f * rand01() - 0.08f;
          std::memcpy(out.data() + data_start + offsets[k] + 4 * i, &v, 4);
        }
      }
    }
    return out;
  }

  template <class T>
  static std::vector<std::byte> bytes_of(T v) {
    std::vector<std::byte> b(sizeof(T));
    std::memcpy(b.data(), &v, sizeof(T));
    return b;
  }

 private:
  static constexpr std::uint64_t kUnset = ~0ull;

  template <class T>
  static void put(std::vector<std::byte>& b, T v) {
    const auto bytes = bytes_of(v);
    b.insert(b.end(), bytes.begin(), bytes.end());
  }
  static void put_string(std::vector<std::byte>& b, const std::string& s) {
    put(b, static_cast<std::uint64_t>(s.size()));
    for (char c : s) b.push_back(static_cast<std::byte>(c));
  }
  std::uint64_t align(std::uint64_t x) const { return (x + alignment_ - 1) / alignment_ * alignment_; }

  const char* magic_ = "GGUF";
  std::uint32_t version_ = 3;
  std::uint64_t alignment_ = 32;
  std::uint64_t kv_count_ = kUnset;
  std::uint64_t tensor_count_ = kUnset;
  std::uint64_t n_kv_ = 0;
  std::uint64_t random_seed_ = 0;
  std::vector<std::byte> kvs_;
  std::vector<Tensor> tensors_;
};

inline std::vector<std::string> tiny_vocab() {
  std::vector<std::string> v = {"a", "b", "\xC3\xA9"};
  while (v.size() < 32) v.push_back("t" + std::to_string(v.size()));
  return v;
}

// A small but complete Llama-shaped file: width `n_embd` (default 64), FFN 96,
// vocab 32. Valid
// input for the parser and the model binder. The defaults (1 layer, 2 heads,
// 1 kv head, F32) are the file the parser tests use; the forward-pass tests
// ask for more layers and kv heads. `matrix_type` is the type of the 2-D
// tensors (0 F32, 8 Q8_0, 2 Q4_0); norms and RoPE factors are always F32.
// Only F32 data can be filled (random_f32_data); other tensors stay zero until
// a test writes them.
// `vocab`: the 32 token texts (default tiny_vocab()). `n_ff`: the feed-forward
// width. `second_type`, if not 0: the type of attn_v, ffn_down and token_embd
// instead of `matrix_type` (as llama-quantize's Q4_K_M gives them Q6_K).
inline GgufBuilder tiny_llama(std::uint32_t n_layer = 1, std::uint32_t n_head = 2, std::uint32_t n_head_kv = 1,
                              std::uint32_t matrix_type = 0, std::uint64_t n_embd = 64,
                              std::uint32_t context_length = 128, const std::vector<std::string>& vocab = tiny_vocab(),
                              std::uint64_t n_ff = 96, std::uint32_t second_type = 0) {
  const std::uint64_t head_dim = n_embd / n_head;
  const std::uint64_t kv_width = n_head_kv * head_dim;
  GgufBuilder b;
  b.kv_string("general.architecture", "llama")
      .kv_string("general.name", "tiny")
      .kv_u32("llama.block_count", n_layer)
      .kv_u32("llama.context_length", context_length)
      .kv_u32("llama.embedding_length", static_cast<std::uint32_t>(n_embd))
      .kv_u32("llama.feed_forward_length", static_cast<std::uint32_t>(n_ff))
      .kv_u32("llama.attention.head_count", n_head)
      .kv_u32("llama.attention.head_count_kv", n_head_kv)
      .kv_f32("llama.rope.freq_base", 10000.0f)
      .kv_f32("llama.attention.layer_norm_rms_epsilon", 1e-5f)
      .kv_u32("llama.vocab_size", 32)
      .kv_bool("tokenizer.ggml.add_bos_token", true)
      .kv_strings("tokenizer.ggml.tokens", vocab);
  auto f32 = [&](const std::string& name, std::vector<std::uint64_t> ne, bool second = false) {
    std::uint64_t n = 1;
    for (std::uint64_t x : ne) n *= x;
    const std::uint32_t type = second && second_type != 0 ? second_type : matrix_type;
    if (ne.size() == 1 || type == 0) {
      b.tensor({name, std::move(ne), 0, n * 4});
    } else {
      const gguf::TypeTraits* t = gguf::type_traits(type);
      b.tensor({name, std::move(ne), type, n / t->block_size * t->type_size});
    }
  };
  f32("token_embd.weight", {n_embd, 32}, true);
  f32("output_norm.weight", {n_embd});
  f32("rope_freqs.weight", {head_dim / 2});
  for (std::uint32_t l = 0; l < n_layer; ++l) {
    const std::string p = "blk." + std::to_string(l) + ".";
    f32(p + "attn_norm.weight", {n_embd});
    f32(p + "attn_q.weight", {n_embd, n_embd});
    f32(p + "attn_k.weight", {n_embd, kv_width});
    f32(p + "attn_v.weight", {n_embd, kv_width}, true);
    f32(p + "attn_output.weight", {n_embd, n_embd});
    f32(p + "ffn_norm.weight", {n_embd});
    f32(p + "ffn_gate.weight", {n_embd, n_ff});
    f32(p + "ffn_up.weight", {n_embd, n_ff});
    f32(p + "ffn_down.weight", {n_ff, n_embd}, true);
  }
  return b;
}

// A small random Qwen2 model (architecture "qwen2"): Llama's layout plus
// F32 biases on Q, K and V, an untied output.weight and no rope_freqs; all
// tensors F32 (fill them with random_f32_data), or the 2-D ones of
// `matrix_type` (zero until a test writes them; the biases and norms stay
// F32). NEOX RoPE is implied by the architecture.
inline GgufBuilder tiny_qwen2(std::uint32_t n_layer = 2, std::uint32_t n_head = 4, std::uint32_t n_head_kv = 2,
                              std::uint64_t n_embd = 64, std::uint64_t n_ff = 96, std::uint32_t matrix_type = 0) {
  const std::uint64_t head_dim = n_embd / n_head;
  const std::uint64_t kv_width = n_head_kv * head_dim;
  GgufBuilder b;
  b.kv_string("general.architecture", "qwen2")
      .kv_string("general.name", "tiny-qwen2")
      .kv_u32("qwen2.block_count", n_layer)
      .kv_u32("qwen2.context_length", 128)
      .kv_u32("qwen2.embedding_length", static_cast<std::uint32_t>(n_embd))
      .kv_u32("qwen2.feed_forward_length", static_cast<std::uint32_t>(n_ff))
      .kv_u32("qwen2.attention.head_count", n_head)
      .kv_u32("qwen2.attention.head_count_kv", n_head_kv)
      .kv_f32("qwen2.rope.freq_base", 1000000.0f)
      .kv_f32("qwen2.attention.layer_norm_rms_epsilon", 1e-6f)
      .kv_bool("tokenizer.ggml.add_bos_token", false)
      .kv_strings("tokenizer.ggml.tokens", tiny_vocab());
  auto f32 = [&](const std::string& name, std::vector<std::uint64_t> ne) {
    std::uint64_t n = 1;
    for (std::uint64_t x : ne) n *= x;
    if (ne.size() == 1 || matrix_type == 0) {
      b.tensor({name, std::move(ne), 0, n * 4});
    } else {
      const gguf::TypeTraits* t = gguf::type_traits(matrix_type);
      b.tensor({name, std::move(ne), matrix_type, n / t->block_size * t->type_size});
    }
  };
  f32("token_embd.weight", {n_embd, 32});
  f32("output_norm.weight", {n_embd});
  f32("output.weight", {n_embd, 32});
  for (std::uint32_t l = 0; l < n_layer; ++l) {
    const std::string p = "blk." + std::to_string(l) + ".";
    f32(p + "attn_norm.weight", {n_embd});
    f32(p + "attn_q.weight", {n_embd, n_embd});
    f32(p + "attn_q.bias", {n_embd});
    f32(p + "attn_k.weight", {n_embd, kv_width});
    f32(p + "attn_k.bias", {kv_width});
    f32(p + "attn_v.weight", {n_embd, kv_width});
    f32(p + "attn_v.bias", {kv_width});
    f32(p + "attn_output.weight", {n_embd, n_embd});
    f32(p + "ffn_norm.weight", {n_embd});
    f32(p + "ffn_gate.weight", {n_embd, n_ff});
    f32(p + "ffn_up.weight", {n_embd, n_ff});
    f32(p + "ffn_down.weight", {n_ff, n_embd});
  }
  return b;
}

}  // namespace hearth::test
