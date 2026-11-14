// Llama-architecture hyper-parameters and weight tensors, from a GGUF file:
// architecture "llama", or "qwen2" (Qwen2/2.5: the same graph with biases on
// Q, K and V and NEOX-style RoPE; docs/architecture/15-qwen.md).
//
// Every tensor in the file must be claimed here, either as a weight the model
// uses or as one it knowingly ignores; an unclaimed tensor is an error, so a
// file with an unexpected structure can never be silently misread.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gguf/gguf.h"

namespace hearth::model {

struct LlamaConfig {
  std::string arch;             // general.architecture: "llama" or "qwen2"
  std::string name;             // general.name (informational)
  std::uint32_t n_layer = 0;    // transformer blocks
  std::uint32_t n_embd = 0;     // model width
  std::uint32_t n_ff = 0;       // feed-forward width
  std::uint32_t n_head = 0;     // query heads
  std::uint32_t n_head_kv = 0;  // key/value heads (GQA groups)
  std::uint32_t head_dim = 0;   // per-head width of Q and K (= V here)
  std::uint32_t n_vocab = 0;
  std::uint32_t n_ctx_train = 0;  // context length the model was trained for
  std::uint32_t rope_dim = 0;     // leading dimensions of each head that RoPE rotates
  float rope_theta = 0;
  float rms_eps = 0;
  bool tied_embeddings = false;  // no output.weight: the LM head reuses token_embd
  bool has_rope_freqs = false;   // Llama-3 per-frequency scaling factors present
  // RoPE rotates dimensions (i, i + rope_dim / 2) instead of (2i, 2i + 1):
  // llama.cpp's LLAMA_ROPE_TYPE_NEOX, which it uses for qwen2.
  bool rope_neox = false;

  static LlamaConfig from_gguf(const gguf::File& f);
};

struct LlamaTensors {
  struct Layer {
    const gguf::TensorInfo* attn_norm;
    const gguf::TensorInfo* attn_q;
    const gguf::TensorInfo* attn_k;
    const gguf::TensorInfo* attn_v;
    const gguf::TensorInfo* attn_output;
    const gguf::TensorInfo* ffn_norm;
    const gguf::TensorInfo* ffn_gate;
    const gguf::TensorInfo* ffn_up;
    const gguf::TensorInfo* ffn_down;
    // Biases added to the Q, K and V products (F32), or nullptr: Qwen2 has
    // them; llama.cpp applies them for "llama" too when a file has them.
    const gguf::TensorInfo* attn_q_bias = nullptr;
    const gguf::TensorInfo* attn_k_bias = nullptr;
    const gguf::TensorInfo* attn_v_bias = nullptr;
  };
  const gguf::TensorInfo* token_embd = nullptr;
  const gguf::TensorInfo* output_norm = nullptr;
  const gguf::TensorInfo* output = nullptr;      // nullptr when embeddings are tied
  const gguf::TensorInfo* rope_freqs = nullptr;  // nullptr when absent
  std::vector<Layer> layers;

  // Finds every tensor by name, checks its shape and type against the
  // config, and checks that nothing in the file was left unclaimed.
  static LlamaTensors bind(const gguf::File& f, const LlamaConfig& c);
};

// Bytes of weights one decode step reads: every matrix, the LM head
// included. Times tokens per second, the memory read rate of decoding.
std::uint64_t decode_weight_bytes(const gguf::File& f);

}  // namespace hearth::model
