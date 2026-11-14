// The reference Llama forward pass: plain code written to be obviously
// correct, and the yardstick every faster backend is held to.
// Design: docs/architecture/05-forward-pass.md.
#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/cpu_features.h"
#include "core/error.h"
#include "core/thread_pool.h"
#include "cpu/kquant.h"
#include "cpu/packed_arena.h"
#include "cpu/qgemm.h"
#include "gguf/gguf.h"
#include "model/llama.h"

namespace hearth::gpu {
struct ContextOptions;
}  // namespace hearth::gpu

namespace hearth::model {

class LlamaGpu;

// Named intermediate tensors recorded during a forward pass, in the format of
// tools/oracle/README.md (index.json + one raw float32 file per tensor), so
// tools/layer_diff.py can compare them with llama.cpp's.
class TensorDump {
 public:
  // The tensors a forward pass can record, named as llama.cpp's graph names
  // them; all but result_norm and result_output exist once per layer.
  static constexpr std::string_view kNames[] = {"attn_norm", "Qcur",        "Kcur",         "Vcur",
                                                "attn_out",  "ffn_inp",     "ffn_norm",     "ffn_out",
                                                "l_out",     "result_norm", "result_output"};
  // Whether `name` is one of kNames, or a per-layer name like "l_out-3".
  static bool known(std::string_view name);

  // `names`: what to keep. A base name ("l_out") selects every layer's tensor,
  // a full name ("l_out-3") one layer. `positions`: absolute positions to
  // keep (sorted and de-duplicated here); empty = all.
  TensorDump(std::vector<std::string> names, std::vector<std::int64_t> positions);

  // Whether tensor `base` of `layer` (-1 outside the layers) is wanted.
  bool wants(std::string_view base, int layer) const;
  bool wants_position(std::int64_t pos) const;
  void record(std::string_view base, int layer, std::int64_t pos, const float* data, std::int64_t n);
  // The requested names that matched nothing (for example a layer the model
  // does not have).
  std::vector<std::string> unmatched() const;
  // `producer` names the backend; `ne` is written as [width, tokens.size()].
  void write(const std::filesystem::path& dir, const std::string& producer, const std::string& model,
             std::span<const std::int32_t> tokens, std::uint32_t ctx) const;

 private:
  struct Tensor {
    std::int64_t width = 0;
    std::map<std::int64_t, std::vector<float>> rows;  // position -> values
  };
  std::vector<std::string> names_;
  std::vector<std::int64_t> positions_;
  std::map<std::string, Tensor> tensors_;
  std::set<std::string> matched_;
};

// How matrix products are computed.
enum class Matmul {
  // cpu-ref: each weight row is decoded to float, FP32 dot products with one
  // accumulator in a fixed order.
  kFp32,
  // The `cpu` backend: Q8_0 and Q4_0 weights are repacked at load time and
  // multiplied in the canonical order of ADR 0009 (cpu/qgemm.h). The
  // activations are quantized to Q8_0 exactly as ggml does, and block sums
  // are exact int32. Q4_K and Q6_K weights are repacked too and multiplied
  // in the canonical order of ADR 0021 (cpu/kquant.h), with the activations
  // quantized to Q8_K exactly as ggml does. Other weight types use kFp32.
  // Attention runs in cpu::attend, which gives cpu-ref's bits.
  kQ8Activations,
  // `gpu`: the whole forward
  // pass on the GPU (LlamaGpu, model/llama_gpu.h) on the same packed
  // weights, imported without a copy. One device runs a
  // model at a time.
  kGpu,
};

class LlamaRef {
 public:
  // `isa` picks the int8 kernels of kQ8Activations: kAvx512, or kScalar.
  // Both compute the canonical order, so they give identical results.
  // `gpu_options` configures the GPU context of kGpu (nullptr: defaults).
  LlamaRef(const gguf::File& file, std::uint32_t n_ctx, ThreadPool& pool, Matmul matmul = Matmul::kFp32,
           Isa isa = isa_supported(Isa::kAvx512) ? Isa::kAvx512 : Isa::kScalar,
           const gpu::ContextOptions* gpu_options = nullptr);
  ~LlamaRef();
  LlamaRef(const LlamaRef&) = delete;
  LlamaRef& operator=(const LlamaRef&) = delete;

  const LlamaConfig& config() const noexcept { return cfg_; }
  Matmul matmul() const noexcept { return matmul_; }
  std::uint32_t n_ctx() const noexcept { return n_ctx_; }
  // kQ8Activations: the packed weight matrices (empty otherwise), and the
  // arena index of a Q8_0/Q4_0 matrix tensor (the GPU imports the arena).
  const cpu::PackedArena& packed_arena() const noexcept { return arena_; }
  std::size_t packed_index(const gguf::TensorInfo* t) const { return packed_index_.at(t); }
  std::uint32_t n_past() const noexcept { return n_past_; }
  void reset() noexcept {
    n_past_ = 0;
    history_.clear();
  }
  // Forgets positions n .. n_past() - 1 (n <= n_past()): the next forward()
  // continues at position n. Nothing is recomputed: positions past n_past()
  // are never read, and their K/V rows are overwritten by the next call, as
  // after a failed call. Speculative decoding drops rejected drafts this way
  // (model/speculative.h).
  void truncate(std::uint32_t n);
  // kGpu: whether a lost GPU (or a watchdog trip) may switch the run to the
  // `cpu` backend, which first re-runs the tokens fed so far (08 doc
  // section 4), instead of failing (every measurement, test and gate). Off
  // by default.
  void set_gpu_fallback(bool allow) noexcept { gpu_fallback_ = allow; }
  // Whether the GPU failed and the run continued on the CPU.
  bool gpu_lost() const noexcept { return gpu_lost_; }
  // kQ8Activations and kGpu: the arena index of the LM head.
  std::size_t lm_head_matrix() const { return packed_index_.at(w_.output != nullptr ? w_.output : w_.token_embd); }
  // kGpu: the GPU forward pass (device, submissions), else nullptr.
  const LlamaGpu* gpu_pass() const noexcept { return gpu_pass_.get(); }
  LlamaGpu* gpu_pass() noexcept { return gpu_pass_.get(); }

  // Feeds `tokens` at positions n_past() .. n_past() + tokens.size() - 1.
  // With `all_logits`, `logits` receives tokens.size() x n_vocab values (one
  // row per token); otherwise n_vocab values for the last token. Throws
  // kInvalidArgument if the context would overflow.
  void forward(std::span<const std::int32_t> tokens, std::span<float> logits, bool all_logits,
               TensorDump* dump = nullptr);

 private:
  // y[t * out + r] = dot(row r of w, x[t * in ...]) for t < T, rows split across threads.
  // `what` and `layer` label the operation in a trace (core/trace.h).
  void matmul(const char* what, int layer, const gguf::TensorInfo* w, const float* x, float* y, std::int64_t T);
  void matmul_q8(const gguf::TensorInfo* w, const float* x, float* y, std::int64_t T);
  // kQ8Activations, Q4_K and Q6_K weights: on the pool only.
  void matmul_k(const gguf::TensorInfo* w, const float* x, float* y, std::int64_t T);
  // Every row of arena matrix `index` on the pool, for the activations in
  // acts_; y[t * rows + r].
  void packed_product(std::size_t index, float* y);
  void forward_chunk(std::span<const std::int32_t> tokens, float* logits, bool all_logits, TensorDump* dump);
  std::vector<float> f32_vector(const gguf::TensorInfo* t) const;
  bool quantized() const noexcept { return matmul_ != Matmul::kFp32; }
  // kGpu: forward() on the GPU pass, in TDR-safe chunks.
  void forward_cpu(std::span<const std::int32_t> tokens, std::span<float> logits, bool all_logits, TensorDump* dump);
  void forward_gpu(std::span<const std::int32_t> tokens, std::span<float> logits, bool all_logits, TensorDump* dump);
  // kGpu, called only from forward()'s catch handler for kDeviceLost: leaks
  // the GPU pass; with fallback the run becomes the `cpu` backend and the
  // tokens fed so far are run again on it, without it the error is rethrown.
  void fall_back_to_cpu(const Error& e);

  const gguf::File& file_;
  LlamaConfig cfg_;
  LlamaTensors w_;
  ThreadPool& pool_;
  Matmul matmul_;
  Isa isa_;
  std::uint32_t n_ctx_;
  std::uint32_t n_past_ = 0;
  std::vector<std::int32_t> history_;  // kGpu: the tokens at positions [0, n_past_), for a fallback
  // Layers k_layer_ floats apart. In a layer: [position][kv_head * head_dim];
  // the quantized backends: [kv_head][cpu::k_index(position, dimension)].
  std::vector<float> k_cache_;
  std::size_t k_layer_ = 0;  // floats of one layer's K (whole 16-position blocks)
  // Layers n_ctx * n_head_kv * head_dim floats apart. In a layer:
  // [position][kv_head * head_dim]; the quantized backends: [kv_head][position][dimension].
  std::vector<float> v_cache_;
  std::vector<float> rope_factors_;
  std::vector<std::vector<float>> attn_norm_, ffn_norm_;
  std::vector<std::vector<float>> q_bias_, k_bias_, v_bias_;  // per layer; empty if the model has none
  std::vector<float> output_norm_;
  // kQ8Activations: every Q8_0/Q4_0 matrix repacked (one copy, made one
  // tensor at a time), and the quantized activations of the current matmul.
  cpu::PackedArena arena_;
  std::unordered_map<const gguf::TensorInfo*, std::size_t> packed_index_;  // matrix -> arena index
  cpu::PackedActs acts_;
  // kQ8Activations: every Q4_K/Q6_K matrix repacked (cpu/kquant.h), and the
  // Q8_K activations of the current product. The GPU has no K-quant kernels,
  // so these live outside the arena.
  std::vector<cpu::KMatrix> kmats_;
  std::unordered_map<const gguf::TensorInfo*, std::size_t> k_index_;  // matrix -> kmats_ index
  cpu::KActs kacts_;
  bool gpu_fallback_ = false, gpu_lost_ = false;
  bool gpu_abandoned_ = false;  // the GPU failed and was leaked (with or without fallback)
  // kGpu. Declared after arena_, so it is destroyed first: the imported GPU
  // memory is released before the pages it wraps.
  std::unique_ptr<LlamaGpu> gpu_pass_;
};

}  // namespace hearth::model
