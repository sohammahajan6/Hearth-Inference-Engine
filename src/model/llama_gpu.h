// The Llama forward pass on the GPU (`--backend gpu`; docs/architecture/
// 08-gpu-backend.md step 6.4). Every operation of a layer runs on the GPU:
// RMSNorm, the bit-exact activation quantizer, the canonical GEMV/GEMM on the
// imported weight arena (no copy), Qwen2's Q/K/V biases, RoPE (Llama's or
// NEOX pairs), the KV cache, attention, SiLU-up and the residual adds. The CPU only writes the tokens' embedding rows
// and reads the logits.
//
// The matrix products give the CPU's bits for the same activations; RMSNorm,
// SiLU and attention are within ACCEPTANCE's element-wise bars (FP32 on the
// GPU where the CPU reference uses double), so the logits differ from the
// `cpu` backend's by the small amounts the quantized gates allow.
// Within the GPU pass, a token's results do not depend on how the prompt is
// split into chunks (prefill = decode), and runs are deterministic.
//
// TDR safety (08 doc section 4, step 6.5): chunks (chunk_tokens) and the
// layers of a chunk are cut into submissions of a bounded predicted cost
// (50 ms), each waited on with a finite timeout. Every submission is timed
// with GPU timestamps; the measurements scale the cost model, a submission
// above 250 ms trips the near-miss watchdog (kDeviceLost), and real losses
// are written to the hang ledger (gpu/hang_ledger.h). While --trace records,
// the same timestamps put each layer on the trace's GPU lane.
#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <vector>

#include "cpu/packed_arena.h"
#include "gguf/gguf.h"
#include "gpu/buffer.h"
#include "gpu/context.h"
#include "gpu/pipeline.h"
#include "gpu/timestamps.h"
#include "gpu/weight_arena.h"
#include "model/llama.h"

namespace hearth::model {

class TensorDump;

// What the GPU pass needs from LlamaRef, which owns all of it and outlives the pass.
struct GpuPassInputs {
  const gguf::File* file = nullptr;
  const LlamaConfig* config = nullptr;
  const LlamaTensors* tensors = nullptr;
  const cpu::PackedArena* arena = nullptr;
  std::function<std::size_t(const gguf::TensorInfo*)> arena_index;
  const std::vector<std::vector<float>>* attn_norm = nullptr;
  const std::vector<std::vector<float>>* ffn_norm = nullptr;
  const std::vector<float>* output_norm = nullptr;
  const std::vector<float>* rope_factors = nullptr;  // empty: none
  // Qwen2's Q, K and V biases, one vector per layer (empty: that layer has none).
  const std::vector<std::vector<float>>* q_bias = nullptr;
  const std::vector<std::vector<float>>* k_bias = nullptr;
  const std::vector<std::vector<float>>* v_bias = nullptr;
  std::uint32_t n_ctx = 0;
};

// The attention buffers' sizes against the device's limits, before anything
// is allocated: each layer's K and V range and the whole score buffer are
// bound as storage buffers (maxStorageBufferRange), and K and V are each one
// VkBuffer (maxBufferSize, 0 = unknown). Throws Error(kUnsupported) naming
// the largest context that fits. Sizes in bytes.
void check_attention_limits(std::uint64_t n_layer, std::uint64_t n_ctx, std::uint64_t kv_width, std::uint64_t chunk,
                            std::uint64_t n_head, std::uint64_t max_buffer_size, std::uint64_t max_storage_range);

class LlamaGpu {
 public:
  static constexpr std::int64_t kChunk = 64;    // at most this many tokens per forward_chunk call
  static constexpr double kBudgetMs = 50.0;     // predicted GPU time per submission
  static constexpr double kWatchdogMs = 250.0;  // measured GPU time that disables the GPU

  LlamaGpu(const GpuPassInputs& in, const gpu::ContextOptions& options);
  ~LlamaGpu();
  LlamaGpu(const LlamaGpu&) = delete;
  LlamaGpu& operator=(const LlamaGpu&) = delete;

  // How many of the next `remaining` tokens (at positions from p0) the next
  // forward_chunk call should take: the most, up to kChunk, whose one-layer
  // submission, and with all logits its LM head, is predicted within
  // kBudgetMs. At least 1.
  std::int64_t chunk_tokens(std::int64_t p0, std::int64_t remaining, bool all_logits) const;

  // Feeds at most kChunk tokens at positions p0 .. p0 + tokens.size() - 1.
  // `logits` (may be null): one row per token with all_logits, else the last
  // token's row. The caller checks the context size and the token ids.
  void forward_chunk(std::span<const std::int32_t> tokens, std::int64_t p0, float* logits, bool all_logits,
                     TensorDump* dump);

  const gpu::DeviceInfo& device() const noexcept { return ctx_.info(); }
  std::uint64_t submissions() const noexcept { return submissions_; }
  std::uint64_t dispatches() const noexcept { return dispatches_; }
  std::uint64_t validation_messages() const noexcept { return ctx_.validation_message_count(); }
  // The longest measured submission of each kind, and the cost model's
  // scales (measured / predicted; 1 before the first measurement).
  double max_layers_ms() const noexcept { return max_layers_ms_; }
  double max_head_ms() const noexcept { return max_head_ms_; }
  double layers_scale() const noexcept { return layers_scale_.value_or(1.0); }
  double head_scale() const noexcept { return head_scale_.value_or(1.0); }

  // Test hooks for the failure paths; real hangs are never used for tests
  // (TDR budget). After `n` more submissions complete, the next one still
  // runs, then reports kDeviceLost as a lost device would (not written to
  // the ledger). A lower watchdog limit makes ordinary submissions trip it
  // (written to the ledger, which a test points at a file of its own).
  void simulate_device_loss_after(std::uint64_t n) noexcept { lose_after_ = submissions_ + n; }
  void set_watchdog_ms(double ms) noexcept { watchdog_ms_ = ms; }

 private:
  struct Matrix {
    std::size_t index = 0;  // in the arena
    cpu::PackedView view;
    VkDescriptorSet gemv = VK_NULL_HANDLE, gemm = VK_NULL_HANDLE;
  };
  struct Layer {
    Matrix q, k, v, o, gate, up, down;
    VkDescriptorSet attn_norm = VK_NULL_HANDLE, ffn_norm = VK_NULL_HANDLE, attention = VK_NULL_HANDLE;
    // The bias adds on q_, k_ and v_ (Qwen2), VK_NULL_HANDLE where the layer has none.
    VkDescriptorSet q_bias = VK_NULL_HANDLE, k_bias = VK_NULL_HANDLE, v_bias = VK_NULL_HANDLE;
  };

  Matrix make_matrix(const gguf::TensorInfo* t, const gpu::Buffer& acts_q, const gpu::Buffer& acts_d,
                     const gpu::Buffer& acts_s, const gpu::Buffer& out);
  void quantize(VkCommandBuffer cmd, VkDescriptorSet set, std::int64_t tokens, std::int64_t k);
  void matmul(VkCommandBuffer cmd, const Matrix& m, std::int64_t tokens, bool single);
  void record_layer(VkCommandBuffer cmd, std::size_t l, std::int64_t T, std::int64_t p0);
  double layer_cost_ms(std::int64_t T, std::int64_t p0) const;
  double head_cost_ms(std::int64_t T) const;
  // Records with `record` between timestamps (record writes timestamps 1..
  // itself through stamp()), submits, waits, then checks the measured time:
  // cost-model scale, watchdog, trace lane. `first_layer` < 0: the LM head.
  void submit(const std::function<void(VkCommandBuffer)>& record, int first_layer, std::size_t layers,
              std::int64_t tokens, double predicted_ms);
  void stamp(VkCommandBuffer cmd) { stamps_.write(cmd, stamp_next_++); }

  GpuPassInputs in_;
  LlamaConfig cfg_;
  gpu::Context ctx_;
  gpu::WeightArena weights_;
  gpu::Pipeline quant_, gemv_q8_, gemv_q4_, gemm_q8_, gemm_q4_, norm_, rope_, silu_, add_, bias_, attention_;
  // Activations of one chunk, [kChunk][width]; x is the residual stream and
  // is written by the CPU (embeddings) before each chunk.
  gpu::Buffer x_, h_, q_, k_, v_, att_, proj_, g_, u_, hlast_;
  gpu::Buffer xq_, xd_, xs_, xq1_, xd1_, xs1_;  // packed activations: the chunk's, and one token's
  gpu::Buffer logits_, kcache_, vcache_, scores_, rope_table_, norms_, biases_;
  gpu::DescriptorPool pool_;
  std::vector<Layer> layers_;
  Matrix head_, head1_;  // the LM head on the chunk's activations, and on one token's
  VkDescriptorSet quant_h_ = VK_NULL_HANDLE, quant_att_ = VK_NULL_HANDLE, quant_g_ = VK_NULL_HANDLE,
                  quant_last_ = VK_NULL_HANDLE, out_norm_ = VK_NULL_HANDLE, rope_q_ = VK_NULL_HANDLE,
                  rope_k_ = VK_NULL_HANDLE, silu_set_ = VK_NULL_HANDLE, add_set_ = VK_NULL_HANDLE;
  std::uint64_t submissions_ = 0, dispatches_ = 0;
  gpu::TimestampPool stamps_;  // the current submission's: start, after each layer (or the head)
  std::uint32_t stamp_next_ = 0;
  std::optional<gpu::ClockMap> clock_;  // --trace only
  bool clock_warned_ = false;
  std::optional<double> layers_scale_, head_scale_;
  double max_layers_ms_ = 0, max_head_ms_ = 0;
  double watchdog_ms_ = kWatchdogMs;
  std::uint64_t lose_after_ = UINT64_MAX;
};

}  // namespace hearth::model
