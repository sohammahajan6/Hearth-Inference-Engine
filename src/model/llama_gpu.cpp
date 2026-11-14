#include "model/llama_gpu.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <format>
#include <string>
#include <utility>

#include "core/clock.h"
#include "core/error.h"
#include "core/log.h"
#include "core/trace.h"
#include "cpu/ref_ops.h"
#include "gpu/hang_ledger.h"
#include "model/dequant.h"
#include "model/llama_ref.h"

namespace hearth::model {
namespace {

// Predicted cost of one submission, the TDR guard (08 doc section 4: <= 100 ms per
// submission). The rates are below the measured ones (bench/gpu/), so the
// prediction errs on the long side; measured times then scale it (submit()).
constexpr double kBytesPerMs = 20e6;  // weight streaming, 20 GB/s
constexpr double kMacsPerMs = 100e6;  // the GEMM, 100 GMAC/s
constexpr double kFlopsPerMs = 10e6;  // attention, 10 GFLOP/s

constexpr double kMinScale = 0.25;                     // the cost model never trusts a rate 4x its constants
constexpr std::int64_t kClockRenewNs = 1'000'000'000;  // --trace: a new clock reading every second

constexpr std::uint32_t kGemvRowsPerWg = 8;
constexpr std::uint32_t kGemmSubgroups = 4, kTileRows = 4, kTileTokens = 4;

std::uint32_t u32(std::int64_t v) { return static_cast<std::uint32_t>(v); }
VkDeviceSize bytes_of(std::int64_t n) { return static_cast<VkDeviceSize>(std::max<std::int64_t>(n, 4)) * 4; }

gpu::PipelineDesc subgroup8(const char* shader, std::uint32_t buffers, std::uint32_t push,
                            std::vector<std::uint32_t> spec) {
  return {.shader = shader,
          .buffers = buffers,
          .push_constant_bytes = push,
          .spec_constants = std::move(spec),
          .required_subgroup_size = 8,
          .require_full_subgroups = true};
}

struct GemvPush {
  std::uint32_t rows, groups, row_begin;
};
struct GemmPush {
  std::uint32_t rows, groups, row_begin, tokens, ldy;
};
struct QuantPush {
  std::uint32_t tokens, groups, blocks;
};
struct NormPush {
  std::uint32_t n;
  float eps;
};
struct RopePush {
  std::uint32_t tokens, heads, head_dim, rope_dim, stride, p0, neox;
};
struct BiasPush {
  std::uint32_t n, width;
};
struct AttentionPush {
  std::uint32_t p0, heads, kv_heads, head_dim, n_ctx;
  float scale;
};

void copy(VkCommandBuffer cmd, const gpu::Buffer& from, VkDeviceSize from_off, const gpu::Buffer& to,
          VkDeviceSize to_off, VkDeviceSize bytes) {
  const VkBufferCopy region{from_off, to_off, bytes};
  vkCmdCopyBuffer(cmd, from.handle(), to.handle(), 1, &region);
}

}  // namespace

void check_attention_limits(std::uint64_t n_layer, std::uint64_t n_ctx, std::uint64_t kv_width, std::uint64_t chunk,
                            std::uint64_t n_head, std::uint64_t max_buffer_size, std::uint64_t max_storage_range) {
  // Bytes per context position of each limited buffer; the largest context
  // that fits is the smallest limit / per-position size.
  struct Item {
    const char* what;
    std::uint64_t per_position;
    std::uint64_t limit;
  };
  const Item items[] = {
      {"one layer's K or V range (maxStorageBufferRange)", kv_width * 4, max_storage_range},
      {"the attention score buffer (maxStorageBufferRange)", chunk * n_head * 4, max_storage_range},
      {"the whole K or V cache (maxBufferSize)", n_layer * kv_width * 4, max_buffer_size},
  };
  for (const Item& it : items) {
    if (it.limit == 0 || it.per_position == 0) continue;
    const std::uint64_t fits = it.limit / it.per_position;
    if (n_ctx > fits) {
      fail(ErrorKind::kUnsupported,
           "--backend gpu: context {} needs {} bytes for {}, above the GPU's limit of {} bytes; use --ctx {} or less",
           n_ctx, n_ctx * it.per_position, it.what, it.limit, fits);
    }
  }
}

LlamaGpu::LlamaGpu(const GpuPassInputs& in, const gpu::ContextOptions& options)
    : in_(in),
      cfg_(*in.config),
      ctx_(options),
      weights_(ctx_, *in.arena),
      quant_(ctx_, {.shader = "quantize_act", .buffers = 4, .push_constant_bytes = sizeof(QuantPush)}),
      gemv_q8_(ctx_, subgroup8("qgemv_q8_0", 5, sizeof(GemvPush), {kGemvRowsPerWg, 8 * kGemvRowsPerWg})),
      gemv_q4_(ctx_, subgroup8("qgemv_q4_0", 6, sizeof(GemvPush), {kGemvRowsPerWg, 8 * kGemvRowsPerWg})),
      gemm_q8_(ctx_, subgroup8("qgemm_q8_0", 5, sizeof(GemmPush),
                               {kGemmSubgroups, 8 * kGemmSubgroups, kTileRows, kTileTokens})),
      gemm_q4_(ctx_, subgroup8("qgemm_q4_0", 6, sizeof(GemmPush),
                               {kGemmSubgroups, 8 * kGemmSubgroups, kTileRows, kTileTokens})),
      norm_(ctx_, {.shader = "rmsnorm", .buffers = 3, .push_constant_bytes = sizeof(NormPush)}),
      rope_(ctx_, {.shader = "rope", .buffers = 2, .push_constant_bytes = sizeof(RopePush)}),
      silu_(ctx_, {.shader = "silu_mul", .buffers = 2, .push_constant_bytes = 4}),
      add_(ctx_, {.shader = "residual_add", .buffers = 2, .push_constant_bytes = 4}),
      bias_(ctx_, {.shader = "bias_add", .buffers = 2, .push_constant_bytes = sizeof(BiasPush)}),
      attention_(ctx_, {.shader = "attention", .buffers = 5, .push_constant_bytes = sizeof(AttentionPush)}),
      pool_(ctx_, 64 + 24 * (cfg_.n_layer + 1), 6 * (64 + 24 * (cfg_.n_layer + 1))),
      stamps_(ctx_, cfg_.n_layer + 2) {
  const LlamaConfig& c = cfg_;
  const std::int64_t C = kChunk, E = c.n_embd, Q = std::int64_t{c.n_head} * c.head_dim,
                     KV = std::int64_t{c.n_head_kv} * c.head_dim, F = c.n_ff, L = c.n_layer;
  const std::int64_t max_k = std::max({E, Q, F});
  const std::int64_t groups = cpu::groups_for(max_k);
  using MK = gpu::MemoryKind;
  x_ = gpu::Buffer(ctx_, bytes_of(C * E), MK::kHostCached);  // the CPU writes embeddings, reads l_out
  h_ = gpu::Buffer(ctx_, bytes_of(C * E), MK::kDeviceLocal);
  q_ = gpu::Buffer(ctx_, bytes_of(C * Q), MK::kDeviceLocal);
  k_ = gpu::Buffer(ctx_, bytes_of(C * KV), MK::kDeviceLocal);
  v_ = gpu::Buffer(ctx_, bytes_of(C * KV), MK::kDeviceLocal);
  att_ = gpu::Buffer(ctx_, bytes_of(C * Q), MK::kDeviceLocal);
  proj_ = gpu::Buffer(ctx_, bytes_of(C * E), MK::kDeviceLocal);
  g_ = gpu::Buffer(ctx_, bytes_of(C * F), MK::kDeviceLocal);
  u_ = gpu::Buffer(ctx_, bytes_of(C * F), MK::kDeviceLocal);
  hlast_ = gpu::Buffer(ctx_, bytes_of(E), MK::kDeviceLocal);
  xq_ = gpu::Buffer(ctx_, static_cast<VkDeviceSize>(C * groups * cpu::kGroupValues), MK::kDeviceLocal);
  xd_ = gpu::Buffer(ctx_, bytes_of(C * groups * cpu::kGroupBlocks), MK::kDeviceLocal);
  xs_ = gpu::Buffer(ctx_, bytes_of(C * groups * cpu::kGroupBlocks), MK::kDeviceLocal);
  xq1_ = gpu::Buffer(ctx_, static_cast<VkDeviceSize>(groups * cpu::kGroupValues), MK::kDeviceLocal);
  xd1_ = gpu::Buffer(ctx_, bytes_of(groups * cpu::kGroupBlocks), MK::kDeviceLocal);
  xs1_ = gpu::Buffer(ctx_, bytes_of(groups * cpu::kGroupBlocks), MK::kDeviceLocal);
  logits_ = gpu::Buffer(ctx_, bytes_of(C * c.n_vocab), MK::kHostCached);
  // The CPU writes x_ and reads x_ and logits_ with no vkFlush/vkInvalidate
  // calls, which is only correct for host-coherent memory. kHostCached prefers
  // a coherent type (type 2 on Iris Xe is cached and coherent); refuse a
  // driver that gives a non-coherent one rather than read stale values.
  for (const gpu::Buffer* b : {&x_, &logits_}) {
    if ((ctx_.info().memory_types[b->memory_type()].flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) {
      fail(ErrorKind::kUnsupported,
           "--backend gpu needs host-cached memory that is also coherent; GPU '{}' gave memory type {}",
           ctx_.info().name, b->memory_type());
    }
  }
  check_attention_limits(static_cast<std::uint64_t>(L), in.n_ctx, static_cast<std::uint64_t>(KV),
                         static_cast<std::uint64_t>(C), c.n_head, ctx_.info().max_buffer_size,
                         ctx_.info().limits.maxStorageBufferRange);
  kcache_ = gpu::Buffer(ctx_, bytes_of(L * in.n_ctx * KV), MK::kDeviceLocal);
  vcache_ = gpu::Buffer(ctx_, bytes_of(L * in.n_ctx * KV), MK::kDeviceLocal);
  scores_ = gpu::Buffer(ctx_, bytes_of(C * c.n_head * std::int64_t{in.n_ctx}), MK::kDeviceLocal);
  {
    const std::vector<float> table = cpu::ref::rope_table(in.n_ctx, c.rope_dim, c.rope_theta,
                                                          in.rope_factors->empty() ? nullptr : in.rope_factors->data());
    rope_table_ = gpu::Buffer(ctx_, bytes_of(static_cast<std::int64_t>(table.size())), MK::kHostCoherent);
    std::memcpy(rope_table_.mapped(), table.data(), table.size() * 4);
  }
  // The norm weights, one row of E floats each: attn_norm per layer, ffn_norm per layer, output_norm.
  norms_ = gpu::Buffer(ctx_, bytes_of((2 * L + 1) * E), MK::kHostCoherent);
  auto* nw = static_cast<float*>(norms_.mapped());
  for (std::int64_t l = 0; l < L; ++l) {
    std::memcpy(nw + l * E, (*in.attn_norm)[static_cast<std::size_t>(l)].data(), static_cast<std::size_t>(E) * 4);
    std::memcpy(nw + (L + l) * E, (*in.ffn_norm)[static_cast<std::size_t>(l)].data(), static_cast<std::size_t>(E) * 4);
  }
  std::memcpy(nw + 2 * L * E, in.output_norm->data(), static_cast<std::size_t>(E) * 4);
  // Qwen2's biases, each slice at an offset the descriptor rules allow: [layer][q, k, v].
  struct BiasSlice {
    const std::vector<float>* values = nullptr;  // null: none
    VkDeviceSize offset = 0;
  };
  std::vector<std::array<BiasSlice, 3>> bias_slices(static_cast<std::size_t>(L));
  {
    const VkDeviceSize align = std::max<VkDeviceSize>(ctx_.info().limits.minStorageBufferOffsetAlignment, 4);
    VkDeviceSize total = 0;
    for (std::size_t l = 0; l < bias_slices.size(); ++l) {
      const std::vector<std::vector<float>>* per_kind[3] = {in.q_bias, in.k_bias, in.v_bias};
      const std::int64_t widths[3] = {Q, KV, KV};
      for (std::size_t w = 0; w < 3; ++w) {
        if (per_kind[w] == nullptr || (*per_kind[w])[l].empty()) continue;
        HEARTH_CHECK(static_cast<std::int64_t>((*per_kind[w])[l].size()) == widths[w]);
        total = (total + align - 1) / align * align;
        bias_slices[l][w] = {&(*per_kind[w])[l], total};
        total += static_cast<VkDeviceSize>(widths[w]) * 4;
      }
    }
    if (total > 0) {
      biases_ = gpu::Buffer(ctx_, total, MK::kHostCoherent);
      for (const auto& layer : bias_slices) {
        for (const BiasSlice& b : layer) {
          if (b.values != nullptr) {
            std::memcpy(static_cast<std::byte*>(biases_.mapped()) + b.offset, b.values->data(), b.values->size() * 4);
          }
        }
      }
    }
  }

  using Range = gpu::DescriptorPool::Range;
  auto whole = [](const gpu::Buffer& b) { return Range{&b, 0, b.size()}; };
  auto row = [&](std::int64_t r) { return Range{&norms_, static_cast<VkDeviceSize>(r * E * 4), bytes_of(E)}; };
  auto set = [&](const gpu::Pipeline& p, std::vector<Range> ranges) { return pool_.allocate(p, ranges); };
  for (std::int64_t l = 0; l < L; ++l) {
    const LlamaTensors::Layer& T = in.tensors->layers[static_cast<std::size_t>(l)];
    Layer y;
    y.q = make_matrix(T.attn_q, xq_, xd_, xs_, q_);
    y.k = make_matrix(T.attn_k, xq_, xd_, xs_, k_);
    y.v = make_matrix(T.attn_v, xq_, xd_, xs_, v_);
    y.o = make_matrix(T.attn_output, xq_, xd_, xs_, proj_);
    y.gate = make_matrix(T.ffn_gate, xq_, xd_, xs_, g_);
    y.up = make_matrix(T.ffn_up, xq_, xd_, xs_, u_);
    y.down = make_matrix(T.ffn_down, xq_, xd_, xs_, proj_);
    y.attn_norm = set(norm_, {whole(x_), row(l), whole(h_)});
    y.ffn_norm = set(norm_, {whole(x_), row(L + l), whole(h_)});
    const VkDeviceSize layer_bytes = static_cast<VkDeviceSize>(std::int64_t{in.n_ctx} * KV * 4);
    y.attention = set(attention_, {whole(q_), Range{&kcache_, static_cast<VkDeviceSize>(l) * layer_bytes, layer_bytes},
                                   Range{&vcache_, static_cast<VkDeviceSize>(l) * layer_bytes, layer_bytes},
                                   whole(scores_), whole(att_)});
    const std::array<BiasSlice, 3>& bs = bias_slices[static_cast<std::size_t>(l)];
    auto bias_set = [&](const BiasSlice& b, const gpu::Buffer& out) {
      if (b.values == nullptr) return VkDescriptorSet{VK_NULL_HANDLE};
      return set(bias_, {whole(out), Range{&biases_, b.offset, static_cast<VkDeviceSize>(b.values->size() * 4)}});
    };
    y.q_bias = bias_set(bs[0], q_);
    y.k_bias = bias_set(bs[1], k_);
    y.v_bias = bias_set(bs[2], v_);
    layers_.push_back(y);
  }
  const gguf::TensorInfo* head = in.tensors->output != nullptr ? in.tensors->output : in.tensors->token_embd;
  head_ = make_matrix(head, xq_, xd_, xs_, logits_);
  head1_ = make_matrix(head, xq1_, xd1_, xs1_, logits_);
  out_norm_ = set(norm_, {whole(x_), row(2 * L), whole(h_)});
  quant_h_ = set(quant_, {whole(h_), whole(xq_), whole(xd_), whole(xs_)});
  quant_att_ = set(quant_, {whole(att_), whole(xq_), whole(xd_), whole(xs_)});
  quant_g_ = set(quant_, {whole(g_), whole(xq_), whole(xd_), whole(xs_)});
  quant_last_ = set(quant_, {whole(hlast_), whole(xq1_), whole(xd1_), whole(xs1_)});
  rope_q_ = set(rope_, {whole(q_), whole(rope_table_)});
  rope_k_ = set(rope_, {whole(k_), whole(rope_table_)});
  silu_set_ = set(silu_, {whole(g_), whole(u_)});
  add_set_ = set(add_, {whole(x_), whole(proj_)});
}

LlamaGpu::~LlamaGpu() = default;

LlamaGpu::Matrix LlamaGpu::make_matrix(const gguf::TensorInfo* t, const gpu::Buffer& acts_q, const gpu::Buffer& acts_d,
                                       const gpu::Buffer& acts_s, const gpu::Buffer& out) {
  // Q8_0 and Q4_0 matrices are imported as they are; Q4_1, Q5_0, Q5_1,
  // Q2_K, Q3_K and Q5_K were converted to Q8_0 into the arena at load
  // (LlamaRef, ADR 0028). Full precision and Q4_K/Q6_K are not in it.
  const auto type = static_cast<gguf::GgmlType>(t->type);
  if (type == gguf::GgmlType::kF32 || type == gguf::GgmlType::kF16 || type == gguf::GgmlType::kBF16 ||
      type == gguf::GgmlType::kQ4_K || type == gguf::GgmlType::kQ6_K) {
    fail(ErrorKind::kUnsupported,
         "--backend gpu cannot read tensor '{}' of type {} (full precision, Q4_K and Q6_K run on the CPU only; "
         "use --backend cpu)",
         t->name, t->traits->name);
  }
  Matrix m;
  m.index = in_.arena_index(t);
  m.view = in_.arena->view(m.index);
  const bool q8 = m.view.type == cpu::QType::kQ8_0;
  std::vector<gpu::DescriptorPool::Range> r = {
      weights_.quants(m.index), weights_.scales(m.index), {&acts_q, 0, acts_q.size()}, {&acts_d, 0, acts_d.size()}};
  if (!q8) r.push_back({&acts_s, 0, acts_s.size()});
  r.push_back({&out, 0, out.size()});
  m.gemv = pool_.allocate(q8 ? gemv_q8_ : gemv_q4_, r);
  m.gemm = pool_.allocate(q8 ? gemm_q8_ : gemm_q4_, r);
  return m;
}

void LlamaGpu::quantize(VkCommandBuffer cmd, VkDescriptorSet set, std::int64_t tokens, std::int64_t k) {
  const QuantPush p{u32(tokens), u32(cpu::groups_for(k)), u32(k / quant::kBlock)};
  quant_.dispatch(cmd, set, &p, sizeof p, u32((tokens * cpu::groups_for(k) * cpu::kGroupBlocks + 63) / 64));
  gpu::compute_barrier(cmd);
  ++dispatches_;
}

void LlamaGpu::matmul(VkCommandBuffer cmd, const Matrix& m, std::int64_t tokens, bool single) {
  const bool q8 = m.view.type == cpu::QType::kQ8_0;
  if (single) {
    const GemvPush p{u32(m.view.rows), u32(m.view.groups), 0};
    (q8 ? gemv_q8_ : gemv_q4_)
        .dispatch(cmd, m.gemv, &p, sizeof p, u32((m.view.rows + kGemvRowsPerWg - 1) / kGemvRowsPerWg));
  } else {
    const GemmPush p{u32(m.view.rows), u32(m.view.groups), 0, u32(tokens), u32(m.view.rows)};
    const std::int64_t rows_per_wg = kGemmSubgroups * kTileRows;
    (q8 ? gemm_q8_ : gemm_q4_)
        .dispatch(cmd, m.gemm, &p, sizeof p, u32((tokens + kTileTokens - 1) / kTileTokens),
                  u32((m.view.rows + rows_per_wg - 1) / rows_per_wg));
  }
  ++dispatches_;
}

void LlamaGpu::record_layer(VkCommandBuffer cmd, std::size_t l, std::int64_t T, std::int64_t p0) {
  const LlamaConfig& c = cfg_;
  const Layer& y = layers_[l];
  const std::int64_t E = c.n_embd, hd = c.head_dim, Q = std::int64_t{c.n_head} * hd,
                     KV = std::int64_t{c.n_head_kv} * hd, F = c.n_ff;
  const bool single = T == 1;
  auto elementwise = [&](const gpu::Pipeline& p, VkDescriptorSet s, std::int64_t n) {
    const auto nu = u32(n);
    p.dispatch(cmd, s, &nu, 4, u32((n + 255) / 256));
    gpu::compute_barrier(cmd);
    ++dispatches_;
  };
  const NormPush norm{u32(E), c.rms_eps};
  // Attention block.
  norm_.dispatch(cmd, y.attn_norm, &norm, sizeof norm, u32(T));
  gpu::compute_barrier(cmd);
  quantize(cmd, quant_h_, T, E);
  matmul(cmd, y.q, T, single);
  matmul(cmd, y.k, T, single);
  matmul(cmd, y.v, T, single);
  gpu::compute_barrier(cmd);
  // Qwen2's biases, before RoPE and the KV cache, as LlamaRef adds them.
  const BiasPush bq{u32(T * Q), u32(Q)}, bkv{u32(T * KV), u32(KV)};
  bool biased = false;
  for (const auto& [bset, push] : {std::pair{y.q_bias, &bq}, std::pair{y.k_bias, &bkv}, std::pair{y.v_bias, &bkv}}) {
    if (bset == VK_NULL_HANDLE) continue;
    bias_.dispatch(cmd, bset, push, sizeof *push, (push->n + 255) / 256);
    ++dispatches_;
    biased = true;
  }
  if (biased) gpu::compute_barrier(cmd);
  const std::uint32_t neox = c.rope_neox ? 1u : 0u;
  const RopePush rq{u32(T), c.n_head, c.head_dim, c.rope_dim, u32(Q), u32(p0), neox};
  const RopePush rk{u32(T), c.n_head_kv, c.head_dim, c.rope_dim, u32(KV), u32(p0), neox};
  rope_.dispatch(cmd, rope_q_, &rq, sizeof rq, u32((T * c.n_head * (c.rope_dim / 2) + 63) / 64));
  rope_.dispatch(cmd, rope_k_, &rk, sizeof rk, u32((T * c.n_head_kv * (c.rope_dim / 2) + 63) / 64));
  gpu::compute_barrier(cmd);
  const VkDeviceSize cache_off = static_cast<VkDeviceSize>((static_cast<std::int64_t>(l) * in_.n_ctx + p0) * KV * 4);
  copy(cmd, k_, 0, kcache_, cache_off, static_cast<VkDeviceSize>(T * KV * 4));
  copy(cmd, v_, 0, vcache_, cache_off, static_cast<VkDeviceSize>(T * KV * 4));
  gpu::compute_barrier(cmd);
  const AttentionPush ap{u32(p0),    c.n_head,  c.n_head_kv,
                         c.head_dim, in_.n_ctx, 1.0f / std::sqrt(static_cast<float>(hd))};
  attention_.dispatch(cmd, y.attention, &ap, sizeof ap, u32(T * c.n_head));
  gpu::compute_barrier(cmd);
  quantize(cmd, quant_att_, T, Q);
  matmul(cmd, y.o, T, single);
  gpu::compute_barrier(cmd);
  elementwise(add_, add_set_, T * E);
  // Feed-forward block.
  norm_.dispatch(cmd, y.ffn_norm, &norm, sizeof norm, u32(T));
  gpu::compute_barrier(cmd);
  quantize(cmd, quant_h_, T, E);
  matmul(cmd, y.gate, T, single);
  matmul(cmd, y.up, T, single);
  gpu::compute_barrier(cmd);
  elementwise(silu_, silu_set_, T * F);
  quantize(cmd, quant_g_, T, F);
  matmul(cmd, y.down, T, single);
  gpu::compute_barrier(cmd);
  elementwise(add_, add_set_, T * E);
  dispatches_ += 5;  // the two norms, the two ropes and attention
}

double LlamaGpu::layer_cost_ms(std::int64_t T, std::int64_t p0) const {
  const Layer& y = layers_[0];
  double bytes = 0, macs = 0;
  for (const Matrix* m : {&y.q, &y.k, &y.v, &y.o, &y.gate, &y.up, &y.down}) {
    bytes += static_cast<double>(in_.arena->location(m->index).bytes);
    macs += static_cast<double>(m->view.rows) * static_cast<double>(m->view.k);
  }
  const double attention_flops =
      4.0 * static_cast<double>(T) * cfg_.n_head * static_cast<double>(p0 + T) * cfg_.head_dim;
  return bytes / kBytesPerMs + static_cast<double>(T) * macs / kMacsPerMs + attention_flops / kFlopsPerMs;
}

double LlamaGpu::head_cost_ms(std::int64_t T) const {
  return static_cast<double>(in_.arena->location(head_.index).bytes) / kBytesPerMs +
         static_cast<double>(T) * static_cast<double>(head_.view.rows) * static_cast<double>(head_.view.k) / kMacsPerMs;
}

std::int64_t LlamaGpu::chunk_tokens(std::int64_t p0, std::int64_t remaining, bool all_logits) const {
  std::int64_t T = std::clamp<std::int64_t>(remaining, 1, kChunk);
  const double ls = layers_scale(), hs = head_scale();
  while (T > 1 && (ls * layer_cost_ms(T, p0) > kBudgetMs || (all_logits && hs * head_cost_ms(T) > kBudgetMs))) --T;
  return T;
}

void LlamaGpu::submit(const std::function<void(VkCommandBuffer)>& record, int first_layer, std::size_t layers,
                      std::int64_t tokens, double predicted_ms) {
  stamps_.reset();
  stamp_next_ = 0;
  gpu::note_device_loss("gpu forward pass", [&] {
    ctx_.run_once([&](VkCommandBuffer cmd) {
      gpu::compute_barrier(cmd);  // after the previous submission's writes
      stamp(cmd);
      record(cmd);
      gpu::compute_to_host_barrier(cmd);
    });
  });
  ++submissions_;
  const bool head = first_layer < 0;
  const std::uint32_t n = stamp_next_;
  HEARTH_CHECK(n == (head ? 2u : static_cast<std::uint32_t>(layers) + 1));
  const std::vector<std::uint64_t> t = stamps_.read(0, n);
  const double ms = stamps_.elapsed_ns(t[0], t[n - 1]) / 1e6;
  // The cost model learns the largest measured / predicted ratio of each kind.
  std::optional<double>& scale = head ? head_scale_ : layers_scale_;
  const double ratio = std::max(ms / predicted_ms, kMinScale);
  scale = scale ? std::max(*scale, ratio) : ratio;
  double& longest = head ? max_head_ms_ : max_layers_ms_;
  longest = std::max(longest, ms);
  if (trace::on()) {
    if (ctx_.calibrated_timestamps_enabled()) {
      if (!clock_ || now_ns() - clock_->cpu_ns() > kClockRenewNs) clock_ = gpu::ClockMap::calibrate(ctx_, 8);
      trace::name_lane(trace::kGpuLane, std::string("GPU: ") + ctx_.info().name);
      for (std::uint32_t i = 0; i + 1 < n; ++i) {
        trace::record_on(trace::kGpuLane, head ? "gpu lm_head" : "gpu layer", clock_->to_cpu_ns(t[i]),
                         clock_->to_cpu_ns(t[i + 1]), head ? -1 : first_layer + static_cast<int>(i),
                         head ? static_cast<std::int64_t>(cfg_.n_vocab) : 0, tokens);
      }
    } else if (!clock_warned_) {
      clock_warned_ = true;
      log_warn("--trace: GPU '{}' cannot relate its timestamps to the CPU clock; the trace has no GPU lane",
               ctx_.info().name);
    }
  }
  if (ms > watchdog_ms_) {
    const std::string what =
        std::format("{} of {} token(s) took {:.1f} ms on the GPU (watchdog {:.0f} ms, predicted {:.1f} ms)",
                    head ? std::string("the LM head")
                         : std::format("layers {}-{}", first_layer, first_layer + static_cast<int>(layers) - 1),
                    tokens, ms, watchdog_ms_, predicted_ms);
    gpu::record_hang("near-miss", what);
    fail(ErrorKind::kDeviceLost, "gpu: near-miss watchdog: {}; the GPU is not used again in this process", what);
  }
  if (submissions_ > lose_after_) fail(ErrorKind::kDeviceLost, "gpu: simulated device loss (test hook)");
}

void LlamaGpu::forward_chunk(std::span<const std::int32_t> tokens, std::int64_t p0, float* logits, bool all_logits,
                             TensorDump* dump) {
  const LlamaConfig& c = cfg_;
  const auto T = static_cast<std::int64_t>(tokens.size());
  HEARTH_CHECK(T >= 1 && T <= kChunk && p0 + T <= in_.n_ctx);
  const std::int64_t E = c.n_embd;
  if (dump != nullptr) {
    for (const char* name :
         {"attn_norm", "Qcur", "Kcur", "Vcur", "attn_out", "ffn_inp", "ffn_norm", "ffn_out", "result_norm"}) {
      for (int l = -1; l < static_cast<int>(c.n_layer); ++l) {
        if (dump->wants(name, l)) {
          fail(ErrorKind::kUnsupported, "--backend gpu can dump l_out and result_output, not {}", name);
        }
      }
    }
  }
  auto* x = static_cast<float*>(x_.mapped());
  for (std::int64_t t = 0; t < T; ++t) {
    dequant_row(*in_.tensors->token_embd, in_.file->tensor_bytes(*in_.tensors->token_embd),
                tokens[static_cast<std::size_t>(t)], x + t * E);
  }
  // Layers in submissions of a bounded predicted cost; one layer per
  // submission when l_out is dumped, so each layer's output can be read.
  const bool dump_layers = dump != nullptr && dump->wants("l_out", 0);
  const double per_layer = layer_cost_ms(T, p0);
  const auto per_submit =
      dump_layers ? std::size_t{1}
                  : static_cast<std::size_t>(std::clamp(kBudgetMs / (per_layer * layers_scale()), 1.0, 1e6));
  for (std::size_t a = 0; a < c.n_layer; a += per_submit) {
    const std::size_t b = std::min<std::size_t>(c.n_layer, a + per_submit);
    submit(
        [&](VkCommandBuffer cmd) {
          for (std::size_t l = a; l < b; ++l) {
            record_layer(cmd, l, T, p0);
            stamp(cmd);
          }
        },
        static_cast<int>(a), b - a, T, per_layer * static_cast<double>(b - a));
    if (dump_layers) {
      for (std::size_t l = a; l < b; ++l) {
        for (std::int64_t t = 0; t < T; ++t) {
          if (dump->wants_position(p0 + t)) dump->record("l_out", static_cast<int>(l), p0 + t, x + t * E, E);
        }
      }
    }
  }
  const bool want_logits = dump != nullptr && dump->wants("result_output", -1);
  if (logits == nullptr && !want_logits) return;
  const bool all = all_logits || want_logits;
  const std::int64_t head_tokens = all ? T : 1;
  submit(
      [&](VkCommandBuffer cmd) {
        const NormPush norm{u32(E), c.rms_eps};
        norm_.dispatch(cmd, out_norm_, &norm, sizeof norm, u32(T));
        gpu::compute_barrier(cmd);
        if (all) {
          quantize(cmd, quant_h_, T, E);
          matmul(cmd, head_, T, T == 1);
        } else {
          copy(cmd, h_, static_cast<VkDeviceSize>((T - 1) * E * 4), hlast_, 0, static_cast<VkDeviceSize>(E * 4));
          gpu::compute_barrier(cmd);
          quantize(cmd, quant_last_, 1, E);
          matmul(cmd, head1_, 1, true);
        }
        stamp(cmd);
      },
      -1, 0, head_tokens, head_cost_ms(head_tokens));
  dispatches_ += 1;
  const auto* out = static_cast<const float*>(logits_.mapped());
  const std::size_t V = c.n_vocab;
  if (want_logits) {
    for (std::int64_t t = 0; t < T; ++t) {
      if (dump->wants_position(p0 + t)) {
        dump->record("result_output", -1, p0 + t, out + static_cast<std::size_t>(t) * V, static_cast<std::int64_t>(V));
      }
    }
  }
  if (logits != nullptr) {
    if (all_logits) {
      std::memcpy(logits, out, static_cast<std::size_t>(T) * V * 4);
    } else {
      std::memcpy(logits, all ? out + static_cast<std::size_t>(T - 1) * V : out, V * 4);
    }
  }
}

}  // namespace hearth::model
