#include "model/llama_ref.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <format>
#include <fstream>
#include <tuple>

#include "core/cpu_features.h"
#include "core/error.h"
#include "core/json_writer.h"
#include "core/log.h"
#include "core/platform.h"
#include "core/quant.h"
#include "core/trace.h"
#include "cpu/attention.h"
#include "cpu/ref_ops.h"
#include "gpu/context.h"
#include "model/dequant.h"
#include "model/llama_gpu.h"

namespace hearth::model {

// ---------------------------------------------------------------- TensorDump

namespace {

// Matrix products of at most this many tokens hand out rows dynamically
// (packed_product); longer ones keep the static, cache-blocked split.
constexpr std::int64_t kDynamicTokens = 32;

bool is_layer_name(std::string_view base) { return base != "result_norm" && base != "result_output"; }

}  // namespace

bool TensorDump::known(std::string_view name) {
  for (std::string_view base : kNames) {
    if (name == base) return true;
    if (!is_layer_name(base) || name.size() <= base.size() + 1 || !name.starts_with(base) || name[base.size()] != '-') {
      continue;
    }
    const std::string_view layer = name.substr(base.size() + 1);
    if (std::all_of(layer.begin(), layer.end(), [](char c) { return c >= '0' && c <= '9'; })) return true;
  }
  return false;
}

TensorDump::TensorDump(std::vector<std::string> names, std::vector<std::int64_t> positions)
    : names_(std::move(names)), positions_(std::move(positions)) {
  std::sort(positions_.begin(), positions_.end());
  positions_.erase(std::unique(positions_.begin(), positions_.end()), positions_.end());
}

bool TensorDump::wants(std::string_view base, int layer) const {
  for (const std::string& n : names_) {
    if (n == base || (layer >= 0 && n == std::format("{}-{}", base, layer))) return true;
  }
  return false;
}

bool TensorDump::wants_position(std::int64_t pos) const {
  return positions_.empty() || std::binary_search(positions_.begin(), positions_.end(), pos);
}

void TensorDump::record(std::string_view base, int layer, std::int64_t pos, const float* data, std::int64_t n) {
  const std::string name = layer < 0 ? std::string(base) : std::format("{}-{}", base, layer);
  for (const std::string& r : names_) {
    if (r == base || r == name) matched_.insert(r);
  }
  Tensor& t = tensors_[name];
  HEARTH_CHECK(t.width == 0 || t.width == n);
  t.width = n;
  t.rows[pos].assign(data, data + n);
}

std::vector<std::string> TensorDump::unmatched() const {
  std::vector<std::string> out;
  for (const std::string& n : names_) {
    if (!matched_.contains(n)) out.push_back(n);
  }
  return out;
}

void TensorDump::write(const std::filesystem::path& dir, const std::string& producer, const std::string& model,
                       std::span<const std::int32_t> tokens, std::uint32_t ctx) const {
  std::filesystem::create_directories(dir);
  // index.json is written last: a directory with one holds a complete dump.
  std::filesystem::remove(dir / "index.json");
  JsonWriter w;
  w.begin_object();
  w.field("producer", producer);
  w.field("model", model);
  w.key("tokens");
  w.begin_array();
  for (std::int32_t t : tokens) w.value(t);
  w.end_array();
  w.field("ctx", ctx);
  w.field("f32_kv", true);
  w.key("tensors");
  w.begin_array();
  for (const auto& [name, t] : tensors_) {
    const std::string file = name + ".bin";
    std::ofstream out(dir / file, std::ios::binary | std::ios::trunc);
    for (const auto& [pos, row] : t.rows) {
      out.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(row.size() * sizeof(float)));
    }
    out.close();
    if (!out) fail(ErrorKind::kIo, "cannot write {}", path_to_utf8(dir / file));
    w.begin_object();
    w.field("name", name);
    w.key("ne");
    w.begin_array();
    w.value(t.width);
    w.value(static_cast<std::int64_t>(tokens.size()));
    w.end_array();
    w.key("positions");
    w.begin_array();
    for (const auto& [pos, row] : t.rows) w.value(pos);
    w.end_array();
    w.field("file", file);
    w.field("count", 1);
    w.end_object();
  }
  w.end_array();
  w.end_object();
  std::ofstream index(dir / "index.json", std::ios::binary | std::ios::trunc);
  index << w.str() << "\n";
  index.close();
  if (!index) fail(ErrorKind::kIo, "cannot write {}", path_to_utf8(dir / "index.json"));
}

// ---------------------------------------------------------------- LlamaRef

LlamaRef::LlamaRef(const gguf::File& file, std::uint32_t n_ctx, ThreadPool& pool, Matmul matmul, Isa isa,
                   const gpu::ContextOptions* gpu_options)
    : file_(file),
      cfg_(LlamaConfig::from_gguf(file)),
      w_(LlamaTensors::bind(file, cfg_)),
      pool_(pool),
      matmul_(matmul),
      isa_(isa),
      n_ctx_(n_ctx) {
  if (isa != Isa::kScalar && isa != Isa::kAvx512) {
    fail(ErrorKind::kInvalidArgument, "the int8 matmul kernels exist as scalar and avx512, not {}", to_string(isa));
  }
  if (!isa_supported(isa)) fail(ErrorKind::kUnsupported, "this CPU cannot run the {} kernels", to_string(isa));
  if (n_ctx == 0 || n_ctx > cfg_.n_ctx_train) {
    fail(ErrorKind::kInvalidArgument, "context {} must be between 1 and the model's trained context {}", n_ctx,
         cfg_.n_ctx_train);
  }
  const std::size_t kv = std::size_t{cfg_.n_layer} * n_ctx * cfg_.n_head_kv * cfg_.head_dim;
  // The `cpu` backend's K cache is in blocks of 16 positions (cpu/attention.h),
  // so a layer's K takes whole blocks; cpu-ref uses the first n_ctx rows of it.
  k_layer_ = static_cast<std::size_t>(cpu::k_head_size(n_ctx, cfg_.head_dim)) * cfg_.n_head_kv;
  // kGpu keeps its KV cache on the GPU (LlamaGpu) and never runs the CPU
  // path, so it needs no CPU cache (268 MB for the 1B at ctx 4096).
  if (matmul_ != Matmul::kGpu) {
    k_cache_.assign(std::size_t{cfg_.n_layer} * k_layer_, 0.0f);
    v_cache_.assign(kv, 0.0f);
  }
  if (w_.rope_freqs != nullptr) rope_factors_ = f32_vector(w_.rope_freqs);
  for (const auto& layer : w_.layers) {
    attn_norm_.push_back(f32_vector(layer.attn_norm));
    ffn_norm_.push_back(f32_vector(layer.ffn_norm));
    q_bias_.push_back(layer.attn_q_bias != nullptr ? f32_vector(layer.attn_q_bias) : std::vector<float>());
    k_bias_.push_back(layer.attn_k_bias != nullptr ? f32_vector(layer.attn_k_bias) : std::vector<float>());
    v_bias_.push_back(layer.attn_v_bias != nullptr ? f32_vector(layer.attn_v_bias) : std::vector<float>());
  }
  output_norm_ = f32_vector(w_.output_norm);
  if (quantized()) {
    // Every Q8_0/Q4_0 matrix goes into one arena (cpu/packed_arena.h), which
    // the GPU can import as a whole. Layers first, then the LM head.
    // Formats without kernels of their own (Q4_1, Q5_0, Q5_1, Q2_K, Q3_K,
    // Q5_K) are converted to Q8_0 here, row by row: decoded as ggml decodes
    // them, then quantized with ggml's Q8_0 rule (Step W3, ADR 0028). Their
    // products then run on the Q8_0 kernels, at their speed, on the CPU and
    // the GPU alike, for the price of Q8_0's size in memory. Q4_K and Q6_K
    // keep their exact CPU kernels and are not converted: converting them
    // for the GPU measured 0.25 % in perplexity off the CPU path.
    std::vector<const gguf::TensorInfo*> mats;
    std::vector<bool> convert;
    std::vector<cpu::MatrixShape> shapes;
    auto add = [&](const gguf::TensorInfo* t) {
      const auto type = static_cast<gguf::GgmlType>(t->type);
      const bool native = type == gguf::GgmlType::kQ8_0 || type == gguf::GgmlType::kQ4_0;
      const bool k_kernel = type == gguf::GgmlType::kQ4_K || type == gguf::GgmlType::kQ6_K;
      const bool to_q8 = !native && !k_kernel && type != gguf::GgmlType::kF32 && type != gguf::GgmlType::kF16 &&
                         type != gguf::GgmlType::kBF16;
      if (!native && !to_q8) return;
      if (packed_index_.contains(t)) return;
      packed_index_.emplace(t, mats.size());
      mats.push_back(t);
      convert.push_back(to_q8);
      shapes.push_back({type == gguf::GgmlType::kQ4_0 ? cpu::QType::kQ4_0 : cpu::QType::kQ8_0,
                        static_cast<std::int64_t>(t->ne[1]), static_cast<std::int64_t>(t->ne[0])});
    };
    for (const auto& layer : w_.layers) {
      for (const gguf::TensorInfo* t : {layer.attn_q, layer.attn_k, layer.attn_v, layer.attn_output, layer.ffn_gate,
                                        layer.ffn_up, layer.ffn_down}) {
        add(t);
      }
    }
    add(w_.output != nullptr ? w_.output : w_.token_embd);  // the LM head
    arena_ = cpu::PackedArena(shapes);
    std::vector<quant::BlockQ8_0> q8;
    for (std::size_t i = 0; i < mats.size(); ++i) {
      if (convert[i]) {
        const gguf::TensorInfo& t = *mats[i];
        const auto rows = static_cast<std::int64_t>(t.ne[1]), k = static_cast<std::int64_t>(t.ne[0]);
        q8.assign(static_cast<std::size_t>(rows * (k / quant::kBlock)), quant::BlockQ8_0{});
        const std::span<const std::byte> bytes = file_.tensor_bytes(t);
        pool_.run([&](int w, int n) {
          std::int64_t r0 = 0, r1 = 0;
          ThreadPool::split_range(rows, n, w, r0, r1);
          std::vector<float> row(static_cast<std::size_t>(k));
          for (std::int64_t r = r0; r < r1; ++r) {
            dequant_row(t, bytes, r, row.data());
            quant::quantize_row_q8_0(row.data(), k, q8.data() + r * (k / quant::kBlock));
          }
        });
        arena_.pack(i, q8.data());
        release_from_working_set(bytes);
        continue;
      }
      arena_.pack(i, file_.tensor_bytes(*mats[i]).data());
      // The packed copy is what the kernels read; the file's pages of this
      // tensor are only needed again for a token's embedding row (tied
      // embeddings), which comes back as a soft fault. Releasing them keeps
      // the working set at one copy of the weights.
      release_from_working_set(file_.tensor_bytes(*mats[i]));
    }
    // Q4_K and Q6_K matrices run on the CPU only; `gpu` refuses
    // them in LlamaGpu.
    auto add_k = [&](const gguf::TensorInfo* t) {
      const auto type = static_cast<gguf::GgmlType>(t->type);
      if (type != gguf::GgmlType::kQ4_K && type != gguf::GgmlType::kQ6_K) return;
      if (matmul_ != Matmul::kQ8Activations || k_index_.contains(t)) return;
      k_index_.emplace(t, kmats_.size());
      const std::span<const std::byte> bytes = file_.tensor_bytes(*t);
      kmats_.emplace_back(type == gguf::GgmlType::kQ4_K ? cpu::KType::kQ4_K : cpu::KType::kQ6_K, bytes.data(),
                          static_cast<std::int64_t>(t->ne[1]), static_cast<std::int64_t>(t->ne[0]));
      release_from_working_set(bytes);
    };
    for (const auto& layer : w_.layers) {
      for (const gguf::TensorInfo* t : {layer.attn_q, layer.attn_k, layer.attn_v, layer.attn_output, layer.ffn_gate,
                                        layer.ffn_up, layer.ffn_down}) {
        add_k(t);
      }
    }
    add_k(w_.output != nullptr ? w_.output : w_.token_embd);
  }
  if (matmul_ == Matmul::kGpu) {
    // The GPU imports the whole arena: the CPU's packed weights, not a copy.
    const gpu::ContextOptions gpu_opts = gpu_options != nullptr ? *gpu_options : gpu::ContextOptions{};
    GpuPassInputs in;
    in.file = &file_;
    in.config = &cfg_;
    in.tensors = &w_;
    in.arena = &arena_;
    in.arena_index = [this](const gguf::TensorInfo* t) { return packed_index_.at(t); };
    in.attn_norm = &attn_norm_;
    in.ffn_norm = &ffn_norm_;
    in.output_norm = &output_norm_;
    in.rope_factors = &rope_factors_;
    in.q_bias = &q_bias_;  // Qwen2; empty vectors for Llama
    in.k_bias = &k_bias_;
    in.v_bias = &v_bias_;
    in.n_ctx = n_ctx_;
    gpu_pass_ = std::make_unique<LlamaGpu>(in, gpu_opts);
  }
}

LlamaRef::~LlamaRef() {
  // After a GPU failure the leaked GPU device still holds its import of the
  // weight pages, so they are kept until the process exits (08 doc section 4).
  if (gpu_abandoned_) static_cast<void>(new cpu::PackedArena(std::move(arena_)));  // NOLINT: deliberate leak
}

std::vector<float> LlamaRef::f32_vector(const gguf::TensorInfo* t) const {
  std::vector<float> v(t->ne[0]);
  dequant_row(*t, file_.tensor_bytes(*t), 0, v.data());
  return v;
}

void LlamaRef::matmul(const char* what, int layer, const gguf::TensorInfo* w, const float* x, float* y,
                      std::int64_t T) {
  const trace::Scope ts(what, layer, static_cast<std::int64_t>(w->ne[1]), T);
  if (quantized() && packed_index_.contains(w)) {  // Q8_0/Q4_0, or converted to Q8_0 at load
    matmul_q8(w, x, y, T);
    return;
  }
  if (quantized() && k_index_.contains(w)) {
    matmul_k(w, x, y, T);
    return;
  }
  const auto in = static_cast<std::int64_t>(w->ne[0]);
  const auto out = static_cast<std::int64_t>(w->ne[1]);
  const std::span<const std::byte> bytes = file_.tensor_bytes(*w);
  pool_.run([&](int i, int n) {
    std::int64_t r0 = 0, r1 = 0;
    ThreadPool::split_range(out, n, i, r0, r1);
    std::vector<float> row(static_cast<std::size_t>(in));
    for (std::int64_t r = r0; r < r1; ++r) {
      dequant_row(*w, bytes, r, row.data());
      for (std::int64_t t = 0; t < T; ++t) {
        const float* xt = x + t * in;
        float acc = 0.0f;
        for (std::int64_t k = 0; k < in; ++k) acc += row[static_cast<std::size_t>(k)] * xt[k];
        y[t * out + r] = acc;
      }
    }
  });
}

void LlamaRef::matmul_q8(const gguf::TensorInfo* w, const float* x, float* y, std::int64_t T) {
  const cpu::PackedView v = arena_.view(packed_index_.at(w));
  // Every activation row is quantized once, as ggml converts its src1 to Q8_0.
  acts_.resize(T, v.k);
  {
    const trace::Scope ts("quantize", -1, v.k, T);
    pool_.run([&](int i, int n) {
      std::int64_t t0 = 0, t1 = 0;
      ThreadPool::split_range(T, n, i, t0, t1);
      acts_.quantize(isa_, x, t0, t1);
    });
  }
  packed_product(packed_index_.at(w), y);
}

void LlamaRef::matmul_k(const gguf::TensorInfo* w, const float* x, float* y, std::int64_t T) {
  const cpu::KPackedView v = kmats_[k_index_.at(w)].view();
  // Every activation row is quantized once, as ggml converts its src1 to Q8_K.
  kacts_.resize(v.type, T, v.k);
  {
    const trace::Scope ts("quantize", -1, v.k, T);
    pool_.run([&](int i, int n) {
      std::int64_t t0 = 0, t1 = 0;
      ThreadPool::split_range(T, n, i, t0, t1);
      kacts_.quantize(isa_, x, t0, t1);
    });
  }
  // The same division of rows as packed_product: chunks handed out
  // dynamically for decode and short passes, a static split in whole 4-row
  // tiles for long ones. Each output is owned by one thread, and its bits do
  // not depend on the split.
  const Isa isa = isa_;
  if (T <= kDynamicTokens) {
    const std::int64_t parts = std::int64_t{pool_.size()} * 16;
    const std::int64_t chunk = std::max<std::int64_t>(8, ((v.rows + parts - 1) / parts + 7) / 8 * 8);
    alignas(64) std::atomic<std::int64_t> next{0};
    pool_.run([&](int, int) {
      for (std::int64_t r = next.fetch_add(chunk, std::memory_order_relaxed); r < v.rows;
           r = next.fetch_add(chunk, std::memory_order_relaxed)) {
        cpu::kgemm(isa, v, kacts_, y, v.rows, r, std::min(r + chunk, v.rows));
      }
    });
    return;
  }
  pool_.run([&](int i, int n) {
    std::int64_t r0 = 0, r1 = 0;
    cpu::tile_range(v.rows, n, i, r0, r1);
    cpu::kgemm(isa, v, kacts_, y, v.rows, r0, r1);
  });
}

void LlamaRef::packed_product(std::size_t index, float* y) {
  const cpu::PackedView v = arena_.view(index);
  // Each output is owned by one thread, and its bits do not depend on how
  // the rows are divided.
  const Isa isa = isa_;
  if (acts_.tokens() <= kDynamicTokens) {
    // Decode and short passes (speculative verification) are bound by
    // memory bandwidth, and fixed shares finish unevenly: the product waits
    // for the slowest thread, and threads that finish early go to sleep,
    // which the next operation pays for in wake-ups.
    // Instead each thread takes chunks of rows (multiples of 8 rows, so
    // the GEMM tiles stay whole; about 16 per thread) until none are left.
    const std::int64_t parts = std::int64_t{pool_.size()} * 16;
    const std::int64_t chunk = std::max<std::int64_t>(8, ((v.rows + parts - 1) / parts + 7) / 8 * 8);
    alignas(64) std::atomic<std::int64_t> next{0};
    pool_.run([&](int, int) {
      for (std::int64_t r = next.fetch_add(chunk, std::memory_order_relaxed); r < v.rows;
           r = next.fetch_add(chunk, std::memory_order_relaxed)) {
        cpu::qgemm(isa, v, acts_, y, v.rows, r, std::min(r + chunk, v.rows));
      }
    });
    return;
  }
  // Longer passes: a static split in whole 4-row tiles.
  pool_.run([&](int i, int n) {
    std::int64_t r0 = 0, r1 = 0;
    cpu::tile_range(v.rows, n, i, r0, r1);
    cpu::qgemm(isa, v, acts_, y, v.rows, r0, r1);
  });
}

void LlamaRef::forward_gpu(std::span<const std::int32_t> tokens, std::span<float> logits, bool all_logits,
                           TensorDump* dump) {
  for (const std::int32_t tok : tokens) {
    if (tok < 0 || static_cast<std::uint32_t>(tok) >= cfg_.n_vocab) {
      fail(ErrorKind::kInvalidArgument, "token id {} is outside the vocabulary (0..{})", tok, cfg_.n_vocab - 1);
    }
  }
  // Chunks shrink as the position grows, so that one layer's submission (and
  // with all logits the LM head's) stays within the TDR budget
  // (LlamaGpu::chunk_tokens). Results do not depend on the chunking.
  const std::size_t before = n_past_;
  for (std::size_t off = 0, n = 0; off < tokens.size(); off += n) {
    n = static_cast<std::size_t>(gpu_pass_->chunk_tokens(n_past_, static_cast<std::int64_t>(tokens.size() - off),
                                                         all_logits || dump != nullptr));
    float* out = all_logits ? logits.data() + off * cfg_.n_vocab : off + n == tokens.size() ? logits.data() : nullptr;
    const trace::Scope ts("forward", -1, 0, static_cast<std::int64_t>(n));  // as the CPU path's forward_chunk
    gpu_pass_->forward_chunk(tokens.subspan(off, n), n_past_, out, all_logits, dump);
    n_past_ += static_cast<std::uint32_t>(n);
    if (off + n < tokens.size()) throw_if_cancelled();  // as on the CPU path
  }
  // Only a complete call enters the history: a failed one is run again whole.
  history_.insert(history_.end(), tokens.begin(), tokens.end());
  HEARTH_CHECK(history_.size() == before + tokens.size());
}

void LlamaRef::fall_back_to_cpu(const Error& e) {
  // Never touched again and never freed, with or without fallback: after a
  // timeout the GPU may still be running (08 doc section 4), so the device,
  // its buffers and its import of the weights stay alive until the process
  // exits (see also ~LlamaRef). Every shader binds the weights readonly, so
  // they need no checksum.
  const std::uint64_t submissions = gpu_pass_->submissions();
  [[maybe_unused]] const LlamaGpu* leaked = gpu_pass_.release();  // deliberately never deleted
  gpu_abandoned_ = true;
  if (!gpu_fallback_) throw;  // called only from forward()'s handler: rethrows the original error
  log_warn(
      "gpu: the GPU failed ({}) after {} submissions; continuing on the CPU (--backend cpu), "
      "re-running the {} token(s) so far",
      e.what(), submissions, history_.size());
  gpu_lost_ = true;
  matmul_ = Matmul::kQ8Activations;
  k_cache_.assign(std::size_t{cfg_.n_layer} * k_layer_, 0.0f);
  v_cache_.assign(std::size_t{cfg_.n_layer} * n_ctx_ * cfg_.n_head_kv * cfg_.head_dim, 0.0f);
  // After a loss the GPU's KV cache is undefined: rebuild the CPU's from the
  // tokens of the completed calls. The failed call then runs from its start.
  const std::vector<std::int32_t> again = std::move(history_);
  history_.clear();
  n_past_ = 0;
  if (!again.empty()) {
    std::vector<float> last(cfg_.n_vocab);
    forward(again, last, false);
  }
}

void LlamaRef::truncate(std::uint32_t n) {
  HEARTH_CHECK(n <= n_past_);
  n_past_ = n;
  // kGpu keeps the tokens of every position, for a fallback to the CPU.
  if (history_.size() > n) history_.resize(n);
}

void LlamaRef::forward(std::span<const std::int32_t> tokens, std::span<float> logits, bool all_logits,
                       TensorDump* dump) {
  if (tokens.empty()) return;
  // A GPU that failed without fallback was abandoned (fall_back_to_cpu):
  // this engine never runs again, rather than continuing on a path it was
  // not built for (no CPU KV cache for kGpu).
  if (gpu_abandoned_ && !gpu_lost_) {
    fail(ErrorKind::kDeviceLost, "the GPU failed earlier in this run and GPU fallback is off: create a new engine");
  }
  if (n_past_ + tokens.size() > n_ctx_) {
    fail(ErrorKind::kInvalidArgument, "the context is full: {} positions used, {} more requested, context size {}",
         n_past_, tokens.size(), n_ctx_);
  }
  const std::size_t need = (all_logits ? tokens.size() : 1) * cfg_.n_vocab;
  HEARTH_CHECK(logits.size() >= need);
  // Chunks of at most 512 tokens bound the activation memory (about 60 MB for
  // the 1B model, plus T x n_vocab logits for a dump of result_output). Results do not
  // depend on the chunk size (every row and position is computed the same
  // way); larger chunks read each weight from memory fewer times.
  if (gpu_pass_) {
    const std::uint32_t gpu_start = n_past_;
    try {
      forward_gpu(tokens, logits, all_logits, dump);
      return;
    } catch (const Error& e) {
      n_past_ = gpu_start;  // as on the CPU path below: a failed call leaves no trace
      if (e.kind() != ErrorKind::kDeviceLost) throw;
      fall_back_to_cpu(e);  // rethrows unless fallback is allowed; then this call runs on the CPU below
    }
  }
  // A call that fails partway leaves the context as it was before the call:
  // positions past n_past_ are never read, so the K/V rows that the completed
  // chunks wrote are simply overwritten by the next call.
  const std::uint32_t start = n_past_;
  try {
    forward_cpu(tokens, logits, all_logits, dump);
  } catch (...) {
    n_past_ = start;
    throw;
  }
}

void LlamaRef::forward_cpu(std::span<const std::int32_t> tokens, std::span<float> logits, bool all_logits,
                           TensorDump* dump) {
  constexpr std::size_t kChunk = 512;
  for (std::size_t off = 0; off < tokens.size(); off += kChunk) {
    const std::size_t n = std::min(kChunk, tokens.size() - off);
    const bool last_chunk = off + n == tokens.size();
    float* out = nullptr;
    if (all_logits)
      out = logits.data() + off * cfg_.n_vocab;
    else if (last_chunk)
      out = logits.data();
    forward_chunk(tokens.subspan(off, n), out, all_logits, dump);
    if (!last_chunk) throw_if_cancelled();  // a long prompt stops at a chunk boundary; forward() rolls back
  }
}

void LlamaRef::forward_chunk(std::span<const std::int32_t> tokens, float* logits, bool all_logits, TensorDump* dump) {
  const LlamaConfig& c = cfg_;
  const auto T = static_cast<std::int64_t>(tokens.size());
  const std::int64_t E = c.n_embd, hd = c.head_dim, nh = c.n_head, nkv = c.n_head_kv;
  const std::int64_t Q = nh * hd, KV = nkv * hd, F = c.n_ff;
  const std::int64_t group = nh / nkv;
  const std::int64_t p0 = n_past_;
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  const trace::Scope forward_scope("forward", -1, 0, T);
  // Element-wise work split across the pool: each index is computed exactly
  // as in a plain loop, so the split never changes a result.
  auto parallel_for = [&](std::int64_t n, auto&& body) {
    pool_.run([&](int i, int parts) {
      std::int64_t a = 0, b = 0;
      ThreadPool::split_range(n, parts, i, a, b);
      for (std::int64_t j = a; j < b; ++j) body(j);
    });
  };
  auto record = [&](const char* base, int layer, const std::vector<float>& data, std::int64_t width) {
    if (dump == nullptr || !dump->wants(base, layer)) return;
    for (std::int64_t t = 0; t < T; ++t) {
      if (dump->wants_position(p0 + t)) dump->record(base, layer, p0 + t, data.data() + t * width, width);
    }
  };

  std::vector<float> x(static_cast<std::size_t>(T * E));
  for (std::int64_t t = 0; t < T; ++t) {
    const std::int32_t tok = tokens[static_cast<std::size_t>(t)];
    if (tok < 0 || static_cast<std::uint32_t>(tok) >= c.n_vocab) {
      fail(ErrorKind::kInvalidArgument, "token id {} is outside the vocabulary (0..{})", tok, c.n_vocab - 1);
    }
    dequant_row(*w_.token_embd, file_.tensor_bytes(*w_.token_embd), tok, x.data() + t * E);
  }
  std::vector<float> h(static_cast<std::size_t>(T * E)), q(static_cast<std::size_t>(T * Q)),
      k(static_cast<std::size_t>(T * KV)), v(static_cast<std::size_t>(T * KV)), att(static_cast<std::size_t>(T * Q)),
      proj(static_cast<std::size_t>(T * E)), g(static_cast<std::size_t>(T * F)), u(static_cast<std::size_t>(T * F));

  for (std::uint32_t l = 0; l < c.n_layer; ++l) {
    const LlamaTensors::Layer& L = w_.layers[l];
    const int il = static_cast<int>(l);
    {
      const trace::Scope ts("attn_norm", il, E, T);
      parallel_for(
          T, [&](std::int64_t t) { cpu::ref::rmsnorm(&x[t * E], attn_norm_[l].data(), &h[t * E], E, c.rms_eps); });
    }
    record("attn_norm", il, h, E);
    matmul("attn_q", il, L.attn_q, h.data(), q.data(), T);
    matmul("attn_k", il, L.attn_k, h.data(), k.data(), T);
    matmul("attn_v", il, L.attn_v, h.data(), v.data(), T);
    // Qwen2's biases, added to the products as ggml adds them (one FP32 add each).
    for (auto [out, bias, width] :
         {std::tuple{&q, &q_bias_[l], Q}, std::tuple{&k, &k_bias_[l], KV}, std::tuple{&v, &v_bias_[l], KV}}) {
      if (bias->empty()) continue;
      for (std::int64_t t = 0; t < T; ++t) {
        for (std::int64_t j = 0; j < width; ++j) (*out)[t * width + j] += (*bias)[j];
      }
    }
    const float* factors = rope_factors_.empty() ? nullptr : rope_factors_.data();
    {
      const trace::Scope ts("rope", il, Q + KV, T);
      parallel_for(T, [&](std::int64_t t) {
        cpu::ref::rope(&q[t * Q], nh, hd, c.rope_dim, p0 + t, c.rope_theta, factors, c.rope_neox);
        cpu::ref::rope(&k[t * KV], nkv, hd, c.rope_dim, p0 + t, c.rope_theta, factors, c.rope_neox);
      });
    }
    record("Qcur", il, q, Q);
    record("Kcur", il, k, KV);
    record("Vcur", il, v, KV);
    float* kc = k_cache_.data() + std::size_t{l} * k_layer_;
    float* vc = v_cache_.data() + std::size_t{l} * n_ctx_ * KV;
    if (quantized()) {
      // K: [kv_head][block of 16 positions][dimension][position in block], for cpu::attend.
      const std::int64_t head = cpu::k_head_size(n_ctx_, hd);
      for (std::int64_t t = 0; t < T; ++t) {
        for (std::int64_t j = 0; j < KV; ++j) kc[j / hd * head + cpu::k_index(p0 + t, j % hd, hd)] = k[t * KV + j];
      }
      // V: [kv_head][position][dimension], so one head's values are one
      // contiguous stream. ([position][kv_head * head_dim] put each position
      // of a head on its own 4 KB page.)
      for (std::int64_t t = 0; t < T; ++t) {
        for (std::int64_t kvh = 0; kvh < nkv; ++kvh) {
          const float* src = v.data() + t * KV + kvh * hd;
          std::copy(src, src + hd, vc + (kvh * n_ctx_ + p0 + t) * hd);
        }
      }
    } else {
      std::copy(k.begin(), k.end(), kc + p0 * KV);
      std::copy(v.begin(), v.end(), vc + p0 * KV);
    }
    // Causal attention: token t attends to positions 0 .. p0 + t. One output
    // (token, head) per call, so results do not depend on the thread count.
    // A task is one token and one KV head: its `group` query heads run one
    // after the other on one thread, so the second and third read that KV
    // head's cache from the core's L2 instead of from memory (dealt per query
    // head, the heads of a group ran on different cores and each read the
    // cache: about 3x the traffic of a long context). Tasks
    // are dealt out round-robin: later tokens attend to more positions, and
    // contiguous ranges would leave the last thread with most of the work.
    {
      const trace::Scope ts("attention", il, p0 + T, T);
      pool_.run([&](int i, int n) {
        std::vector<float> scores(static_cast<std::size_t>(p0 + T));
        for (std::int64_t task = i; task < T * nkv; task += n) {
          const std::int64_t t = task / nkv, kvh = task % nkv;
          const std::int64_t len = p0 + t + 1;
          for (std::int64_t head = kvh * group; head < (kvh + 1) * group; ++head) {
            const float* qh = &q[t * Q + head * hd];
            if (quantized()) {
              cpu::attend(isa_, {qh, kc + kvh * cpu::k_head_size(n_ctx_, hd), vc + kvh * n_ctx_ * hd, hd, len, hd,
                                 scale, scores.data(), &att[t * Q + head * hd]});
              continue;
            }
            for (std::int64_t s = 0; s < len; ++s) {
              const float* ks = kc + s * KV + kvh * hd;
              float dot = 0.0f;
              for (std::int64_t d = 0; d < hd; ++d) dot += qh[d] * ks[d];
              scores[static_cast<std::size_t>(s)] = dot * scale;
            }
            cpu::ref::softmax(scores.data(), len);
            float* o = &att[t * Q + head * hd];
            std::fill(o, o + hd, 0.0f);
            for (std::int64_t s = 0; s < len; ++s) {
              const float p = scores[static_cast<std::size_t>(s)];
              const float* vs = vc + s * KV + kvh * hd;
              for (std::int64_t d = 0; d < hd; ++d) o[d] += p * vs[d];
            }
          }
        }
      });
    }
    matmul("attn_output", il, L.attn_output, att.data(), proj.data(), T);
    record("attn_out", il, proj, E);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] += proj[i];
    record("ffn_inp", il, x, E);
    {
      const trace::Scope ts("ffn_norm", il, E, T);
      parallel_for(T,
                   [&](std::int64_t t) { cpu::ref::rmsnorm(&x[t * E], ffn_norm_[l].data(), &h[t * E], E, c.rms_eps); });
    }
    record("ffn_norm", il, h, E);
    matmul("ffn_gate", il, L.ffn_gate, h.data(), g.data(), T);
    matmul("ffn_up", il, L.ffn_up, h.data(), u.data(), T);
    {
      const trace::Scope ts("silu_mul", il, F, T);
      parallel_for(T * F, [&](std::int64_t i) {
        g[static_cast<std::size_t>(i)] =
            cpu::ref::silu(g[static_cast<std::size_t>(i)]) * u[static_cast<std::size_t>(i)];
      });
    }
    matmul("ffn_down", il, L.ffn_down, g.data(), proj.data(), T);
    record("ffn_out", il, proj, E);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] += proj[i];
    record("l_out", il, x, E);
  }
  for (std::int64_t t = 0; t < T; ++t) cpu::ref::rmsnorm(&x[t * E], output_norm_.data(), &h[t * E], E, c.rms_eps);
  record("result_norm", -1, h, E);
  n_past_ += static_cast<std::uint32_t>(T);

  const gguf::TensorInfo* head = w_.output != nullptr ? w_.output : w_.token_embd;
  const bool want_dump = dump != nullptr && dump->wants("result_output", -1);
  if (logits == nullptr && !want_dump) return;
  if (all_logits && !want_dump) {
    matmul("lm_head", -1, head, h.data(), logits, T);
  } else if (all_logits || want_dump) {
    std::vector<float> all(static_cast<std::size_t>(T) * c.n_vocab);
    matmul("lm_head", -1, head, h.data(), all.data(), T);
    record("result_output", -1, all, c.n_vocab);
    if (logits != nullptr) {
      if (all_logits)
        std::copy(all.begin(), all.end(), logits);
      else
        std::copy(all.end() - c.n_vocab, all.end(), logits);
    }
  } else {
    matmul("lm_head", -1, head, h.data() + (T - 1) * E, logits, 1);
  }
}

}  // namespace hearth::model
