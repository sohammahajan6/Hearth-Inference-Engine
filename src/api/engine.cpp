#include "hearth/engine.h"

#include <format>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "api/engine_internal.h"
#include "core/error.h"
#include "core/log.h"
#include "core/platform.h"
#include "gpu/adapters.h"
#include "gpu/context.h"
#include "model/generate.h"
#include "model/llama_gpu.h"
#include "model/speculative.h"
#include "sampling/sampler.h"
#include "tokenizer/chat.h"

namespace hearth {
namespace {

// gpu with fallback but without a Vulkan GPU runs the cpu backend (08 doc
// section 5); the user is told.
model::Matmul effective_matmul(const api::LoadOptions& o) {
  if (!o.gpu_fallback || gpu::vulkan_available()) return o.matmul;
  if (o.matmul == model::Matmul::kGpu) {
    log_warn("gpu: no Vulkan GPU is available; running --backend cpu instead");
    return model::Matmul::kQ8Activations;
  }
  return o.matmul;
}

// With fallback, a Vulkan loader that has no usable GPU behind it (no
// device, a missing feature, device creation failing) also runs the cpu
// backend, not only a missing loader.
model::LlamaRef make_llm(const gguf::File& file, const api::LoadOptions& o, ThreadPool& pool) {
  const model::Matmul m = effective_matmul(o);
  const bool gpu_backend = m == model::Matmul::kGpu;
  try {
    return model::LlamaRef(file, o.ctx, pool, m, o.isa);
  } catch (const Error& e) {
    const bool gpu_error =
        e.kind() == ErrorKind::kGpu || e.kind() == ErrorKind::kUnsupported || e.kind() == ErrorKind::kDeviceLost;
    if (!o.gpu_fallback || !gpu_backend || !gpu_error) throw;
    log_warn("gpu: the GPU cannot be used ({}); running on the CPU only (--backend cpu)", e.what());
  }
  return model::LlamaRef(file, o.ctx, pool, model::Matmul::kQ8Activations, o.isa);
}

// engine.h promises that every failure is a hearth::Error.
template <class F>
decltype(auto) translate_oom(F&& f) {
  try {
    return f();
  } catch (const std::bad_alloc&) {
    fail(ErrorKind::kOutOfMemory, "out of memory (a smaller context or a shorter prompt needs less)");
  }
}

StopReason to_public(model::Stop s) {
  switch (s) {
    case model::Stop::kEndOfGeneration:
      return StopReason::kEndOfGeneration;
    case model::Stop::kLimit:
      return StopReason::kLimit;
    case model::Stop::kContextFull:
      return StopReason::kContextFull;
    case model::Stop::kCancelled:
      return StopReason::kCancelled;
  }
  return StopReason::kLimit;
}

api::LoadOptions from_public(const Engine::Options& p) {
  api::LoadOptions o;
  o.model = path_to_utf8(p.model);
  if (o.model.empty()) fail(ErrorKind::kInvalidArgument, "Engine::Options::model is empty");
  if (p.threads < 1 || p.threads > 64) fail(ErrorKind::kInvalidArgument, "threads must be 1..64 (got {})", p.threads);
  if (p.context == 0) fail(ErrorKind::kInvalidArgument, "context must be at least 1");
  o.threads = p.threads;
  o.ctx = p.context;
  switch (p.kernels) {
    case Kernels::kAuto:
      o.isa = isa_supported(Isa::kAvx512) ? Isa::kAvx512 : Isa::kScalar;
      break;
    case Kernels::kScalar:
      o.isa = Isa::kScalar;
      break;
    case Kernels::kAvx512:
      o.isa = Isa::kAvx512;
      break;
  }
  // An enum value from another version of this header (the removed kHetero
  // was 3) must not silently run another backend.
  if (p.backend != Backend::kCpuRef && p.backend != Backend::kCpu && p.backend != Backend::kGpu) {
    fail(ErrorKind::kInvalidArgument, "unknown backend {} (kCpuRef, kCpu or kGpu)", static_cast<int>(p.backend));
  }
  switch (p.backend) {
    case Backend::kCpuRef:
      o.matmul = model::Matmul::kFp32;
      break;
    case Backend::kCpu:
      o.matmul = model::Matmul::kQ8Activations;
      break;
    case Backend::kGpu:
      o.matmul = model::Matmul::kGpu;
      break;
  }
  o.gpu_fallback = p.gpu_fallback;
  o.draft_model = path_to_utf8(p.draft_model);
  o.speculative.draft_max = p.draft_max;
  o.speculative.draft_p_min = p.draft_p_min;
  return o;
}

// The draft model of speculative decoding, on the cpu backend and the main
// model's threads.
struct DraftModel {
  gguf::File file;
  model::LlamaRef llm;
  DraftModel(const api::LoadOptions& o, ThreadPool& pool)
      : file(gguf::File::open(path_from_utf8(o.draft_model))),
        llm(file, o.ctx, pool, model::Matmul::kQ8Activations, o.isa) {}
};

// Checked before any file is opened; returns the options.
api::LoadOptions check_speculative(api::LoadOptions o) {
  if (o.draft_model.empty()) return o;
  if (o.speculative.draft_max < 1 || o.speculative.draft_max > 16) {
    fail(ErrorKind::kInvalidArgument, "the draft length must be 1..16 (got {})", o.speculative.draft_max);
  }
  if (!(o.speculative.draft_p_min <= 1.0f)) {  // negative: automatic
    fail(ErrorKind::kInvalidArgument, "the draft's minimum probability must be at most 1 (got {})",
         o.speculative.draft_p_min);
  }
  if (o.matmul != model::Matmul::kQ8Activations && o.matmul != model::Matmul::kFp32) {
    fail(ErrorKind::kInvalidArgument, "a draft model needs the cpu or cpu-ref backend");
  }
  return o;
}

// The draft must spell every token as the main model does, or its proposals
// would be other words with the same IDs.
void check_same_vocabulary(const tokenizer::Tokenizer& main, const tokenizer::Tokenizer& draft,
                           const std::string& draft_path) {
  bool same = main.n_vocab() == draft.n_vocab() && main.bos() == draft.bos() && main.eos() == draft.eos();
  for (std::int32_t id = 0; same && id < main.n_vocab(); ++id) {
    same = main.text(id) == draft.text(id) && main.is_eog(id) == draft.is_eog(id);
  }
  if (!same) {
    fail(ErrorKind::kInvalidArgument, "the draft model {} does not use the main model's vocabulary", draft_path);
  }
}

}  // namespace

struct Engine::Impl {
  api::LoadOptions o;
  gguf::File file;
  tokenizer::Tokenizer tok;
  ThreadPool pool;
  model::LlamaRef llm;
  std::unique_ptr<DraftModel> draft;                    // speculative decoding, else nullptr
  model::SpeculativeState draft_costs;                  // costs, guard and the draft's lag, across generate() calls
  mutable std::optional<tokenizer::ChatTemplate> chat;  // detected on first use
  std::optional<sampling::Sampler> sampler;             // kept across generate() calls with equal options
  GenerateOptions sampler_options;

  GenerateResult generate(std::span<const std::int32_t> prompt, const GenerateOptions& options,
                          const TokenCallback& on_token);

  explicit Impl(api::LoadOptions opts)
      : o(check_speculative(std::move(opts))),
        file(gguf::File::open(path_from_utf8(o.model))),
        tok(tokenizer::Tokenizer::from_gguf(file)),
        pool(o.threads),
        llm(make_llm(file, o, pool)) {
    if (!o.draft_model.empty()) {
      draft = std::make_unique<DraftModel>(o, pool);
      check_same_vocabulary(tok, tokenizer::Tokenizer::from_gguf(draft->file), o.draft_model);
    }
  }

  ~Impl() {
    // Proof of execution (08 doc section 4): how many GPU submissions ran.
    if (const model::LlamaGpu* g = llm.gpu_pass()) {
      log_info(
          "gpu: {} GPU submissions, {} dispatches; longest submission: layers {:.1f} ms, LM head {:.1f} ms; "
          "cost-model scale: layers {:.2f}, LM head {:.2f}",
          g->submissions(), g->dispatches(), g->max_layers_ms(), g->max_head_ms(), g->layers_scale(), g->head_scale());
    }
  }
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  const tokenizer::ChatTemplate& chat_template() const {
    if (!chat) chat = tokenizer::ChatTemplate::detect(file, tok);
    return *chat;
  }
};

Engine::Engine(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {
  Impl& m = *impl_;
  model::LlamaRef& llm = m.llm;
  const api::LoadOptions& o = m.o;
  if (llm.matmul() == model::Matmul::kGpu) llm.set_gpu_fallback(o.gpu_fallback);
  if (const model::LlamaGpu* g = llm.gpu_pass()) {
    log_info("gpu: forward pass on {} (driver {}, LUID {})", g->device().name, g->device().driver_version_text,
             gpu::luid_pdh_string(g->device().luid));
  }
}

Engine::Engine(const Options& options)
    : Engine(translate_oom([&] { return std::make_unique<Impl>(from_public(options)); })) {}
Engine::~Engine() = default;
Engine::Engine(Engine&&) noexcept = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

std::vector<std::int32_t> Engine::tokenize(std::string_view text, bool add_bos) const {
  return translate_oom([&] { return impl_->tok.encode(text, add_bos); });
}

std::vector<std::int32_t> Engine::chat_prompt(std::string_view system, std::span<const ChatMessage> messages,
                                              std::string_view date) const {
  std::vector<tokenizer::Message> m;
  m.reserve(messages.size());
  for (const ChatMessage& x : messages) m.push_back({x.role, x.content});
  const std::string d = date.empty() ? tokenizer::template_date_today() : tokenizer::template_date(date);
  return impl_->chat_template().prompt(impl_->tok, system, m, d);
}

std::vector<std::int32_t> Engine::chat_next_turn(std::string_view user) const {
  return impl_->chat_template().next_turn(impl_->tok, user);
}

GenerateResult Engine::generate(std::span<const std::int32_t> prompt, const GenerateOptions& options,
                                const TokenCallback& on_token) {
  return translate_oom([&] { return impl_->generate(prompt, options, on_token); });
}

GenerateResult Engine::Impl::generate(std::span<const std::int32_t> prompt, const GenerateOptions& options,
                                      const TokenCallback& on_token) {
  if (options.max_new_tokens < 0) fail(ErrorKind::kInvalidArgument, "max_new_tokens must not be negative");
  const GenerateOptions& last = sampler_options;
  if (!sampler || last.temperature != options.temperature || last.top_k != options.top_k ||
      last.top_p != options.top_p || last.seed != options.seed) {
    sampler.emplace(sampling::Params{options.temperature, options.top_k, options.top_p, options.seed});
    sampler_options = options;
  }
  sampling::Sampler& active = *sampler;
  tokenizer::Utf8Stream stream;
  std::string bytes;
  const auto emit = [&](std::int32_t id) {
    bytes.clear();
    tok.decode(id, bytes, options.special_text);
    const std::string text = stream.push(bytes);
    if (on_token) on_token(id, text);
  };
  const model::GenerateResult r =
      draft ? model::generate_speculative(llm, draft->llm, tok, active, prompt, options.max_new_tokens, o.speculative,
                                          emit, &draft_costs)
            : model::generate(llm, tok, active, prompt, options.max_new_tokens, emit);
  GenerateResult out;
  out.prompt_tokens = r.prompt_tokens;
  out.prefill_seconds = r.prefill_seconds;
  out.generated_tokens = r.generated;
  out.decode_seconds = r.decode_seconds;
  out.stop = to_public(r.stop);
  out.end_token = r.eog_token;
  out.trailing_text = stream.flush();
  out.draft_tokens = r.drafted;
  out.draft_accepted = r.accepted;
  return out;
}

// A fresh start: the same seed then gives the same text again.
void Engine::reset() {
  impl_->llm.reset();
  if (impl_->draft) impl_->draft->llm.reset();
  impl_->draft_costs.forget_context();
  impl_->sampler.reset();
}
std::uint32_t Engine::context_used() const { return impl_->llm.n_past(); }
std::uint32_t Engine::context_size() const { return impl_->llm.n_ctx(); }

std::string Engine::backend_description() const {
  const api::LoadOptions& o = impl_->o;
  switch (impl_->llm.matmul()) {
    case model::Matmul::kFp32:
      return "cpu-ref";
    case model::Matmul::kQ8Activations:
      return std::format("cpu ({} kernels){}", to_string(o.isa),
                         o.matmul != model::Matmul::kQ8Activations ? " (GPU unavailable or lost)" : "");
    case model::Matmul::kGpu:
      return "gpu";
  }
  return "?";
}

std::string Engine::gpu_device() const {
  if (const model::LlamaGpu* g = impl_->llm.gpu_pass()) return g->device().name;
  return {};
}

bool Engine::gpu_lost() const { return impl_->llm.gpu_lost(); }

std::string Engine::draft_description() const {
  const Impl& m = *impl_;
  if (!m.draft) return {};
  const float p = m.o.speculative.draft_p_min;
  const int read = m.o.speculative.draft_read_max;
  return std::format(
      "draft {} (cpu, {} kernels), up to {} tokens, {}, {}, {}",
      path_to_utf8(path_from_utf8(m.o.draft_model).filename()), to_string(m.o.isa), m.o.speculative.draft_max,
      p < 0    ? std::string("length by measured cost")
      : p == 0 ? std::string("fixed length")
               : std::format("stops below probability {}", p),
      read > 0 ? std::format("reads at most {} new tokens at once", read) : std::string("reads everything"),
      m.o.speculative.lookup_ngram > 0 ? std::format("prompt lookup of {} tokens", m.o.speculative.lookup_ngram)
                                       : std::string("no prompt lookup"));
}

namespace detail {

Engine EngineAccess::load(const api::LoadOptions& o) { return Engine(std::make_unique<Engine::Impl>(o)); }
const api::LoadOptions& EngineAccess::options(const Engine& e) { return e.impl_->o; }
gguf::File& EngineAccess::file(Engine& e) { return e.impl_->file; }
tokenizer::Tokenizer& EngineAccess::tokenizer(Engine& e) { return e.impl_->tok; }
ThreadPool& EngineAccess::pool(Engine& e) { return e.impl_->pool; }
model::LlamaRef& EngineAccess::llama(Engine& e) { return e.impl_->llm; }
model::LlamaRef* EngineAccess::draft(Engine& e) { return e.impl_->draft ? &e.impl_->draft->llm : nullptr; }

}  // namespace detail
}  // namespace hearth
