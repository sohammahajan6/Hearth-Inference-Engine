#include "session.h"

#include <charconv>
#include <format>
#include <random>

#include "core/error.h"
#include "core/platform.h"

namespace hearth::cli {

ModelOptions model_options(Args& args, std::uint32_t default_ctx, const char* default_backend) {
  ModelOptions o;
  const auto model = args.value("model");
  if (!model) fail(ErrorKind::kInvalidArgument, "--model F is required");
  o.model = *model;
  const std::string backend = args.value_or("backend", default_backend);
  const std::string isa = args.value_or("isa", "auto");
  o.threads = static_cast<int>(args.int_or("threads", 4, 1, 64));
  o.ctx = static_cast<std::uint32_t>(args.int_or("ctx", default_ctx, 1, 1 << 20));
  if (backend == "cpu") {
    o.matmul = model::Matmul::kQ8Activations;
  } else if (backend == "gpu") {
    // The whole forward pass on the GPU (08 doc step 6.4).
    o.matmul = model::Matmul::kGpu;
  } else if (backend != "cpu-ref") {
    fail(ErrorKind::kInvalidArgument, "--backend must be cpu, gpu or cpu-ref (got '{}')", backend);
  }
  // The kernels of the cpu backend. Both compute the canonical order of ADR 0009, so they give the
  // same results; scalar exists to check that, and for CPUs without AVX-512.
  o.isa = isa_supported(Isa::kAvx512) ? Isa::kAvx512 : Isa::kScalar;
  if (isa != "auto" && (!parse_isa(isa, o.isa) || o.isa == Isa::kAvx2)) {
    fail(ErrorKind::kInvalidArgument, "--isa must be auto, scalar or avx512 (got '{}')", isa);
  }
  o.backend = backend == "cpu"   ? std::format("cpu ({} kernels)", to_string(o.isa))
              : backend == "gpu" ? std::string("gpu")
                                 : std::string("cpu-ref");
  return o;
}

Session::Session(const ModelOptions& o)
    : engine(detail::EngineAccess::load(o)),
      file(detail::EngineAccess::file(engine)),
      tok(detail::EngineAccess::tokenizer(engine)),
      pool(detail::EngineAccess::pool(engine)),
      llm(detail::EngineAccess::llama(engine)) {}

void draft_options(Args& args, ModelOptions& o) {
  if (const auto draft = args.value("draft")) o.draft_model = *draft;
}

sampling::Params sampling_options(Args& args, std::string& seed_note) {
  sampling::Params p;
  p.temperature = static_cast<float>(args.double_or("temp", 0.8, 0.0, 100.0));
  p.top_k = static_cast<int>(args.int_or("top-k", 40, 0, 1 << 20));
  p.top_p = static_cast<float>(args.double_or("top-p", 0.95, 1e-6, 1.0));
  if (const auto seed = args.value("seed")) {
    const auto [end, ec] = std::from_chars(seed->data(), seed->data() + seed->size(), p.seed);
    if (seed->empty() || ec != std::errc() || end != seed->data() + seed->size()) {
      fail(ErrorKind::kInvalidArgument, "--seed must be an integer in 0..2^64-1 (got '{}')", *seed);
    }
    seed_note = std::format("seed {}", p.seed);
  } else {
    p.seed = (std::uint64_t{std::random_device{}()} << 32) | std::random_device{}();
    seed_note = std::format("seed {} (random; pass --seed {} to repeat)", p.seed, p.seed);
  }
  if (p.temperature <= 0.0f) seed_note = "greedy";
  return p;
}

}  // namespace hearth::cli
