// Step 6.1 on the real models: the weight arena of a LlamaRef is imported into
// the GPU without a copy (gpu::WeightArena), the GPU reads exactly the bytes
// the CPU wrote (the XOR canary over the whole arena), and the canonical GEMV
// shaders reading matrices straight from the imported arena give the CPU's
// canonical_dot bits. The 3B Q8_0 file is the largest import (one 3.4 GB slice).
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <windows.h>

#include <psapi.h>

#include "core/pdh.h"
#include "core/platform.h"
#include "core/quant.h"
#include "core/thread_pool.h"
#include "cpu/qgemm.h"
#include "gguf/gguf.h"
#include "gpu/buffer.h"
#include "gpu/context.h"
#include "gpu/pipeline.h"
#include "gpu/weight_arena.h"
#include "model/llama_ref.h"
#include "test_util.h"

using namespace hearth;

namespace {

struct GemvPush {
  std::uint32_t rows, groups, row_begin;
};
constexpr std::uint32_t kSubgroups = 8;

// Rows of matrix `name` computed on the GPU from the imported arena, against
// canonical_dot on the CPU's view of the same bytes. Returns the rows compared.
std::int64_t check_matrix(gpu::Context& ctx, const gpu::WeightArena& ga, const model::LlamaRef& llama,
                          const gguf::File& f, const std::string& name) {
  const std::size_t idx = llama.packed_index(f.find_tensor(name));
  const cpu::PackedView v = llama.packed_arena().view(idx);
  const bool q8 = v.type == cpu::QType::kQ8_0;
  gpu::Pipeline pipe(ctx, {.shader = q8 ? "qgemv_q8_0" : "qgemv_q4_0",
                           .buffers = q8 ? 5u : 6u,
                           .push_constant_bytes = sizeof(GemvPush),
                           .spec_constants = {kSubgroups, 8 * kSubgroups},
                           .required_subgroup_size = 8,
                           .require_full_subgroups = true});
  gpu::DescriptorPool pool(ctx, 1, 6);
  std::mt19937 rng(static_cast<unsigned>(idx));
  std::normal_distribution<float> n(0.0f, 1.0f);
  std::vector<float> x(static_cast<std::size_t>(v.k));
  for (auto& e : x) e = n(rng);
  cpu::PackedActs a;
  a.resize(1, v.k);
  a.quantize(Isa::kScalar, x.data(), 0, 1);
  const auto groups = static_cast<std::size_t>(v.groups);
  auto upload = [&](const void* p, std::size_t bytes) {
    gpu::Buffer b(ctx, bytes, gpu::MemoryKind::kHostCoherent);
    std::memcpy(b.mapped(), p, bytes);
    return b;
  };
  gpu::Buffer xq = upload(a.q(0, 0), groups * 512), xd = upload(a.d(0, 0), groups * 64),
              xs = upload(a.s(0, 0), groups * 64);
  const auto rows = static_cast<std::size_t>(v.rows);
  gpu::Buffer y(ctx, rows * 4, gpu::MemoryKind::kHostCached);
  std::fill_n(static_cast<float*>(y.mapped()), rows, std::numeric_limits<float>::quiet_NaN());
  std::vector<gpu::DescriptorPool::Range> r = {
      ga.quants(idx), ga.scales(idx), {&xq, 0, xq.size()}, {&xd, 0, xd.size()}};
  if (!q8) r.push_back({&xs, 0, xs.size()});
  r.push_back({&y, 0, y.size()});
  const VkDescriptorSet set = pool.allocate(pipe, r);
  const GemvPush push{static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(groups), 0};
  ctx.run_once([&](VkCommandBuffer cmd) {
    pipe.dispatch(cmd, set, &push, sizeof(push), static_cast<std::uint32_t>((rows + kSubgroups - 1) / kSubgroups));
    gpu::compute_to_host_barrier(cmd);
  });
  std::vector<float> want(rows);
  cpu::qgemm(Isa::kScalar, v, a, want.data(), static_cast<std::int64_t>(rows), 0, static_cast<std::int64_t>(rows));
  const auto* got = static_cast<const float*>(y.mapped());
  for (std::size_t i = 0; i < rows; ++i) {
    if (std::memcmp(&got[i], &want[i], 4) != 0) {
      FAIL(std::format("{} row {}: gpu {:a}, cpu {:a}", name, i, got[i], want[i]));
    }
  }
  return static_cast<std::int64_t>(rows);
}

// Process commit (PrivateUsage) and this process's GPU memory counters, in MiB,
// after the 1.5 s the GPU counters need to update.
struct MemSnap {
  double commit = 0, shared = 0, dedicated = 0;
};
MemSnap mem_snapshot(PdhQuery& pdh, int shared, int dedicated) {
  precise_sleep_us(1'500'000);
  pdh.collect();
  PROCESS_MEMORY_COUNTERS_EX pmc{};
  pmc.cb = sizeof(pmc);
  GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc));
  const std::string filter = std::format("pid_{}_", GetCurrentProcessId());
  constexpr double mib = 1024.0 * 1024.0;
  return {static_cast<double>(pmc.PrivateUsage) / mib, pdh.sum(shared, {filter}) / mib,
          pdh.sum(dedicated, {filter}) / mib};
}

}  // namespace

TEST_SUITE("model") {
  // ADR 0015: the zero-copy gate in the terms Windows counts. A = the arena's
  // bytes. Packing commits A; the import adds no commit and makes the GPU's
  // shared usage A (the same pages); a deliberate copy into Vulkan type 2
  // adds A to both. So commit is about 1 x A with the import and 2 x A with
  // the copy; adding commit and shared usage would count the same pages twice.
  TEST_CASE("gpu zero-copy: the import adds no memory, a deliberate copy adds one copy") {
    if (!gpu::vulkan_available()) return;
    for (const char* file : {"Llama-3.2-1B-Instruct-Q8_0-pure.gguf", "Llama-3.2-1B-Instruct-Q4_0-pure.gguf"}) {
      CAPTURE(file);
      const auto path = test::repo_dir() / "models" / file;
      if (!std::filesystem::exists(path)) {
        const std::string note = std::string("missing (run scripts\\fetch-models.ps1): ") + file;
        MESSAGE(note);
        continue;
      }
      PdhQuery pdh;
      const int shared = pdh.add(L"\\GPU Process Memory(*)\\Shared Usage");
      const int dedicated = pdh.add(L"\\GPU Process Memory(*)\\Dedicated Usage");
      REQUIRE(shared >= 0);
      gpu::Context ctx(test::gpu_test_options());
      const gguf::File f = gguf::File::open(path);
      ThreadPool pool(4);
      const MemSnap s0 = mem_snapshot(pdh, shared, dedicated);
      const model::LlamaRef llama(f, 64, pool, model::Matmul::kQ8Activations);
      const MemSnap s1 = mem_snapshot(pdh, shared, dedicated);
      const gpu::WeightArena ga(ctx, llama.packed_arena());
      const MemSnap s2 = mem_snapshot(pdh, shared, dedicated);
      gpu::Buffer copy(ctx, ga.bytes(), gpu::MemoryKind::kHostCached);  // the deliberate copy mode
      for (std::size_t sl = 0, off = 0; sl < llama.packed_arena().slices(); ++sl) {
        std::memcpy(static_cast<std::byte*>(copy.mapped()) + off, llama.packed_arena().slice(sl).data(),
                    llama.packed_arena().slice(sl).size());
        off += llama.packed_arena().slice(sl).size();
      }
      const MemSnap s3 = mem_snapshot(pdh, shared, dedicated);
      const double a = static_cast<double>(ga.bytes()) / (1024.0 * 1024.0);
      const std::string summary = std::format(
          "{}: A = {:.0f} MiB. commit +{:.0f} (pack), {:+.0f} (import), {:+.0f} (copy); GPU shared {:+.0f} (pack), "
          "{:+.0f} (import), {:+.0f} (copy); dedicated {:+.0f} overall. Copies by commit: import {:.3f}, copy {:.3f}",
          file, a, s1.commit - s0.commit, s2.commit - s1.commit, s3.commit - s2.commit, s1.shared - s0.shared,
          s2.shared - s1.shared, s3.shared - s2.shared, s3.dedicated - s0.dedicated, (s2.commit - s0.commit) / a,
          (s3.commit - s0.commit) / a);
      MESSAGE(summary);
      // Under ASan, process commit also holds the sanitizer's shadow memory
      // (an eighth of every range touched: the copy step adds 1.125 x A), so
      // the commit checks run in the other builds, among them the Release
      // model test tier. The GPU's shared
      // usage has no shadow and is checked in every build.
      if (test::kAddressSanitizer) {
        MESSAGE("ASan build: process commit includes the sanitizer's shadow memory; commit checks not applied here");
      } else {
        // Packing commits the arena once (plus a few MiB of KV cache and buffers).
        CHECK((s1.commit - s0.commit) / a >= 0.98);
        CHECK((s1.commit - s0.commit) / a <= 1.05);
        // The import adds no commit.
        CHECK(std::abs(s2.commit - s1.commit) / a <= 0.02);
        // The deliberate copy adds one more copy.
        CHECK((s3.commit - s2.commit) / a >= 0.95);
        CHECK((s3.commit - s2.commit) / a <= 1.05);
      }
      // The import makes the same pages visible to the GPU; the copy adds a second set.
      CHECK((s2.shared - s1.shared) / a >= 0.95);
      CHECK((s2.shared - s1.shared) / a <= 1.05);
      CHECK((s3.shared - s2.shared) / a >= 0.95);
      CHECK((s3.shared - s2.shared) / a <= 1.05);
    }
  }

  TEST_CASE("gpu weight arena: imported without a copy, canary equal, GEMV from the arena equals the CPU") {
    if (!gpu::vulkan_available()) return;
    for (const char* file : {"Llama-3.2-1B-Instruct-Q4_0-pure.gguf", "Llama-3.2-1B-Instruct-Q8_0-pure.gguf",
                             "Llama-3.2-3B-Instruct-Q8_0-pure.gguf"}) {
      CAPTURE(file);
      const auto path = test::repo_dir() / "models" / file;
      if (!std::filesystem::exists(path)) {
        const std::string note = std::string("missing (run scripts\\fetch-models.ps1): ") + file;
        MESSAGE(note);
        continue;
      }
      const gguf::File f = gguf::File::open(path);
      ThreadPool pool(4);
      const model::LlamaRef llama(f, 64, pool, model::Matmul::kQ8Activations);
      gpu::Context ctx(test::gpu_test_options());
      const gpu::HeapBudget before = gpu::heap_budget(ctx);
      const gpu::WeightArena ga(ctx, llama.packed_arena());
      const gpu::HeapBudget after = gpu::heap_budget(ctx);
      const gpu::WeightArena::Canary c = ga.canary(ctx);
      CHECK(c.bytes == llama.packed_arena().bytes());
      CHECK(c.gpu_xor == c.cpu_xor);
      std::int64_t rows = 0;
      for (const char* name : {"blk.0.attn_q.weight", "blk.0.ffn_down.weight", "output.weight", "token_embd.weight"}) {
        const gguf::TensorInfo* t = f.find_tensor(name);
        if (t == nullptr) continue;
        if (name == std::string("token_embd.weight") && f.find_tensor("output.weight") != nullptr) continue;
        rows += check_matrix(ctx, ga, llama, f, name);
      }
      const std::string summary = std::format(
          "{}: {} slice(s), {} MiB imported; canary {:016x} over {} MiB in {:.0f} ms; heap usage {} -> {} MiB "
          "(budget {} MiB); {} GEMV rows from the arena bit-identical",
          file, llama.packed_arena().slices(), ga.bytes() >> 20, c.gpu_xor, c.bytes >> 20, c.gpu_ms, before.usage >> 20,
          after.usage >> 20, after.budget >> 20, rows);
      MESSAGE(summary);
      CHECK(rows > 0);
      CHECK(ctx.validation_message_count() == 0);
    }
  }
}
