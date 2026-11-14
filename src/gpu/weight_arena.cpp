#include "gpu/weight_arena.h"

#include <algorithm>
#include <chrono>

#include "core/error.h"
#include "gpu/context.h"

namespace hearth::gpu {
namespace {

// Room left for everything else that uses the heap (exchange buffers, the
// KV cache, other processes): the import may use at most 90 % of the budget.
constexpr double kBudgetShare = 0.9;
// The canary reads at most this much per dispatch (well inside the 1 GiB
// storage range and the TDR budget: ~10 ms at the GPU's measured read rate).
constexpr std::size_t kCanaryChunk = std::size_t{256} << 20;
constexpr std::uint32_t kGroups = 256;
constexpr std::uint32_t kInvocations = kGroups * 256;

struct ReadPush {
  std::uint32_t n, magic, write_all, base, blocked;
};

std::uint64_t cpu_xor(const std::byte* p, std::size_t bytes) {
  const auto* w = reinterpret_cast<const std::uint64_t*>(p);
  std::uint64_t x = 0;
  for (std::size_t i = 0; i < bytes / 8; ++i) x ^= w[i];
  return x;
}

}  // namespace

HeapBudget heap_budget(const Context& ctx) {
  if (!ctx.info().ext_memory_budget) return {};
  VkPhysicalDeviceMemoryBudgetPropertiesEXT b{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
  VkPhysicalDeviceMemoryProperties2 p{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
  p.pNext = &b;
  vkGetPhysicalDeviceMemoryProperties2(ctx.physical_device(), &p);
  const std::uint32_t heap = p.memoryProperties.memoryTypes[0].heapIndex;
  return {b.heapBudget[heap], b.heapUsage[heap]};
}

WeightArena::WeightArena(Context& ctx, const cpu::PackedArena& arena) : arena_(&arena), bytes_(arena.bytes()) {
  if (!ctx.external_memory_host_enabled()) {
    fail(ErrorKind::kUnsupported, "the GPU cannot import host memory (VK_EXT_external_memory_host); use --backend cpu");
  }
  const HeapBudget hb = heap_budget(ctx);
  if (hb.budget != 0 && static_cast<double>(hb.usage + bytes_) > kBudgetShare * static_cast<double>(hb.budget)) {
    fail(ErrorKind::kOutOfMemory,
         "the weights ({} MiB) do not fit in the GPU heap's budget ({} MiB, {} MiB already used, {:.0f} % usable); "
         "close other programs or use --backend cpu",
         bytes_ >> 20, hb.budget >> 20, hb.usage >> 20, kBudgetShare * 100);
  }
  const VkDeviceSize align = ctx.info().limits.minStorageBufferOffsetAlignment;
  for (std::size_t i = 0; i < arena.size(); ++i) {
    const auto& l = arena.location(i);
    if (l.offset % align != 0 || (l.offset + l.quant_bytes) % align != 0 ||
        l.quant_bytes > ctx.info().limits.maxStorageBufferRange) {
      fail(ErrorKind::kUnsupported, "weight matrix {} cannot be bound as a storage range on this GPU", i);
    }
  }
  for (std::size_t s = 0; s < arena.slices(); ++s) {
    const std::span<std::byte> sl = arena.slice(s);
    slices_.push_back(Buffer::import_host(ctx, sl.data(), sl.size()));
  }
}

DescriptorPool::Range WeightArena::quants(std::size_t i) const {
  const auto& l = arena_->location(i);
  return {&slices_.at(l.slice), l.offset, l.quant_bytes};
}

DescriptorPool::Range WeightArena::scales(std::size_t i) const {
  const auto& l = arena_->location(i);
  return {&slices_.at(l.slice), l.offset + l.quant_bytes, l.bytes - l.quant_bytes};
}

WeightArena::Canary WeightArena::canary(Context& ctx) const {
  Canary c;
  Pipeline pipe(ctx, {.shader = "probe_read", .buffers = 2, .push_constant_bytes = sizeof(ReadPush)});
  Buffer dst(ctx, kInvocations * 16, MemoryKind::kHostCached);
  std::size_t chunks = 0;
  for (std::size_t s = 0; s < slices_.size(); ++s)
    chunks += (arena_->slice(s).size() + kCanaryChunk - 1) / kCanaryChunk;
  DescriptorPool pool(ctx, static_cast<std::uint32_t>(chunks), static_cast<std::uint32_t>(2 * chunks));
  const auto t0 = std::chrono::steady_clock::now();
  for (std::size_t s = 0; s < slices_.size(); ++s) {
    const std::span<std::byte> sl = arena_->slice(s);
    for (std::size_t off = 0; off < sl.size(); off += kCanaryChunk) {
      const std::size_t n = std::min(kCanaryChunk, sl.size() - off);  // a multiple of 4 KiB
      const DescriptorPool::Range ranges[] = {{&slices_[s], off, n}, {&dst, 0, dst.size()}};
      const VkDescriptorSet set = pool.allocate(pipe, ranges);
      const ReadPush push{static_cast<std::uint32_t>(n / 16), 0, 1, 0, 0};
      ctx.run_once([&](VkCommandBuffer cmd) {
        pipe.dispatch(cmd, set, &push, sizeof(push), kGroups);
        compute_to_host_barrier(cmd);
      });
      const auto* w = static_cast<const std::uint64_t*>(dst.mapped());
      for (std::uint32_t i = 0; i < kInvocations * 2; ++i) c.gpu_xor ^= w[i];
      c.bytes += n;
    }
  }
  c.gpu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  for (std::size_t s = 0; s < slices_.size(); ++s)
    c.cpu_xor ^= cpu_xor(arena_->slice(s).data(), arena_->slice(s).size());
  return c;
}

}  // namespace hearth::gpu
