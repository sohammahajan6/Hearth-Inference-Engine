// The GPU's view of the CPU's weight arena (cpu/packed_arena.h):
// docs/architecture/08-gpu-backend.md section 1. Each slice is imported once
// (VK_EXT_external_memory_host), so the CPU and the GPU read the same pages:
// one copy of the weights. The CPU must not change the arena while it is
// imported (imported memory is type 0, which the host cannot flush).
#pragma once

#include <cstdint>
#include <vector>

#include "cpu/packed_arena.h"
#include "gpu/buffer.h"
#include "gpu/pipeline.h"

namespace hearth::gpu {

class Context;

class WeightArena {
 public:
  // Fails with kUnsupported without VK_EXT_external_memory_host, and with
  // kOutOfMemory if the import would exceed the heap's budget
  // (VK_EXT_memory_budget) minus a margin. The arena must outlive this.
  WeightArena(Context& ctx, const cpu::PackedArena& arena);

  // Descriptor ranges of matrix i: its quants and its fp16 scales.
  DescriptorPool::Range quants(std::size_t i) const;
  DescriptorPool::Range scales(std::size_t i) const;

  struct Canary {
    std::size_t bytes = 0;
    std::uint64_t gpu_xor = 0, cpu_xor = 0;
    double gpu_ms = 0;
  };
  // XOR of every imported byte, read by the GPU and by the CPU. Equal values
  // show that the GPU sees exactly what the CPU wrote, including the matrix
  // packed last, whose pages may still be dirty in the CPU caches.
  Canary canary(Context& ctx) const;

  std::size_t bytes() const noexcept { return bytes_; }

 private:
  const cpu::PackedArena* arena_;
  std::vector<Buffer> slices_;
  std::size_t bytes_ = 0;
};

// The heap budget and usage reported by VK_EXT_memory_budget for the heap of
// memory type 0, in bytes (both 0 if the extension is not enabled).
struct HeapBudget {
  std::uint64_t budget = 0, usage = 0;
};
HeapBudget heap_budget(const Context& ctx);

}  // namespace hearth::gpu
