// GPU decode matvec y = W x (shaders matvec_*.comp; arithmetic of ADR 0004,
// identical in meaning to cpu/matvec.h).
//
// Descriptor bindings (one set, 6 storage buffers):
//   0  AoS: the weight blocks        SoA: the fp16 scales
//   1  AoS: (bind the weights again) SoA: the quants
//   2  activation quants (32 int8 per block)
//   3  activation scales (float, one per block)
//   4  activation quant sums (int32, one per block)
//   5  output y (float, one per row)
#pragma once

#include <cstdint>
#include <memory>

#include "gpu/pipeline.h"

namespace hearth::gpu {

class Context;

enum class WeightFormat { kQ8_0, kQ4_0 };
enum class WeightLayout { kAos, kSoa };

struct MatvecPush {
  std::uint32_t nb;         // blocks per row (k / 32)
  std::uint32_t row_begin;  // rows [row_begin, row_end) are computed
  std::uint32_t row_end;
  std::uint32_t w_base0;  // element offset into binding 0 (AoS: 16-bit words; SoA: scales)
  std::uint32_t w_base1;  // element offset into binding 1 (SoA: uvec4 of quants)
};

class MatvecPipelines {
 public:
  explicit MatvecPipelines(Context& ctx, std::uint32_t rows_per_workgroup = 4);

  const Pipeline& get(WeightFormat f, WeightLayout l) const;
  std::uint32_t rows_per_workgroup() const noexcept { return rows_; }

  // Records the dispatch for push.row_begin .. push.row_end.
  void record(VkCommandBuffer cmd, WeightFormat f, WeightLayout l, VkDescriptorSet set, const MatvecPush& push) const;

 private:
  std::uint32_t rows_;
  std::unique_ptr<Pipeline> pipes_[2][2];
};

}  // namespace hearth::gpu
