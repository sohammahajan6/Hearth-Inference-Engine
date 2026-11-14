#include "gpu/matvec.h"

#include "core/error.h"
#include "gpu/context.h"

namespace hearth::gpu {

MatvecPipelines::MatvecPipelines(Context& ctx, std::uint32_t rows_per_workgroup) : rows_(rows_per_workgroup) {
  if (rows_ == 0 || rows_ > 256 || (rows_ & (rows_ - 1)) != 0) {
    fail(ErrorKind::kInvalidArgument, "matvec rows per workgroup must be a power of two in [1, 256], got {}", rows_);
  }
  const char* names[2][2] = {{"matvec_q8_0_aos", "matvec_q8_0_soa"}, {"matvec_q4_0_aos", "matvec_q4_0_soa"}};
  for (int f = 0; f < 2; ++f) {
    for (int l = 0; l < 2; ++l) {
      pipes_[f][l] = std::make_unique<Pipeline>(ctx, PipelineDesc{.shader = names[f][l],
                                                                  .buffers = 6,
                                                                  .push_constant_bytes = sizeof(MatvecPush),
                                                                  .spec_constants = {rows_},
                                                                  .required_subgroup_size = 0});
    }
  }
}

const Pipeline& MatvecPipelines::get(WeightFormat f, WeightLayout l) const {
  return *pipes_[f == WeightFormat::kQ8_0 ? 0 : 1][l == WeightLayout::kAos ? 0 : 1];
}

void MatvecPipelines::record(VkCommandBuffer cmd, WeightFormat f, WeightLayout l, VkDescriptorSet set,
                             const MatvecPush& push) const {
  HEARTH_CHECK(push.row_end >= push.row_begin);
  const std::uint32_t rows = push.row_end - push.row_begin;
  if (rows == 0) return;
  get(f, l).dispatch(cmd, set, &push, sizeof(push), (rows + rows_ - 1) / rows_);
}

}  // namespace hearth::gpu
