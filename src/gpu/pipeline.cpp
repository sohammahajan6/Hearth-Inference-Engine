#include "gpu/pipeline.h"

#include <vector>

#include "core/error.h"
#include "gpu/buffer.h"
#include "gpu/context.h"

namespace hearth::gpu {

Pipeline::Pipeline(Context& ctx, const PipelineDesc& desc) : device_(ctx.device()), desc_(desc) {
  const std::span<const std::uint32_t> code = find_shader(desc.shader.c_str());
  if (code.empty()) fail(ErrorKind::kInternal, "shader '{}' is not embedded in this build", desc.shader);
  if (desc.push_constant_bytes % 4 != 0 || desc.push_constant_bytes > 128) {
    fail(ErrorKind::kInternal, "shader '{}': push constant size {} must be a multiple of 4 and <= 128", desc.shader,
         desc.push_constant_bytes);
  }
  if (desc.required_subgroup_size != 0 && (desc.required_subgroup_size < ctx.info().min_subgroup_size ||
                                           desc.required_subgroup_size > ctx.info().max_subgroup_size)) {
    fail(ErrorKind::kUnsupported, "shader '{}' needs subgroup size {}, GPU supports {}..{}", desc.shader,
         desc.required_subgroup_size, ctx.info().min_subgroup_size, ctx.info().max_subgroup_size);
  }
  if (desc.require_full_subgroups && !ctx.info().compute_full_subgroups) {
    fail(ErrorKind::kUnsupported, "shader '{}' needs full subgroups (computeFullSubgroups), which this GPU lacks",
         desc.shader);
  }
  VkShaderModule module = VK_NULL_HANDLE;
  try {
    std::vector<VkDescriptorSetLayoutBinding> bindings(desc.buffers);
    for (std::uint32_t i = 0; i < desc.buffers; ++i) {
      bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    }
    VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dl.bindingCount = desc.buffers;
    dl.pBindings = bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(device_, &dl, nullptr, &set_layout_));

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, desc.push_constant_bytes};
    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &set_layout_;
    pl.pushConstantRangeCount = desc.push_constant_bytes > 0 ? 1u : 0u;
    pl.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(device_, &pl, nullptr, &layout_));

    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = code.size_bytes();
    smci.pCode = code.data();
    VK_CHECK(vkCreateShaderModule(device_, &smci, nullptr, &module));

    std::vector<VkSpecializationMapEntry> entries(desc.spec_constants.size());
    for (std::uint32_t i = 0; i < entries.size(); ++i) entries[i] = {i, i * 4u, 4u};
    VkSpecializationInfo spec{static_cast<std::uint32_t>(entries.size()), entries.data(),
                              desc.spec_constants.size() * 4u, desc.spec_constants.data()};

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rss{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    rss.requiredSubgroupSize = desc.required_subgroup_size;
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.pNext = desc.required_subgroup_size != 0 ? &rss : nullptr;
    cpci.stage.flags =
        desc.require_full_subgroups
            ? static_cast<VkPipelineShaderStageCreateFlags>(VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT)
            : VkPipelineShaderStageCreateFlags{0};
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = module;
    cpci.stage.pName = "main";
    cpci.stage.pSpecializationInfo = entries.empty() ? nullptr : &spec;
    cpci.layout = layout_;
    VK_CHECK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline_));
    vkDestroyShaderModule(device_, module, nullptr);
  } catch (...) {
    if (module != VK_NULL_HANDLE) vkDestroyShaderModule(device_, module, nullptr);
    release();
    throw;
  }
}

Pipeline::~Pipeline() { release(); }

void Pipeline::release() noexcept {
  if (pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, pipeline_, nullptr);
  if (layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, layout_, nullptr);
  if (set_layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device_, set_layout_, nullptr);
  pipeline_ = VK_NULL_HANDLE;
  layout_ = VK_NULL_HANDLE;
  set_layout_ = VK_NULL_HANDLE;
}

void Pipeline::dispatch(VkCommandBuffer cmd, VkDescriptorSet set, const void* push, std::uint32_t push_bytes,
                        std::uint32_t groups_x, std::uint32_t groups_y, std::uint32_t groups_z) const {
  HEARTH_CHECK(push_bytes == desc_.push_constant_bytes);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &set, 0, nullptr);
  if (push_bytes > 0) vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes, push);
  vkCmdDispatch(cmd, groups_x, groups_y, groups_z);
}

DescriptorPool::DescriptorPool(Context& ctx, std::uint32_t max_sets, std::uint32_t max_buffers)
    : device_(ctx.device()) {
  VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, max_buffers};
  VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  ci.maxSets = max_sets;
  ci.poolSizeCount = 1;
  ci.pPoolSizes = &size;
  VK_CHECK(vkCreateDescriptorPool(device_, &ci, nullptr, &pool_));
}

DescriptorPool::~DescriptorPool() {
  if (pool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(device_, pool_, nullptr);
}

VkDescriptorSet DescriptorPool::allocate(const Pipeline& pipeline, std::initializer_list<const Buffer*> buffers) {
  std::vector<Range> ranges;
  for (const Buffer* b : buffers) ranges.push_back({b, 0, VK_WHOLE_SIZE});
  return allocate(pipeline, std::span<const Range>(ranges));
}

VkDescriptorSet DescriptorPool::allocate(const Pipeline& pipeline, std::span<const Range> ranges) {
  HEARTH_CHECK(ranges.size() == pipeline.desc().buffers);
  VkDescriptorSetLayout layout = pipeline.set_layout();
  VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  ai.descriptorPool = pool_;
  ai.descriptorSetCount = 1;
  ai.pSetLayouts = &layout;
  VkDescriptorSet set = VK_NULL_HANDLE;
  VK_CHECK(vkAllocateDescriptorSets(device_, &ai, &set));
  update(set, ranges);
  return set;
}

void DescriptorPool::update(VkDescriptorSet set, std::span<const Range> ranges) const {
  std::vector<VkDescriptorBufferInfo> infos(ranges.size());
  std::vector<VkWriteDescriptorSet> writes(ranges.size());
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    infos[i] = {ranges[i].buffer->handle(), ranges[i].offset, ranges[i].size};
    writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writes[i].dstSet = set;
    writes[i].dstBinding = static_cast<std::uint32_t>(i);
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &infos[i];
  }
  vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void compute_barrier(VkCommandBuffer cmd) {
  VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
  mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  mb.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
  mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  mb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT |
                     VK_ACCESS_2_TRANSFER_WRITE_BIT;
  VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  di.memoryBarrierCount = 1;
  di.pMemoryBarriers = &mb;
  vkCmdPipelineBarrier2(cmd, &di);
}

void compute_to_host_barrier(VkCommandBuffer cmd) {
  VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
  mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  mb.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
  mb.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
  mb.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
  VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  di.memoryBarrierCount = 1;
  di.pMemoryBarriers = &mb;
  vkCmdPipelineBarrier2(cmd, &di);
}

}  // namespace hearth::gpu
