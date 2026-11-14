// A compute pipeline built from an embedded SPIR-V shader, and descriptor
// sets that bind storage buffers to it.
//
// Every shader in this project uses the same simple interface: set 0 holds N
// storage buffers at bindings 0..N-1, and small per-dispatch values travel as
// push constants. One interface means one pipeline builder and no per-shader
// boilerplate.
#pragma once

#include <cstdint>
#include <initializer_list>
#include <span>
#include <string>
#include <vector>

#include "gpu/vk.h"

namespace hearth::gpu {

class Context;
class Buffer;

// Embedded SPIR-V by shader file name without extension ("hello_add").
// Empty span if no such shader was compiled in.
std::span<const std::uint32_t> find_shader(const char* name);

struct PipelineDesc {
  std::string shader;                         // embedded shader name
  std::uint32_t buffers = 0;                  // storage buffers at bindings 0..buffers-1
  std::uint32_t push_constant_bytes = 0;      // multiple of 4, at most 128
  std::vector<std::uint32_t> spec_constants;  // constant_id i = spec_constants[i]
  std::uint32_t required_subgroup_size = 0;   // 0 = let the driver choose
  // Every subgroup of a workgroup is full (VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT),
  // so gl_SubgroupID / gl_SubgroupInvocationID index whole subgroups. Needs computeFullSubgroups.
  bool require_full_subgroups = false;
};

class Pipeline {
 public:
  Pipeline(Context& ctx, const PipelineDesc& desc);
  ~Pipeline();
  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;

  VkPipeline handle() const noexcept { return pipeline_; }
  VkPipelineLayout layout() const noexcept { return layout_; }
  VkDescriptorSetLayout set_layout() const noexcept { return set_layout_; }
  const PipelineDesc& desc() const noexcept { return desc_; }

  // Records bind + push constants + dispatch.
  void dispatch(VkCommandBuffer cmd, VkDescriptorSet set, const void* push, std::uint32_t push_bytes,
                std::uint32_t groups_x, std::uint32_t groups_y = 1, std::uint32_t groups_z = 1) const;

 private:
  void release() noexcept;

  VkDevice device_ = VK_NULL_HANDLE;
  PipelineDesc desc_;
  VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout layout_ = VK_NULL_HANDLE;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
};

// A descriptor pool that hands out storage-buffer descriptor sets.
class DescriptorPool {
 public:
  DescriptorPool(Context& ctx, std::uint32_t max_sets, std::uint32_t max_buffers);
  ~DescriptorPool();
  DescriptorPool(const DescriptorPool&) = delete;
  DescriptorPool& operator=(const DescriptorPool&) = delete;

  // A set for `pipeline` with binding i = whole buffer i.
  VkDescriptorSet allocate(const Pipeline& pipeline, std::initializer_list<const Buffer*> buffers);

  struct Range {
    const Buffer* buffer;
    VkDeviceSize offset;
    VkDeviceSize size;
  };
  // A set for `pipeline` with binding i = ranges[i] (offsets must respect
  // minStorageBufferOffsetAlignment, sizes maxStorageBufferRange).
  VkDescriptorSet allocate(const Pipeline& pipeline, std::span<const Range> ranges);
  // Points binding i of `set` at ranges[i], with the same rules. The set must
  // not be used by a submission that has not finished.
  void update(VkDescriptorSet set, std::span<const Range> ranges) const;

 private:
  VkDevice device_ = VK_NULL_HANDLE;
  VkDescriptorPool pool_ = VK_NULL_HANDLE;
};

// Full memory barrier between compute dispatches (write -> read/write).
void compute_barrier(VkCommandBuffer cmd);

// Makes compute-shader writes visible to the CPU once the submission's fence
// or semaphore has signalled. Record it last before reading results on the host.
void compute_to_host_barrier(VkCommandBuffer cmd);

}  // namespace hearth::gpu
