// `hearth info`: what this machine offers the engine.
#include <cstdio>
#include <string>

#include "commands.h"
#include "core/cpu_features.h"
#include "core/error.h"
#include "core/platform.h"
#include "gpu/adapters.h"
#include "gpu/context.h"
#include "hearth/version.h"

namespace hearth::cli {
namespace {

std::string yes_no(bool b) { return b ? "yes" : "no"; }

std::string uuid_string(const std::array<std::uint8_t, 16>& u) {
  std::string s;
  for (std::size_t i = 0; i < u.size(); ++i) {
    if (i == 4 || i == 6 || i == 8 || i == 10) s += '-';
    s += std::format("{:02x}", u[i]);
  }
  return s;
}

std::string memory_flags(VkMemoryPropertyFlags f) {
  std::string s;
  auto add = [&](VkMemoryPropertyFlags bit, const char* name) {
    if ((f & bit) != 0) s += std::string(s.empty() ? "" : "|") + name;
  };
  add(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "DEVICE_LOCAL");
  add(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, "HOST_VISIBLE");
  add(VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, "HOST_COHERENT");
  add(VK_MEMORY_PROPERTY_HOST_CACHED_BIT, "HOST_CACHED");
  return s.empty() ? "(none)" : s;
}

const char* device_type(VkPhysicalDeviceType t) {
  switch (t) {
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
      return "integrated";
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
      return "discrete";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
      return "virtual";
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
      return "cpu";
    default:
      return "other";
  }
}

double gib(unsigned long long bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0); }

}  // namespace

int cmd_info(Args& args) {
  const auto model = args.value("model");
  const bool list_tensors = args.flag("tensors");
  const bool json = args.flag("json");
  args.finish();
  if (model) return model_info(*model, list_tensors, json);
  if (list_tensors || json) fail(ErrorKind::kInvalidArgument, "--tensors and --json need --model FILE");
  const BuildInfo& b = build_info();
  std::printf("hearth %s (%s, %s %s)\n", b.version, b.commit, b.compiler, b.build_type);
  std::printf("os:     Windows %s, power: %s\n", os_version_string().c_str(), to_string(power_source()));
  const MemoryStatus mem = memory_status();
  std::printf("memory: %.1f GiB total, %.1f GiB available\n", gib(mem.total_bytes), gib(mem.available_bytes));

  const CpuFeatures& f = cpu_features();
  std::printf("cpu:    %s, %d cores / %d threads\n", f.brand.c_str(), f.physical_cores, f.logical_cores);
  std::printf("        kernels: avx512=%s avx2=%s (best: %s)\n", yes_no(isa_supported(Isa::kAvx512)).c_str(),
              yes_no(isa_supported(Isa::kAvx2)).c_str(), to_string(best_isa()));
  std::printf("        avx2=%s fma=%s f16c=%s avx512f=%s bw=%s dq=%s vl=%s avx512_vnni=%s avx_vnni=%s avx512_bf16=%s\n",
              yes_no(f.avx2).c_str(), yes_no(f.fma).c_str(), yes_no(f.f16c).c_str(), yes_no(f.avx512f).c_str(),
              yes_no(f.avx512bw).c_str(), yes_no(f.avx512dq).c_str(), yes_no(f.avx512vl).c_str(),
              yes_no(f.avx512vnni).c_str(), yes_no(f.avx_vnni).c_str(), yes_no(f.avx512bf16).c_str());

  for (const gpu::DxgiAdapter& a : gpu::list_dxgi_adapters()) {
    std::printf("dxgi:   %s (vendor 0x%04x, device 0x%04x, LUID %s%s)\n", a.name.c_str(), a.vendor_id, a.device_id,
                gpu::luid_pdh_string(a.luid).c_str(), a.software ? ", software" : "");
  }
  if (!gpu::vulkan_available()) {
    std::printf("gpu:    Vulkan is not available (vulkan-1.dll not found); the GPU backends cannot run\n");
    return 0;
  }
  const auto devices = gpu::list_devices();
  if (devices.empty()) std::printf("gpu:    no Vulkan device found\n");
  for (std::size_t i = 0; i < devices.size(); ++i) {
    const gpu::DeviceInfo& d = devices[i];
    std::printf("gpu %zu:  %s (%s, Vulkan %u.%u.%u, driver %s)\n", i, d.name.c_str(), device_type(d.type),
                VK_API_VERSION_MAJOR(d.api_version), VK_API_VERSION_MINOR(d.api_version),
                VK_API_VERSION_PATCH(d.api_version), d.driver_version_text.c_str());
    std::printf("        uuid %s, LUID %s (%s), duplicates by UUID: %d\n", uuid_string(d.uuid).c_str(),
                d.luid_valid ? gpu::luid_pdh_string(d.luid).c_str() : "invalid",
                d.luid_matches_dxgi ? "matches a DXGI hardware adapter" : "no DXGI match", d.duplicates);
    std::printf("        subgroups %u..%u (default %u), int8 dot accelerated: mixed=%s signed=%s\n",
                d.min_subgroup_size, d.max_subgroup_size, d.subgroup_size, yes_no(d.int8_dot_mixed_accelerated).c_str(),
                yes_no(d.int8_dot_signed_accelerated).c_str());
    std::printf("        fp16=%s int8=%s int16=%s int64=%s fp64=%s storage8=%s storage16=%s timeline=%s\n",
                yes_no(d.shader_float16).c_str(), yes_no(d.shader_int8).c_str(), yes_no(d.shader_int16).c_str(),
                yes_no(d.shader_int64).c_str(), yes_no(d.shader_float64).c_str(), yes_no(d.storage_buffer_8bit).c_str(),
                yes_no(d.storage_buffer_16bit).c_str(), yes_no(d.timeline_semaphore).c_str());
    std::printf("        maxStorageBufferRange %.3f GiB, maxBufferSize %.3f GiB, maxAllocation %.3f GiB\n",
                gib(d.limits.maxStorageBufferRange), gib(d.max_buffer_size), gib(d.max_memory_allocation_size));
    std::printf("        shared memory %u KiB, max workgroup %u, timestamp %.2f ns (%u bits)\n",
                d.limits.maxComputeSharedMemorySize / 1024, d.limits.maxComputeWorkGroupInvocations,
                static_cast<double>(d.timestamp_period_ns), d.timestamp_valid_bits);
    std::printf("        external_memory_host=%s (alignment %llu B), memory_budget=%s\n",
                yes_no(d.ext_external_memory_host).c_str(),
                static_cast<unsigned long long>(d.min_imported_host_pointer_alignment),
                yes_no(d.ext_memory_budget).c_str());
    for (std::size_t h = 0; h < d.memory_heaps.size(); ++h) {
      std::printf("        heap %zu: %.2f GiB%s\n", h, gib(d.memory_heaps[h].size),
                  (d.memory_heaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0 ? " (device-local)" : "");
    }
    for (std::size_t t = 0; t < d.memory_types.size(); ++t) {
      std::printf("        memory type %zu: heap %u, %s\n", t, d.memory_types[t].heap_index,
                  memory_flags(d.memory_types[t].flags).c_str());
    }
  }
  return 0;
}

}  // namespace hearth::cli
