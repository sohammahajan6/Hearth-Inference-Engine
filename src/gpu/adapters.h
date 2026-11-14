// The display adapters Windows knows about (DXGI). Used to map a Vulkan
// physical device to the real hardware adapter: on this machine the Intel
// driver exposes the Iris Xe twice with the same UUID, and only the LUID
// (a per-boot adapter id) tells them apart. The same LUID names the GPU's
// performance counters (GPU Engine, GPU Process Memory) in PDH.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace hearth::gpu {

struct DxgiAdapter {
  std::string name;
  std::uint32_t vendor_id = 0;
  std::uint32_t device_id = 0;
  std::array<std::uint8_t, 8> luid{};  // same byte layout as VkPhysicalDeviceIDProperties::deviceLUID
  bool software = false;
  std::uint64_t dedicated_video_memory = 0;
  std::uint64_t shared_system_memory = 0;
};

std::vector<DxgiAdapter> list_dxgi_adapters();

// "0x00000000_0x0000BAFD": the form PDH counter instance names use.
std::string luid_pdh_string(const std::array<std::uint8_t, 8>& luid);

}  // namespace hearth::gpu
