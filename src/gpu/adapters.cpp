#include "gpu/adapters.h"

#include <windows.h>
// (separate block: dxgi.h needs windows.h first, and the formatter sorts within a block)
#include <dxgi.h>

#include <cstring>
#include <format>

namespace hearth::gpu {
namespace {

std::string narrow(const wchar_t* w) {
  const int len = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
  if (len <= 1) return {};
  std::string s(static_cast<std::size_t>(len - 1), '\0');
  WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), len, nullptr, nullptr);
  return s;
}

}  // namespace

std::vector<DxgiAdapter> list_dxgi_adapters() {
  std::vector<DxgiAdapter> out;
  IDXGIFactory1* factory = nullptr;
  if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory)))) return out;
  for (UINT i = 0;; ++i) {
    IDXGIAdapter1* adapter = nullptr;
    if (factory->EnumAdapters1(i, &adapter) == DXGI_ERROR_NOT_FOUND) break;
    DXGI_ADAPTER_DESC1 desc{};
    if (SUCCEEDED(adapter->GetDesc1(&desc))) {
      DxgiAdapter a;
      a.name = narrow(desc.Description);
      a.vendor_id = desc.VendorId;
      a.device_id = desc.DeviceId;
      static_assert(sizeof(desc.AdapterLuid) == 8);
      std::memcpy(a.luid.data(), &desc.AdapterLuid, 8);
      a.software = (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;
      a.dedicated_video_memory = desc.DedicatedVideoMemory;
      a.shared_system_memory = desc.SharedSystemMemory;
      out.push_back(std::move(a));
    }
    adapter->Release();
  }
  factory->Release();
  return out;
}

std::string luid_pdh_string(const std::array<std::uint8_t, 8>& luid) {
  LUID l;
  std::memcpy(&l, luid.data(), 8);
  return std::format("0x{:08X}_0x{:08X}", static_cast<std::uint32_t>(l.HighPart),
                     static_cast<std::uint32_t>(l.LowPart));
}

}  // namespace hearth::gpu
