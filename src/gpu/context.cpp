#include "gpu/context.h"

#include <windows.h>

#include <algorithm>
#include <cstring>
#include <format>
#include <mutex>
#include <string>
#include <utility>

#include "core/clock.h"
#include "core/error.h"
#include "core/log.h"
#include "gpu/adapters.h"

namespace hearth::gpu {
namespace {

constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";

std::once_flag g_loader_once;
VkResult g_loader_result = VK_ERROR_INITIALIZATION_FAILED;

bool init_loader() {
  std::call_once(g_loader_once, [] {
    // OBS registers an implicit layer that the loader injects into every
    // Vulkan process on this machine. A compute engine gains nothing from it
    // and it is one more thing that can break, so implicit layers are off
    // unless HEARTH_ALLOW_IMPLICIT_LAYERS is set or the user already chose a
    // VK_LOADER_LAYERS_DISABLE value. This changes only this process.
    wchar_t probe[2];
    if (GetEnvironmentVariableW(L"VK_LOADER_LAYERS_DISABLE", probe, 2) == 0 &&
        GetEnvironmentVariableW(L"HEARTH_ALLOW_IMPLICIT_LAYERS", probe, 2) == 0) {
      SetEnvironmentVariableW(L"VK_LOADER_LAYERS_DISABLE", L"~implicit~");
    }
    g_loader_result = volkInitialize();
  });
  return g_loader_result == VK_SUCCESS;
}

std::string decode_driver_version(std::uint32_t vendor, std::uint32_t v) {
  if (vendor == 0x8086) return std::format("{}.{}", v >> 14, v & 0x3fffu);  // Intel on Windows
  if (vendor == 0x10de) {
    return std::format("{}.{}.{}.{}", (v >> 22) & 0x3ffu, (v >> 14) & 0xffu, (v >> 6) & 0xffu, v & 0x3fu);
  }
  return std::format("{}.{}.{}", VK_API_VERSION_MAJOR(v), VK_API_VERSION_MINOR(v), VK_API_VERSION_PATCH(v));
}

bool has_extension(const std::vector<VkExtensionProperties>& exts, const char* name) {
  return std::any_of(exts.begin(), exts.end(),
                     [&](const VkExtensionProperties& e) { return std::strcmp(e.extensionName, name) == 0; });
}

std::vector<VkExtensionProperties> device_extensions(VkPhysicalDevice pd) {
  std::uint32_t n = 0;
  VK_CHECK(vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, nullptr));
  std::vector<VkExtensionProperties> exts(n);
  VK_CHECK(vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, exts.data()));
  exts.resize(n);
  return exts;
}

DeviceInfo describe(VkPhysicalDevice pd, const std::vector<DxgiAdapter>& adapters) {
  DeviceInfo d;
  const auto exts = device_extensions(pd);
  d.ext_external_memory_host = has_extension(exts, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
  d.ext_memory_budget = has_extension(exts, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
  d.ext_calibrated_timestamps = has_extension(exts, VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);

  VkPhysicalDeviceProperties base{};
  vkGetPhysicalDeviceProperties(pd, &base);
  d.api_version = base.apiVersion;
  const bool vk13 = base.apiVersion >= VK_API_VERSION_1_3;

  // The ID properties are in every chain (UUID/LUID are needed even to
  // report an unsupported device); the 1.1/1.3 aggregates only on 1.3 devices.
  VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
  VkPhysicalDeviceVulkan11Properties p11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES};
  VkPhysicalDeviceVulkan13Properties p13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
  VkPhysicalDeviceExternalMemoryHostPropertiesEXT emh{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
  VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  void** tail = &props.pNext;
  if (vk13) {
    // VkPhysicalDeviceVulkan11Properties already contains the ID properties;
    // the spec forbids chaining both, so ID comes from p11 on 1.3 devices.
    *tail = &p11;
    p11.pNext = &p13;
    tail = &p13.pNext;
  } else {
    *tail = &id;
    tail = &id.pNext;
  }
  if (d.ext_external_memory_host) *tail = &emh;
  vkGetPhysicalDeviceProperties2(pd, &props);
  if (vk13) {
    std::memcpy(id.deviceUUID, p11.deviceUUID, VK_UUID_SIZE);
    std::memcpy(id.deviceLUID, p11.deviceLUID, VK_LUID_SIZE);
    id.deviceLUIDValid = p11.deviceLUIDValid;
  }

  d.name = props.properties.deviceName;
  d.vendor_id = props.properties.vendorID;
  d.device_id = props.properties.deviceID;
  d.driver_version = props.properties.driverVersion;
  d.driver_version_text = decode_driver_version(d.vendor_id, d.driver_version);
  d.type = props.properties.deviceType;
  d.limits = props.properties.limits;
  d.timestamp_period_ns = props.properties.limits.timestampPeriod;
  std::memcpy(d.uuid.data(), id.deviceUUID, VK_UUID_SIZE);
  std::memcpy(d.luid.data(), id.deviceLUID, VK_LUID_SIZE);
  d.luid_valid = id.deviceLUIDValid == VK_TRUE;
  if (vk13) {
    d.subgroup_size = p11.subgroupSize;
    d.max_memory_allocation_size = p11.maxMemoryAllocationSize;
    d.min_subgroup_size = p13.minSubgroupSize;
    d.max_subgroup_size = p13.maxSubgroupSize;
    d.max_buffer_size = p13.maxBufferSize;
    d.int8_dot_mixed_accelerated = p13.integerDotProduct4x8BitPackedMixedSignednessAccelerated == VK_TRUE;
    d.int8_dot_signed_accelerated = p13.integerDotProduct4x8BitPackedSignedAccelerated == VK_TRUE;
  }
  if (d.ext_external_memory_host) d.min_imported_host_pointer_alignment = emh.minImportedHostPointerAlignment;
  if (d.luid_valid) {
    d.luid_matches_dxgi = std::any_of(adapters.begin(), adapters.end(),
                                      [&](const DxgiAdapter& a) { return !a.software && a.luid == d.luid; });
  }

  if (vk13) {
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &f11;
    f11.pNext = &f12;
    f12.pNext = &f13;
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    d.shader_int16 = f2.features.shaderInt16 == VK_TRUE;
    d.shader_int64 = f2.features.shaderInt64 == VK_TRUE;
    d.shader_float64 = f2.features.shaderFloat64 == VK_TRUE;
    d.storage_buffer_16bit = f11.storageBuffer16BitAccess == VK_TRUE;
    d.storage_buffer_8bit = f12.storageBuffer8BitAccess == VK_TRUE;
    d.shader_int8 = f12.shaderInt8 == VK_TRUE;
    d.shader_float16 = f12.shaderFloat16 == VK_TRUE;
    d.timeline_semaphore = f12.timelineSemaphore == VK_TRUE;
    d.host_query_reset = f12.hostQueryReset == VK_TRUE;
    d.synchronization2 = f13.synchronization2 == VK_TRUE;
    d.subgroup_size_control = f13.subgroupSizeControl == VK_TRUE;
    d.compute_full_subgroups = f13.computeFullSubgroups == VK_TRUE;
    d.integer_dot_product = f13.shaderIntegerDotProduct == VK_TRUE;
    d.maintenance4 = f13.maintenance4 == VK_TRUE;
  }

  VkPhysicalDeviceMemoryProperties mem{};
  vkGetPhysicalDeviceMemoryProperties(pd, &mem);
  for (std::uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
    d.memory_types.push_back({mem.memoryTypes[i].propertyFlags, mem.memoryTypes[i].heapIndex});
  }
  for (std::uint32_t i = 0; i < mem.memoryHeapCount; ++i) d.memory_heaps.push_back(mem.memoryHeaps[i]);

  std::uint32_t qn = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
  std::vector<VkQueueFamilyProperties> qf(qn);
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qf.data());
  // Prefer a compute-only family (async compute) if the GPU has one; Iris Xe has a single universal family.
  for (int pass = 0; pass < 2 && d.compute_queue_family == UINT32_MAX; ++pass) {
    for (std::uint32_t i = 0; i < qn; ++i) {
      const bool compute = (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
      const bool graphics = (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
      if (compute && (pass == 1 || !graphics)) {
        d.compute_queue_family = i;
        d.compute_queue_count = qf[i].queueCount;
        d.timestamp_valid_bits = qf[i].timestampValidBits;
        break;
      }
    }
  }
  return d;
}

struct Candidate {
  VkPhysicalDevice handle;
  DeviceInfo info;
};

// All physical devices, deduplicated by UUID, in enumeration order.
std::vector<Candidate> enumerate(VkInstance instance) {
  std::uint32_t n = 0;
  VK_CHECK(vkEnumeratePhysicalDevices(instance, &n, nullptr));
  std::vector<VkPhysicalDevice> pds(n);
  VK_CHECK(vkEnumeratePhysicalDevices(instance, &n, pds.data()));
  pds.resize(n);
  const auto adapters = list_dxgi_adapters();
  std::vector<Candidate> unique;
  for (VkPhysicalDevice pd : pds) {
    Candidate c{pd, describe(pd, adapters)};
    auto same =
        std::find_if(unique.begin(), unique.end(), [&](const Candidate& u) { return u.info.uuid == c.info.uuid; });
    if (same == unique.end()) {
      unique.push_back(std::move(c));
      continue;
    }
    const int dups = same->info.duplicates + 1;
    if (!same->info.luid_matches_dxgi && c.info.luid_matches_dxgi) *same = std::move(c);
    same->info.duplicates = dups;
  }
  return unique;
}

VkInstance create_instance(bool validation, VkDebugUtilsMessengerCreateInfoEXT* messenger_info) {
  std::uint32_t api = 0;
  VK_CHECK(vkEnumerateInstanceVersion(&api));
  if (api < VK_API_VERSION_1_3) {
    fail(ErrorKind::kUnsupported, "the Vulkan loader is version {}.{}; 1.3 or newer is required",
         VK_API_VERSION_MAJOR(api), VK_API_VERSION_MINOR(api));
  }
  std::vector<const char*> layers;
  std::vector<const char*> exts;
  VkBool32 on = VK_TRUE;
  VkLayerSettingEXT settings[] = {
      // Synchronization validation: reports missing barriers (read-after-write
      // hazards), the bug class most likely to corrupt results silently.
      {kValidationLayer, "validate_sync", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &on},
      // Without this, the pinned layer (1.4.363) ignores every access a shader
      // makes through descriptors (layers/sync/sync_command_buffer.cpp:449),
      // so a missing barrier between two compute dispatches - the main hazard
      // in a compute-only engine - would go unreported. Verified by the
      // "missing barrier is caught" test.
      {kValidationLayer, "syncval_shader_accesses_heuristic", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &on},
  };
  VkLayerSettingsCreateInfoEXT settings_info{VK_STRUCTURE_TYPE_LAYER_SETTINGS_CREATE_INFO_EXT};
  if (validation) {
    std::uint32_t n = 0;
    VK_CHECK(vkEnumerateInstanceLayerProperties(&n, nullptr));
    std::vector<VkLayerProperties> props(n);
    VK_CHECK(vkEnumerateInstanceLayerProperties(&n, props.data()));
    const bool found = std::any_of(props.begin(), props.begin() + n, [](const VkLayerProperties& p) {
      return std::strcmp(p.layerName, kValidationLayer) == 0;
    });
    if (!found) {
      fail(ErrorKind::kGpu,
           "Vulkan validation was requested but {} was not found. Build it with "
           "`pwsh -File scripts\\setup.ps1 -Only vvl`; scripts\\test.ps1 points VK_ADD_LAYER_PATH at it.",
           kValidationLayer);
    }
    layers.push_back(kValidationLayer);
    exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    std::uint32_t en = 0;
    VK_CHECK(vkEnumerateInstanceExtensionProperties(kValidationLayer, &en, nullptr));
    std::vector<VkExtensionProperties> layer_exts(en);
    VK_CHECK(vkEnumerateInstanceExtensionProperties(kValidationLayer, &en, layer_exts.data()));
    if (has_extension(layer_exts, VK_EXT_LAYER_SETTINGS_EXTENSION_NAME)) {
      exts.push_back(VK_EXT_LAYER_SETTINGS_EXTENSION_NAME);
      settings_info.settingCount = static_cast<std::uint32_t>(std::size(settings));
      settings_info.pSettings = settings;
      settings_info.pNext = messenger_info;
    }
  }
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "hearth";
  app.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
  app.pEngineName = "hearth";
  app.engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
  app.apiVersion = VK_API_VERSION_1_3;
  VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ci.pApplicationInfo = &app;
  ci.enabledLayerCount = static_cast<std::uint32_t>(layers.size());
  ci.ppEnabledLayerNames = layers.data();
  ci.enabledExtensionCount = static_cast<std::uint32_t>(exts.size());
  ci.ppEnabledExtensionNames = exts.data();
  if (validation) {
    // The messenger chained here also reports problems during instance creation/destruction.
    ci.pNext = settings_info.settingCount > 0 ? static_cast<const void*>(&settings_info) : messenger_info;
  }
  VkInstance instance = VK_NULL_HANDLE;
  VK_CHECK(vkCreateInstance(&ci, nullptr, &instance));
  volkLoadInstanceOnly(instance);
  return instance;
}

}  // namespace

bool vulkan_available() noexcept {
  try {
    return init_loader();
  } catch (...) {
    return false;
  }
}

std::vector<DeviceInfo> list_devices() {
  if (!init_loader()) return {};
  VkInstance instance = create_instance(false, nullptr);
  std::vector<DeviceInfo> out;
  try {
    for (Candidate& c : enumerate(instance)) out.push_back(std::move(c.info));
  } catch (...) {
    vkDestroyInstance(instance, nullptr);
    throw;
  }
  vkDestroyInstance(instance, nullptr);
  return out;
}

VKAPI_ATTR VkBool32 VKAPI_CALL Context::debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                       VkDebugUtilsMessageTypeFlagsEXT /*types*/,
                                                       const VkDebugUtilsMessengerCallbackDataEXT* data, void* user) {
  auto* self = static_cast<Context*>(user);
  const char* msg = (data != nullptr && data->pMessage != nullptr) ? data->pMessage : "(no message)";
  // The loader reports through the same messenger. On this machine it
  // complains about stale layer manifests left by uninstalled programs
  // (Steam, Epic, Rockstar) and notes that the OBS layer is disabled. Those
  // say nothing about our API usage, so they are logged but not counted.
  if (data != nullptr && data->pMessageIdName != nullptr && std::strcmp(data->pMessageIdName, "Loader Message") == 0) {
    log_debug("vulkan loader: {}", msg);
    return VK_FALSE;
  }
  self->validation_messages_.fetch_add(1);
  if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0) {
    log_error("vulkan validation: {}", msg);
  } else {
    log_warn("vulkan validation: {}", msg);
  }
  return VK_FALSE;  // never abort the call that triggered the message
}

Context::Context(const ContextOptions& options) : validation_(options.validation) {
  if (!init_loader()) {
    fail(ErrorKind::kGpu, "Vulkan is not available: vulkan-1.dll could not be loaded ({})",
         vk_result_name(g_loader_result));
  }
  try {
    VkDebugUtilsMessengerCreateInfoEXT mci{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    mci.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    mci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                      VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    mci.pfnUserCallback = &Context::debug_callback;
    mci.pUserData = this;
    instance_ = create_instance(validation_, validation_ ? &mci : nullptr);
    if (validation_) VK_CHECK(vkCreateDebugUtilsMessengerEXT(instance_, &mci, nullptr, &messenger_));

    std::vector<Candidate> devices = enumerate(instance_);
    if (devices.empty()) fail(ErrorKind::kGpu, "no Vulkan GPU was found");
    std::size_t pick = 0;
    if (options.device_index >= 0) {
      if (static_cast<std::size_t>(options.device_index) >= devices.size()) {
        fail(ErrorKind::kInvalidArgument, "GPU index {} does not exist ({} GPU(s) found; see `hearth info`)",
             options.device_index, devices.size());
      }
      pick = static_cast<std::size_t>(options.device_index);
    } else {
      auto score = [](const DeviceInfo& d) {
        return (d.type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2 : 0) + (d.luid_matches_dxgi ? 1 : 0);
      };
      for (std::size_t i = 1; i < devices.size(); ++i) {
        if (score(devices[i].info) > score(devices[pick].info)) pick = i;
      }
    }
    physical_ = devices[pick].handle;
    info_ = std::move(devices[pick].info);

    if (info_.api_version < VK_API_VERSION_1_3) {
      fail(ErrorKind::kUnsupported, "GPU '{}' supports Vulkan {}.{}; 1.3 is required", info_.name,
           VK_API_VERSION_MAJOR(info_.api_version), VK_API_VERSION_MINOR(info_.api_version));
    }
    std::string missing;
    auto require = [&](bool ok, const char* name) {
      if (!ok) missing += std::string(missing.empty() ? "" : ", ") + name;
    };
    require(info_.storage_buffer_16bit, "storageBuffer16BitAccess");
    require(info_.storage_buffer_8bit, "storageBuffer8BitAccess");
    require(info_.timeline_semaphore, "timelineSemaphore");
    require(info_.synchronization2, "synchronization2");
    require(info_.subgroup_size_control, "subgroupSizeControl");
    require(info_.host_query_reset, "hostQueryReset");
    require(info_.compute_queue_family != UINT32_MAX, "a compute queue");
    if (!missing.empty())
      fail(ErrorKind::kUnsupported, "GPU '{}' lacks required Vulkan features: {}", info_.name, missing);

    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &f11;
    f11.pNext = &f12;
    f12.pNext = &f13;
    f2.features.shaderInt16 = info_.shader_int16 ? VK_TRUE : VK_FALSE;
    f11.storageBuffer16BitAccess = VK_TRUE;
    f12.storageBuffer8BitAccess = VK_TRUE;
    f12.timelineSemaphore = VK_TRUE;
    f12.hostQueryReset = VK_TRUE;
    f12.shaderInt8 = info_.shader_int8 ? VK_TRUE : VK_FALSE;
    f12.shaderFloat16 = info_.shader_float16 ? VK_TRUE : VK_FALSE;
    f13.synchronization2 = VK_TRUE;
    f13.subgroupSizeControl = VK_TRUE;
    f13.computeFullSubgroups = info_.compute_full_subgroups ? VK_TRUE : VK_FALSE;
    f13.shaderIntegerDotProduct = info_.integer_dot_product ? VK_TRUE : VK_FALSE;
    f13.maintenance4 = info_.maintenance4 ? VK_TRUE : VK_FALSE;

    // Optional extensions. (VK_EXT_external_memory_host depends on
    // VK_KHR_external_memory, which is core since Vulkan 1.1.)
    std::vector<const char*> exts;
    if (info_.ext_external_memory_host) {
      exts.push_back(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
      ext_memory_host_enabled_ = true;
    }
    if (info_.ext_memory_budget) exts.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    if (info_.ext_calibrated_timestamps) {
      std::uint32_t nd = 0;
      VK_CHECK(vkGetPhysicalDeviceCalibrateableTimeDomainsEXT(physical_, &nd, nullptr));
      std::vector<VkTimeDomainEXT> domains(nd);
      VK_CHECK(vkGetPhysicalDeviceCalibrateableTimeDomainsEXT(physical_, &nd, domains.data()));
      const auto has = [&](VkTimeDomainEXT t) { return std::find(domains.begin(), domains.end(), t) != domains.end(); };
      if (has(VK_TIME_DOMAIN_DEVICE_EXT) && has(VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_EXT)) {
        exts.push_back(VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
        calibrated_timestamps_enabled_ = true;
      }
    }

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = info_.compute_queue_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f2;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = static_cast<std::uint32_t>(exts.size());
    dci.ppEnabledExtensionNames = exts.data();
    VK_CHECK(vkCreateDevice(physical_, &dci, nullptr, &device_));
    volkLoadDevice(device_);
    vkGetDeviceQueue(device_, info_.compute_queue_family, 0, &queue_);

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = info_.compute_queue_family;
    VK_CHECK(vkCreateCommandPool(device_, &pci, nullptr, &once_pool_));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = once_pool_;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(device_, &cai, &once_cmd_));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK(vkCreateFence(device_, &fci, nullptr, &once_fence_));
    log_debug("gpu: using '{}' (driver {}, {} duplicate(s) by UUID, LUID {}{})", info_.name, info_.driver_version_text,
              info_.duplicates, luid_pdh_string(info_.luid), info_.luid_matches_dxgi ? " = DXGI hardware adapter" : "");
  } catch (...) {
    destroy();
    throw;
  }
}

Context::~Context() { destroy(); }

void Context::destroy() noexcept {
  if (device_ != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(device_);  // result ignored: on a lost device there is nothing left to wait for
    if (once_fence_ != VK_NULL_HANDLE) vkDestroyFence(device_, once_fence_, nullptr);
    if (once_pool_ != VK_NULL_HANDLE) vkDestroyCommandPool(device_, once_pool_, nullptr);
    vkDestroyDevice(device_, nullptr);
  }
  if (messenger_ != VK_NULL_HANDLE) vkDestroyDebugUtilsMessengerEXT(instance_, messenger_, nullptr);
  if (instance_ != VK_NULL_HANDLE) vkDestroyInstance(instance_, nullptr);
  once_fence_ = VK_NULL_HANDLE;
  once_pool_ = VK_NULL_HANDLE;
  once_cmd_ = VK_NULL_HANDLE;
  device_ = VK_NULL_HANDLE;
  messenger_ = VK_NULL_HANDLE;
  instance_ = VK_NULL_HANDLE;
}

std::uint32_t Context::find_memory_type(std::uint32_t type_bits, VkMemoryPropertyFlags required,
                                        VkMemoryPropertyFlags preferred) const {
  std::uint32_t fallback = UINT32_MAX;
  for (std::uint32_t i = 0; i < info_.memory_types.size(); ++i) {
    if ((type_bits & (1u << i)) == 0) continue;
    const VkMemoryPropertyFlags f = info_.memory_types[i].flags;
    if ((f & required) != required) continue;
    if ((f & preferred) == preferred) return i;
    if (fallback == UINT32_MAX) fallback = i;
  }
  if (fallback == UINT32_MAX) {
    fail(ErrorKind::kUnsupported, "GPU '{}' has no memory type with flags 0x{:x} (allowed types 0x{:x})", info_.name,
         static_cast<unsigned>(required), type_bits);
  }
  return fallback;
}

void Context::run_once(const std::function<void(VkCommandBuffer)>& record, std::uint64_t timeout_ns) {
  VK_CHECK(vkResetCommandBuffer(once_cmd_, 0));
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK(vkBeginCommandBuffer(once_cmd_, &bi));
  record(once_cmd_);
  VK_CHECK(vkEndCommandBuffer(once_cmd_));
  VK_CHECK(vkResetFences(device_, 1, &once_fence_));
  VkCommandBufferSubmitInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
  cbi.commandBuffer = once_cmd_;
  VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
  si.commandBufferInfoCount = 1;
  si.pCommandBufferInfos = &cbi;
  VK_CHECK(vkQueueSubmit2(queue_, 1, &si, once_fence_));
  wait_fence(device_, once_fence_, timeout_ns);
}

Context::ClockPair Context::sample_clocks() const {
  HEARTH_CHECK(calibrated_timestamps_enabled_);
  VkCalibratedTimestampInfoEXT infos[2] = {{VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT},
                                           {VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT}};
  infos[0].timeDomain = VK_TIME_DOMAIN_DEVICE_EXT;
  infos[1].timeDomain = VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_EXT;
  std::uint64_t ts[2] = {};
  std::uint64_t deviation = 0;
  VK_CHECK(vkGetCalibratedTimestampsEXT(device_, 2, infos, ts, &deviation));
  // The QPC domain returns raw counter ticks; convert exactly as now_ns() does.
  const std::int64_t f = qpc_frequency();
  const auto ticks = static_cast<std::int64_t>(ts[1]);
  const std::int64_t ns = (ticks / f) * 1'000'000'000 + (ticks % f) * 1'000'000'000 / f;
  const std::uint64_t mask = info_.timestamp_valid_bits >= 64 ? ~0ull : ((1ull << info_.timestamp_valid_bits) - 1);
  return {ts[0] & mask, ns, deviation};
}

void wait_fence(VkDevice device, VkFence fence, std::uint64_t timeout_ns) {
  const VkResult r = vkWaitForFences(device, 1, &fence, VK_TRUE, timeout_ns);
  if (r == VK_TIMEOUT) {
    fail(ErrorKind::kDeviceLost, "the GPU did not finish within {:.1f} s; treating it as hung",
         static_cast<double>(timeout_ns) / 1e9);
  }
  VK_CHECK(r);
}

}  // namespace hearth::gpu
