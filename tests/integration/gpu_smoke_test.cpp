// GPU smoke test: the whole Vulkan path end to end, under the validation
// layer when scripts/test.ps1 enables it.
#include <doctest/doctest.h>

#include <cstring>
#include <vector>

#include "core/log.h"
#include "gpu/buffer.h"
#include "gpu/context.h"
#include "gpu/pipeline.h"
#include "test_util.h"

using namespace hearth;

TEST_SUITE("quick") {
  TEST_CASE("gpu: hello_add computes c = a + 2b with zero validation messages") {
    if (!gpu::vulkan_available()) {
      MESSAGE("Vulkan not available; skipped");
      return;
    }
    gpu::Context ctx(test::gpu_test_options());
    constexpr std::uint32_t n = 1000;  // deliberately not a multiple of the workgroup size (64)
    gpu::Buffer a(ctx, n * sizeof(float), gpu::MemoryKind::kHostCoherent);
    gpu::Buffer b(ctx, n * sizeof(float), gpu::MemoryKind::kHostCoherent);
    gpu::Buffer c(ctx, n * sizeof(float), gpu::MemoryKind::kHostCached);
    auto* pa = static_cast<float*>(a.mapped());
    auto* pb = static_cast<float*>(b.mapped());
    for (std::uint32_t i = 0; i < n; ++i) {
      pa[i] = static_cast<float>(i);
      pb[i] = 0.25f * static_cast<float>(i) - 7.0f;
    }
    std::memset(c.mapped(), 0xff, n * sizeof(float));  // NaN pattern: unwritten outputs cannot pass

    gpu::Pipeline pipe(ctx, {.shader = "hello_add", .buffers = 3, .push_constant_bytes = 4});
    gpu::DescriptorPool pool(ctx, 1, 3);
    VkDescriptorSet set = pool.allocate(pipe, {&a, &b, &c});
    ctx.run_once([&](VkCommandBuffer cmd) {
      pipe.dispatch(cmd, set, &n, sizeof(n), (n + 63) / 64);
      gpu::compute_to_host_barrier(cmd);
    });
    const auto* pc = static_cast<const float*>(c.mapped());
    for (std::uint32_t i = 0; i < n; ++i) {
      REQUIRE(pc[i] == pa[i] + 2.0f * pb[i]);  // exact: small integers and quarters
    }
    CHECK(ctx.validation_message_count() == 0);
    // (doctest's MESSAGE expands to `builder * expr`, so build the text first.)
    const std::string note =
        std::string("validation layer ") + (ctx.validation_enabled() ? "on" : "off") + ", device " + ctx.info().name;
    MESSAGE(note);
  }

  TEST_CASE("gpu: the validation layer really reports invalid API usage") {
    // Core validation: a zero-size buffer violates VUID-VkBufferCreateInfo-size-00912.
    if (!gpu::vulkan_available() || !test::gpu_test_options().validation) {
      MESSAGE("validation layer not enabled; skipped");
      return;
    }
    gpu::Context ctx(test::gpu_test_options());
    // The report is expected: catch it and check its text instead of printing it.
    const LogCapture log;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = 0;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkBuffer buffer = VK_NULL_HANDLE;
    if (vkCreateBuffer(ctx.device(), &bci, nullptr, &buffer) == VK_SUCCESS)
      vkDestroyBuffer(ctx.device(), buffer, nullptr);
    CHECK(ctx.validation_message_count() > 0);
    CHECK(log.count("vulkan validation: ") == static_cast<int>(ctx.validation_message_count()));
    CHECK(log.count("VUID-VkBufferCreateInfo-size-00912") > 0);
    CHECK(log.lines() == static_cast<int>(ctx.validation_message_count()));
  }

  TEST_CASE("gpu: the validation layer really reports problems (missing barrier is caught)") {
    // A "0 validation messages" result only means something if the layer is
    // wired up and synchronization validation is on. Two dispatches that
    // write the same buffer with no barrier between them are a
    // write-after-write hazard: harmless for the GPU, but it must be reported.
    if (!gpu::vulkan_available() || !test::gpu_test_options().validation) {
      MESSAGE("validation layer not enabled; skipped");
      return;
    }
    gpu::Context ctx(test::gpu_test_options());
    constexpr std::uint32_t n = 256;
    gpu::Buffer a(ctx, n * sizeof(float), gpu::MemoryKind::kHostCoherent);
    gpu::Buffer c(ctx, n * sizeof(float), gpu::MemoryKind::kHostCoherent);
    std::memset(a.mapped(), 0, n * sizeof(float));
    gpu::Pipeline pipe(ctx, {.shader = "hello_add", .buffers = 3, .push_constant_bytes = 4});
    gpu::DescriptorPool pool(ctx, 1, 3);
    VkDescriptorSet set = pool.allocate(pipe, {&a, &a, &c});
    // The hazard reports are expected: catch them and check their text.
    const LogCapture log;
    ctx.run_once([&](VkCommandBuffer cmd) {
      vkCmdFillBuffer(cmd, c.handle(), 0, VK_WHOLE_SIZE, 0);
      vkCmdFillBuffer(cmd, c.handle(), 0, VK_WHOLE_SIZE, 1);  // no barrier: transfer write-after-write
    });
    const std::uint32_t after_transfer = ctx.validation_message_count();
    CHECK(after_transfer > 0);
    CHECK(log.count("vkCmdFillBuffer(): WRITE_AFTER_WRITE hazard") > 0);
    ctx.run_once([&](VkCommandBuffer cmd) {
      pipe.dispatch(cmd, set, &n, sizeof(n), n / 64);
      pipe.dispatch(cmd, set, &n, sizeof(n), n / 64);  // no barrier: compute write-after-write on c
    });
    CHECK(ctx.validation_message_count() > after_transfer);
    CHECK(log.count("vkCmdDispatch(): WRITE_AFTER_WRITE hazard") > 0);
    CHECK(log.count("vulkan validation: ") == static_cast<int>(ctx.validation_message_count()));
    CHECK(log.lines() == static_cast<int>(ctx.validation_message_count()));
  }

  TEST_CASE("gpu: device list is deduplicated by UUID") {
    if (!gpu::vulkan_available()) return;
    const auto devices = gpu::list_devices();
    for (std::size_t i = 0; i < devices.size(); ++i) {
      for (std::size_t j = i + 1; j < devices.size(); ++j) CHECK(devices[i].uuid != devices[j].uuid);
    }
  }
}
