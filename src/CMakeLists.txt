# volk loads Vulkan functions at run time (no link-time dependency on
# vulkan-1.lib). Third-party C code: built without our strict warning flags.
add_library(hearth_volk STATIC "${HEARTH_THIRD_PARTY}/volk/volk.c")
target_include_directories(hearth_volk SYSTEM PUBLIC
  "${HEARTH_THIRD_PARTY}/volk" "${HEARTH_THIRD_PARTY}/Vulkan-Headers/include")
target_compile_definitions(hearth_volk PUBLIC VK_NO_PROTOTYPES)

add_library(hearth_core STATIC
  core/clock.cpp
  core/cpu_features.cpp
  core/environment.cpp
  core/error.cpp
  core/json_writer.cpp
  core/log.cpp
  core/mapped_file.cpp
  core/page_buffer.cpp
  core/pdh.cpp
  core/platform.cpp
  core/kquant.cpp
  core/quant.cpp
  core/thread_pool.cpp
  core/trace.cpp
  core/utf8.cpp
  cpu/attention.cpp
  cpu/attention_avx512.cpp
  cpu/matvec.cpp
  cpu/ref_ops.cpp
  cpu/matvec_avx512.cpp
  cpu/packed_arena.cpp
  cpu/kquant.cpp
  cpu/kquant_avx512.cpp
  cpu/qgemm.cpp
  cpu/qgemm_avx512.cpp
  cpu/probe_kernels.cpp
  cpu/probe_kernels_avx2.cpp
  cpu/probe_kernels_avx512.cpp
  gguf/gguf.cpp
  gguf/types.cpp
  api/engine.cpp
  gpu/adapters.cpp
  gpu/buffer.cpp
  gpu/context.cpp
  gpu/hang_ledger.cpp
  gpu/matvec.cpp
  gpu/pipeline.cpp
  gpu/weight_arena.cpp
  gpu/sync.cpp
  gpu/timestamps.cpp
  gpu/vk.cpp
  model/dequant.cpp
  model/generate.cpp
  model/speculative.cpp
  model/llama.cpp
  model/llama_gpu.cpp
  model/llama_ref.cpp
  sampling/sampler.cpp
  tokenizer/chat.cpp
  tokenizer/tokenizer.cpp
  tokenizer/unicode.cpp
  tokenizer/unicode_tables.cpp
  probe/layout.cpp
  probe/probe.cpp
  probe/probe_cpu.cpp
  probe/probe_qgemm.cpp
  probe/probe_gpu.cpp
  probe/probe_gpu_qgemm.cpp
  probe/probe_gpu_placement.cpp
  probe/report.cpp
  "${HEARTH_BUILD_INFO_CPP}")
add_dependencies(hearth_core hearth_build_info)
target_include_directories(hearth_core PUBLIC "${PROJECT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(hearth_core PUBLIC hearth_volk PRIVATE dxgi synchronization pdh)
hearth_set_defaults(hearth_core)

hearth_isa_sources(avx2 cpu/probe_kernels_avx2.cpp)
hearth_isa_sources(avx512 cpu/attention_avx512.cpp cpu/probe_kernels_avx512.cpp cpu/matvec_avx512.cpp
  cpu/qgemm_avx512.cpp cpu/kquant_avx512.cpp)

hearth_add_shaders(hearth_core
  gpu/shaders/attention.comp
  gpu/shaders/bias_add.comp
  gpu/shaders/hello_add.comp
  gpu/shaders/matvec_q4_0_aos.comp
  gpu/shaders/matvec_q4_0_soa.comp
  gpu/shaders/matvec_q8_0_aos.comp
  gpu/shaders/matvec_q8_0_soa.comp
  gpu/shaders/probe_dot8.comp
  gpu/shaders/probe_empty.comp
  gpu/shaders/probe_fma16.comp
  gpu/shaders/probe_fma32.comp
  gpu/shaders/probe_fma_detect.comp
  gpu/shaders/probe_div_exact.comp
  gpu/shaders/probe_read.comp
  gpu/shaders/qgemm_q4_0.comp
  gpu/shaders/qgemm_q8_0.comp
  gpu/shaders/qgemv_q4_0.comp
  gpu/shaders/qgemv_q8_0.comp
  gpu/shaders/residual_add.comp
  gpu/shaders/rmsnorm.comp
  gpu/shaders/rope.comp
  gpu/shaders/silu_mul.comp
  gpu/shaders/quantize_act.comp)

# The local app server of `hearth ui`:
# the HTTP request parser (fuzzed: tests/fuzz/fuzz_http.cpp) and the loopback
# server. Winsock and BCrypt are part of Windows.
add_library(hearth_ui STATIC
  ui/command_line.cpp
  ui/http.cpp
  ui/server.cpp)
target_include_directories(hearth_ui PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(hearth_ui PUBLIC hearth_core PRIVATE ws2_32 bcrypt)
hearth_set_defaults(hearth_ui)
