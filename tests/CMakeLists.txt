# One test executable, organised in doctest test suites that map to the
# tiers in scripts/test.ps1:
#   quick - seconds; no model files needed (the GPU smoke tests are here too)
#   model - needs the files from scripts/fetch-models.ps1
add_executable(hearth_tests
  test_main.cpp
  unit/chat_sampling_test.cpp
  unit/cpu_features_test.cpp
  unit/fp16_test.cpp
  unit/fp16_hw_test.cpp
  unit/fuzz_replay_test.cpp
  unit/engine_test.cpp
  unit/gguf_test.cpp
  unit/hang_ledger_test.cpp
  unit/json_writer_test.cpp
  unit/log_test.cpp
  unit/llama_ref_test.cpp
  unit/probe_report_test.cpp
  unit/speculative_test.cpp
  unit/thread_pool_test.cpp
  unit/tokenizer_test.cpp
  unit/trace_test.cpp
  unit/ui_app_test.cpp
  unit/ui_http_test.cpp
  integration/gpu_arena_test.cpp
  integration/gpu_prefill_decode_test.cpp
  integration/gpu_probe_shaders_test.cpp
  integration/ui_server_test.cpp
  kernels/attention_test.cpp
  kernels/dequant_more_test.cpp
  kernels/gpu_ops_test.cpp
  kernels/gpu_qgemm_test.cpp
  kernels/gpu_qgemv_test.cpp
  kernels/gpu_quantize_test.cpp
  kernels/gpu_trace_test.cpp
  kernels/kquant_test.cpp
  kernels/matvec_test.cpp
  kernels/packed_arena_test.cpp
  kernels/qgemm_test.cpp
  kernels/quant_hw_test.cpp
  integration/gpu_smoke_test.cpp
  hw_reference_avx2.cpp
  # The app of `hearth ui`, tested through its routes (unit/ui_app_test.cpp).
  "${PROJECT_SOURCE_DIR}/cli/ui_app.cpp")
hearth_isa_sources(avx2 hw_reference_avx2.cpp)
hearth_embed_file(hearth_tests "${PROJECT_SOURCE_DIR}/cli/ui/index.html" ui_index_html)
target_include_directories(hearth_tests SYSTEM PRIVATE "${HEARTH_THIRD_PARTY}/doctest")
target_include_directories(hearth_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/cli")
target_link_libraries(hearth_tests PRIVATE hearth_core hearth_ui ws2_32 shell32 ole32 uuid)
hearth_set_defaults(hearth_tests)
target_compile_definitions(hearth_tests PRIVATE
  "HEARTH_REPO_DIR=\"${PROJECT_SOURCE_DIR}\"")

add_test(NAME quick COMMAND hearth_tests --test-suite=quick)

# Fuzz targets. Each compiles the sources it
# tests itself instead of linking hearth_core, so that in the ASan build only
# the code under test carries coverage instrumentation. Without the clang
# sanitizer build, the same files build a replay tool (main() in the target).
function(hearth_add_fuzz_target name main)
  set(sources)
  foreach(s IN LISTS ARGN)
    list(APPEND sources "${PROJECT_SOURCE_DIR}/src/${s}")
  endforeach()
  add_executable(${name} ${main} ${sources})
  target_include_directories(${name} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/include")
  target_compile_definitions(${name} PRIVATE "HEARTH_REPO_DIR=\"${PROJECT_SOURCE_DIR}\"")
  hearth_set_defaults(${name})
  if(HEARTH_SANITIZE AND NOT MSVC)
    # The clang driver refuses -fsanitize=fuzzer for the windows-gnu target, so
    # the coverage instrumentation is requested directly and llvm-mingw's
    # libFuzzer runtime (which provides main) is linked by path.
    execute_process(COMMAND "${CMAKE_CXX_COMPILER}" --print-resource-dir
                    OUTPUT_VARIABLE clang_resource_dir OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(libfuzzer "${clang_resource_dir}/lib/windows/libclang_rt.fuzzer-x86_64.a")
    if(NOT EXISTS "${libfuzzer}")
      message(FATAL_ERROR "libFuzzer runtime not found at ${libfuzzer}")
    endif()
    target_compile_definitions(${name} PRIVATE HEARTH_LIBFUZZER)
    # Beyond the default ASan/UBSan set: unsigned wrap-around and implicit
    # truncation are not undefined behaviour, but in parsers of sizes, offsets
    # and UTF-8 they are exactly the bugs to find. libc++'s hash functions wrap
    # on purpose, so its headers are excluded.
    target_compile_options(${name} PRIVATE
      -fsanitize-coverage=inline-8bit-counters,pc-table,trace-cmp
      -fsanitize=unsigned-integer-overflow,implicit-integer-truncation
      "-fsanitize-ignorelist=${CMAKE_CURRENT_SOURCE_DIR}/fuzz/ubsan-ignorelist.txt"
      -fno-sanitize-recover=all)
    target_link_options(${name} PRIVATE -fsanitize=unsigned-integer-overflow,implicit-integer-truncation)
    target_link_libraries(${name} PRIVATE "${libfuzzer}")
  endif()
endfunction()

hearth_add_fuzz_target(hearth_fuzz_gguf fuzz/fuzz_gguf.cpp
  core/error.cpp core/json_writer.cpp core/mapped_file.cpp core/platform.cpp core/utf8.cpp
  gguf/gguf.cpp gguf/types.cpp model/llama.cpp)
# The HTTP request parser of the `hearth ui` server.
hearth_add_fuzz_target(hearth_fuzz_http fuzz/fuzz_http.cpp
  core/error.cpp core/platform.cpp core/utf8.cpp ui/http.cpp)
# The tokenizer target loads the real vocabulary from the model file.
hearth_add_fuzz_target(hearth_fuzz_tokenizer fuzz/fuzz_tokenizer.cpp
  core/error.cpp core/mapped_file.cpp core/platform.cpp core/utf8.cpp gguf/gguf.cpp gguf/types.cpp
  tokenizer/tokenizer.cpp tokenizer/unicode.cpp tokenizer/unicode_tables.cpp)
