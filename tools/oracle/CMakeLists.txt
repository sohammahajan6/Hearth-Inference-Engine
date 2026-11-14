# hetero-oracle-dump: llama.cpp's own forward pass, with named intermediate
# tensors written to disk for layer-by-layer comparison with hetero
# (docs/architecture/05-forward-pass.md, "The oracle tools"; the output format
# is in README.md next to this file).
#
# A standalone CMake project, separate from the hetero build. It compiles the
# pinned llama.cpp source as static libraries with the CPU backend only and
# links one executable against them. Build it with
#   pwsh -File scripts\setup.ps1 -Only oracle
# which uses MSVC + Ninja, builds in third_party\build\oracle and installs
# third_party\install\oracle\bin\hetero-oracle-dump.exe.
cmake_minimum_required(VERSION 3.28)
project(hetero_oracle LANGUAGES C CXX)

set(LLAMA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/llama.cpp"
    CACHE PATH "llama.cpp source tree at the pinned commit (scripts/pins.psd1)")
file(REAL_PATH "${LLAMA_SOURCE_DIR}" llama_src)
if(NOT EXISTS "${llama_src}/include/llama.h" OR NOT EXISTS "${llama_src}/CMakeLists.txt")
  message(FATAL_ERROR "LLAMA_SOURCE_DIR='${LLAMA_SOURCE_DIR}' is not a llama.cpp source tree. "
                      "Fetch the pinned one with: pwsh -File scripts\\setup.ps1 -Only deps")
endif()

# What the oracle is (FORCE: a stale cache or a stray -D cannot change it):
# static libraries, the CPU backend only, llama.cpp's core library and its
# `llama-common` library (for the chat template's Jinja engine), without its
# tools, examples, server, tests, subprocesses or downloads.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
foreach(backend CUDA HIP MUSA VULKAN SYCL OPENCL WEBGPU BLAS RPC)
  set(GGML_${backend} OFF CACHE BOOL "" FORCE)
endforeach()
foreach(opt LLAMA_BUILD_TESTS LLAMA_BUILD_TOOLS LLAMA_BUILD_EXAMPLES LLAMA_BUILD_SERVER
            LLAMA_BUILD_APP LLAMA_BUILD_MTMD LLAMA_CURL LLAMA_OPENSSL LLAMA_LLGUIDANCE LLAMA_SUBPROCESS
            GGML_BUILD_TESTS GGML_BUILD_EXAMPLES)
  set(${opt} OFF CACHE BOOL "" FORCE)
endforeach()
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)

# Defaults that can be overridden with -D (for another machine, say).
#
# Instruction set: fixed instead of probed. With MSVC, GGML_NATIVE probes the
# build machine and only finds AVX-512F/BW. This set equals ggml's "icelake"
# CPU variant, which is the variant the official b11960 CPU release loads on
# Ice Lake / Tiger Lake CPUs. AVX-VNNI stays off: those CPUs have AVX-512 VNNI but
# not the VEX-encoded AVX-VNNI.
set(GGML_NATIVE OFF CACHE BOOL "ggml: optimize the build for the current system")
foreach(isa SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VBMI AVX512_VNNI)
  set(GGML_${isa} ON CACHE BOOL "ggml: enable ${isa}")
endforeach()
foreach(isa AVX_VNNI AVX512_BF16 AMX_TILE AMX_INT8 AMX_BF16)
  set(GGML_${isa} OFF CACHE BOOL "ggml: enable ${isa}")
endforeach()
# ggml's own thread pool rather than OpenMP: no OpenMP runtime DLL is needed.
set(GGML_OPENMP OFF CACHE BOOL "ggml: use OpenMP")
set(GGML_CCACHE OFF CACHE BOOL "ggml: use ccache if available")

# SYSTEM: llama.cpp's headers do not take part in this project's warning level.
add_subdirectory("${llama_src}" llama.cpp EXCLUDE_FROM_ALL SYSTEM)

# llama.cpp's own source files compile with no warnings shown (/w). They are
# pinned third-party code that this project must not edit (it is the oracle:
# changing it would change what we compare against), and its MSVC warnings
# (C4068 on its GCC pragmas, C4244 inside the standard library for its
# int64 -> uint32 pairs) are upstream's to fix. Warning switches do not change
# the generated code. The two oracle executables below, which are this
# project's code, keep /W4 /WX.
# /w alone is not enough: MSVC's standard library re-enables level-3 warnings
# inside its own headers (yvals_core.h: #pragma warning(push,
# _STL_WARNING_LEVEL)), so a conversion in a template that llama.cpp
# instantiates is still reported there. _STL_EXTRA_DISABLED_WARNINGS is the
# library's documented hook for that list: C4244 and C4267, the conversion
# warnings llama.cpp's code triggers.
function(hetero_oracle_targets_in dir out)
  get_property(targets DIRECTORY "${dir}" PROPERTY BUILDSYSTEM_TARGETS)
  get_property(subdirs DIRECTORY "${dir}" PROPERTY SUBDIRECTORIES)
  foreach(sub IN LISTS subdirs)
    hetero_oracle_targets_in("${sub}" sub_targets)
    list(APPEND targets ${sub_targets})
  endforeach()
  set(${out} ${targets} PARENT_SCOPE)
endfunction()
if(MSVC)
  hetero_oracle_targets_in("${llama_src}" llama_targets)
  foreach(t IN LISTS llama_targets)
    get_target_property(type ${t} TYPE)
    if(type MATCHES "^(STATIC_LIBRARY|SHARED_LIBRARY|MODULE_LIBRARY|OBJECT_LIBRARY|EXECUTABLE)$")
      target_compile_options(${t} PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:/w>)
      target_compile_definitions(${t} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:_STL_EXTRA_DISABLED_WARNINGS=4244 4267>")
    endif()
  endforeach()
endif()

add_executable(hetero-oracle-dump dump.cpp)
target_link_libraries(hetero-oracle-dump PRIVATE llama)
target_compile_features(hetero-oracle-dump PRIVATE cxx_std_20)
if(MSVC)
  # The same warning and floating-point settings as hetero's own code
  # (cmake/HeteroOptions.cmake). The manifest makes UTF-8 the process code
  # page, as for hetero.exe, so argv and file names are UTF-8.
  target_compile_definitions(hetero-oracle-dump PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
  target_compile_options(hetero-oracle-dump PRIVATE
    /W4 /WX /permissive- /utf-8 /EHsc /Zc:__cplusplus /Zc:preprocessor /fp:precise)
  target_sources(hetero-oracle-dump PRIVATE hetero-oracle-dump.manifest)
endif()

# hetero-oracle-chat: llama.cpp's own rendering of the chat template (its
# Jinja engine is part of the `llama-common` library) and its tokenization of
# the result. An unknown name in target_link_libraries would silently become
# a plain library file name, so the target is checked first.
foreach(t llama llama-common)
  if(NOT TARGET ${t})
    message(FATAL_ERROR "llama.cpp did not define the CMake target '${t}'; did the pinned version rename it?")
  endif()
endforeach()
add_executable(hetero-oracle-chat chat.cpp)
target_link_libraries(hetero-oracle-chat PRIVATE llama-common llama)
target_compile_features(hetero-oracle-chat PRIVATE cxx_std_20)
if(MSVC)
  target_compile_definitions(hetero-oracle-chat PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
  target_compile_options(hetero-oracle-chat PRIVATE
    /W4 /WX /permissive- /utf-8 /EHsc /Zc:__cplusplus /Zc:preprocessor /fp:precise)
  target_sources(hetero-oracle-chat PRIVATE hetero-oracle-dump.manifest)
endif()

install(TARGETS hetero-oracle-dump hetero-oracle-chat RUNTIME DESTINATION bin COMPONENT oracle)
