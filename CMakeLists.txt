cmake_minimum_required(VERSION 3.28)
project(hearth VERSION 2.0.0 LANGUAGES C CXX)

# Use scripts/build.ps1 instead of calling CMake directly: it selects the
# pinned toolchain (MSVC 14.44 or llvm-mingw clang) and the right options.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# MSVC: the C++ runtime is linked statically, so hearth.exe needs no Visual
# C++ runtime DLLs on the PC it runs on.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

option(HEARTH_SANITIZE "AddressSanitizer + UndefinedBehaviorSanitizer (clang build only)" OFF)
option(HEARTH_ANALYZE "MSVC /analyze static analysis" OFF)

set(HEARTH_THIRD_PARTY "${PROJECT_SOURCE_DIR}/third_party" CACHE PATH "Pinned third-party sources and tools")
set(HEARTH_GLSLANG "${HEARTH_THIRD_PARTY}/install/glslang/bin/glslang.exe" CACHE FILEPATH "glslang shader compiler")

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Windows")
  message(FATAL_ERROR "hearth targets Windows only.")
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(HearthOptions)
include(HearthShaders)
include(HearthBuildInfo)
include(HearthEmbed)

add_subdirectory(src)
add_subdirectory(cli)

enable_testing()
add_subdirectory(tests)
