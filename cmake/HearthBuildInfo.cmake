# Build identity for `hearth --version`: git commit, dirty flag, compiler,
# build type and flags. Regenerated on every build (cheaply), and the output
# file is only rewritten when the content changes, so it rebuilds nothing
# unless the identity actually changed.

set(HEARTH_BUILD_INFO_CPP "${CMAKE_BINARY_DIR}/generated/build_info.cpp")

if(CMAKE_BUILD_TYPE)
  string(TOUPPER "${CMAKE_BUILD_TYPE}" _bt)
  set(_flags "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_bt}}")
else()
  set(_flags "${CMAKE_CXX_FLAGS}")
endif()
string(STRIP "${_flags}" _flags)

add_custom_target(hearth_build_info
  COMMAND ${CMAKE_COMMAND}
    -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
    -DOUTPUT=${HEARTH_BUILD_INFO_CPP}
    -DVERSION=${PROJECT_VERSION}
    "-DCOMPILER=${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
    -DBUILD_TYPE=${CMAKE_BUILD_TYPE}
    "-DFLAGS=${_flags}"
    -DSANITIZE=${HEARTH_SANITIZE}
    -P ${PROJECT_SOURCE_DIR}/cmake/WriteBuildInfo.cmake
  BYPRODUCTS ${HEARTH_BUILD_INFO_CPP}
  COMMENT "build info"
  VERBATIM)
