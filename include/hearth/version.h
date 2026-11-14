// Build identity of the hearth library, as printed by `hearth --version`.
#pragma once

namespace hearth {

struct BuildInfo {
  const char* version;     // SemVer of the project, from CMakeLists.txt
  const char* commit;      // short git hash, with "-dirty" if uncommitted changes were built
  const char* compiler;    // e.g. "MSVC 19.44.35207.0"
  const char* build_type;  // Release, Debug, RelWithDebInfo
  const char* flags;       // compiler flags for this build type
  bool sanitized;          // built with ASan + UBSan
};

const BuildInfo& build_info();

}  // namespace hearth
