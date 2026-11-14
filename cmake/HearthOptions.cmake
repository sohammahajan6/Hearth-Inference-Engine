# Compiler settings shared by every first-party target.
#
# Two compilers build this code: MSVC (the product and all benchmarks) and
# llvm-mingw clang (sanitizers and lint). Both must be warning-clean.
# No fast-math anywhere: results are compared against llama.cpp with tight
# tolerances, and reassociating float math would break bit-reproducibility.

function(hearth_set_defaults target)
  target_compile_features(${target} PUBLIC cxx_std_20)
  target_compile_definitions(${target} PRIVATE
    NOMINMAX WIN32_LEAN_AND_MEAN UNICODE _UNICODE _WIN32_WINNT=0x0A00)
  if(MSVC)
    # C4324 ("structure was padded due to alignment specifier") only reports
    # padding that an alignas() asked for, e.g. to keep hot atomics on
    # separate cache lines; it never indicates a bug.
    target_compile_options(${target} PRIVATE
      /W4 /WX /permissive- /utf-8 /EHsc /Zc:__cplusplus /Zc:preprocessor
      /Zc:inline /Zc:throwingNew /volatile:iso /fp:precise /Zi /Gy /diagnostics:caret
      /wd4324)
    if(HEARTH_ANALYZE)
      # C26xxx Core Guidelines checks are off: they flag idiomatic low-level
      # code (raw pointers into mapped files, unions in kernels) wholesale.
      target_compile_options(${target} PRIVATE /analyze /analyze:external- /wd26812)
    endif()
    target_link_options(${target} PRIVATE /DEBUG:FULL /OPT:REF /OPT:ICF)
  else()
    target_compile_options(${target} PRIVATE
      -Wall -Wextra -Wpedantic -Werror -Wshadow -Wnon-virtual-dtor
      -Wimplicit-fallthrough -Wno-missing-field-initializers -g
      -ffp-contract=off)
    if(HEARTH_SANITIZE)
      # use-after-return=never: clang's default (=runtime) gives instrumented
      # functions a dynamically allocated frame, so every call gets its own
      # `sub rsp, 32` ... `add rsp, 32`. When such a call throws, the Windows
      # unwinder resumes the landing pad with the call-site stack pointer and
      # the 32 bytes are never given back: a function that catches exceptions
      # in a loop overflows its 1 MiB stack after ~31,000 catches
      #. Nothing is lost: the runtime check is off by
      # default on Windows anyway.
      target_compile_options(${target} PRIVATE -fsanitize=address,undefined
        -fno-sanitize-recover=undefined -fno-omit-frame-pointer
        -fsanitize-address-use-after-return=never)
      target_link_options(${target} PRIVATE -fsanitize=address,undefined)
    endif()
  endif()
endfunction()

# Per-ISA translation units: the base code is plain x86-64, and kernels for
# newer instruction sets live in their own .cpp files compiled with these
# flags. They are only called after CPUID + XGETBV confirm that the CPU and OS
# support them, so the binary never executes an instruction the CPU lacks.
function(hearth_isa_sources isa)
  if(isa STREQUAL "avx2")
    if(MSVC)
      set(flags /arch:AVX2)
    else()
      set(flags -mavx2 -mfma -mf16c -mbmi2)
    endif()
  elseif(isa STREQUAL "avx512")
    if(MSVC)
      set(flags /arch:AVX512)
    else()
      set(flags -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni -mfma -mf16c -mbmi2)
    endif()
  else()
    message(FATAL_ERROR "unknown ISA '${isa}'")
  endif()
  set_source_files_properties(${ARGN} PROPERTIES COMPILE_OPTIONS "${flags}")
endfunction()
