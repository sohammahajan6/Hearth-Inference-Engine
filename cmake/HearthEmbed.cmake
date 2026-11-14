# Embed a file in the executable as a byte array, regenerated when the file
# changes (the page of `hearth ui`, cli/ui/index.html).
#
# hearth_embed_file(<target> <file> <function>)
#   generates <function>.cpp defining `std::string_view hearth::cli::<function>()`.

function(hearth_embed_file target file function)
  get_filename_component(abs "${file}" ABSOLUTE)
  set(out "${CMAKE_CURRENT_BINARY_DIR}/embedded/${function}.cpp")
  add_custom_command(
    OUTPUT "${out}"
    COMMAND "${CMAKE_COMMAND}" "-DIN=${abs}" "-DOUT=${out}" "-DFUNCTION=${function}"
            -P "${PROJECT_SOURCE_DIR}/cmake/embed_file.cmake"
    DEPENDS "${abs}" "${PROJECT_SOURCE_DIR}/cmake/embed_file.cmake"
    COMMENT "embed ${file}"
    VERBATIM)
  target_sources(${target} PRIVATE "${out}")
endfunction()
