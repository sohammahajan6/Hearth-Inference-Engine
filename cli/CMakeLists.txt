add_executable(hearth
  main.cpp
  args.cpp
  cmd_bench.cpp
  cmd_chat.cpp
  cmd_dump.cpp
  cmd_info.cpp
  cmd_model_info.cpp
  cmd_ppl.cpp
  cmd_probe.cpp
  cmd_run.cpp
  cmd_tokenize.cpp
  cmd_ui.cpp
  session.cpp
  ui_app.cpp)
target_link_libraries(hearth PRIVATE hearth_core hearth_ui ole32 shell32 uuid)
# The page of `hearth ui`, compiled in.
hearth_embed_file(hearth ui/index.html ui_index_html)
hearth_set_defaults(hearth)
target_include_directories(hearth PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

# Application manifest: UTF-8 as the process code page (so argv and the C
# runtime speak UTF-8), long-path awareness, and Windows 10/11 compatibility.
if(MSVC)
  target_sources(hearth PRIVATE hearth.manifest)
else()
  enable_language(RC)
  target_sources(hearth PRIVATE hearth.rc)
endif()
