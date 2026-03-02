option(STDEXEC_ENABLE_TASKFLOW "Enable TaskFlow targets" OFF)

if(STDEXEC_ENABLE_TASKFLOW)
  rapids_cpm_find(Taskflow 3.7.0
  CPM_ARGS
    GITHUB_REPOSITORY taskflow/taskflow
    GIT_TAG v3.7.0
    OPTIONS "TF_BUILD_TESTS OFF"
  )
  file(GLOB_RECURSE taskflowexec_headers CONFIGURE_DEPENDS include/exec/taskflow/*.hpp)
  add_library(taskflowexec INTERFACE ${taskflowexec_headers})
  list(APPEND stdexec_export_targets taskflowexec)
  add_library(STDEXEC::taskflowexec ALIAS taskflowexec)

  # These aliases are provided for backwards compatibility with the old target names
  add_library(taskflow_pool ALIAS taskflowexec)
  add_library(STDEXEC::taskflow_pool ALIAS taskflowexec)

  target_sources(taskflowexec
  PUBLIC
    FILE_SET headers
    TYPE HEADERS
    BASE_DIRS include
    FILES ${taskflowexec_headers}
  )
  target_compile_definitions(taskflowexec INTERFACE STDEXEC_ENABLE_TASKFLOW)

  target_link_libraries(stdexec INTERFACE
    Taskflow
  )

  target_link_libraries(taskflowexec INTERFACE
    STDEXEC::stdexec
  )
endif()
