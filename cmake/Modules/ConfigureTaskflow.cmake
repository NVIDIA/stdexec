option(STDEXEC_ENABLE_TASKFLOW "Enable TaskFlow targets" OFF)

if(STDEXEC_ENABLE_TASKFLOW)
  rapids_cpm_find(Taskflow 3.7.0
  CPM_ARGS
    GITHUB_REPOSITORY taskflow/taskflow
    GIT_TAG v3.7.0
    OPTIONS "TF_BUILD_TESTS OFF"
  )
  file(GLOB_RECURSE taskflow_pool include/execpools/taskflow/*.hpp)
  add_library(taskflow_pool INTERFACE ${taskflowexec_sources})
  target_compile_definitions(taskflow_pool INTERFACE STDEXEC_ENABLE_TASKFLOW)
  list(APPEND stdexec_export_targets taskflow_pool)
  add_library(STDEXEC::taskflow_pool ALIAS taskflow_pool)

  target_link_libraries(taskflow_pool
    INTERFACE
    STDEXEC::stdexec
    Taskflow
  )
endif()
