option(STDEXEC_ENABLE_TBB "Enable TBB targets" OFF)

include(CheckIncludeFileCXX)
rapids_find_package(
  TBB QUIET
  COMPONENTS tbb
  BUILD_EXPORT_SET stdexec-exports
  INSTALL_EXPORT_SET stdexec-exports
)

# If TBB is available and the stdlib is libstdc++, then a #include of <execution> will
# pull in TBB headers, creating a link-time dependency on TBB.
if (TBB_FOUND)
  if (NOT STDEXEC_ENABLE_TBB)
    include(CheckCXXSymbolExists)
    check_cxx_symbol_exists(__GLIBCXX__ "ciso646" _glibcxx_version_defined)
    if (_glibcxx_version_defined)
      set(STDEXEC_ENABLE_TBB ON)
      message(STATUS "Enabling TBB support in stdexec because <tbb/tbb.h> is available and libstdc++ is used.")
    endif()
  endif()
else()
  set(STDEXEC_ENABLE_TBB OFF)
endif()

# TBB does not sanitize cleanly. Disable TBB support if any sanitizer is enabled.
string(FIND "${CMAKE_CXX_FLAGS}" "-fsanitize" _fsanitize_flag_index)
if (_fsanitize_flag_index GREATER -1)
  message(STATUS "Disabling TBB support in stdexec because sanitizers are enabled.")
  set(STDEXEC_ENABLE_TBB OFF)
endif()

if (STDEXEC_ENABLE_TBB)
  # CONFIGURE_DEPENDS ensures that CMake reconfigures when a relevant hpp file is
  # added or removed.
  file(GLOB_RECURSE tbbpool_headers CONFIGURE_DEPENDS include/execpools/tbb/*.hpp)
  add_library(tbbpool INTERFACE)
  list(APPEND stdexec_export_targets tbbpool)
  add_library(STDEXEC::tbbpool ALIAS tbbpool)
  target_sources(tbbpool
  PUBLIC
    FILE_SET headers
    TYPE HEADERS
    BASE_DIRS include
    FILES ${tbbpool_headers}
  )
  target_compile_definitions(tbbpool INTERFACE
    STDEXEC_ENABLE_TBB
  )

  target_link_libraries(stdexec INTERFACE
    TBB::tbb
  )

  target_link_libraries(tbbpool INTERFACE
    STDEXEC::stdexec
  )

  install(TARGETS tbbpool
    EXPORT stdexec-exports
    FILE_SET headers
  )
endif()
