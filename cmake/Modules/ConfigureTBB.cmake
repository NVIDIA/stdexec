option(STDEXEC_ENABLE_TBB "Enable TBB targets" OFF)

include(CheckIncludeFileCXX)
rapids_find_package(
  TBB
  QUIET
  COMPONENTS
  tbb
  BUILD_EXPORT_SET
  stdexec-exports
  INSTALL_EXPORT_SET
  stdexec-exports)

# If TBB is available and the stdlib is libstdc++, then a #include of
# <execution> will pull in TBB headers, creating a link-time dependency on TBB.
if(TBB_FOUND)
  if(NOT STDEXEC_ENABLE_TBB)
    include(CheckCXXSymbolExists)
    check_cxx_symbol_exists(__GLIBCXX__ "ciso646" _glibcxx_version_defined)
    if(_glibcxx_version_defined)
      set(STDEXEC_ENABLE_TBB ON)
      message(
        STATUS
          "Enabling TBB support in stdexec because <tbb/tbb.h> is available and libstdc++ is used."
      )
    endif()
  endif()
else()
  set(STDEXEC_ENABLE_TBB OFF)
endif()

# Some distro-packaged TBB configs (e.g., Debian/Ubuntu's libtbb-dev) list
# /usr/include in INTERFACE_INCLUDE_DIRECTORIES. That makes CMake pass
# `-isystem /usr/include`, which precedes the libstdc++ header directories and
# breaks the `#include_next <math.h>` in <cmath> (seen with clang's CUDA mode).
# System directories are already on the implicit include path, so strip them.
if(TBB_FOUND)
  foreach(_tbb_target tbb tbbmalloc tbbmalloc_proxy tbbbind_2_5 tbbbind_2_0
                      tbbbind)
    if(TARGET TBB::${_tbb_target})
      get_target_property(_tbb_incs TBB::${_tbb_target}
                          INTERFACE_INCLUDE_DIRECTORIES)
      if(_tbb_incs)
        list(REMOVE_ITEM _tbb_incs /usr/include /usr/local/include)
        set_property(TARGET TBB::${_tbb_target} PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                                         "${_tbb_incs}")
      endif()
    endif()
  endforeach()
endif()

# TBB does not sanitize cleanly. Disable TBB support if any sanitizer is
# enabled.
string(FIND "${CMAKE_CXX_FLAGS}" "-fsanitize" _fsanitize_flag_index)
if(_fsanitize_flag_index GREATER -1)
  message(
    STATUS "Disabling TBB support in stdexec because sanitizers are enabled.")
  set(STDEXEC_ENABLE_TBB OFF)
endif()

if(STDEXEC_ENABLE_TBB)
  # CONFIGURE_DEPENDS ensures that CMake reconfigures when a relevant hpp file
  # is added or removed.
  file(GLOB_RECURSE tbbexec_headers CONFIGURE_DEPENDS include/exec/tbb/*.hpp)
  add_library(tbbexec INTERFACE)
  list(APPEND stdexec_export_targets tbbexec)
  add_library(STDEXEC::tbbexec ALIAS tbbexec)

  # These aliases are provided for backwards compatibility with the old target
  # names
  add_library(tbbpool ALIAS tbbexec)
  add_library(STDEXEC::tbbpool ALIAS tbbexec)

  target_sources(
    tbbexec
    PUBLIC FILE_SET
           headers
           TYPE
           HEADERS
           BASE_DIRS
           include
           FILES
           ${tbbexec_headers})
  target_compile_definitions(tbbexec INTERFACE STDEXEC_ENABLE_TBB)

  target_link_libraries(stdexec INTERFACE TBB::tbb)

  target_link_libraries(tbbexec INTERFACE STDEXEC::stdexec)

  if(STDEXEC_INSTALL)
    install(
      TARGETS tbbexec
      EXPORT stdexec-exports
      FILE_SET headers)
  endif()
endif()
