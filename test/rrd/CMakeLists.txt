include(FetchContent)

# Relacy has its own runtime checks and does not compose with compiler sanitizers.
# Strip any -fsanitize* flags in this directory so Relacy-linked targets run cleanly.
function(_stdexec_strip_sanitize_flags out_var in_value)
  separate_arguments(_flags UNIX_COMMAND "${in_value}")
  list(FILTER _flags EXCLUDE REGEX "^-fsanitize")
  string(JOIN " " _joined ${_flags})
  set(${out_var} "${_joined}" PARENT_SCOPE)
endfunction()

foreach(_cfg "" "_DEBUG" "_RELEASE" "_RELWITHDEBINFO" "_MINSIZEREL")
  set(_var "CMAKE_CXX_FLAGS${_cfg}")
  if(DEFINED ${_var} AND NOT "${${_var}}" STREQUAL "")
    _stdexec_strip_sanitize_flags(_stripped "${${_var}}")
    set(${_var} "${_stripped}")
  endif()
endforeach()

FetchContent_Declare(
  relacy
  GIT_REPOSITORY https://github.com/dvyukov/relacy
  GIT_TAG master
)

FetchContent_Populate(relacy)

add_library(relacy INTERFACE)
target_include_directories(relacy INTERFACE
  $<BUILD_INTERFACE:${relacy_SOURCE_DIR}/relacy/fakestd>
  $<BUILD_INTERFACE:${relacy_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
)

function(add_relacy_test target_name)
  add_executable(${target_name} ${target_name}.cpp)

  target_link_libraries(${target_name} PRIVATE relacy)
  # Disable TBB - Relacy simulates all threads with fibers anyway...
  target_compile_definitions(${target_name} PRIVATE _GLIBCXX_USE_TBB_PAR_BACKEND=0)
  target_include_directories(${target_name} PRIVATE ../../include)
  target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  set_target_properties(${target_name} PROPERTIES
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF)

  add_test(NAME relacy-${target_name} COMMAND ${target_name})
  set_tests_properties(relacy-${target_name} PROPERTIES TIMEOUT 120)
endfunction()

set(relacy_tests
  async_scope
  bwos_lifo_queue
  intrusive_mpsc_queue
  split
  sync_wait
)

foreach(test ${relacy_tests})
  add_relacy_test(${test})
endforeach()

# Target to build all relacy tests
add_custom_target(relacy-tests DEPENDS ${relacy_tests})
