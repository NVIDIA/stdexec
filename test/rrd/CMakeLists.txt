include(FetchContent)

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
endfunction()

set(relacy_tests
  async_scope
  intrusive_mpsc_queue
  split
  sync_wait
)

foreach(test ${relacy_tests})
  add_relacy_test(${test})
endforeach()

# Target to build all relacy tests
add_custom_target(relacy-tests DEPENDS ${relacy_tests})
