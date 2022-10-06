if (CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC" OR P2300_ENABLE_CUDA)
  if(CUDAToolkit_FOUND)
      file(GLOB_RECURSE EXAMPLES
              RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
              CONFIGURE_DEPENDS
              *.cpp)

      foreach(example ${EXAMPLES})
          file(TO_CMAKE_PATH "${example}" example_prefix)
          string(REPLACE "/" "." example_prefix "${example_prefix}")
          get_filename_component(example_name "${example_prefix}" NAME_WLE)

          add_executable(${example_name} ${example})
          target_link_libraries(${example_name} PRIVATE CUDASchedulers)

          target_include_directories(${example_name} PRIVATE {CMAKE_CURRENT_LIST_DIR})
      endforeach()
  endif()
endif()

