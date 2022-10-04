cmake_minimum_required(VERSION 3.17.0)
project(CUDAExample CXX)

find_package(CUDAToolkit)

if (CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
  if(CUDAToolkit_FOUND)
      file(GLOB_RECURSE SCHED_SOURCES schedulers/*.cuh)
      add_library(CUDASchedulers INTERFACE ${SCHED_SOURCES})
      target_link_libraries(CUDASchedulers INTERFACE p2300 p2300_executable_flags CUDA::cudart)
      target_include_directories(CUDASchedulers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
      target_compile_features(CUDASchedulers INTERFACE cxx_std_20)

      if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(CUDASchedulers INTERFACE -x cu -Wno-unknown-cuda-version)
        foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
          target_compile_options(CUDASchedulers INTERFACE --cuda-gpu-arch=sm_${arch})
        endforeach()
      elseif (CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
        target_compile_options(CUDASchedulers INTERFACE -stdpar -gpu=nomanaged)
        target_link_options(CUDASchedulers INTERFACE -stdpar -gpu=nomanaged)
        foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
          target_compile_options(CUDASchedulers INTERFACE -gpu=cc${arch})
        endforeach()
      endif()

      # Tests
      file(GLOB_RECURSE TEST_SOURCES
              RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
              CONFIGURE_DEPENDS
              test/*.cpp)

      add_executable(test.CUDA ${TEST_SOURCES})

      target_include_directories(test.CUDA PRIVATE test)
      target_link_libraries(test.CUDA CUDASchedulers Catch2::Catch2)
      catch_discover_tests(test.CUDA)

      # Examples
      file(GLOB_RECURSE EXAMPLES
              RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
              CONFIGURE_DEPENDS
              examples/*.cpp)

      foreach(example ${EXAMPLES})
          file(TO_CMAKE_PATH "${example}" example_prefix)
          string(REPLACE "/" "." example_prefix "${example_prefix}")
          get_filename_component(example_name "${example_prefix}" NAME_WLE)

          add_executable(${example_name} ${example})
          target_link_libraries(${example_name} PRIVATE CUDASchedulers)

          get_filename_component(PARENT_DIR ../ ABSOLUTE)
          target_include_directories(${example_name} PRIVATE ${PARENT_DIR})
      endforeach()
  endif()
endif()

