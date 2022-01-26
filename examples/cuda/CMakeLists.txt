cmake_minimum_required(VERSION 3.17.0)
project(CUDAExample CXX)

find_package(CUDAToolkit)

if(CUDAToolkit_FOUND)
    file(GLOB_RECURSE SCHED_SOURCES schedulers/*.hpp)
    add_library(CUDASchedulers INTERFACE ${SCHED_SOURCES})
    target_link_libraries(CUDASchedulers INTERFACE P2300 CUDA::cudart)
    target_compile_options(CUDASchedulers INTERFACE -x cuda -Wno-unknown-cuda-version)
    foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
        target_compile_options(CUDASchedulers INTERFACE --cuda-gpu-arch=sm_${arch})
    endforeach()

    add_executable(test.CUDA test/test_main.cpp test/graph/test_then.cpp)

    target_include_directories(test.CUDA PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} test)
    target_link_libraries(test.CUDA CUDASchedulers P2300 CONAN_PKG::catch2)
    catch_discover_tests(test.CUDA)
endif()
