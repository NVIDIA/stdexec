if (CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC" OR STDEXEC_ENABLE_CUDA)
    file(GLOB_RECURSE TEST_SOURCES
            RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
            CONFIGURE_DEPENDS
            *.cpp)

    add_executable(test.CUDA ${TEST_SOURCES})

    target_include_directories(test.CUDA PRIVATE test)
    target_link_libraries(test.CUDA CUDASchedulers Catch2::Catch2)
    catch_discover_tests(test.CUDA)
endif()

