if(STDEXEC_ENABLE_CUDA)
    file(GLOB_RECURSE TEST_SOURCES
            RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
            CONFIGURE_DEPENDS
            *.cpp)

    add_executable(test.CUDA ${TEST_SOURCES})
    if(NOT (CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC"))
        set_source_files_properties(${TEST_SOURCES} PROPERTIES LANGUAGE CUDA)
    endif()
    target_include_directories(test.CUDA PRIVATE ${STDEXEC_SOURCE_DIR}/test)
    target_link_libraries(test.CUDA CUDASchedulers Catch2::Catch2)
    catch_discover_tests(test.CUDA)
endif()
